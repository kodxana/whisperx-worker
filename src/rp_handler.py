import base64
import os
import sys
import shutil
import logging
import tempfile

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login, whoami
import torch
import numpy as np
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output
from speaker_profiles import load_embeddings, relabel
from speaker_processing import (
    process_diarized_output, enroll_profiles, identify_speakers_on_segments,
    load_known_speakers_from_samples, identify_speaker, relabel_speakers_by_avg_similarity,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Hugging Face authentication
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())
raw_token = os.environ.get("HF_TOKEN", "")
hf_token = raw_token.strip()

if hf_token and not hf_token.startswith("hf_"):
    logger.warning("HF_TOKEN does not start with 'hf_' prefix - token may be malformed")

if hf_token:
    try:
        logger.debug(f"HF_TOKEN Loaded: {repr(hf_token[:10])}...")
        login(token=hf_token, add_to_git_credential=False)
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face Authenticated as: {user['name']}")
    except Exception as e:
        logger.error("Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.warning("No Hugging Face token found in HF_TOKEN environment variable.")




MODEL = Predictor()
MODEL.setup()

DATA_URI_AUDIO_EXTENSIONS = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/webm": ".webm",
}


def audio_suffix_from_data_uri(audio_input):
    if not audio_input.startswith("data:") or "," not in audio_input:
        return ".wav"

    media_type = audio_input[5:audio_input.find(",")].split(";", 1)[0].lower()
    return DATA_URI_AUDIO_EXTENSIONS.get(media_type, ".wav")


def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception as e:
            logger.error(f"Error removing job directory {job_path}: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Job directory not found: {job_path}")


def cleanup_worker_state(job_id):
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

# --------------------------------------------------------------------
# main serverless entry-point
# --------------------------------------------------------------------
def run(job):
    job_id     = job["id"]
    job_input  = job["input"]

    # ------------- validate basic schema ----------------------------
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    request_hf_token = job_input.get("huggingface_access_token")
    if isinstance(request_hf_token, str):
        request_hf_token = request_hf_token.strip()
    effective_hf_token = request_hf_token or hf_token

    # ------------- 1) resolve audio input (URL or base64) -----------
    audio_input = job_input["audio_file"]
    try:
        if "://" in audio_input:
            # Standard URL — download as before
            audio_file_path = download_files_from_urls(job_id, [audio_input])[0]
            logger.debug(f"Audio downloaded → {audio_file_path}")
        else:
            # Treat as base64-encoded audio data
            # Strip optional data-URI prefix (e.g. "data:audio/wav;base64,")
            audio_suffix = audio_suffix_from_data_uri(audio_input)
            if "," in audio_input:
                audio_input = audio_input.split(",", 1)[1]
            audio_bytes = base64.b64decode(audio_input)
            os.makedirs(f"/jobs/{job_id}", exist_ok=True)
            audio_file_path = f"/jobs/{job_id}/audio_input{audio_suffix}"
            with open(audio_file_path, "wb") as f:
                f.write(audio_bytes)
            logger.debug(f"Audio decoded from base64 → {audio_file_path} ({len(audio_bytes)} bytes)")
    except Exception as e:
        logger.error("Audio input failed", exc_info=True)
        cleanup_worker_state(job_id)
        return {"error": f"audio input: {e}"}

    # ------------- 2) download speaker profiles (optional) ----------
    speaker_profiles = job_input.get("speaker_samples", [])
    speaker_verification = job_input.get("speaker_verification", False)
    embeddings = {}
    if speaker_verification and speaker_profiles:
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_profiles,
                huggingface_access_token=effective_hf_token
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)
            embeddings = {}  # graceful degradation: proceed without profiles
    elif speaker_profiles:
        logger.info("speaker_samples provided but speaker_verification is false; skipping speaker enrollment.")

    # ------------- 3) call WhisperX / VAD / diarization -------------
    predict_input = {
        "audio_file"               : audio_file_path,
        "language"                 : job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt"           : job_input.get("initial_prompt"),
        "batch_size"               : job_input.get("batch_size", 64),
        "temperature"              : job_input.get("temperature", 0),
        "vad_onset"                : job_input.get("vad_onset", 0.50),
        "vad_offset"               : job_input.get("vad_offset", 0.363),
        "align_output"             : job_input.get("align_output", False),
        "diarization"              : job_input.get("diarization", False),
        "huggingface_access_token" : effective_hf_token,
        "min_speakers"             : job_input.get("min_speakers"),
        "max_speakers"             : job_input.get("max_speakers"),
        "debug"                    : job_input.get("debug", False),
    }

    try:
        result = MODEL.predict(**predict_input)             # <-- heavy job
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        cleanup_worker_state(job_id)
        return {"error": f"prediction: {e}"}

    output_dict = {
        "segments"         : result.segments,
        "detected_language": result.detected_language
    }
    # ------------------------------------------------embedding-info----------------
    # 4) speaker verification (optional)
    if embeddings:
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.1  # Adjust threshold as needed
            )
            #output_dict["segments"] = segments_with_speakers
            segments_with_final_labels = relabel_speakers_by_avg_similarity(segments_with_speakers)
            output_dict["segments"] = segments_with_final_labels
            logger.info("Speaker identification completed successfully.")
        except Exception as e:
            logger.error("Speaker identification failed", exc_info=True)
            output_dict["warning"] = f"Speaker identification skipped: {e}"
    else:
        logger.info("No enrolled embeddings available; skipping speaker identification.")

    # 4-Cleanup and return output_dict normally
    cleanup_worker_state(job_id)

    return output_dict

runpod.serverless.start({"handler": run})
