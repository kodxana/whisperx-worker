import os
import shutil
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output

MODEL = Predictor()
MODEL.setup()

def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            print(f"Removed job directory: {job_path}")
        except Exception as e:
            print(f"Error removing job directory {job_path}: {str(e)}")
    else:
        print(f"Job directory not found: {job_path}")

def run(job):
    job_input = job['input']
    job_id = job['id']
    # Input validation
    validated_input = validate(job_input, INPUT_VALIDATIONS)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    
    # Download audio file
    audio_file_path = download_files_from_urls(job['id'], [job_input['audio_file']])[0]
    
    # Prepare input for prediction
    predict_input = {
        'audio_file': audio_file_path,
        'language': job_input.get('language'),
        'language_detection_min_prob': job_input.get('language_detection_min_prob', 0),
        'language_detection_max_tries': job_input.get('language_detection_max_tries', 5),
        'initial_prompt': job_input.get('initial_prompt'),
        'batch_size': job_input.get('batch_size', 64),
        'temperature': job_input.get('temperature', 0),
        'vad_onset': job_input.get('vad_onset', 0.500),
        'vad_offset': job_input.get('vad_offset', 0.363),
        'align_output': job_input.get('align_output', False),
        'diarization': job_input.get('diarization', False),
        'huggingface_access_token': job_input.get('huggingface_access_token'),
        'min_speakers': job_input.get('min_speakers'),
        'max_speakers': job_input.get('max_speakers'),
        'debug': job_input.get('debug', False)
    }
    
    # Run prediction
    try:
        result = MODEL.predict(**predict_input)
        
        # Convert Output model to dict for JSON serialization
        output_dict = {
            "segments": result.segments,
            "detected_language": result.detected_language
        }
        
        # Cleanup downloaded files
        rp_cleanup.clean(['input_objects'])
        cleanup_job_files(job_id)
        
        return output_dict
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": run})
