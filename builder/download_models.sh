#!/bin/bash

set -e

CACHE_DIR="/cache/models"
MODELS_DIR="/models"

mkdir -p /root/.cache/torch/hub/checkpoints

download() {
  local file_url="$1"
  local destination_path="$2"
  local cache_path="${CACHE_DIR}/${destination_path##*/}"

  mkdir -p "$(dirname "$cache_path")"
  mkdir -p "$(dirname "$destination_path")"
  
  if [ ! -e "$cache_path" ]; then
    echo "Downloading $file_url to cache..."
    wget -O "$cache_path" "$file_url"
  else
    echo "Using cached version of ${cache_path##*/}"
  fi

  cp "$cache_path" "$destination_path"
}

faster_whisper_model_dir="${MODELS_DIR}/faster-whisper-large-v3"
mkdir -p $faster_whisper_model_dir

download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json" "$faster_whisper_model_dir/config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin" "$faster_whisper_model_dir/model.bin"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json" "$faster_whisper_model_dir/preprocessor_config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json" "$faster_whisper_model_dir/tokenizer.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json" "$faster_whisper_model_dir/vocabulary.json"

vad_model_dir="${MODELS_DIR}/vad"
mkdir -p $vad_model_dir

download $(python3 /builder/get_vad_model_url.py) "$vad_model_dir/whisperx-vad-segmentation.bin"
download "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth" "/root/.cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='speechbrain/spkrec-ecapa-voxceleb')
"

echo "All models downloaded successfully."