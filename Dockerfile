FROM runpod/pytorch:3.10-2.0.0-117

SHELL ["/bin/bash", "-c"]
WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create cache directory
RUN mkdir -p /cache/models

# Copy only requirements file first to leverage Docker cache
COPY builder/requirements.txt /builder/requirements.txt

# Install Python dependencies (Worker Template)
RUN pip install --upgrade pip && \
    pip install -r /builder/requirements.txt

# Download VAD model
RUN python -c "import whisperx; from whisperx.vad import load_vad_model; load_vad_model('cpu')"

# Copy the rest of the builder files
COPY builder /builder

# Download Faster Whisper Models
RUN chmod +x /builder/download_models.sh
RUN --mount=type=cache,target=/cache/models \
    /builder/download_models.sh

# Copy source code
COPY src .

CMD [ "python", "-u", "/rp_handler.py" ]