FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache
ENV TORCH_HOME=/runpod-volume/torch_cache
ENV TOPAUDIO_BASE_DIR=/runpod-volume/topaudio_ai
ENV MODEL_ID=Qwen/Qwen2-Audio-7B-Instruct

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg libsndfile1 curl wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
