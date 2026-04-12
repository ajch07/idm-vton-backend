# RunPod pre-built image: Python 3.11, PyTorch 2.4, CUDA 12.4
# torch, torchvision, torchaudio, CUDA all pre-installed (~5GB saved)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System libs for Pillow/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# NO xformers needed — PyTorch 2.4 has built-in SDPA (same perf on RTX 4090)
# This saves 900MB+ download and avoids timeout issues

# Install Python deps with generous retry/timeout for large downloads
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 600 -r requirements-runpod.txt

# Copy application code
COPY app/ ./app/

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
RUN mkdir -p /app/hf_cache

# Start FLUX try-on RunPod handler
CMD ["python", "-m", "app.services.runpod_handler"]
