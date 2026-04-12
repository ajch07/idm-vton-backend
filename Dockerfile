# RunPod pre-built image: Python 3.11, PyTorch 2.4, CUDA 12.4
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System libs for Pillow/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Remove flash_attn if pre-installed — PEP604 annotations crash PyTorch 2.4 infer_schema
# Belt-and-suspenders: pip uninstall + delete files + handler also blocks at Python level
RUN pip uninstall -y flash_attn flash-attn 2>/dev/null; \
    find / -type d -name "flash_attn" 2>/dev/null | xargs -r rm -rf 2>/dev/null; \
    echo "flash_attn cleanup done"

# Install Python deps
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
