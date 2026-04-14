# RunPod pre-built image: Python 3.11, PyTorch 2.4, CUDA 12.4
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System libs for Pillow/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Phase 1: Remove flash_attn + upgrade PyTorch ----
# ROOT CAUSE: PyTorch 2.4's infer_schema() cannot parse PEP 604 string annotations
# like (q: 'torch.Tensor', softmax_scale: 'float | None').
# flash_attn (baked into base image's torch registry) uses these annotations.
# Removing flash_attn files alone doesn't help — the ops are registered in torch itself.
# PyTorch 2.5+ natively handles PEP 604 string annotations. This is THE fix.
RUN pip uninstall -y flash_attn flash-attn 2>/dev/null || true && \
    find / -name '*flash_attn*' -exec rm -rf {} + 2>/dev/null || true && \
    find / -name '*flash-attn*' -exec rm -rf {} + 2>/dev/null || true && \
    pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    echo "Phase 1: flash_attn removed + PyTorch upgraded to 2.5.1"

# ---- Phase 2: Install Python deps ----
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 600 -r requirements-runpod.txt

# ---- Phase 3: Post-install cleanup + .pth blocker (safety net) ----
RUN pip uninstall -y flash_attn flash-attn 2>/dev/null || true && \
    find / -name '*flash_attn*' -exec rm -rf {} + 2>/dev/null || true && \
    python -c "\
import site; \
pth = site.getsitepackages()[0] + '/00_block_fa.pth'; \
open(pth, 'w').write('import sys; sys.modules.update({m: None for m in [\"flash_attn\", \"flash_attn.flash_attn_interface\", \"flash_attn.bert_padding\", \"flash_attn.flash_attn_triton\", \"flash_attn.ops\", \"flash_attn.ops.fused_dense\", \"flash_attn_2_cuda\", \"flash_attn_cuda\", \"flash_attn_3_cuda\"]})\n')"

# ---- Phase 4: BUILD-TIME IMPORT TEST ----
# This is the critical test — if FluxKontextInpaintPipeline can't be imported, the build FAILS here
# instead of failing silently at runtime on RunPod.
RUN python -c "\
import torch; print(f'PyTorch {torch.__version__}'); \
from diffusers import FluxKontextInpaintPipeline; \
print('BUILD VERIFIED: FluxKontextInpaintPipeline imports successfully')"

# Copy application code
COPY app/ ./app/

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
RUN mkdir -p /app/hf_cache

# Start FLUX try-on RunPod handler
CMD ["python", "-m", "app.services.runpod_handler"]
