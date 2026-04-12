# RunPod pre-built image: Python 3.11, PyTorch 2.4, CUDA 12.4
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System libs for Pillow/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Phase 1: Remove flash_attn from base image BEFORE pip install ----
# The base image ships flash_attn whose PEP604 annotations crash PyTorch 2.4 infer_schema.
RUN pip uninstall -y flash_attn flash-attn 2>/dev/null || true && \
    find / -name '*flash_attn*' -exec rm -rf {} + 2>/dev/null || true && \
    find / -name '*flash-attn*' -exec rm -rf {} + 2>/dev/null || true && \
    echo "Phase 1: flash_attn removed from base image"

# ---- Phase 2: Install Python deps ----
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 600 -r requirements-runpod.txt

# ---- Phase 3: Remove flash_attn AGAIN after pip install ----
# In case any dependency silently pulled it back in.
RUN pip uninstall -y flash_attn flash-attn 2>/dev/null || true && \
    find / -name '*flash_attn*' -exec rm -rf {} + 2>/dev/null || true && \
    find / -name '*flash-attn*' -exec rm -rf {} + 2>/dev/null || true && \
    echo "Phase 3: flash_attn post-install cleanup done"

# ---- Phase 4: Create .pth startup blocker ----
# .pth files in site-packages execute during site.py init — BEFORE any user code,
# before torch loads, before diffusers loads. This is the earliest possible block.
RUN python -c "\
import site; \
pth = site.getsitepackages()[0] + '/00_block_fa.pth'; \
open(pth, 'w').write('import sys; sys.modules.update({m: None for m in [\"flash_attn\", \"flash_attn.flash_attn_interface\", \"flash_attn.bert_padding\", \"flash_attn.flash_attn_triton\", \"flash_attn.ops\", \"flash_attn.ops.fused_dense\", \"flash_attn_2_cuda\", \"flash_attn_cuda\", \"flash_attn_3_cuda\"]})\n'); \
print('Phase 4: Created startup blocker at ' + pth)"

# ---- Phase 5: Verify ----
RUN echo '--- FLASH_ATTN VERIFICATION ---' && \
    (pip list 2>/dev/null | grep -i flash && echo 'WARN: flash_attn in pip list' || echo 'OK: not in pip list') && \
    (find / -name '*flash_attn*.so' 2>/dev/null | head -5 | grep . && echo 'WARN: .so files found on disk' || echo 'OK: no .so files on disk') && \
    (python -c "import flash_attn" 2>/dev/null && echo 'WARN: import succeeded!' || echo 'OK: import blocked') && \
    echo '--- END VERIFICATION ---'

# Copy application code
COPY app/ ./app/

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
RUN mkdir -p /app/hf_cache

# Start FLUX try-on RunPod handler
CMD ["python", "-m", "app.services.runpod_handler"]
