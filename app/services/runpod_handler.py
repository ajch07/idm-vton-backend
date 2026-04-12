"""
RunPod serverless handler for FLUX-based virtual try-on.

Uses black-forest-labs/FLUX.1-Fill-dev (official Flux inpainting model).
Requires HF_TOKEN env var (FLUX.1-Fill-dev is a gated model).

Verified against:
  - diffusers 0.37.1 docs: FluxFillPipeline API
  - FLUX.1-Fill-dev model card: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
  - PyTorch 2.4 (RunPod base image)
"""

# ============================================================
# CRITICAL: Block flash_attn BEFORE any torch/diffusers imports
# ============================================================
# RunPod PyTorch 2.4 base image ships flash_attn pre-installed.
# flash_attn uses PEP 604 string annotations (e.g. q: 'torch.Tensor')
# which crash PyTorch 2.4 infer_schema() at import time.
# Setting sys.modules entries to None makes import flash_attn
# raise ImportError. diffusers detects this and falls back to
# PyTorch native SDPA attention (same quality, no crash).
import sys

sys.modules["flash_attn"] = None
sys.modules["flash_attn.flash_attn_interface"] = None
sys.modules["flash_attn.bert_padding"] = None
sys.modules["flash_attn.flash_attn_triton"] = None
sys.modules["flash_attn.ops"] = None
sys.modules["flash_attn.ops.fused_dense"] = None
# ============================================================

import base64
import io
import os
import time

import torch
from PIL import Image, ImageDraw

# --- HF Token: read from env and register globally ---
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
print(f"[TryOn] HF_TOKEN present: {bool(_hf_token)} (len={len(_hf_token) if _hf_token else 0})")

if _hf_token:
    # Register token globally so ALL huggingface_hub calls use it automatically
    try:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
        print("[TryOn] HuggingFace login successful")
    except Exception as e:
        print(f"[TryOn] HuggingFace login warning: {e}")
else:
    print("[TryOn] WARNING: No HF_TOKEN found! Gated model download will fail.")

MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
TARGET_HEIGHT = 1024
TARGET_WIDTH = 768


class FluxTryOnInference:
    """FLUX.1-Fill-dev inpainting for virtual try-on."""

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load_model(self):
        if self.pipe is not None:
            return

        print(f"[TryOn] Loading FLUX.1-Fill-dev... (device={self.device})")
        t0 = time.time()

        # Lazy import - runs AFTER flash_attn is blocked above
        from diffusers import FluxFillPipeline

        self.pipe = FluxFillPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=_hf_token,  # also passed explicitly as backup
        )
        # CPU offload keeps VRAM safe on RTX 4090 (24 GB)
        self.pipe.enable_model_cpu_offload()

        print(f"[TryOn] Model loaded in {time.time() - t0:.1f}s")

    def _make_upper_body_mask(self, width: int, height: int) -> Image.Image:
        """White (255) = inpaint (clothing area), Black (0) = keep."""
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(
            [
                int(width * 0.1),
                int(height * 0.10),
                int(width * 0.9),
                int(height * 0.65),
            ],
            fill=255,
        )
        return mask

    async def generate(
        self,
        person_image: Image.Image,
        garment_desc: str = "a garment",
        num_steps: int = 28,
        guidance_scale: float = 30.0,
        seed: int = 42,
    ) -> Image.Image:
        if self.pipe is None:
            await self.load_model()

        t0 = time.time()

        person_image = person_image.resize(
            (TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS
        )
        mask = self._make_upper_body_mask(TARGET_WIDTH, TARGET_HEIGHT)

        prompt = (
            f"a photo of a person wearing {garment_desc}, "
            f"high quality, realistic, well-fitted clothing, fashion photography"
        )

        generator = torch.Generator("cpu").manual_seed(seed)

        # FluxFillPipeline.__call__ parameters verified against diffusers 0.37.1 docs:
        # prompt, image, mask_image, height, width, num_inference_steps,
        # guidance_scale (default 30.0), max_sequence_length (default 512), generator
        result = self.pipe(
            prompt=prompt,
            image=person_image,
            mask_image=mask,
            height=TARGET_HEIGHT,
            width=TARGET_WIDTH,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            generator=generator,
        ).images[0]

        print(f"[TryOn] Generated in {time.time() - t0:.1f}s ({num_steps} steps)")
        return result


# ---------------------------------------------------------------------------
# Singleton - models persist across requests on same RunPod worker
# ---------------------------------------------------------------------------
_handler_instance = None


class TryOnHandler:
    def __init__(self):
        self.inference = FluxTryOnInference()

    @staticmethod
    def _decode_image(b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    @staticmethod
    def _encode_image(img: Image.Image, quality: int = 90) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    async def handle(self, event: dict) -> dict:
        try:
            start = time.time()

            user_b64 = event.get("user_image_base64") or event.get("user_image")
            if not user_b64:
                return {"success": False, "error": "Missing user_image"}

            garment_name = event.get("garment_name", "a garment")
            garment_desc = event.get("prompt") or garment_name
            num_steps = min(int(event.get("num_steps", 28)), 50)
            seed = int(event.get("seed", 42))

            person_image = self._decode_image(user_b64)

            print(f"[TryOn] Generating | garment={event.get('garment_id', 'unknown')}")

            result = await self.inference.generate(
                person_image=person_image,
                garment_desc=garment_desc,
                num_steps=num_steps,
                seed=seed,
            )

            return {
                "success": True,
                "image_base64": self._encode_image(result),
                "model_used": "flux-fill-dev",
                "processing_time_ms": int((time.time() - start) * 1000),
                "garment_id": event.get("garment_id", ""),
                "garment_name": garment_name,
            }

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}


async def async_runpod_handler(job):
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = TryOnHandler()
    return await _handler_instance.handle(job["input"])


if __name__ == "__main__":
    import runpod

    print("[TryOn] Starting RunPod serverless handler (FLUX.1-Fill-dev)...")
    runpod.serverless.start({"handler": async_runpod_handler})
