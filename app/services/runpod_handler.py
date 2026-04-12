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
# Single panel size — composite will be 2x this width
PANEL_HEIGHT = 768
PANEL_WIDTH = 576

# Keywords to detect garment region
_LOWER_BODY_KW = {"skirt", "pant", "pants", "trouser", "trousers", "jeans", "shorts",
                   "legging", "leggings", "bottom", "bottoms", "palazzo", "culottes"}
_FULL_BODY_KW = {"dress", "gown", "jumpsuit", "romper", "saree", "sari", "suit",
                  "onepiece", "one-piece", "co-ord", "coord"}


def _detect_region(garment_name: str, category: str | None = None) -> str:
    """Return 'upper', 'lower', or 'full' based on garment name/category."""
    text = f"{garment_name} {category or ''}".lower()
    tokens = set(text.replace("-", " ").split())
    if tokens & _LOWER_BODY_KW:
        return "lower"
    if tokens & _FULL_BODY_KW:
        return "full"
    return "upper"


class FluxTryOnInference:
    """FLUX.1-Fill-dev inpainting for virtual try-on using composite image technique."""

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load_model(self):
        if self.pipe is not None:
            return

        print(f"[TryOn] Loading FLUX.1-Fill-dev... (device={self.device})")
        t0 = time.time()

        from diffusers import FluxFillPipeline

        self.pipe = FluxFillPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=_hf_token,
        )
        # Sequential offload: each layer moves to GPU one at a time — slower but fits in 24GB
        self.pipe.enable_sequential_cpu_offload()
        # VAE optimizations to reduce peak VRAM
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        print(f"[TryOn] Model loaded in {time.time() - t0:.1f}s")

    @staticmethod
    def _make_mask(region: str) -> Image.Image:
        """Build a mask for the composite image (garment left | person right).

        White (255) = inpaint, Black (0) = keep.
        Left half is ALWAYS black (keep garment reference visible).
        Right half is white only over the clothing area.
        """
        w = PANEL_WIDTH * 2  # full composite width
        h = PANEL_HEIGHT
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        # Right-half clothing region coordinates
        rx0 = PANEL_WIDTH + int(PANEL_WIDTH * 0.05)
        rx1 = PANEL_WIDTH + int(PANEL_WIDTH * 0.95)

        if region == "upper":
            ry0 = int(h * 0.12)
            ry1 = int(h * 0.60)
        elif region == "lower":
            ry0 = int(h * 0.45)
            ry1 = int(h * 0.95)
        else:  # full
            ry0 = int(h * 0.12)
            ry1 = int(h * 0.95)

        draw.rectangle([rx0, ry0, rx1, ry1], fill=255)
        return mask

    @staticmethod
    def _build_composite(
        person: Image.Image, garment: Image.Image
    ) -> Image.Image:
        """Side-by-side: [garment | person] so FLUX sees the reference garment."""
        garment_panel = garment.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.LANCZOS)
        person_panel = person.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.LANCZOS)
        composite = Image.new("RGB", (PANEL_WIDTH * 2, PANEL_HEIGHT))
        composite.paste(garment_panel, (0, 0))
        composite.paste(person_panel, (PANEL_WIDTH, 0))
        return composite

    async def generate(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_name: str = "a garment",
        category: str | None = None,
        num_steps: int = 28,
        guidance_scale: float = 30.0,
        seed: int = 42,
    ) -> Image.Image:
        if self.pipe is None:
            await self.load_model()

        t0 = time.time()
        region = _detect_region(garment_name, category)
        print(f"[TryOn] Region={region} | garment={garment_name}")

        composite = self._build_composite(person_image, garment_image)
        mask = self._make_mask(region)

        prompt = (
            f"The person on the right is wearing the exact same garment shown on the left. "
            f"The garment is a {garment_name}. Preserve the person's face, hair, pose, and "
            f"background exactly. The clothing fits naturally, with realistic fabric texture, "
            f"folds, and shadows. High quality fashion photography."
        )

        generator = torch.Generator("cpu").manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=composite,
            mask_image=mask,
            height=PANEL_HEIGHT,
            width=PANEL_WIDTH * 2,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            generator=generator,
        ).images[0]

        # Crop right half — that's the person with new garment
        output = result.crop((PANEL_WIDTH, 0, PANEL_WIDTH * 2, PANEL_HEIGHT))

        print(f"[TryOn] Generated in {time.time() - t0:.1f}s ({num_steps} steps, region={region})")
        return output


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

            garment_b64 = event.get("garment_image_base64") or event.get("garment_image")
            if not garment_b64:
                return {"success": False, "error": "Missing garment_image"}

            garment_name = event.get("garment_name", "a garment")
            category = event.get("category")
            num_steps = min(int(event.get("num_steps", 28)), 50)
            seed = int(event.get("seed", 42))

            person_image = self._decode_image(user_b64)
            garment_image = self._decode_image(garment_b64)

            print(f"[TryOn] Generating | garment={event.get('garment_id', 'unknown')} | name={garment_name}")

            result = await self.inference.generate(
                person_image=person_image,
                garment_image=garment_image,
                garment_name=garment_name,
                category=category,
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
