"""
RunPod serverless handler for FLUX-based virtual try-on.

Uses black-forest-labs/FLUX.1-Kontext-dev via Diffusers' Kontext inpaint pipeline.
Requires HF_TOKEN env var (FLUX.1-Kontext-dev is a gated model).

Verified against:
  - diffusers 0.37.1 docs: FluxKontextInpaintPipeline API
  - FLUX.1-Kontext-dev model card: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
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

from .prompt_adjuster import DEFAULT_NEGATIVE_PROMPT

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

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
# Single working canvas size for masked editing
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
    """FLUX.1-Kontext-dev inpainting for virtual try-on using garment reference images."""

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load_model(self):
        if self.pipe is not None:
            return

        print(f"[TryOn] Loading FLUX.1-Kontext-dev... (device={self.device})")
        t0 = time.time()

        from diffusers import FluxKontextInpaintPipeline

        self.pipe = FluxKontextInpaintPipeline.from_pretrained(
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
        """Build a mask for the person image.

        White (255) = inpaint, Black (0) = keep.
        """
        mask = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 0)
        draw = ImageDraw.Draw(mask)

        x0 = int(PANEL_WIDTH * 0.05)
        x1 = int(PANEL_WIDTH * 0.95)

        if region == "upper":
            y0 = int(PANEL_HEIGHT * 0.12)
            y1 = int(PANEL_HEIGHT * 0.60)
        elif region == "lower":
            y0 = int(PANEL_HEIGHT * 0.45)
            y1 = int(PANEL_HEIGHT * 0.95)
        else:  # full
            y0 = int(PANEL_HEIGHT * 0.12)
            y1 = int(PANEL_HEIGHT * 0.95)

        draw.rounded_rectangle([x0, y0, x1, y1], radius=24, fill=255)
        return mask

    @staticmethod
    def _prepare_person_image(person: Image.Image) -> Image.Image:
        return person.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.LANCZOS)

    @staticmethod
    def _prepare_reference_image(garment: Image.Image) -> Image.Image:
        return garment.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.LANCZOS)

    @staticmethod
    def _build_default_prompt(garment_name: str) -> str:
        return (
            f"Edit the input person photo so the person is wearing the exact same {garment_name} "
            f"from the reference image. Keep identity, face, hair, hands, pose, body shape, "
            f"background, framing, and lighting unchanged. Match garment color, print, silhouette, "
            f"fabric texture, fit, folds, and shadows realistically. Output a realistic ecommerce try-on photo."
        )

    async def generate(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_name: str = "a garment",
        category: str | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        num_steps: int = 28,
        guidance_scale: float = 2.5,
        seed: int = 42,
    ) -> Image.Image:
        if self.pipe is None:
            await self.load_model()

        t0 = time.time()
        region = _detect_region(garment_name, category)
        print(f"[TryOn] Region={region} | garment={garment_name}")

        source = self._prepare_person_image(person_image)
        image_reference = self._prepare_reference_image(garment_image)
        mask = self._make_mask(region)
        if hasattr(self.pipe, "mask_processor"):
            mask = self.pipe.mask_processor.blur(mask, blur_factor=12)

        prompt_text = (prompt or "").strip() or self._build_default_prompt(garment_name)
        negative_text = (negative_prompt or "").strip() or DEFAULT_NEGATIVE_PROMPT
        final_prompt = f"{prompt_text}\n\nHard constraints: {negative_text}"

        generator = torch.Generator("cpu").manual_seed(seed)

        result = self.pipe(
            prompt=final_prompt,
            image=source,
            mask_image=mask,
            image_reference=image_reference,
            height=PANEL_HEIGHT,
            width=PANEL_WIDTH,
            strength=1.0,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            generator=generator,
        ).images[0]

        print(f"[TryOn] Generated in {time.time() - t0:.1f}s ({num_steps} steps, region={region})")
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

            garment_b64 = event.get("garment_image_base64") or event.get("garment_image")
            if not garment_b64:
                return {"success": False, "error": "Missing garment_image"}

            garment_name = event.get("garment_name", "a garment")
            category = event.get("category")
            prompt = event.get("prompt")
            negative_prompt = event.get("negative_prompt")
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
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=num_steps,
                seed=seed,
            )

            return {
                "success": True,
                "image_base64": self._encode_image(result),
                "model_used": "flux-kontext-dev",
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

    print("[TryOn] Starting RunPod serverless handler (FLUX.1-Kontext-dev)...")
    runpod.serverless.start({"handler": async_runpod_handler})
