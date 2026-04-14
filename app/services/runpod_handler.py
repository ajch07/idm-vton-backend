"""
RunPod serverless handler for FireRed-based virtual try-on.

Uses FireRedTeam/FireRed-Image-Edit-1.1 via Diffusers' QwenImageEditPlusPipeline.
Verified against:
  - FireRed official inference example using QwenImageEditPlusPipeline
  - FireRed model card on Hugging Face
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
from PIL import Image

from .prompt_adjuster import build_runpod_prompt
from .tryon_interface import TryOnMetadata

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
    print("[TryOn] WARNING: No HF_TOKEN found! Model download may fail for gated assets.")

MODEL_ID = "FireRedTeam/FireRed-Image-Edit-1.1"
TARGET_HEIGHT = 1024
TARGET_WIDTH = 768


class FireRedTryOnInference:
    """FireRed-Image-Edit-1.1 editing pipeline for virtual try-on."""

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load_model(self):
        if self.pipe is not None:
            return

        print(f"[TryOn] Loading FireRed-Image-Edit-1.1... (device={self.device})")
        t0 = time.time()

        from diffusers import QwenImageEditPlusPipeline

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=_hf_token,
        )
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        print(f"[TryOn] Model loaded in {time.time() - t0:.1f}s")

    @staticmethod
    def _prepare_person_image(person: Image.Image) -> Image.Image:
        return person.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

    async def generate(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_name: str = "a garment",
        category: str | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        num_steps: int = 40,
        true_cfg_scale: float = 4.0,
        seed: int = 42,
    ) -> Image.Image:
        if self.pipe is None:
            await self.load_model()

        t0 = time.time()
        print(f"[TryOn] Garment={garment_name} | category={category}")

        person = self._prepare_person_image(person_image)
        images = [person, garment_image.convert("RGB")]
        prompt_text = (prompt or "").strip()
        negative_text = (negative_prompt or "").strip() or " "
        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            image=images,
            prompt=prompt_text,
            negative_prompt=negative_text,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_steps,
            num_images_per_prompt=1,
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
        self.inference = FireRedTryOnInference()

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
            prompt, negative_prompt = build_runpod_prompt(
                TryOnMetadata(
                    garment_name=garment_name,
                    garment_id=event.get("garment_id", ""),
                    user_prompt=event.get("prompt"),
                    user_negative_prompt=event.get("negative_prompt"),
                    category=category,
                )
            )
            num_steps = min(int(event.get("num_steps", 40)), 50)
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
                "model_used": "firered-image-edit-1.1",
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

    print("[TryOn] Starting RunPod serverless handler (FireRed-Image-Edit-1.1)...")
    runpod.serverless.start({"handler": async_runpod_handler})
