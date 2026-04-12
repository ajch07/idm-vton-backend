"""
RunPod serverless handler for IDM-VTON virtual try-on.

Uses the proper IDM-VTON pipeline from yisol/IDM-VTON with:
  - Custom UNet (tryon) with 13-channel input (latent + mask + masked_img + pose)
  - Garment encoder UNet (garmnet) for clothing feature extraction
  - CLIP Vision for IP-adapter garment embedding
  - VAE, dual CLIP text encoders, tokenizers, DDPM scheduler

Preprocessing (simplified — no DensePose/detectron2):
  - Mask: upper-body rectangle (TODO: add human parsing for better masks)
  - Pose: black image placeholder (TODO: add DensePose for better quality)

Target: ~15-20s inference on RTX 4090/A40 with 20 steps.

NOTE: IDM-VTON source files are downloaded lazily on first inference
(from yisol/IDM-VTON HF Space), so Docker build is instant.
"""

import asyncio
import base64
import io
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login as hf_login
    hf_login(token=_hf_token)

MODEL_ID = "yisol/IDM-VTON"
TARGET_HEIGHT = 1024
TARGET_WIDTH = 768
CACHE_DIR = Path("/app/hf_cache")


def _ensure_idm_vton_source():
    """Lazily download IDM-VTON source files on first use."""
    src_dir = CACHE_DIR / "idm-vton-src"
    
    # If already downloaded, just add to path and return
    # Files end up at: src_dir/src/tryon_pipeline.py
    if (src_dir / "src" / "tryon_pipeline.py").exists():
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        return
    
    print("[IDM-VTON] Downloading source files from HuggingFace Space...")
    try:
        from huggingface_hub import snapshot_download
        
        # Download from the SPACE repo (not model repo!)
        # The model repo has weights; the Space repo has src/ with pipeline code.
        snapshot_download(
            MODEL_ID,
            repo_type="space",
            allow_patterns=["src/*"],  # Only download src/ folder
            local_dir=str(src_dir),
        )
        
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        print("[IDM-VTON] Source files ready!")
        
    except Exception as e:
        # If sparse download fails, just clone the full Space once
        print(f"[IDM-VTON] Sparse download failed ({e}), cloning Space repo...")
        os.system(
            f"git clone --depth 1 https://huggingface.co/spaces/yisol/IDM-VTON {src_dir}"
        )
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))


class IDMVTONInference:
    """IDM-VTON inference engine for virtual try-on."""

    def __init__(self):
        self.pipe = None
        self.unet_encoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self.device == "cuda" else torch.float32
        # ToTensor converts PIL [0,255] → tensor [0,1] (matches HF Space behavior)
        # Defer torchvision import to avoid C++ extension crash at module load time
        import torchvision.transforms as transforms
        self._to_tensor = transforms.Compose([
            transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH)),
            transforms.ToTensor(),
        ])

    async def load_model(self):
        if self.pipe is not None:
            return

        # Ensure IDM-VTON source is available (one-time, cached)
        _ensure_idm_vton_source()

        print(f"[IDM-VTON] Loading model... (device={self.device})")
        load_start = time.time()

        # Custom IDM-VTON pipeline components (from HF Space source)
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet
        from src.unet_hacked_tryon import UNet2DConditionModel as TryOnUNet

        # 1) Load custom try-on UNet (13-channel input)
        print("[IDM-VTON]   Loading try-on UNet...")
        unet = TryOnUNet.from_pretrained(
            MODEL_ID, subfolder="unet", torch_dtype=self._dtype,
        )

        # 2) Load garment encoder UNet
        print("[IDM-VTON]   Loading garment encoder...")
        self.unet_encoder = GarmentUNet.from_pretrained(
            MODEL_ID, subfolder="unet_encoder", torch_dtype=self._dtype,
        )

        # 3) Load full pipeline (VAE, CLIP text encoders, tokenizers,
        #    CLIP vision image_encoder, scheduler) with our custom UNet
        print("[IDM-VTON]   Loading pipeline (VAE, CLIP, tokenizers, scheduler)...")
        self.pipe = TryonPipeline.from_pretrained(
            MODEL_ID,
            unet=unet,
            torch_dtype=self._dtype,
        )
        self.pipe.unet_encoder = self.unet_encoder

        # Move to device
        self.pipe = self.pipe.to(self.device)
        self.unet_encoder = self.unet_encoder.to(self.device)

        # Speed optimizations — use PyTorch native SDPA (built-in, no xformers needed)
        # On RTX 4090 with PyTorch 2.4+, SDPA uses FlashAttention2 automatically
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            print("[IDM-VTON]   PyTorch SDPA (FlashAttention2) enabled")
        except Exception as e:
            print(f"[IDM-VTON]   SDPA setup skipped: {e}")

        elapsed = time.time() - load_start
        print(f"[IDM-VTON]   Model loaded in {elapsed:.1f}s")

    def _create_upper_body_mask(self, width: int, height: int) -> Image.Image:
        """Create a simple upper-body rectangle mask.
        White (255) = area to inpaint (clothing region).
        Black (0) = preserve.

        TODO: Replace with human parsing (ATR/LIP ONNX) for precise masks.
        """
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        left = int(width * 0.15)
        top = int(height * 0.12)
        right = int(width * 0.85)
        bottom = int(height * 0.60)
        draw.rectangle([left, top, right, bottom], fill=255)
        return mask

    async def generate(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_desc: str = "a garment",
        negative_prompt: str = "",
        num_steps: int = 20,
        guidance_scale: float = 2.0,
        seed: int = 42,
    ) -> Image.Image:
        if self.pipe is None:
            await self.load_model()

        t0 = time.time()

        # Resize to IDM-VTON's expected resolution (768×1024)
        person_image = person_image.resize(
            (TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS
        )
        garment_image = garment_image.resize(
            (TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS
        )

        # Create mask (simple upper-body rectangle)
        mask = self._create_upper_body_mask(TARGET_WIDTH, TARGET_HEIGHT)

        # Create pose image placeholder (black = no explicit pose info)
        # The model still gets body shape from masked_image_latents.
        # TODO: Add DensePose output for significantly better quality.
        pose_img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0))

        # Prepare tensors (ToTensor gives [0,1] — matches HF Space behavior)
        cloth_tensor = (
            self._to_tensor(garment_image)
            .unsqueeze(0)
            .to(self.device, dtype=self._dtype)
        )
        pose_tensor = (
            self._to_tensor(pose_img)
            .unsqueeze(0)
            .to(self.device, dtype=self._dtype)
        )

        # Garment text prompt (used for garment encoder conditioning)
        prompt_cloth = f"a photo of {garment_desc}"

        # Main prompt (used for denoising UNet)
        prompt_main = f"model is wearing {garment_desc}"

        with torch.no_grad():
            # Encode garment text (no classifier-free guidance)
            prompt_embeds_c, _, _, _ = self.pipe.encode_prompt(
                prompt_cloth,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                device=self.device,
            )

            # Encode main prompt (with CFG)
            (
                prompt_embeds,
                neg_embeds,
                pooled_embeds,
                neg_pooled_embeds,
            ) = self.pipe.encode_prompt(
                prompt_main,
                negative_prompt=(
                    negative_prompt
                    or "low quality, bad quality, deformed, ugly, blurry"
                ),
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                device=self.device,
            )

            # Run IDM-VTON pipeline
            result = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pooled_embeds,
                negative_pooled_prompt_embeds=neg_pooled_embeds,
                num_inference_steps=num_steps,
                generator=torch.Generator(self.device).manual_seed(seed),
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c,
                cloth=cloth_tensor,
                mask_image=mask,
                image=person_image,
                height=TARGET_HEIGHT,
                width=TARGET_WIDTH,
                ip_adapter_image=garment_image,
                guidance_scale=guidance_scale,
            )[0]

        output_image = result[0] if isinstance(result, list) else result
        elapsed = time.time() - t0
        print(f"[IDM-VTON] Generated in {elapsed:.1f}s ({num_steps} steps)")
        return output_image


# Singleton — models persist across requests on same RunPod worker
_handler_instance = None


class TryOnHandler:
    """RunPod request handler wrapping IDM-VTON inference."""

    def __init__(self):
        self.inference = IDMVTONInference()

    @staticmethod
    def _decode_image(image_base64: str) -> Image.Image:
        image_data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    @staticmethod
    def _encode_image(image: Image.Image, quality: int = 90) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def handle(self, event: dict) -> dict:
        try:
            start_time = time.time()

            user_image_b64 = (
                event.get("user_image_base64") or event.get("user_image")
            )
            garment_image_b64 = (
                event.get("garment_image_base64") or event.get("garment_image")
            )
            garment_id = event.get("garment_id", "unknown")
            garment_name = event.get("garment_name", "a garment")
            custom_prompt = event.get("prompt")
            custom_negative = event.get("negative_prompt")
            num_steps = event.get("num_steps", 20)
            seed = event.get("seed", 42)

            if not user_image_b64 or not garment_image_b64:
                return {
                    "success": False,
                    "error": "Missing user_image or garment_image",
                }

            user_image = self._decode_image(user_image_b64)
            garment_image = self._decode_image(garment_image_b64)

            garment_desc = custom_prompt or garment_name

            print(f"[IDM-VTON] Generating try-on | garment={garment_id}")
            result_image = await self.inference.generate(
                person_image=user_image,
                garment_image=garment_image,
                garment_desc=garment_desc,
                negative_prompt=custom_negative or "",
                num_steps=min(int(num_steps), 30),  # Cap at 30
                seed=int(seed),
            )

            result_base64 = self._encode_image(result_image)
            processing_time_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "image_base64": result_base64,
                "model_used": "idm-vton",
                "processing_time_ms": processing_time_ms,
                "garment_id": garment_id,
                "garment_name": garment_name,
            }

        except Exception as e:
            import traceback
            print(f"[IDM-VTON] Error: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}


async def async_runpod_handler(job):
    """Async RunPod handler — reuses singleton so models stay loaded."""
    global _handler_instance
    try:
        if _handler_instance is None:
            _handler_instance = TryOnHandler()
        return await _handler_instance.handle(job["input"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    try:
        import runpod
        print("[IDM-VTON] Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": async_runpod_handler})
    except ImportError:
        print("[IDM-VTON] RunPod SDK not available, running in test mode")
