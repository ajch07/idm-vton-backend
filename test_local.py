#!/usr/bin/env python3
"""
Local test script for virtual try-on models.
Tests IDM-VTON and Flux without needing Runpod deployment.

Usage:
    python test_local.py --person <path_to_person.jpg> --garment <path_to_garment.jpg>
    
Or with sample images (will be created if missing):
    python test_local.py

Timeline (first run):
    - IDM-VTON download: ~15-20 mins
    - Flux download: ~5-10 mins
    - First inference: ~5-10 mins total
    
Timeline (subsequent runs):
    - Model loading from cache: ~1-2 mins
    - Inference: ~3-5 mins total
    
Requirements:
    - GPU with 12GB+ VRAM (RTX 3080 / 4060 / 4090)
    - Or CPU (much slower, 30+ mins per image)
"""

import asyncio
import base64
import io
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "app" / "services"))

from runpod_handler import TryOnHandler


def create_dummy_image(width=768, height=1024, text="", color=(200, 200, 200)):
    """Create a simple dummy image for testing."""
    image = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(image)

    # Draw circles/shapes to simulate a person or garment
    draw.ellipse([(300, 100), (468, 250)], fill=(220, 180, 160))  # Head
    draw.rectangle([(250, 250), (518, 600)], fill=(100, 150, 255))  # Torso/Garment
    draw.rectangle([(280, 600), (388, 900)], fill=(220, 180, 160))  # Left leg
    draw.rectangle([(380, 600), (488, 900)], fill=(220, 180, 160))  # Right leg

    # Add text
    if text:
        draw.text((350, 950), text, fill=(0, 0, 0))

    return image


def load_or_create_image(
    image_path: Optional[str], name: str, is_person: bool = False
) -> Image.Image:
    """Load image from path or create dummy if path not provided."""
    if image_path and Path(image_path).exists():
        print(f"  Loading {name} from: {image_path}")
        return Image.open(image_path).convert("RGB")
    else:
        print(f"  Creating dummy {name}")
        color = (220, 180, 160) if is_person else (100, 150, 255)
        text = "Person" if is_person else "Garment"
        return create_dummy_image(color=color, text=text)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


async def test_tryon(
    person_image_path: Optional[str] = None,
    garment_image_path: Optional[str] = None,
    use_flux_only: bool = False,
):
    """Test virtual try-on with local images."""

    print("\n" + "=" * 80)
    print("VIRTUAL TRY-ON LOCAL TEST")
    print("=" * 80)

    print(f"\nDevice: {'cuda' if _has_cuda() else 'cpu'}")
    print(f"Using model: {'Flux only' if use_flux_only else 'IDM-VTON (with Flux fallback)'}")

    # Load images
    print("\nLoading images...")
    person_image = load_or_create_image(person_image_path, "person image", is_person=True)
    garment_image = load_or_create_image(
        garment_image_path, "garment image", is_person=False
    )

    print(f"  Person image: {person_image.size}")
    print(f"  Garment image: {garment_image.size}")

    # Convert to base64
    person_b64 = image_to_base64(person_image)
    garment_b64 = image_to_base64(garment_image)

    # Create handler
    print("\nInitializing handler...")
    handler = TryOnHandler()

    # Build event
    event = {
        "user_image": person_b64,
        "garment_image": garment_b64,
        "garment_id": "test_garment_001",
        "garment_name": "Test Summer Dress",
        "prompt": None,
        "negative_prompt": None,
        "use_flux": use_flux_only,
    }

    # Run handler
    print("\n" + "-" * 80)
    print("GENERATING TRY-ON IMAGE...")
    print(
        "This may take 5-30+ minutes on first run (model downloads from HuggingFace)"
    )
    print("-" * 80 + "\n")

    result = await handler.handle(event)

    print("\n" + "-" * 80)
    print("RESULT")
    print("-" * 80)
    print(f"Success: {result.get('success')}")
    print(f"Model used: {result.get('model_used')}")
    print(f"Processing time: {result.get('processing_time_ms')}ms")

    if result.get("success"):
        # Save result image
        output_path = Path("test_output.png")
        result_image = base64_to_image(result["image_base64"])
        result_image.save(output_path)
        print(f"Result saved to: {output_path}")
        print(f"Output image size: {result_image.size}")
    else:
        print(f"Error: {result.get('error')}")

    print("\n" + "=" * 80)


def _has_cuda():
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test virtual try-on models locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_local.py
    (Uses dummy images)
  
  python test_local.py --person my_photo.jpg --garment dress.jpg
    (Uses your own images)
  
  python test_local.py --flux-only
    (Test Flux model only, skips IDM-VTON)
        """,
    )

    parser.add_argument(
        "--person",
        type=str,
        default=None,
        help="Path to person/user image (default: creates dummy image)",
    )
    parser.add_argument(
        "--garment",
        type=str,
        default=None,
        help="Path to garment image (default: creates dummy image)",
    )
    parser.add_argument(
        "--flux-only",
        action="store_true",
        help="Test Flux model only (skip IDM-VTON)",
    )

    args = parser.parse_args()

    # Run test
    asyncio.run(
        test_tryon(
            person_image_path=args.person,
            garment_image_path=args.garment,
            use_flux_only=args.flux_only,
        )
    )


if __name__ == "__main__":
    main()
