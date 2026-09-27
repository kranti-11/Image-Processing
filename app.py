"""
E-Commerce Product Image Processing Pipeline - Main Application Entry Point.

Usage:
    python app.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

from src.image_pipeline import ProductImagePipeline

# Configure clean logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("app")


def create_sample_product_image(sample_path: Path):
    """
    Creates a sample tilted product image in input/ for demo purposes if no input image exists.
    """
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generating sample product image at: %s", sample_path)

    # Create synthetic product image (e.g., a tilted product bottle/box on a studio background)
    canvas = np.full((600, 600, 3), (240, 240, 245), dtype=np.uint8)

    # Draw a synthetic product (a blue product bottle with label)
    prod_img = Image.new("RGBA", (200, 360), (0, 0, 0, 0))
    draw = ImageDraw.Draw(prod_img)
    # Bottle body
    draw.rounded_rectangle([20, 60, 180, 340], radius=25, fill=(30, 90, 180, 255), outline=(10, 50, 120, 255), width=3)
    # Bottle neck & cap
    draw.rectangle([70, 20, 130, 60], fill=(220, 220, 225, 255), outline=(180, 180, 190, 255), width=2)
    draw.rectangle([65, 5, 135, 20], fill=(200, 30, 30, 255))
    # Product label
    draw.rectangle([35, 120, 165, 260], fill=(255, 255, 255, 255))
    draw.text((50, 150), "ECOMMERCE\nPRODUCT", fill=(20, 20, 20, 255))

    # Introduce a 15-degree tilt to demonstrate rotation detection & correction
    tilted_prod = prod_img.rotate(15, expand=True, resample=Image.Resampling.BICUBIC)

    # Paste onto canvas
    pil_canvas = Image.fromarray(canvas)
    paste_x = (600 - tilted_prod.width) // 2
    paste_y = (600 - tilted_prod.height) // 2
    pil_canvas.paste(tilted_prod, (paste_x, paste_y), tilted_prod)

    pil_canvas.save(sample_path, format="JPEG", quality=95)
    logger.info("Sample image created successfully at %s", sample_path)


def main():
    print("==================================================================")
    print("       E-Commerce Product Image Processing Pipeline              ")
    print("==================================================================")

    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for input images
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    input_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]

    if not input_files:
        sample_file = input_dir / "product.jpg"
        create_sample_product_image(sample_file)
        input_files = [sample_file]

    # Initialize Pipeline
    pipeline = ProductImagePipeline(
        target_size=(1000, 1000),
        padding_pct=0.10,          # 10% padding around product
        min_angle_threshold=0.5,   # Minimum angle tilt threshold
        crop_margin=5,             # 5px crop margin around bounding box
        sharpen=True,              # Mild unsharp masking
        contrast_correction=True,  # LAB CLAHE contrast optimization
        noise_reduction=False      # Bilateral noise reduction
    )

    print(f"\nFound {len(input_files)} input image(s) to process.\n")

    for img_path in input_files:
        output_filename = f"{img_path.stem}_processed.png"
        output_path = output_dir / output_filename

        try:
            print(f"--> Processing: {img_path.name}...")
            result = pipeline.process(img_path, output_path)

            print("\n[OK] SUCCESS!")
            print(f"    - Input:           {result['input_file']}")
            print(f"    - Output:          {result['output_file']}")
            print(f"    - Original Size:   {result['original_size'][0]}x{result['original_size'][1]} px")
            print(f"    - Rotated Angle:   {result['rotation_angle_deg']} deg")
            print(f"    - Final Size:      {result['final_size'][0]}x{result['final_size'][1]} px")
            print(f"    - Time Taken:      {result['elapsed_seconds']}s\n")

        except Exception as e:
            logger.error("Failed to process %s: %s", img_path.name, e, exc_info=True)

    print("==================================================================")
    print("Processing complete. All output images saved in 'output/' folder.")
    print("==================================================================")


if __name__ == "__main__":
    main()
