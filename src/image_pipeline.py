"""
Modular E-Commerce Product Image Processing Pipeline.
"""

import os
import time
import logging
from pathlib import Path
from PIL import Image
import numpy as np

from src.background_removal import remove_background
from src.orientation import detect_and_correct_rotation
from src.cropping import crop_product, center_and_pad, apply_white_background
from src.enhancement import enhance_quality

logger = logging.getLogger(__name__)


class ProductImagePipeline:
    """
    End-to-end modular e-commerce product image processing pipeline.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (1000, 1000),
        padding_pct: float = 0.10,
        min_angle_threshold: float = 0.5,
        crop_margin: int = 5,
        sharpen: bool = True,
        contrast_correction: bool = True,
        noise_reduction: bool = False
    ):
        """
        Configures pipeline parameters.
        
        Args:
            target_size: Output dimensions (width, height), default (1000, 1000).
            padding_pct: Percentage of canvas margin, default 0.10 (10%).
            min_angle_threshold: Tilt angle threshold in degrees to trigger rotation.
            crop_margin: Pixels margin around product bounding box.
            sharpen: Enable unsharp mask sharpening.
            contrast_correction: Enable LAB CLAHE contrast correction.
            noise_reduction: Enable bilateral filter noise reduction.
        """
        self.target_size = target_size
        self.padding_pct = padding_pct
        self.min_angle_threshold = min_angle_threshold
        self.crop_margin = crop_margin
        self.sharpen = sharpen
        self.contrast_correction = contrast_correction
        self.noise_reduction = noise_reduction

    def validate_image(self, input_path: str | Path) -> Image.Image:
        """
        Validates input image file existence and integrity.
        
        Args:
            input_path: Path to input image file.
            
        Returns:
            Image.Image: Loaded PIL Image object in RGB format.
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path.absolute()}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported file format '{path.suffix}'. Supported formats: {valid_extensions}")

        try:
            pil_img = Image.open(path)
            pil_img.verify()  # Verify image integrity
            # Re-open after verify() as required by PIL
            pil_img = Image.open(path)
            logger.info("Successfully validated and loaded input image: %s (%s, %dx%d)",
                        path.name, pil_img.format, pil_img.width, pil_img.height)
            return pil_img
        except Exception as e:
            raise ValueError(f"Failed to load or validate image file {path.name}: {e}")

    def process(self, input_path: str | Path, output_path: str | Path) -> dict:
        """
        Runs full image processing pipeline:
        1. Validate Image
        2. Background Removal (rembg)
        3. Extract Foreground Mask
        4. Detect Contour & Orientation (PCA + MinAreaRect)
        5. Correct Tilt / Rotation
        6. Crop Product (recalculates bounding box post-rotation)
        7. Center Product & Add Padding (Aspect ratio preserved)
        8. Enhance Quality (Traditional OpenCV)
        9. Composite onto Pure White Background (RGB 255, 255, 255)
        10. Save Processed 1000x1000 RGB PNG Image
        
        Args:
            input_path: Path to source image.
            output_path: Path where processed PNG will be saved.
            
        Returns:
            dict: Processing metadata summary.
        """
        start_time = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info("Starting pipeline execution for: %s", input_path.name)

        # 1. Validate Image
        pil_img = self.validate_image(input_path)

        # 2 & 3. Remove Background & Extract Mask
        rgba_img, mask = remove_background(pil_img)

        # 4 & 5. Detect & Correct Rotation (PCA + MinAreaRect)
        rotated_rgba, rotated_mask, applied_angle = detect_and_correct_rotation(
            rgba_img, mask, min_angle_threshold=self.min_angle_threshold
        )

        # 6. Recalculate Bounding Box and Crop Product Foreground
        cropped_rgba, cropped_mask, bbox = crop_product(
            rotated_rgba, rotated_mask, margin=self.crop_margin
        )

        # 7. Center Product & Add Padding to 1000x1000 Canvas
        centered_canvas = center_and_pad(
            cropped_rgba, target_size=self.target_size, padding_pct=self.padding_pct
        )

        # 8. Enhance Quality (Traditional OpenCV)
        enhanced_canvas = enhance_quality(
            centered_canvas,
            sharpen=self.sharpen,
            contrast_correction=self.contrast_correction,
            noise_reduction=self.noise_reduction
        )

        # 9. Composite product onto Pure White RGB Background (255, 255, 255)
        final_rgb = apply_white_background(enhanced_canvas)

        # 10. Save Output 1000x1000 RGB Image (No transparency)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_pil = Image.fromarray(final_rgb, mode="RGB")
        final_pil.save(output_path, format="PNG", optimize=True)

        elapsed = time.time() - start_time
        logger.info("Pipeline finished in %.2fs. Processed image saved to: %s", elapsed, output_path.absolute())

        return {
            "status": "SUCCESS",
            "input_file": str(input_path),
            "output_file": str(output_path),
            "original_size": (pil_img.width, pil_img.height),
            "rotation_angle_deg": round(applied_angle, 2),
            "crop_bbox": bbox,
            "final_size": self.target_size,
            "elapsed_seconds": round(elapsed, 3)
        }
