"""
Image Quality Enhancement Module using Traditional OpenCV Techniques.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def enhance_quality(
    rgba_image: np.ndarray,
    sharpen: bool = True,
    sharpen_strength: float = 0.25,
    contrast_correction: bool = True,
    noise_reduction: bool = False
) -> np.ndarray:
    """
    Enhances image quality using classical OpenCV techniques: unsharp masking, CLAHE contrast optimization,
    and bilateral edge-preserving noise reduction. Does NOT alter colors artificially or generate fake details.
    
    Args:
        rgba_image: Input RGBA product image array (H, W, 4).
        sharpen: Whether to apply mild unsharp masking.
        sharpen_strength: Weight factor for sharpening (0.1 - 0.4 recommended).
        contrast_correction: Whether to apply CLAHE to luminance channel in LAB space.
        noise_reduction: Whether to apply mild bilateral noise reduction.
        
    Returns:
        np.ndarray: Enhanced RGBA image array (H, W, 4).
    """
    logger.info("Applying traditional image quality enhancement...")

    rgb = rgba_image[:, :, :3].copy()
    alpha = rgba_image[:, :, 3].copy()
    foreground_mask = alpha > 0

    if not np.any(foreground_mask):
        logger.warning("Empty image provided to enhancement. Returning original.")
        return rgba_image

    # 1. Noise Reduction (Bilateral Filter preserves crisp edges while removing pixel noise)
    if noise_reduction:
        logger.info("Applying mild bilateral noise reduction...")
        rgb = cv2.bilateralFilter(rgb, d=3, sigmaColor=10, sigmaSpace=10)

    # 2. Subtle Contrast Adjustment via CLAHE on L channel in LAB color space
    if contrast_correction:
        logger.info("Applying subtle CLAHE contrast correction in LAB color space...")
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Low clipLimit guarantees natural colors are preserved without over-exposure
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # 3. Mild Sharpening via Unsharp Masking
    if sharpen and sharpen_strength > 0.0:
        logger.info("Applying unsharp mask sharpening (strength: %.2f)...", sharpen_strength)
        blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.2)
        # Formula: sharpened = rgb * (1 + strength) - blurred * strength
        sharpened_float = cv2.addWeighted(
            rgb.astype(np.float32), 1.0 + sharpen_strength,
            blurred.astype(np.float32), -sharpen_strength,
            0
        )
        rgb = np.clip(sharpened_float, 0, 255).astype(np.uint8)

    # Ensure background area outside alpha mask remains clean
    enhanced_rgba = np.zeros_like(rgba_image)
    enhanced_rgba[:, :, :3] = rgb
    enhanced_rgba[:, :, 3] = alpha

    logger.info("Quality enhancement completed.")
    return enhanced_rgba
