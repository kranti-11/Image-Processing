"""
Background Removal Module using rembg and classical fallback techniques.
"""

import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


def remove_background(image: Image.Image | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes background from product image using local rembg.
    
    Args:
        image: Input image as PIL Image or NumPy array (RGB/RGBA).
        
    Returns:
        tuple (rgba_image, mask):
            - rgba_image: NumPy array (H, W, 4) in RGBA format with background removed.
            - mask: NumPy uint8 array (H, W) where 255 represents foreground product and 0 represents background.
    """
    logger.info("Starting background removal step...")

    # Convert input to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:
            pil_img = Image.fromarray(image, mode="RGBA")
        else:
            pil_img = Image.fromarray(image, mode="RGB")
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise ValueError(f"Unsupported image input type: {type(image)}")

    # Ensure RGB base
    rgb_pil = pil_img.convert("RGB")

    try:
        import rembg
        logger.info("Removing background with rembg (u2netp model)...")
        session = rembg.new_session("u2netp")
        result_pil = rembg.remove(rgb_pil, session=session)
        rgba_image = np.array(result_pil.convert("RGBA"), dtype=np.uint8)

        # Extract alpha channel
        alpha = rgba_image[:, :, 3]

        # Binarize mask
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        
        logger.info("rembg background removal completed successfully.")
        return rgba_image, mask

    except Exception as e:
        logger.warning("rembg background removal encountered an issue: %s. Using classical OpenCV fallback...", e)
        return _fallback_background_removal(np.array(rgb_pil))


def _fallback_background_removal(rgb_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical background removal using Otsu thresholding and GrabCut algorithm.
    """
    logger.info("Executing classical GrabCut background removal fallback...")
    h, w, _ = rgb_image.shape

    # Initial mask estimation via GrabCut with margin rectangle
    margin_w = int(w * 0.05)
    margin_h = int(h * 0.05)
    rect = (margin_w, margin_h, max(1, w - 2 * margin_w), max(1, h - 2 * margin_h))

    grabcut_mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(rgb_image, grabcut_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        binary_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception as err:
        logger.warning("GrabCut failed: %s. Using color thresholding...", err)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine RGB with extracted alpha mask
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_image[:, :, :3] = rgb_image
    rgba_image[:, :, 3] = binary_mask

    return rgba_image, binary_mask
