"""
Cropping, Centering, Consistent Padding, and Background Compositing Module.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def crop_product(
    rgba_image: np.ndarray,
    mask: np.ndarray,
    margin: int = 5
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Crops the image and mask tightly around the foreground product bounding box.
    
    Args:
        rgba_image: Input RGBA image array (H, W, 4).
        mask: Binary foreground mask array (H, W).
        margin: Small safety pixel margin around product bounding box.
        
    Returns:
        tuple (cropped_rgba, cropped_mask, bbox):
            - cropped_rgba: Tight crop of product image.
            - cropped_mask: Tight crop of mask.
            - bbox: (xmin, ymin, xmax, ymax) coordinates in original image.
    """
    logger.info("Cropping product foreground...")

    h, w = mask.shape[:2]
    y_indices, x_indices = np.where(mask > 10)

    if len(y_indices) == 0 or len(x_indices) == 0:
        logger.warning("No foreground pixels detected in mask. Returning uncropped image.")
        return rgba_image, mask, (0, 0, w, h)

    ymin = max(0, int(np.min(y_indices)) - margin)
    ymax = min(h, int(np.max(y_indices)) + 1 + margin)
    xmin = max(0, int(np.min(x_indices)) - margin)
    xmax = min(w, int(np.max(x_indices)) + 1 + margin)

    cropped_rgba = rgba_image[ymin:ymax, xmin:xmax]
    cropped_mask = mask[ymin:ymax, xmin:xmax]

    logger.info("Product cropped from (%d, %d) to (%d, %d) (Size: %dx%d).",
                xmin, ymin, xmax, ymax, xmax - xmin, ymax - ymin)

    return cropped_rgba, cropped_mask, (xmin, ymin, xmax, ymax)


def center_and_pad(
    rgba_image: np.ndarray,
    target_size: tuple[int, int] = (1000, 1000),
    padding_pct: float = 0.10
) -> np.ndarray:
    """
    Centers the product on a square canvas with consistent padding while strictly preserving aspect ratio.
    
    Args:
        rgba_image: Input cropped product RGBA image (H, W, 4).
        target_size: Canvas size tuple (width, height), default (1000, 1000).
        padding_pct: Percentage of canvas width/height reserved as margin/padding (e.g. 0.10 = 10% on each side).
        
    Returns:
        np.ndarray: Centered RGBA product image of dimensions target_size.
    """
    target_w, target_h = target_size
    logger.info("Centering product on %dx%d canvas with %.1f%% padding...", target_w, target_h, padding_pct * 100)

    orig_h, orig_w = rgba_image.shape[:2]

    if orig_h == 0 or orig_w == 0:
        raise ValueError("Invalid product image size for centering (0 height or width).")

    # Available printable area inside canvas
    max_w = int(round(target_w * (1.0 - 2.0 * padding_pct)))
    max_h = int(round(target_h * (1.0 - 2.0 * padding_pct)))

    # Compute scaling factor to fit within max bounds preserving aspect ratio
    scale = min(max_w / orig_w, max_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    logger.info("Resizing product from %dx%d to %dx%d (Scale factor: %.3f)...",
                orig_w, orig_h, new_w, new_h, scale)

    # High-quality resize using LANCZOS4
    resized_product = cv2.resize(rgba_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create target RGBA canvas (transparent background)
    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)

    # Calculate centering offsets
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    # Paste resized product onto canvas center
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_product

    logger.info("Product successfully centered at canvas offset (%d, %d).", offset_x, offset_y)
    return canvas


def apply_white_background(rgba_canvas: np.ndarray) -> np.ndarray:
    """
    Composites RGBA product image canvas onto a solid pure white RGB canvas (255, 255, 255).
    
    Args:
        rgba_canvas: Centered product canvas of shape (H, W, 4) in RGBA uint8 format.
        
    Returns:
        np.ndarray: Final RGB image array of shape (H, W, 3) in uint8 format with pure white background.
    """
    logger.info("Compositing product image onto solid pure white RGB background (255, 255, 255)...")

    rgb = rgba_canvas[:, :, :3].astype(np.float32)
    alpha = (rgba_canvas[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    white_bg = np.full_like(rgb, 255.0)

    # Smooth alpha blend onto white background: Product * Alpha + White * (1 - Alpha)
    composite = (rgb * alpha) + (white_bg * (1.0 - alpha))
    composite_rgb = np.clip(composite, 0, 255).astype(np.uint8)

    logger.info("Pure white background compositing completed. Output format: RGB (No transparency).")
    return composite_rgb
