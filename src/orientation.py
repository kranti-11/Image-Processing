"""
Product Orientation Detection and Rotation Correction Module using OpenCV & PCA.
"""

import logging
import math
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def detect_and_correct_rotation(
    rgba_image: np.ndarray,
    mask: np.ndarray,
    min_angle_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Detects product tilt/orientation angle from foreground mask using contour analysis and PCA,
    and rotates image/mask to correct orientation.
    
    Args:
        rgba_image: Input RGBA image (H, W, 4).
        mask: Binary mask (H, W) of the foreground product (255=product).
        min_angle_threshold: Minimum angle in degrees required to trigger rotation.
        
    Returns:
        tuple (rotated_rgba, rotated_mask, applied_angle):
            - rotated_rgba: Rotated RGBA NumPy array.
            - rotated_mask: Rotated binary mask.
            - applied_angle: The rotation angle in degrees applied to the image.
    """
    logger.info("Detecting product orientation...")

    # Find contours in binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("No contours found in mask. Skipping rotation correction.")
        return rgba_image, mask, 0.0

    # Filter contours to remove small noise fragments
    max_area = max(cv2.contourArea(c) for c in contours)
    if max_area < 100:
        logger.warning("Contour area too small for reliable orientation detection. Skipping rotation.")
        return rgba_image, mask, 0.0

    # Filter out secondary noise contours (< 10% of max contour area)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 0.10 * max_area]
    main_contour = max(valid_contours, key=cv2.contourArea)

    # Compute tilt angle using robust PCA + MinAreaRect contour analysis
    rotation_angle = _calculate_product_tilt(mask, main_contour, min_angle_threshold)

    logger.info("Detected tilt correction angle: %.2f deg", rotation_angle)

    if abs(rotation_angle) < min_angle_threshold:
        logger.info("Product tilt angle (%.2f deg) is below threshold (%.2f deg). No rotation needed.",
                    rotation_angle, min_angle_threshold)
        return rgba_image, mask, 0.0

    # Perform rotation with expanded canvas to prevent clipping product edges
    rotated_rgba, rotated_mask = _rotate_image_and_mask(rgba_image, mask, rotation_angle)

    logger.info("Successfully corrected product rotation by %.2f deg.", rotation_angle)
    return rotated_rgba, rotated_mask, rotation_angle


def _calculate_product_tilt(mask: np.ndarray, contour: np.ndarray, threshold: float) -> float:
    """
    Calculates tilt angle using PCA on foreground mask points combined with cv2.minAreaRect.
    """
    # 1. MinAreaRect analysis on main contour
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rect_w, rect_h), rect_angle = rect
    box_pts = cv2.boxPoints(rect)

    edge1 = box_pts[1] - box_pts[0]
    edge2 = box_pts[2] - box_pts[1]
    len1 = np.linalg.norm(edge1)
    len2 = np.linalg.norm(edge2)

    # Aspect ratio check: Nearly square/round products shouldn't be rotated
    aspect_ratio = max(len1, len2) / max(1.0, min(len1, len2))
    if aspect_ratio < 1.08:
        logger.info("Product aspect ratio (%.2f) is nearly 1:1. Skipping rotation.", aspect_ratio)
        return 0.0

    # 2. PCA analysis on foreground mask points for highly robust principal axis direction
    y_pts, x_pts = np.where(mask > 10)
    if len(x_pts) < 50:
        return 0.0

    pts = np.column_stack((x_pts, y_pts)).astype(np.float32)
    mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)

    # Principal direction vector (vx, vy)
    vx, vy = eigenvectors[0, 0], eigenvectors[0, 1]
    pca_angle_deg = math.degrees(math.atan2(vy, vx))

    # Normalize PCA angle to [-90, 90]
    angle_deg = pca_angle_deg
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180

    # Calculate tilt required to align major axis upright (vertical, ±90°) or flat (horizontal, 0°)
    if abs(angle_deg) > 45.0:
        # Product is oriented closer to vertical (Y-axis)
        if angle_deg > 0:
            tilt = 90.0 - angle_deg
        else:
            tilt = -90.0 - angle_deg
    else:
        # Product is oriented closer to horizontal (X-axis)
        tilt = -angle_deg

    # Bound tilt to [-45, 45] to avoid unintended 90-degree flips
    if tilt > 45.0:
        tilt -= 90.0
    elif tilt < -45.0:
        tilt += 90.0

    if abs(tilt) < threshold:
        return 0.0

    return float(tilt)


def _rotate_image_and_mask(
    rgba: np.ndarray,
    mask: np.ndarray,
    angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotates RGBA image and binary mask around center, expanding canvas so no pixels are clipped.
    """
    h, w = rgba.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])

    new_w = int(round((h * sin_a) + (w * cos_a)))
    new_h = int(round((h * cos_a) + (w * sin_a)))

    # Adjust transformation matrix translation
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    rotated_rgba = cv2.warpAffine(
        rgba, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    rotated_mask = cv2.warpAffine(
        mask, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated_rgba, rotated_mask
