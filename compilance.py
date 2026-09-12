import cv2
import numpy as np


def check_white_purity(image, threshold=245):
    """
    Checks how much of the image background is close to pure white.
    Returns percentage of near-white pixels.
    """

    if image is None:
        return 0.0

    # Convert RGB → grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Count pixels above the white threshold
    white_pixels = np.sum(gray >= threshold)

    total_pixels = gray.size

    percentage = (white_pixels / total_pixels) * 100

    return round(float(percentage), 2)


def flag_text(image):
    """
    Basic text/edge detection.
    This does NOT read the text.
    It only identifies areas that may contain text or fine details.
    """

    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect strong edges
    edges = cv2.Canny(gray, 100, 200)

    edge_pixels = np.count_nonzero(edges)

    total_pixels = edges.size

    edge_ratio = edge_pixels / total_pixels

    # High edge density can indicate text/details
    return edge_ratio > 0.02