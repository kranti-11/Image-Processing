import cv2
import numpy as np


def remove_shadows(image_rgb):
    """
    Reduces soft gray shadows from the white background
    while preserving the product.
    """

    # Convert RGB to LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    # Estimate slowly varying illumination
    background = cv2.GaussianBlur(
        l,
        (0, 0),
        sigmaX=25
    )

    # Normalize illumination
    corrected_l = cv2.divide(
        l,
        background,
        scale=255
    )

    corrected_l = np.clip(
        corrected_l,
        0,
        255
    ).astype(np.uint8)

    corrected_lab = cv2.merge(
        [corrected_l, a, b]
    )

    result = cv2.cvtColor(
        corrected_lab,
        cv2.COLOR_LAB2RGB
    )

    return result