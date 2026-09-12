import sys

import torchvision.transforms.functional as functional

# Compatibility fix for older BasicSR / newer TorchVision
sys.modules["torchvision.transforms.functional_tensor"] = functional

import streamlit as st
from PIL import Image
from rembg import remove, new_session
import numpy as np
import cv2
import io
import os
import torch

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vipto — Product Image Cleaner",
    page_icon="📦",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("Vipto — Product Image Cleaner")

st.write(
    "Upload a product image to remove the background, "
    "enhance image quality, and generate a catalog-ready image."
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "weights",
    "RealESRGAN_x4plus.pth"
)


# ============================================================
# LOAD BACKGROUND REMOVAL MODEL
# ============================================================

@st.cache_resource
def load_background_model():
    return new_session(
        "isnet-general-use"
    )


session = load_background_model()


# ============================================================
# LOAD REAL-ESRGAN MODEL
# ============================================================

@st.cache_resource
def load_realesrgan():

    # RRDBNet architecture used by RealESRGAN_x4plus
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    # Apple Silicon MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )

    return upsampler


# ============================================================
# IMAGE ENHANCEMENT FUNCTIONS
# ============================================================

def white_balance(image_rgb, strength=0.35):
    """
    Mild Gray-World white balance.
    """

    image = image_rgb.astype(
        np.float32
    )

    avg_r = np.mean(
        image[:, :, 0]
    )

    avg_g = np.mean(
        image[:, :, 1]
    )

    avg_b = np.mean(
        image[:, :, 2]
    )

    avg_gray = (
        avg_r +
        avg_g +
        avg_b
    ) / 3.0

    avg_r = max(
        avg_r,
        1.0
    )

    avg_g = max(
        avg_g,
        1.0
    )

    avg_b = max(
        avg_b,
        1.0
    )

    r_gain = avg_gray / avg_r
    g_gain = avg_gray / avg_g
    b_gain = avg_gray / avg_b

    balanced = image.copy()

    balanced[:, :, 0] *= (
        1.0 +
        (r_gain - 1.0) *
        strength
    )

    balanced[:, :, 1] *= (
        1.0 +
        (g_gain - 1.0) *
        strength
    )

    balanced[:, :, 2] *= (
        1.0 +
        (b_gain - 1.0) *
        strength
    )

    return np.clip(
        balanced,
        0,
        255
    ).astype(np.uint8)


def denoise_image(image_rgb):
    """
    Reduces noise while preserving product edges.
    """

    return cv2.bilateralFilter(
        image_rgb,
        d=7,
        sigmaColor=35,
        sigmaSpace=35
    )


def sharpen_image(image_rgb, strength=0.35):
    """
    Mild unsharp-mask sharpening.
    """

    blurred = cv2.GaussianBlur(
        image_rgb,
        (0, 0),
        sigmaX=2
    )

    sharpened = cv2.addWeighted(
        image_rgb,
        1.0 + strength,
        blurred,
        -strength,
        0
    )

    return np.clip(
        sharpened,
        0,
        255
    ).astype(np.uint8)


def restore_white_background(image_rgb):
    """
    Restores pixels that are already very close to white.
    """

    result = image_rgb.copy()

    white_mask = np.all(
        result >= 245,
        axis=2
    )

    result[white_mask] = [
        255,
        255,
        255
    ]

    return result


# ============================================================
# CONSERVATIVE SHADOW HANDLING
# ============================================================

def reduce_background_shadows(image_rgb):
    """
    Mild shadow reduction.
    Avoids aggressive changes to product colors.
    """

    result = image_rgb.copy()

    gray = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2GRAY
    )

    shadow_mask = gray < 225

    corrected = result.astype(
        np.float32
    )

    for channel in range(3):

        channel_data = corrected[
            :,
            :,
            channel
        ]

        channel_data[
            shadow_mask
        ] = np.minimum(
            channel_data[
                shadow_mask
            ] * 1.05,
            255
        )

        corrected[
            :,
            :,
            channel
        ] = channel_data

    return np.clip(
        corrected,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# REAL-ESRGAN UPSCALING
# ============================================================

def upscale_with_realesrgan(
    image_rgb,
    upsampler
):
    """
    Upscales the processed product image using Real-ESRGAN.
    """

    # RGB -> BGR
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    output, _ = upsampler.enhance(
        image_bgr,
        outscale=4
    )

    # BGR -> RGB
    output_rgb = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )

    return output_rgb


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a product image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    img = Image.open(
        uploaded_file
    ).convert("RGBA")

    st.subheader(
        "Original Image"
    )

    st.image(
        img,
        caption="Original",
        width=300
    )


    # ========================================================
    # BACKGROUND REMOVAL
    # ========================================================

    with st.spinner(
        "Removing background..."
    ):

        foreground = remove(
            img,
            session=session
        )


    foreground_np = np.array(
        foreground
    )

    rgb = foreground_np[
        :,
        :,
        :3
    ]

    alpha = foreground_np[
        :,
        :,
        3
    ]


    # ========================================================
    # CLEAN ALPHA MASK
    # ========================================================

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    alpha_clean = cv2.morphologyEx(
        alpha,
        cv2.MORPH_OPEN,
        kernel
    )

    alpha_clean = cv2.morphologyEx(
        alpha_clean,
        cv2.MORPH_CLOSE,
        kernel
    )

    alpha_feathered = cv2.GaussianBlur(
        alpha_clean,
        (5, 5),
        0
    )


    # ========================================================
    # WHITE BACKGROUND
    # ========================================================

    alpha_norm = (
        alpha_feathered[
            :,
            :,
            None
        ] / 255.0
    )

    white = np.ones_like(
        rgb
    ) * 255

    result = (
        rgb * alpha_norm
        +
        white * (
            1 -
            alpha_norm
        )
    ).astype(
        np.uint8
    )


    # ========================================================
    # CROP PRODUCT
    # ========================================================

    result_bgr = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        result_bgr,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        250,
        255,
        cv2.THRESH_BINARY_INV
    )

    coords = cv2.findNonZero(
        thresh
    )


    if coords is not None:

        x, y, w, h = cv2.boundingRect(
            coords
        )

        cropped = result_bgr[
            y:y + h,
            x:x + w
        ]

    else:

        cropped = result_bgr


    # ========================================================
    # PADDING
    # ========================================================

    h, w = cropped.shape[:2]

    pad = int(
        max(w, h) * 0.10
    )

    padded = cv2.copyMakeBorder(
        cropped,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=[
            255,
            255,
            255
        ]
    )


    # ========================================================
    # SQUARE CANVAS
    # ========================================================

    h2, w2 = padded.shape[:2]

    side = max(
        h2,
        w2
    )

    top = (
        side - h2
    ) // 2

    bottom = (
        side -
        h2 -
        top
    )

    left = (
        side - w2
    ) // 2

    right = (
        side -
        w2 -
        left
    )

    square = cv2.copyMakeBorder(
        padded,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[
            255,
            255,
            255
        ]
    )


    # ========================================================
    # RESIZE TO 1000 × 1000
    # ========================================================

    final = cv2.resize(
        square,
        (1000, 1000),
        interpolation=cv2.INTER_AREA
    )

    final_rgb = cv2.cvtColor(
        final,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # SHADOW HANDLING
    # ========================================================

    st.subheader(
        "Shadow Handling"
    )

    with st.spinner(
        "Reducing background shadows..."
    ):

        shadow_corrected = (
            reduce_background_shadows(
                final_rgb
            )
        )


    st.image(
        shadow_corrected,
        caption="After shadow handling",
        width=300
    )


    # ========================================================
    # IMAGE ENHANCEMENT
    # ========================================================

    st.subheader(
        "Image Enhancement"
    )


    # --------------------------------------------------------
    # WHITE BALANCE
    # --------------------------------------------------------

    with st.spinner(
        "Adjusting white balance..."
    ):

        balanced = white_balance(
            shadow_corrected,
            strength=0.35
        )


    # --------------------------------------------------------
    # DENOISING
    # --------------------------------------------------------

    with st.spinner(
        "Reducing image noise..."
    ):

        denoised = denoise_image(
            balanced
        )


    # --------------------------------------------------------
    # SHARPENING
    # --------------------------------------------------------

    with st.spinner(
        "Improving sharpness..."
    ):

        enhanced = sharpen_image(
            denoised,
            strength=0.35
        )


    # --------------------------------------------------------
    # RESTORE WHITE BACKGROUND
    # --------------------------------------------------------

    enhanced = restore_white_background(
        enhanced
    )


    st.image(
        enhanced,
        caption="Enhanced image",
        width=300
    )


    # ========================================================
    # REAL-ESRGAN
    # ========================================================

    st.subheader(
        "AI Upscaling — Real-ESRGAN"
    )


    if not os.path.exists(
        MODEL_PATH
    ):

        st.error(
            "Real-ESRGAN model not found."
        )

        st.write(
            "Expected model path:"
        )

        st.code(
            MODEL_PATH
        )

        st.stop()


    with st.spinner(
        "Upscaling with Real-ESRGAN..."
    ):

        try:

            upsampler = load_realesrgan()

            upscaled = (
                upscale_with_realesrgan(
                    enhanced,
                    upsampler
                )
            )

        except Exception as e:

            st.error(
                "Real-ESRGAN processing failed."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # UPSCALED RESULT
    # ========================================================

    st.image(
        upscaled,
        caption="Real-ESRGAN upscaled result",
        width=500
    )


    # ========================================================
    # SIZE INFORMATION
    # ========================================================

    h_final, w_final = upscaled.shape[:2]

    st.write(
        f"Final image size: "
        f"**{w_final} × {h_final} pixels**"
    )


    # ========================================================
    # FINAL CATALOG RESULT
    # ========================================================

    st.subheader(
        "Catalog-ready Result"
    )

    st.image(
        upscaled,
        caption="Final Vipto product image",
        width=500
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    final_pil = Image.fromarray(
        upscaled
    )

    buffer = io.BytesIO()

    final_pil.save(
        buffer,
        format="JPEG",
        quality=95
    )

    st.download_button(
        label="⬇️ Download Catalog Image",
        data=buffer.getvalue(),
        file_name="vipto_catalog_ready.jpg",
        mime="image/jpeg"
    )