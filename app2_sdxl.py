import sys
import io
import os

import torchvision.transforms.functional as functional

# Compatibility fix for older BasicSR / newer TorchVision
sys.modules["torchvision.transforms.functional_tensor"] = functional

import streamlit as st
from PIL import Image
from rembg import remove, new_session
import numpy as np
import cv2
import torch

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# SDXL
from diffusers import StableDiffusionXLPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vipto — Product Image Cleaner",
    page_icon="📦",
    layout="centered"
)

st.title("Vipto — Product Image Cleaner")

st.write(
    "Upload a product image to remove the background, generate "
    "an AI background, enhance image quality, and generate a "
    "catalog-ready image."
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "weights",
    "RealESRGAN_x4plus.pth"
)

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


DEVICE = get_device()


# ============================================================
# BACKGROUND REMOVAL MODEL
# ============================================================

@st.cache_resource
def load_background_model():
    return new_session(
        "isnet-general-use"
    )


session = load_background_model()


# ============================================================
# SDXL MODEL
# ============================================================

@st.cache_resource
def load_sdxl():

    dtype = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True
    )

    pipe = pipe.to(DEVICE)

    # Reduces memory usage when supported.
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe


# ============================================================
# SDXL BACKGROUND GENERATION
# ============================================================

def generate_background(pipe, prompt, width, height):

    # SDXL works best with dimensions divisible by 8.
    width = max(512, min(width, 1024))
    height = max(512, min(height, 1024))

    width = (width // 8) * 8
    height = (height // 8) * 8

    negative_prompt = (
        "product, object, person, people, text, logo, watermark, "
        "letters, distorted objects, blurry, low quality"
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=25,
        guidance_scale=7.0
    ).images[0]

    return result


# ============================================================
# REAL-ESRGAN MODEL
# ============================================================

@st.cache_resource
def load_realesrgan():

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=(device.type == "cuda"),
        device=device
    )

    return upsampler


# ============================================================
# IMAGE ENHANCEMENT FUNCTIONS
# ============================================================

def white_balance(image_rgb, strength=0.35):

    image = image_rgb.astype(np.float32)

    avg_r = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_b = np.mean(image[:, :, 2])

    avg_gray = (avg_r + avg_g + avg_b) / 3.0

    avg_r = max(avg_r, 1.0)
    avg_g = max(avg_g, 1.0)
    avg_b = max(avg_b, 1.0)

    r_gain = avg_gray / avg_r
    g_gain = avg_gray / avg_g
    b_gain = avg_gray / avg_b

    balanced = image.copy()

    balanced[:, :, 0] *= 1.0 + (r_gain - 1.0) * strength
    balanced[:, :, 1] *= 1.0 + (g_gain - 1.0) * strength
    balanced[:, :, 2] *= 1.0 + (b_gain - 1.0) * strength

    return np.clip(
        balanced,
        0,
        255
    ).astype(np.uint8)


def denoise_image(image_rgb):

    return cv2.bilateralFilter(
        image_rgb,
        d=7,
        sigmaColor=35,
        sigmaSpace=35
    )


def sharpen_image(image_rgb, strength=0.35):

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


def reduce_background_shadows(image_rgb):

    result = image_rgb.copy()

    gray = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2GRAY
    )

    shadow_mask = gray < 225

    corrected = result.astype(np.float32)

    for channel in range(3):

        channel_data = corrected[:, :, channel]

        channel_data[shadow_mask] = np.minimum(
            channel_data[shadow_mask] * 1.05,
            255
        )

        corrected[:, :, channel] = channel_data

    return np.clip(
        corrected,
        0,
        255
    ).astype(np.uint8)


def upscale_with_realesrgan(image_rgb, upsampler):

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    output, _ = upsampler.enhance(
        image_bgr,
        outscale=4
    )

    output_rgb = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )

    return output_rgb


# ============================================================
# PRODUCT + AI BACKGROUND COMPOSITING
# ============================================================

def create_product_mask(alpha, threshold=20):

    mask = alpha.copy()

    # Remove tiny noise.
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep edges smooth.
    mask = cv2.GaussianBlur(
        mask,
        (5, 5),
        0
    )

    return mask


def composite_product_on_background(
    product_rgb,
    alpha,
    background_pil
):

    product_h, product_w = product_rgb.shape[:2]

    background = background_pil.convert("RGB").resize(
        (product_w, product_h),
        Image.Resampling.LANCZOS
    )

    background_rgb = np.array(background)

    alpha_norm = (
        alpha.astype(np.float32)[:, :, None] / 255.0
    )

    result = (
        product_rgb.astype(np.float32) * alpha_norm
        +
        background_rgb.astype(np.float32) * (1.0 - alpha_norm)
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


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

    img = Image.open(
        uploaded_file
    ).convert("RGBA")

    st.subheader("Original Image")

    st.image(
        img,
        caption="Original",
        width=300
    )


    # ========================================================
    # BACKGROUND REMOVAL
    # ========================================================

    with st.spinner("Removing background..."):

        foreground = remove(
            img,
            session=session
        )

    foreground_np = np.array(
        foreground
    )

    rgb = foreground_np[:, :, :3]

    alpha = foreground_np[:, :, 3]

    alpha_clean = create_product_mask(
        alpha
    )


    # ========================================================
    # SHOW EXTRACTED PRODUCT
    # ========================================================

    product_rgba = np.dstack(
        (
            rgb,
            alpha_clean
        )
    )

    st.subheader("Extracted Product")

    st.image(
        product_rgba,
        caption="Product after background removal",
        width=300
    )


    # ========================================================
    # AI BACKGROUND SETTINGS
    # ========================================================

    st.subheader("AI Background Generation")

    background_prompt = st.text_area(
        "Background prompt",
        value=(
            "premium e-commerce product photography background, "
            "clean white studio, subtle light gray gradient, "
            "soft realistic floor shadow, professional softbox lighting, "
            "minimal luxury catalog style, clean and realistic, "
            "empty background, no product"
        )
    )


    # ========================================================
    # GENERATE AI BACKGROUND
    # ========================================================

    if st.button("Generate AI Background"):

        with st.spinner(
            f"Loading SDXL and generating background on {DEVICE}..."
        ):

            try:

                sdxl = load_sdxl()

                # Use a square background for the catalog image.
                ai_background = generate_background(
                    sdxl,
                    background_prompt,
                    1024,
                    1024
                )

            except Exception as e:

                st.error(
                    "SDXL background generation failed."
                )

                st.exception(e)

                st.stop()


        st.image(
            ai_background,
            caption="AI-generated background",
            width=500
        )


        # ====================================================
        # COMPOSITE PRODUCT ON AI BACKGROUND
        # ====================================================

        with st.spinner(
            "Placing original product on AI background..."
        ):

            composited = composite_product_on_background(
                rgb,
                alpha_clean,
                ai_background
            )


        st.subheader(
            "Product + AI Background"
        )

        st.image(
            composited,
            caption="AI background with original product",
            width=500
        )


        # ====================================================
        # CROP / SQUARE / 1000x1000
        # ====================================================

        result_bgr = cv2.cvtColor(
            composited,
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            result_bgr,
            cv2.COLOR_BGR2GRAY
        )

        # Since the AI background is not white, use the
        # original alpha mask to find the product bounding box.
        coords = cv2.findNonZero(
            alpha_clean
        )

        if coords is not None:

            x, y, w, h = cv2.boundingRect(
                coords
            )

            # Add some product-side padding.
            pad = int(
                max(w, h) * 0.10
            )

            x1 = max(
                0,
                x - pad
            )

            y1 = max(
                0,
                y - pad
            )

            x2 = min(
                composited.shape[1],
                x + w + pad
            )

            y2 = min(
                composited.shape[0],
                y + h + pad
            )

            cropped = result_bgr[
                y1:y2,
                x1:x2
            ]

        else:

            cropped = result_bgr


        # ====================================================
        # SQUARE CANVAS
        # ====================================================

        h2, w2 = cropped.shape[:2]

        side = max(
            h2,
            w2
        )

        top = (
            side - h2
        ) // 2

        bottom = (
            side - h2 - top
        )

        left = (
            side - w2
        ) // 2

        right = (
            side - w2 - left
        )

        square = cv2.copyMakeBorder(
            cropped,
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


        # ====================================================
        # RESIZE
        # ====================================================

        final = cv2.resize(
            square,
            (1000, 1000),
            interpolation=cv2.INTER_AREA
        )

        final_rgb = cv2.cvtColor(
            final,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # IMAGE ENHANCEMENT
        # ====================================================

        st.subheader(
            "Image Enhancement"
        )

        with st.spinner(
            "Adjusting white balance..."
        ):

            balanced = white_balance(
                final_rgb,
                strength=0.35
            )

        with st.spinner(
            "Reducing image noise..."
        ):

            denoised = denoise_image(
                balanced
            )

        with st.spinner(
            "Improving sharpness..."
        ):

            enhanced = sharpen_image(
                denoised,
                strength=0.35
            )


        # Do not force the AI background to pure white.
        # Only restore pixels that are already almost white.
        enhanced = restore_white_background(
            enhanced
        )

        st.image(
            enhanced,
            caption="Enhanced image",
            width=500
        )


        # ====================================================
        # REAL-ESRGAN
        # ====================================================

        st.subheader(
            "AI Upscaling — Real-ESRGAN"
        )

        if not os.path.exists(
            MODEL_PATH
        ):

            st.warning(
                "Real-ESRGAN model not found. "
                "Skipping Real-ESRGAN and using the enhanced image."
            )

            upscaled = enhanced

        else:

            with st.spinner(
                "Upscaling with Real-ESRGAN..."
            ):

                try:

                    upsampler = load_realesrgan()

                    upscaled = upscale_with_realesrgan(
                        enhanced,
                        upsampler
                    )

                except Exception as e:

                    st.warning(
                        "Real-ESRGAN failed. "
                        "Using the enhanced image instead."
                    )

                    st.exception(e)

                    upscaled = enhanced


        # ====================================================
        # FINAL RESULT
        # ====================================================

        st.subheader(
            "Catalog-ready Result"
        )

        st.image(
            upscaled,
            caption="Final Vipto product image",
            width=500
        )

        h_final, w_final = upscaled.shape[:2]

        st.write(
            f"Final image size: "
            f"**{w_final} × {h_final} pixels**"
        )


        # ====================================================
        # DOWNLOAD
        # ====================================================

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
