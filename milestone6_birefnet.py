import os
import io
import torch
import numpy as np

from PIL import Image, ImageFilter
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


# ============================================================
# MILESTONE 6 — BiRefNet Background Removal
# Mac Apple Silicon / MPS
# ============================================================

MODEL_NAME = "ZhengPeng7/BiRefNet"

INPUT_PATH = "i3.jpg"

OUTPUT_DIR = "final.jpg"

OUTPUT_TRANSPARENT = os.path.join(
    OUTPUT_DIR,
    "birefnet_transparent.png"
)

OUTPUT_WHITE = os.path.join(
    OUTPUT_DIR,
    "birefnet_white.jpg"
)

MASK_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "birefnet_mask.png"
)


# ============================================================
# 1. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. CHECK DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Device: CPU")


# ============================================================
# 3. LOAD IMAGE
# ============================================================

print("Loading image...")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"\nInput image not found:\n{INPUT_PATH}\n\n"
        "Make sure the image exists at:\n"
        "milestone6/images/hard_product.jpg"
    )

original = Image.open(INPUT_PATH).convert("RGB")

original_size = original.size

print(f"Original image size: {original_size}")


# ============================================================
# 4. LOAD BiRefNet
# ============================================================

print("Loading BiRefNet...")

model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model.to(device)
model.eval()

# BiRefNet model weights are FP16.
model.half()

print("BiRefNet loaded successfully.")


# ============================================================
# 5. PREPROCESS IMAGE
# ============================================================

print("Preparing image...")

image_size = (1024, 1024)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

input_tensor = transform(original)

# Add batch dimension
input_tensor = input_tensor.unsqueeze(0)

# IMPORTANT:
# Input must have the same dtype as the FP16 model.
input_tensor = input_tensor.to(device).half()

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Input dtype: {input_tensor.dtype}")


# ============================================================
# 6. RUN BiRefNet
# ============================================================

print("Running BiRefNet...")

with torch.no_grad():

    prediction = model(input_tensor)

    # BiRefNet returns multiple predictions.
    # The final prediction is used for the segmentation mask.
    prediction = prediction[-1]

    prediction = prediction.sigmoid()

    # Remove batch/channel dimensions
    prediction = prediction[0, 0]

    # Move tensor from MPS → CPU
    prediction = prediction.detach().float().cpu()


print("BiRefNet inference complete.")


# ============================================================
# 7. CONVERT PREDICTION → PIL MASK
# ============================================================

print("Creating mask...")

mask_array = prediction.numpy()

# Convert 0–1 → 0–255
mask_array = (mask_array * 255).clip(0, 255).astype(np.uint8)

mask = Image.fromarray(mask_array, mode="L")

# Resize mask back to ORIGINAL image dimensions
mask = mask.resize(
    original_size,
    Image.Resampling.BILINEAR
)

# Small smoothing for cleaner edges
mask = mask.filter(
    ImageFilter.GaussianBlur(radius=0.5)
)


# ============================================================
# 8. SAVE MASK
# ============================================================

mask.save(MASK_OUTPUT)

print(f"Mask saved: {MASK_OUTPUT}")


# ============================================================
# 9. CREATE TRANSPARENT RESULT
# ============================================================

print("Creating transparent result...")

transparent = original.convert("RGBA")

transparent.putalpha(mask)

transparent.save(
    OUTPUT_TRANSPARENT,
    format="PNG"
)

print(
    f"Transparent result saved: "
    f"{OUTPUT_TRANSPARENT}"
)


# ============================================================
# 10. CREATE WHITE BACKGROUND RESULT
# ============================================================

print("Creating white-background result...")

white_background = Image.new(
    "RGBA",
    original_size,
    (255, 255, 255, 255)
)

white_background.alpha_composite(
    transparent
)

white_rgb = white_background.convert("RGB")

white_rgb.save(
    OUTPUT_WHITE,
    format="JPEG",
    quality=95
)

print(
    f"White-background result saved: "
    f"{OUTPUT_WHITE}"
)


# ============================================================
# 11. FINAL INFORMATION
# ============================================================

print("\n========================================")
print("MILESTONE 6 COMPLETE")
print("========================================")

print(f"Input:        {INPUT_PATH}")
print(f"Mask:         {MASK_OUTPUT}")
print(f"Transparent:  {OUTPUT_TRANSPARENT}")
print(f"White result: {OUTPUT_WHITE}")

print("\nBiRefNet successfully processed the image.")