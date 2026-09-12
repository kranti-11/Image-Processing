from PIL import Image
from rembg import remove, new_session
import numpy as np
import cv2

input_path = "i2.jpg"
output_path = "milestone2i_output.jpg"
transparent_path = "milestone2i_transparent.png"

img = Image.open(input_path).convert("RGBA")

print("Original size:", img.size, flush=True)

session = new_session("u2net")

print("Removing background...", flush=True)

foreground = remove(
    img,
    session=session
)

print("Background removed.", flush=True)

data = np.array(foreground)

rgb = data[:, :, :3]
alpha = data[:, :, 3]

# -----------------------------------------
# CREATE MASK
# -----------------------------------------

mask = np.where(alpha > 10, 255, 0).astype(np.uint8)

# Smooth small gaps
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------------------
# KEEP MAIN OBJECT
# -----------------------------------------

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mask,
    connectivity=8
)

if num_labels > 1:
    # Ignore background label 0
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

# -----------------------------------------
# SAVE TRANSPARENT PNG
# -----------------------------------------

transparent = np.dstack((rgb, mask))
Image.fromarray(transparent).save(transparent_path)

# -----------------------------------------
# NUMPY WHITE BACKGROUND
# -----------------------------------------

white = np.full_like(rgb, 255)

alpha_float = mask[:, :, None].astype(np.float32) / 255.0

result = (
    rgb.astype(np.float32) * alpha_float
    + white.astype(np.float32) * (1 - alpha_float)
).astype(np.uint8)

Image.fromarray(result).save(
    output_path,
    quality=95
)

print("Saved:", transparent_path, flush=True)
print("Saved:", output_path, flush=True)