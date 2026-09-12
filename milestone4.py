from PIL import Image
import numpy as np

# -----------------------------
# SETTINGS
# -----------------------------

input_path = "milestonei3_transparent.png"
output_path = "milestonei3_output.png"

# Final fixed canvas size
CANVAS_SIZE = (800, 800)

# Maximum product size inside canvas
MAX_PRODUCT_SIZE = (650, 650)


# -----------------------------
# LOAD IMAGE
# -----------------------------

img = Image.open(input_path).convert("RGBA")

print("Original size:", img.size)


# -----------------------------
# FIND PRODUCT BOUNDING BOX
# -----------------------------

# Get alpha channel
alpha = np.array(img)[:, :, 3]

# Find pixels that are not transparent
ys, xs = np.where(alpha > 10)

if len(xs) == 0:
    raise ValueError("No product detected in image.")

left = xs.min()
top = ys.min()
right = xs.max() + 1
bottom = ys.max() + 1

print("Bounding box:")
print("Left:", left)
print("Top:", top)
print("Right:", right)
print("Bottom:", bottom)


# -----------------------------
# CROP PRODUCT
# -----------------------------

cropped = img.crop((left, top, right, bottom))

print("Cropped size:", cropped.size)


# -----------------------------
# RESIZE WHILE KEEPING
# ASPECT RATIO
# -----------------------------

cropped.thumbnail(
    MAX_PRODUCT_SIZE,
    Image.Resampling.LANCZOS
)

print("Resized product:", cropped.size)


# -----------------------------
# CREATE WHITE CANVAS
# -----------------------------

canvas = Image.new(
    "RGBA",
    CANVAS_SIZE,
    (255, 255, 255, 255)
)


# -----------------------------
# CENTER PRODUCT
# -----------------------------

canvas_x = (CANVAS_SIZE[0] - cropped.width) // 2
canvas_y = (CANVAS_SIZE[1] - cropped.height) // 2

print("Product position:", canvas_x, canvas_y)


# -----------------------------
# COMPOSITE PRODUCT
# -----------------------------

canvas.alpha_composite(
    cropped,
    (canvas_x, canvas_y)
)


# -----------------------------
# SAVE
# -----------------------------

canvas.convert("RGB").save(
    output_path,
    quality=95
)

print("Final image saved:", output_path)
print("Final size:", canvas.size)