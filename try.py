from PIL import Image
from rembg import remove, new_session
import numpy as np
import cv2

input_path = "packet.jpg"
output_path = "try_output.jpg"

img = Image.open(input_path).convert("RGBA")
print("Original size:", img.size)

session = new_session("isnet-general-use")

print("Removing background...", flush=True)
foreground = remove(img, session=session)   # no alpha_matting argument
print("Background removed.", flush=True)

foreground_np = np.array(foreground)
rgb = foreground_np[:, :, :3]
alpha = foreground_np[:, :, 3]   # keep as 0-255 for OpenCV

# --- mask refinement (what we covered earlier) ---
kernel = np.ones((5, 5), np.uint8)
alpha_clean = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)   # remove specks
alpha_clean = cv2.morphologyEx(alpha_clean, cv2.MORPH_CLOSE, kernel)  # fill holes
alpha_feathered = cv2.GaussianBlur(alpha_clean, (5, 5), 0)      # soften edges
# ---------------------------------------------------

alpha_norm = alpha_feathered[:, :, None] / 255.0
white = np.ones_like(rgb) * 255
result = (rgb * alpha_norm + white * (1 - alpha_norm)).astype(np.uint8)

Image.fromarray(result).save(output_path)
print("Saved:", output_path)