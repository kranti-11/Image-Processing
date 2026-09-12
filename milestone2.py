import cv2
import numpy as np

# Load image
img = cv2.imread("packet.jpg")

if img is None:
    print("Error: packet.jpg not found")
    exit()

print("Original image shape:", img.shape)

# Create initial mask
mask = np.zeros(img.shape[:2], np.uint8)

# Rectangle around the object
height, width = img.shape[:2]

margin_x = int(width * 0.05)
margin_y = int(height * 0.05)

rect = (
    margin_x,
    margin_y,
    width - 2 * margin_x,
    height - 2 * margin_y
)

# Background and foreground models
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# GrabCut
cv2.grabCut(
    img,
    mask,
    rect,
    bgd_model,
    fgd_model,
    5,
    cv2.GC_INIT_WITH_RECT
)

# Keep probable/definite foreground
foreground = np.where(
    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
    255,
    0
).astype("uint8")

# Create white background
white = np.ones_like(img) * 255

# Convert mask to 3 channels
foreground_3ch = cv2.cvtColor(
    foreground,
    cv2.COLOR_GRAY2BGR
)

# Composite foreground onto white
result = np.where(
    foreground_3ch == 255,
    img,
    white
)

# Save
cv2.imwrite("milestone2_output.jpg", result)

print("OpenCV background removal completed.")
print("Saved: milestone2_output.jpg")