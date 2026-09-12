import cv2
import numpy as np

# Use the white-background result from Milestone 3
img = cv2.imread("milestonei3_output.jpg")

# Find where the product is (anything not pure white)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

coords = cv2.findNonZero(thresh)
x, y, w, h = cv2.boundingRect(coords)

# Crop tightly to the product
cropped = img[y:y+h, x:x+w]

# Add consistent white padding (10% of the larger dimension)
pad = int(max(w, h) * 0.1)
padded = cv2.copyMakeBorder(cropped, pad, pad, pad, pad,
                             cv2.BORDER_CONSTANT, value=[255, 255, 255])

# Make it square by padding the shorter side, then resize to a standard size
h2, w2 = padded.shape[:2]
side = max(h2, w2)
square = cv2.copyMakeBorder(
    padded,
    (side - h2)//2, side - h2 - (side - h2)//2,
    (side - w2)//2, side - w2 - (side - w2)//2,
    cv2.BORDER_CONSTANT, value=[255, 255, 255]
)

resized = cv2.resize(square, (1000, 1000))
cv2.imwrite("milestone4_output.jpg", resized)
print("Saved milestone4_output.jpg")