from PIL import Image
import numpy as np

# Load image
img = Image.open("test.jpg").convert("RGB")

# Convert image to NumPy array
img_array = np.array(img)

print("Image size:", img.size)
print("Array shape:", img_array.shape)
print("Data type:", img_array.dtype)

# Paint a rectangular region white
img_array[100:300, 100:300] = [255, 255, 255]

# Convert back to image
result = Image.fromarray(img_array)

# Save result
result.save("milestone1_output.jpg")

print("Saved: milestone1_output.jpg")
