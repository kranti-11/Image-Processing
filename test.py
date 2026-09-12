from PIL import Image, ImageOps
import numpy as np

img = Image.open("test.jpg")

print("Original image:")
print("Size:", img.size)
print("Format:", img.format)
print("Mode:", img.mode)

# Fix EXIF rotation
img = ImageOps.exif_transpose(img)

# Convert to RGB
img = img.convert("RGB")

# Convert to NumPy
arr = np.array(img)

print("\nAfter preprocessing:")
print("Size:", img.size)
print("Mode:", img.mode)
print("Array shape:", arr.shape)

# Upscale low-resolution images
width, height = img.size

if width < 800 or height < 800:
    print("\nLow resolution detected. Upscaling 2x...")

    img = img.resize(
        (width * 2, height * 2),
        Image.Resampling.LANCZOS
    )

    print("Upscaled size:", img.size)
else:
    print("\nResolution is sufficient. No upscaling needed.")

gray = np.mean(arr, axis=2)
brightness = gray.mean()

print("Average brightness:", round(brightness, 2))

if brightness < 50:
    print("Warning: Image is too dark")
elif brightness > 220:
    print("Warning: Image is too bright")
else:
    print("Brightness: Good")

# Save processed image
img.save("prepared.jpg", format="JPEG", quality=95)

#display the image
#img.show()

print("\nPrepared image saved successfully.")