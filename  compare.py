# compare_results.py
from PIL import Image

isnet_img = Image.open("compare_isnet_output.png")
birefnet_img = Image.open("compare_birefnet_output.png")

# Resize both to the same height for a fair side-by-side
h = 500
isnet_img = isnet_img.resize((int(isnet_img.width * h / isnet_img.height), h))
birefnet_img = birefnet_img.resize((int(birefnet_img.width * h / birefnet_img.height), h))

combined = Image.new("RGBA", (isnet_img.width + birefnet_img.width + 20, h), (255, 255, 255, 255))
combined.paste(isnet_img, (0, 0))
combined.paste(birefnet_img, (isnet_img.width + 20, 0))
combined.save("comparison_side_by_side.png")
print("Saved comparison_side_by_side.png")