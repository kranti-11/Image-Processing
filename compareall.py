# compare_all.py
from PIL import Image, ImageDraw

# Match whatever numbers you actually have output pairs for
image_ids = ["i1", "i2", "i3", "i5", "i6", "i7"]

row_height = 300
label_height = 30
rows = []

for img_id in image_ids:
    try:
        isnet_img = Image.open(f"compare_isnet_output{img_id}.png").convert("RGBA")
        birefnet_img = Image.open(f"compare_birefnet_output{img_id}.png").convert("RGBA")
    except FileNotFoundError as e:
        print(f"Skipping {img_id}: {e}")
        continue

    isnet_img = isnet_img.resize((int(isnet_img.width * row_height / isnet_img.height), row_height))
    birefnet_img = birefnet_img.resize((int(birefnet_img.width * row_height / birefnet_img.height), row_height))

    row_width = isnet_img.width + birefnet_img.width + 20
    row = Image.new("RGBA", (row_width, row_height + label_height), (255, 255, 255, 255))

    draw = ImageDraw.Draw(row)
    draw.text((10, 5), f"{img_id} — isnet (left) vs BiRefNet (right)", fill=(0, 0, 0, 255))

    row.paste(isnet_img, (0, label_height))
    row.paste(birefnet_img, (isnet_img.width + 20, label_height))
    rows.append(row)

# Stack all rows into one tall image
max_width = max(r.width for r in rows)
total_height = sum(r.height for r in rows) + (10 * (len(rows) - 1))

grid = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 255))
y = 0
for row in rows:
    grid.paste(row, (0, y))
    y += row.height + 10

grid.save("comparison_grid_all.png")
print(f"Saved comparison_grid_all.png with {len(rows)} test images")