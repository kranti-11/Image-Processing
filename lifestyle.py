import cv2
import numpy as np
from PIL import Image

def add_shadow(canvas, product_mask, position, scale_factor):
    """Soft blurred ellipse shadow beneath the product."""
    h, w = product_mask.shape[:2]
    shadow_layer = np.zeros(canvas.shape[:2], dtype=np.uint8)
    px, py = position
    ellipse_w = int(w * scale_factor * 0.6)
    ellipse_h = int(ellipse_w * 0.25)
    center = (px + int(w * scale_factor / 2), py + int(h * scale_factor) - ellipse_h // 3)
    cv2.ellipse(shadow_layer, center, (ellipse_w, ellipse_h), 0, 0, 360, 120, -1)
    shadow_layer = cv2.GaussianBlur(shadow_layer, (31, 31), 0)
    shadow_rgb = cv2.cvtColor(shadow_layer, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    canvas[:] = (canvas * (1 - shadow_rgb * 0.4)).astype(np.uint8)
    return canvas

def match_lighting(product_rgb, background, position, scale_factor):
    """Nudge product brightness/warmth toward the background's tone, so it doesn't look pasted."""
    h, w = product_rgb.shape[:2]
    px, py = position
    bg_patch = background[py:py+int(h*scale_factor), px:px+int(w*scale_factor)]
    if bg_patch.size == 0:
        return product_rgb
    bg_avg = bg_patch.reshape(-1, 3).mean(axis=0)
    product_avg = product_rgb.reshape(-1, 3).mean(axis=0)
    shift = (bg_avg - product_avg) * 0.15
    matched = np.clip(product_rgb.astype(float) + shift, 0, 255).astype(np.uint8)
    return matched

def composite_product_on_background(product_path, background_path, output_path,
                                      scale_factor=0.5, surface_y=0.75):
    product = Image.open(product_path).convert("RGBA")
    background = Image.open(background_path).convert("RGB")

    bg_w, bg_h = 1200, 1200
    background = background.resize((bg_w, bg_h))
    bg_np = np.array(background)

    prod_w = int(bg_w * scale_factor)
    prod_h = int(product.height * (prod_w / product.width))
    product = product.resize((prod_w, prod_h))
    product_np = np.array(product)

    alpha_raw = product_np[:, :, 3]
    kernel = np.ones((5, 5), np.uint8)
    alpha_clean = cv2.morphologyEx(alpha_raw, cv2.MORPH_CLOSE, kernel)
    alpha_clean = cv2.morphologyEx(alpha_clean, cv2.MORPH_OPEN, kernel)
    product_np[:, :, 3] = alpha_clean

    px = (bg_w - prod_w) // 2
    py = int(bg_h * surface_y) - prod_h

    canvas = bg_np.copy()
    canvas = add_shadow(canvas, product_np[:, :, 3], (px, py), 1.0)

    product_rgb = product_np[:, :, :3]
    product_rgb = match_lighting(product_rgb, canvas, (px, py), 1.0)
    alpha = product_np[:, :, 3:4] / 255.0

    roi = canvas[py:py+prod_h, px:px+prod_w]
    blended = (product_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    canvas[py:py+prod_h, px:px+prod_w] = blended

    Image.fromarray(canvas).save(output_path)
    print(f"Saved: {output_path}")

# Test on one background first
composite_product_on_background(
    "milestone2i_transparent.png",
    "pexels1.jpg",
    "lifestyle_test1_2.jpg",
    scale_factor=0.4,
    surface_y=0.87
)
