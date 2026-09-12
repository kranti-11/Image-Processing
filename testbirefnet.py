# benchmark_birefnet.py
import time
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageSegmentation

input_path = "i7.jpg"   # same image you used for the U²-Net test
output_path = "compare_birefnet_outputi7.png"

# --- Load model (MIT-licensed original BiRefNet, not Bria's RMBG) ---
print("Loading BiRefNet...")
device = "mps" if torch.backends.mps.is_available() else "cpu"

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(device)
birefnet.eval()

# Match the model's dtype (this is the fix for your earlier FP16/FP32 mismatch)
model_dtype = next(birefnet.parameters()).dtype
print(f"Model dtype: {model_dtype}, device: {device}")

# --- Preprocess the image ---
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open(input_path).convert("RGB")
input_tensor = transform_image(image).unsqueeze(0).to(device, dtype=model_dtype)

# --- Run inference, timed ---
print("Running BiRefNet...")
start = time.time()
with torch.no_grad():
    preds = birefnet(input_tensor)[-1].sigmoid().cpu()
elapsed = time.time() - start

# --- Convert prediction tensor to a proper Pillow mask ---
pred = preds[0].squeeze()                     # drop batch/channel dims
pred = pred.float()                            # back to float32 for PIL conversion
mask_pil = transforms.ToPILImage()(pred)
mask_resized = mask_pil.resize(image.size)     # match original image size

# --- Composite onto white using the mask (same approach as your rembg pipeline) ---
image_rgba = image.convert("RGBA")
image_rgba.putalpha(mask_resized)

print(f"BiRefNet: {elapsed:.2f} seconds")
image_rgba.save(output_path)
print(f"Saved: {output_path}")