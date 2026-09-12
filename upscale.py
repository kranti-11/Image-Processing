from PIL import Image
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

input_path = "test.jpg"
output_path = "upscaled.jpg"
model_path = "weights/RealESRGAN_x4plus.pth"

# Load image
img = Image.open(input_path).convert("RGB")
img_np = np.array(img)

print("Original size:", img.size, flush=True)

# Use Apple Silicon GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device, flush=True)
# 
# Real-ESRGAN model
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

print("Model created. Loading weights...", flush=True)

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

print("Model loaded.", flush=True)
print("Starting upscale...", flush=True)

output, _ = upsampler.enhance(
    img_np,
    outscale=2
)

print("Upscale complete.", flush=True)

Image.fromarray(output).save(output_path)

print("Saved:", output_path, flush=True)
print("Output size:", Image.open(output_path).size, flush=True)
output, _ = upsampler.enhance(img_np, outscale=2)

# Save
Image.fromarray(output).save(output_path)

print("Upscaled size:", Image.open(output_path).size)
print("Saved:", output_path)
