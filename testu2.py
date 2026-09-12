# benchmark_isnet.py
import time
from PIL import Image
from rembg import remove, new_session
import numpy as np
import cv2

input_path = "i1.jpg"   # use the same hard test image both times
session = new_session("isnet-general-use")

img = Image.open(input_path).convert("RGBA")

start = time.time()
foreground = remove(img, session=session)
elapsed = time.time() - start

foreground.save("compare_isnet_outputi1.png")
print(f"isnet-general-use: {elapsed:.2f} seconds")
print(f"Processing: {input_path}")