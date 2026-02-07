import os
import sys
from PIL import Image, ImageDraw

# Create dummy image
try:
    img = Image.new('RGB', (400, 100), color='white')
    d = ImageDraw.Draw(img)
    d.text((10, 40), "Hello World Test", fill='black')
except Exception as e:
    print(f"Image creation failed: {e}")
    sys.exit(1)

print("--- Testing TrOCR ---", flush=True)
try:
    from transformers import pipeline
    # Use CPU explicitly
    pipe = pipeline("image-to-text", model="microsoft/trocr-base-handwritten", device=-1)
    res = pipe(img)
    print("TrOCR result:", res, flush=True)
except Exception as e:
    print("TrOCR failed:", e, flush=True)

print("\n--- Testing PaddleOCR ---", flush=True)
try:
    from paddleocr import PaddleOCR
    # Minimal init
    ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False) 
    result = ocr.ocr(img, cls=False)
    print("PaddleOCR result:", result, flush=True)
except Exception as e:
    print("PaddleOCR failed:", e, flush=True)
