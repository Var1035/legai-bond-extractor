import requests
import json
from PIL import Image, ImageDraw, ImageFont
import os

# 1. Generate Image
os.makedirs("uploads", exist_ok=True)
img_path = "uploads/test_image.png"
img = Image.new('RGB', (800, 200), color=(255, 255, 255))
d = ImageDraw.Draw(img)
# Use default font or try to load one
try:
    font = ImageFont.truetype("arial.ttf", 24)
except IOError:
    font = ImageFont.load_default()

text = "This is a sample lease agreement for 5 years between A and B."
d.text((10, 80), text, fill=(0, 0, 0), font=font)
img.save(img_path)
print(f"Created test image at {img_path}")

# 2. Upload
url = "http://127.0.0.1:8000/upload"
files = {'file': open(img_path, 'rb')}
try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print("Response JSON:", data)
        with open("self_test_result.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Saved result to self_test_result.json")
    except json.JSONDecodeError:
        print("Failed to decode JSON. Response text:", response.text)
except requests.exceptions.ConnectionError:
    print("Could not connect to server. Is it running?")
