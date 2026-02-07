import sys
import os
# Add the project directory to path so we can import app
sys.path.append(os.path.join(os.getcwd(), 'bond-extractor'))

from app import extract_fields_from_text, ml_entities, ml_purpose_scores, hybrid_ocr_image
from PIL import Image

# Path to the uploaded image
IMAGE_PATH = r"C:/Users/varun/.gemini/antigravity/brain/767b95dc-75a2-4ef1-ab7d-c6c24737451b/uploaded_image_1766934549545.png"

def test_extraction():
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found at {IMAGE_PATH}")
        return

    print(f"Loading image: {IMAGE_PATH}")
    try:
        img = Image.open(IMAGE_PATH)
    except Exception as e:
        print(f"Failed to open image: {e}")
        return

    print("Running OCR...")
    # Use just tesseract for speed if others not set up, or hybrid if possible
    res = hybrid_ocr_image(img, engines=['tesseract']) 
    text = res.get("text", "")
    print("\n--- Testing Purpose Detection ---")
    fields = extract_fields_from_text(text)
    
    print(f"Detected Purpose: {fields['purpose']}")
    
    ml_purp = ml_purpose_scores(text)
    print(f"ML Purpose Scores: {ml_purp}")

    print("\n--- Testing Parties Extraction ---")
    print(f"Parties: {fields['parties']}")

    print("\n--- Testing Entities ---")
    ents = ml_entities(text)

    with open("debug_log.txt", "w", encoding="utf-8") as f:
        f.write(f"--- OCR Text ---\n{text}\n----------------\n")
        f.write(f"Purpose: {fields['purpose']}\n")
        f.write(f"Parties: {fields['parties']}\n")
        f.write(f"ML Purpose: {ml_purp}\n")
        f.write(f"Entities: {ents}\n")
    print(f"Entities Found: {ents}")
    if ents:
        first = ents[0]
        if 'type' not in first or 'value' not in first:
            print("FAILURE: Entity keys 'type' and 'value' missing! Frontend will show undefined.")
        else:
            print("SUCCESS: Entity keys present.")
            
    # Assertions
    if fields['purpose'] != "Rental Agreement":
        print(f"FAILURE: Purpose is {fields['purpose']}, expected 'Rental Agreement'")
        sys.exit(1)
    else:
        print("SUCCESS: Purpose is Rental Agreement")

    parties_str = str(fields['parties'])
    if "Gemini" in parties_str:
         print("FAILURE: 'Gemini' found in parties!")
         sys.exit(1)
    else:
         print("SUCCESS: Parties look clean (no Gemini)")


if __name__ == "__main__":
    test_extraction()
