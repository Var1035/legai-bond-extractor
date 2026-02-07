import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"Torch: {torch.__version__}")
except ImportError as e:
    print(f"Torch Import Error: {e}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"Transformers Import Error: {e}")

try:
    from transformers import pipeline
    print("Pipeline imported.")
    
    print("Loading model...")
    pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=50)
    print("SUCCESS: Model loaded.")
    
    out = pipe("Test prompt")
    print(f"Output: {out}")

except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
