
import os
import json
from dotenv import load_dotenv
from mistralai import Mistral
from legal_risk_engine import evaluate_document
from app import identify_ipc_sections, extract_legal_entities
from fastapi import FastAPI

# Mock app state
app = FastAPI()
class MockState:
    pass
app.state = MockState()

def test_ai_components():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    print(f"API Key present: {bool(api_key)}")
    
    if not api_key:
        print("CRITICAL: MISTRAL_API_KEY missing.")
        return

    try:
        client = Mistral(api_key=api_key)
        app.state.mistral_client = client
        print("Mistral Client initialized.")
        
        # Test 1: Simple Chat
        print("\n--- Test 1: Simple Chat ---")
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Say 'Hello from Debug Script'"}]
        )
        print(f"Response: {resp.choices[0].message.content}")

        # Test 2: Identify IPC Sections
        print("\n--- Test 2: Identify IPC Sections ---")
        sample_text = "The contractor shall be liable under IPC 420 for cheating and Section 27 of Contract Act."
        # mocking ipc_model for fallback regex or allowing Mistral to run
        # Mistral should catch this even if local model is missing because we removed the guard
        # But identify_ipc_sections checks for app.state.ipc_model.
        # Let's mock it as None to force Mistral path (if logic allows) or see if it returns empty.
        # Wait, the code returns [] if ipc_model is missing? 
        # Yes: if not hasattr(app.state, "ipc_model") or not app.state.ipc_model: return []
        # So we MUST have ipc_model loaded for that function to run at all?
        # Let's check app.py code again.
        
    except Exception as e:
        print(f"AI Test Failed: {e}")

if __name__ == "__main__":
    test_ai_components()
