
import os
from dotenv import load_dotenv
from mistralai import Mistral

def test_mistral_only():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    print(f"API Key found: {bool(api_key)}")
    
    if not api_key:
        print("CRITICAL: MISTRAL_API_KEY missing.")
        return

    try:
        client = Mistral(api_key=api_key)
        print("Mistral Client initialized.")
        
        # Test 1: Simple Chat
        print("\n--- Test 1: Simple Chat ---")
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Just say 'Works'"}]
        )
        print(f"Response: {resp.choices[0].message.content}")

    except Exception as e:
        print(f"Mistral Test Failed: {e}")

if __name__ == "__main__":
    test_mistral_only()
