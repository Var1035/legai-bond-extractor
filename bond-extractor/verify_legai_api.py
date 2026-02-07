import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_legai_flow():
    print("--- Testing LEGAI API ---")
    
    # 1. Upload logic (Simulated via Chat for direct prompt testing)
    # Since we modified the internal pipeline, we can test /chat-legai directly 
    # with raw text to see if models load and respond.
    
    dummy_text = """
    LEASE AGREEMENT
    
    This Lease Agreement is made on 10th January 2023 between:
    Mr. Rajesh Kumar (hereinafter called Landlord)
    AND
    Ms. Priya Sharma (hereinafter called Tenant)
    
    The monthly rent shall be Rs. 25,000.
    The Security Deposit is Rs. 1,00,000.
    
    If the Tenant commits theft under IPC 378, they shall be liable.
    """
    
    question = "Who is the tenant and what is the rent?"
    
    payload = {
        "text": dummy_text,
        "question": question
    }
    
    print(f"Sending Request to {BASE_URL}/chat-legai...")
    print(f"Question: {question}")
    
    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/chat-legai", json=payload, timeout=300) # Long timeout for model load
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\nSuccess! (Time: {elapsed:.2f}s)")
            print("-" * 30)
            print("Answer:", data.get("answer"))
            print("Entities:", data.get("legal_entities"))
            print("IPC Sections:", data.get("identified_bns_sections"))
            print("-" * 30)
            
            # Simple assertions
            ans = data.get("answer", "").lower()
            if "priya" in ans or "25,000" in ans:
                print("✅ Answer looks relevant.")
            else:
                print("⚠️ Answer might be generic, please check.")
                
        else:
            print(f"❌ Error: {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("Make sure 'python app.py' is running!")

if __name__ == "__main__":
    test_legai_flow()
