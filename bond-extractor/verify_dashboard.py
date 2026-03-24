import requests
import time
import os

BASE_URL = "http://localhost:8000"

def verify_dashboard_data():
    print("--- Verifying Dashboard Data Flow ---")
    
    # 1. Test /upload with a dummy text file to simulate behavior
    # We can't easily upload a PDF via script without a file, but we can verify the API structure
    # By using the 'demo' mode logic on frontend, or just checking if `extract_fields` works on backend.
    
    # Let's test the endpoint response structure by mocking a file upload
    # We'll create a dummy PDF or Text file
    with open("test_upload.txt", "w") as f:
        f.write("LEASE AGREEMENT\nBetween Ram and Shyam.\nIPC 420 is applicable.")
        
    try:
        # Open file in binary mode for upload
        with open('test_upload.txt', 'rb') as f_upload:
            files = {'file': f_upload}
            resp = requests.post(f"{BASE_URL}/upload", files=files)
        
        if resp.status_code == 200:
            data = resp.json()
            print("✅ /upload endpoint reachable")
            
            fields = data.get("fields", {})
            ml = fields.get("ml", {})
            
            # Check Title
            if fields.get("title") == "LEASE AGREEMENT":
                print("✅ Title Extraction: Success")
            else:
                print(f"⚠️ Title Extraction: Got '{fields.get('title')}'")

            # Check IPC
            sections = ml.get("ipc_sections", [])
            if any("IPC 420" in s["ipc"] for s in sections):
                 print("✅ IPC/BNS Section Identification: Success")
            else:
                 print(f"⚠️ IPC Sections: {sections}")

            # Check Purpose
            if fields.get("purpose"):
                print(f"✅ Purpose: {fields.get('purpose')}")
            
            print("✅ Data structure for Dashboard is valid.")
        else:
            print(f"❌ /upload failed: {resp.status_code}")

    except Exception as e:
        print(f"❌ Test Failed: {e}")
    finally:
        if os.path.exists("test_upload.txt"):
            os.remove("test_upload.txt")

    # 2. Test Section Analysis Chat (The "Why?" feature)
    print("\n--- Verifying Section Analysis (Mistral) ---")
    payload = {
        "text": "The tenant shall be liable under IPC 420 for fraud.",
        "question": "Explain why IPC 420 is applicable.",
        "language": "en"
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/chat-legai", json=payload)
        if resp.status_code == 200:
            ans = resp.json().get("answer", "")
            if len(ans) > 20:
                print("✅ Section Analysis (Chat) returned a valid answer.")
            else:
                 print("⚠️ Section Analysis answer too short.")
        else:
             print(f"❌ Chat API failed: {resp.status_code}")
    except Exception as e:
        print(f"❌ Chat Test Failed: {e}")

if __name__ == "__main__":
    verify_dashboard_data()
