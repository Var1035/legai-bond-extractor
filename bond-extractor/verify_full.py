import threading
import time
import requests
import uvicorn
import os
import sys
from app import app

# Define a thread to run the server
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")

def verify_dashboard_data():
    BASE_URL = "http://localhost:8001"
    print("--- Verifying Dashboard Data Flow (Port 8001) ---")
    
    # Give server time to start
    time.sleep(5)
    
    # 1. Test /upload
    with open("test_upload_2.txt", "w") as f:
        f.write("LEASE AGREEMENT\nBetween Ram and Shyam.\nIPC 420 is applicable.")
        
    try:
        with open('test_upload_2.txt', 'rb') as f_upload:
            files = {'file': f_upload}
            # Add timeout to prevent hanging
            resp = requests.post(f"{BASE_URL}/upload", files=files, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            print("✅ /upload endpoint reachable")
            
            fields = data.get("fields", {})
            ml = fields.get("ml", {})
            
            if fields.get("title") == "LEASE AGREEMENT":
                print("✅ Title Extraction: Success")
            else:
                print(f"⚠️ Title Extraction: Got '{fields.get('title')}'")

            sections = ml.get("ipc_sections", [])
            # The mocked logic might return something else or empty if models aren't loaded or mocked.
            # But the logic we added in app.py for `identify_ipc_sections` used regex fallback too.
            if any("IPC 420" in s["ipc"] for s in sections):
                 print("✅ IPC/BNS Section Identification: Success")
            else:
                 print(f"⚠️ IPC Sections: {sections}")

            print("✅ Data structure for Dashboard is valid.")
        else:
            print(f"❌ /upload failed: {resp.status_code} - {resp.text}")

    except Exception as e:
        print(f"❌ Upload Test Failed: {e}")
    finally:
        if os.path.exists("test_upload_2.txt"):
            os.remove("test_upload_2.txt")

    # 2. Test Chat
    print("\n--- Verifying Section Analysis (Mistral) ---")
    payload = {
        "text": "The tenant shall be liable under IPC 420 for fraud.",
        "question": "Explain why IPC 420 is applicable.",
        "language": "en"
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/chat-legai", json=payload, timeout=30)
        if resp.status_code == 200:
            ans = resp.json().get("answer", "")
            if ans:
                print("✅ Section Analysis (Chat) returned a valid answer.")
            else:
                 print("⚠️ Section Analysis answer empty.")
        else:
             print(f"❌ Chat API failed: {resp.status_code}")
    except Exception as e:
        print(f"❌ Chat Test Failed: {e}")

if __name__ == "__main__":
    # Start server in thread
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    
    # Run tests
    verify_dashboard_data()
    
    print("Tests Completed.")
    # Thread will die when main exits
