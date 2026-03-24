import requests

def test_post_upload_with_unsupported_file_type():
    base_url = "http://localhost:8000"
    upload_url = f"{base_url}/upload"
    
    # Prepare a dummy .exe file content for upload (unsupported media type)
    file_content = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00"
    files = {
        "file": ("dummy.exe", file_content, "application/x-msdownload")
    }
    
    try:
        response = requests.post(upload_url, files=files, timeout=30)
    except requests.RequestException as e:
        assert False, f"Request failed: {e}"
    
    assert response.status_code == 400, f"Expected status code 400 but got {response.status_code}"
    
    try:
        resp_json = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"
    
    assert "error" in resp_json, "Response JSON missing 'error' key"
    assert "message" in resp_json, "Response JSON missing 'message' key"
    assert resp_json["error"] == "unsupported_media_type", f"Expected error 'unsupported_media_type' but got {resp_json['error']}"
    assert "Only PDF/DOCX/Images are supported" in resp_json["message"], f"Error message does not indicate unsupported media type: {resp_json['message']}"

test_post_upload_with_unsupported_file_type()