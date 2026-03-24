import requests

def test_post_upload_with_very_large_file_exceeding_limits():
    url = "http://localhost:8000/upload"
    # Generate a large dummy file content exceeding typical limits (e.g., 100MB)
    large_content = b'0' * 1024 * 1024 * 100  # 100 MB
    files = {
        'file': ('large_test_file.pdf', large_content, 'application/pdf')
    }
    try:
        response = requests.post(url, files=files, timeout=30)
    except requests.exceptions.RequestException as e:
        assert False, f"Request failed: {e}"

    assert response.status_code == 413, f"Expected status code 413, got {response.status_code}"

    # The client should display 'file too large'. We check response text or JSON if available
    # Since no JSON schema defined for this error, check if message or indication is present in text or JSON
    content_type = response.headers.get('Content-Type', '')
    if 'application/json' in content_type:
        try:
            json_resp = response.json()
            # The error message should indicate file too large
            messages = [json_resp.get('message', ''), json_resp.get('error', '')]
            assert any('file too large' in str(msg).lower() for msg in messages), \
                f"Response JSON does not contain 'file too large' message: {json_resp}"
        except Exception:
            # If JSON parsing fails, fallback to text check
            assert 'file too large' in response.text.lower(), \
                f"Response text does not contain 'file too large' message: {response.text}"
    else:
        # If not JSON, fallback to text check
        assert 'file too large' in response.text.lower(), \
            f"Response text does not contain 'file too large' message: {response.text}"

test_post_upload_with_very_large_file_exceeding_limits()