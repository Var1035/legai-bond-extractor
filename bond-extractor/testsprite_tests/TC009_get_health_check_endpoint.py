import requests

def test_health_check_endpoint():
    url = "http://localhost:8000/health"
    headers = {
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        assert False, f"Request to /health endpoint failed: {e}"

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    assert "status" in data, "Response JSON missing 'status'"
    assert data["status"] == "ok", f"Expected status 'ok', got {data['status']}"

    assert "tesseract" in data, "Response JSON missing 'tesseract'"
    assert data["tesseract"] == "available", f"Expected 'tesseract' to be 'available', got {data['tesseract']}"

    assert "version" in data, "Response JSON missing 'version'"
    version = data["version"]
    assert isinstance(version, str) and len(version) > 0, f"Version should be a non-empty string, got: {version}"

test_health_check_endpoint()