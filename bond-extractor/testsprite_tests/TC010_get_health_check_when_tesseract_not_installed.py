import requests

def test_get_health_check_when_tesseract_not_installed():
    url = "http://localhost:8000/health"
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as e:
        assert False, f"Request to {url} failed: {e}"

    # The service should return a 500 status code when tesseract is not installed
    assert response.status_code == 500, f"Expected status code 500, got {response.status_code}"

    try:
        json_data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Validate expected JSON content
    assert json_data.get("status") == "error", f"Expected status 'error', got {json_data.get('status')}"
    assert json_data.get("component") == "tesseract", f"Expected component 'tesseract', got {json_data.get('component')}"
    message = json_data.get("message")
    assert isinstance(message, str) and "tesseract" in message.lower(), "Expected error message mentioning 'tesseract'"

test_get_health_check_when_tesseract_not_installed()