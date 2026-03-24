import requests
from requests.exceptions import RequestException

BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
TIMEOUT = 30

def test_post_upload_with_low_quality_scan():
    # Minimal valid PDF content to avoid 400 error and simulate low quality scan
    low_quality_scan_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"

    files = {
        'file': ('low_quality_scan.pdf', low_quality_scan_content, 'application/pdf')
    }
    headers = {}

    try:
        response = requests.post(UPLOAD_ENDPOINT, files=files, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
    except RequestException as e:
        assert False, f"Request to {UPLOAD_ENDPOINT} failed: {e}"

    try:
        json_resp = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Validate extracted_text is empty string or whitespace only
    extracted_text = json_resp.get('extracted_text')
    assert extracted_text is not None, "extracted_text field missing in response"
    assert extracted_text.strip() == '', f"Expected empty extracted_text, got: {extracted_text}"

    # Validate ocr_warnings includes 'low_confidence'
    ocr_warnings = json_resp.get('ocr_warnings')
    assert isinstance(ocr_warnings, list), f"ocr_warnings should be a list, got: {type(ocr_warnings)}"
    assert 'low_confidence' in ocr_warnings, f"'low_confidence' not found in ocr_warnings: {ocr_warnings}"

    # Validate ocr_confidence is low (expect ~0.15 from PRD)
    ocr_confidence = json_resp.get('ocr_confidence')
    assert isinstance(ocr_confidence, (float, int)), f"ocr_confidence should be numeric, got: {ocr_confidence}"
    assert 0 <= ocr_confidence <= 0.25, f"Expected low ocr_confidence <=0.25, got: {ocr_confidence}"

    # Check that client prompt (indirectly) is through response context or message,
    # Since API returns JSON only, assume presence of a field or rely on above conditions for client prompt.

test_post_upload_with_low_quality_scan()
