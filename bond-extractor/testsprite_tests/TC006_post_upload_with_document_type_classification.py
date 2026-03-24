import requests

BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
DASHBOARD_ENDPOINT = f"{BASE_URL}/dashboard"
TIMEOUT = 30

def test_post_upload_with_document_type_classification():
    # Use fixed file name assuming it is in the current working directory
    sample_pdf_path = "employment_agreement_sample.pdf"

    # Open file with context manager to ensure proper closing
    with open(sample_pdf_path, "rb") as f:
        files = {"file": ("employment_agreement_sample.pdf", f, "application/pdf")}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=TIMEOUT)

    # Assert upload response status code
    assert response.status_code == 200, f"Expected status 200, got {response.status_code}"

    json_response = response.json()

    # Assert document_type and type_confidence present and validate types and reasonable confidence score
    assert "document_type" in json_response, "Response missing 'document_type'"
    assert isinstance(json_response["document_type"], str), "'document_type' is not a string"
    assert json_response["document_type"].lower() == "employment agreement", f"Unexpected document_type: {json_response['document_type']}"

    assert "type_confidence" in json_response, "Response missing 'type_confidence'"
    assert isinstance(json_response["type_confidence"], (float, int)), "'type_confidence' is not a number"
    assert 0 <= json_response["type_confidence"] <= 1, f"Invalid type_confidence value: {json_response['type_confidence']}"

    # After upload, fetch dashboard
    dash_response = requests.get(DASHBOARD_ENDPOINT, timeout=TIMEOUT)
    assert dash_response.status_code == 200, f"Dashboard GET returned status {dash_response.status_code}"
    dash_text = dash_response.text

    # Assert the dashboard HTML includes the expected document type
    assert "Employment Agreement" in dash_text, "Dashboard does not display the correct document type 'Employment Agreement'"

test_post_upload_with_document_type_classification()
