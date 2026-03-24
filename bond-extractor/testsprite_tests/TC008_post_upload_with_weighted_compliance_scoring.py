import requests
import os

BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
DASHBOARD_ENDPOINT = f"{BASE_URL}/dashboard"
TIMEOUT = 30

# Sample PDF file with multiple detected clauses for testing compliance score
TEST_FILE_PATH = "test_docs/multi_clause_contract.pdf"


def test_post_upload_with_weighted_compliance_scoring():
    files = {}
    if not os.path.isfile(TEST_FILE_PATH):
        raise FileNotFoundError(f"Test file not found: {TEST_FILE_PATH}")

    files = {
        "file": ("multi_clause_contract.pdf", open(TEST_FILE_PATH, "rb"), "application/pdf")
    }

    try:
        # POST /upload with the test document
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200 but got {response.status_code}"
        json_resp = response.json()

        # Verify compliance_score, total_weight, detected_weight in response
        assert "compliance_score" in json_resp, "Response missing compliance_score"
        assert isinstance(json_resp["compliance_score"], (int, float)), "Invalid type for compliance_score"
        assert "total_weight" in json_resp, "Response missing total_weight"
        assert isinstance(json_resp["total_weight"], (int, float)), "Invalid type for total_weight"
        assert "detected_weight" in json_resp, "Response missing detected_weight"
        assert isinstance(json_resp["detected_weight"], (int, float)), "Invalid type for detected_weight"

        # GET /dashboard to verify compliance score display
        dash_response = requests.get(DASHBOARD_ENDPOINT, timeout=TIMEOUT)
        assert dash_response.status_code == 200, f"Dashboard GET failed with status {dash_response.status_code}"
        dashboard_html = dash_response.text

        # Check compliance score presence in dashboard text (string representation)
        compliance_score_str = str(json_resp["compliance_score"])
        assert compliance_score_str in dashboard_html, "Compliance score not displayed in dashboard"

    finally:
        if "file" in files:
            files["file"][1].close()


test_post_upload_with_weighted_compliance_scoring()
