import requests
import io

BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
DASHBOARD_ENDPOINT = f"{BASE_URL}/dashboard"
TIMEOUT = 30


def test_post_upload_with_valid_pdf_or_image_file():
    # Prepare a small valid PDF file binary content (simple minimal PDF)
    pdf_content = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< "
        b"/Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 100 Td\n(Hello PDF) Tj\nET\n"
        b"endstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000067 00000 n \n0000000120 00000 n "
        b"\n0000000215 00000 n \ntrailer\n<< /Root 1 0 R /Size 5 >>\nstartxref\n310\n%%EOF\n"
    )

    files = {
        'file': ('test.pdf', io.BytesIO(pdf_content), 'application/pdf')
    }

    response = None
    try:
        # POST /upload with the PDF file
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200 but got {response.status_code}"
        json_data = response.json()
        # Validate JSON response keys and values
        assert 'job_id' in json_data, "Response JSON missing 'job_id'"
        assert json_data.get('status') == 'completed', f"Expected status 'completed', got {json_data.get('status')}"
        assert json_data.get('processed') is True, f"Expected processed True, got {json_data.get('processed')}"
        assert 'results_summary_url' in json_data, "Response JSON missing 'results_summary_url'"

        # GET /dashboard and verify it returns 200 and contains likely document summary text
        dashboard_resp = requests.get(DASHBOARD_ENDPOINT, timeout=TIMEOUT)
        assert dashboard_resp.status_code == 200, f"Dashboard GET expected 200 but got {dashboard_resp.status_code}"
        # Check if content contains expected summary markers; as per PRD likely HTML with summary text
        content = dashboard_resp.text.lower()
        expected_keywords = ['document', 'summary', 'compliance', 'risk', 'clauses']
        assert any(keyword in content for keyword in expected_keywords), "Dashboard content missing document summary indicators"
    finally:
        # Attempt to clean up uploaded document if an API for deletion existed.
        # PRD doesn't specify delete endpoint, so no action here.
        pass


test_post_upload_with_valid_pdf_or_image_file()
