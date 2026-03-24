import requests

BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
DASHBOARD_ENDPOINT = f"{BASE_URL}/dashboard"
TIMEOUT = 30

def test_post_upload_scanned_pdf_ocr_extraction():
    scanned_pdf_path = "scanned_sample.pdf"
    # Provide a minimal valid PDF content for upload with proper structure.
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 100 700 Td (Scanned PDF) Tj ET\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000212 00000 n \ntrailer<< /Root 1 0 R /Size 5 >>\nstartxref\n293\n%%EOF\n"
    )

    files = {
        'file': ('scanned_sample.pdf', pdf_bytes, 'application/pdf')
    }

    response_upload = None
    try:
        response_upload = requests.post(UPLOAD_ENDPOINT, files=files, timeout=TIMEOUT)
        assert response_upload.status_code == 200, f"Expected 200 OK, got {response_upload.status_code}"

        json_data = response_upload.json()
        # Validate presence and content of OCR results
        assert 'extracted_text' in json_data, "Response missing 'extracted_text'"
        extracted_text = json_data['extracted_text']
        assert isinstance(extracted_text, str), "'extracted_text' should be a string"
        assert len(extracted_text.strip()) > 0, "'extracted_text' should not be empty"

        assert 'ocr_engine' in json_data, "Response missing 'ocr_engine'"
        ocr_engine = json_data['ocr_engine']
        assert ocr_engine in ('tesseract', 'trocr', 'paddle'), f"Unexpected OCR engine: {ocr_engine}"

        assert 'ocr_confidence' in json_data, "Response missing 'ocr_confidence'"
        ocr_confidence = json_data['ocr_confidence']
        assert isinstance(ocr_confidence, (float, int)), "'ocr_confidence' should be a number"
        assert ocr_confidence > 0.7, f"OCR confidence too low: {ocr_confidence}"

        # Verify dashboard shows extracted text preview
        dashboard_resp = requests.get(DASHBOARD_ENDPOINT, timeout=TIMEOUT)
        assert dashboard_resp.status_code == 200, f"Dashboard GET failed with status {dashboard_resp.status_code}"
        dashboard_html = dashboard_resp.text
        assert extracted_text[:30].strip() in dashboard_html or extracted_text.strip() in dashboard_html, \
            "Extracted text preview not found in dashboard output"

    finally:
        # There is no delete endpoint specified in PRD to clean uploaded documents.
        # If such endpoint is available, resource cleanup code should be added here.
        pass

test_post_upload_scanned_pdf_ocr_extraction()
