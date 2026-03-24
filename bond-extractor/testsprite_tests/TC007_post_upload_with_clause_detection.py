import requests

BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
DASHBOARD_ENDPOINT = f"{BASE_URL}/dashboard"
TIMEOUT = 30

def test_post_upload_with_clause_detection():
    # Prepare a sample PDF content with legal clauses (simulate minimal PDF binary content)
    # In real scenario, you would load a real PDF file.
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 55 >>\nstream\n"
        b"BT /F1 24 Tf 100 700 Td (Termination Clause Text Page 5) Tj ET\n"
        b"BT /F1 24 Tf 100 680 Td (Arbitration Clause Text) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n"
        b"0000000101 00000 n \n0000000178 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n260\n%%EOF"
    )
    files = {
        "file": ("test_contract_clauses.pdf", pdf_content, "application/pdf")
    }

    upload_response = None
    try:
        upload_response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=TIMEOUT)
        assert upload_response.status_code == 200, f"Expected 200 OK but got {upload_response.status_code}"
        json_resp = upload_response.json()

        # Validate that response contains clauses with required fields and missing_clauses as empty list
        assert "clauses" in json_resp, "Response JSON missing 'clauses' key"
        assert isinstance(json_resp["clauses"], list), "'clauses' is not a list"
        assert "missing_clauses" in json_resp, "Response JSON missing 'missing_clauses' key"
        assert isinstance(json_resp["missing_clauses"], list), "'missing_clauses' is not a list"
        assert len(json_resp["missing_clauses"]) == 0, "Expected 'missing_clauses' to be empty list"

        for clause in json_resp["clauses"]:
            assert "name" in clause and isinstance(clause["name"], str) and clause["name"], "Clause missing valid 'name'"
            assert "text" in clause and isinstance(clause["text"], str) and clause["text"], "Clause missing valid 'text'"
            # page is optional according to PRD example, but if present should be int
            if "page" in clause:
                assert isinstance(clause["page"], int), "'page' in clause is not integer"
            assert "confidence" in clause and isinstance(clause["confidence"], (float, int)), "Clause missing valid 'confidence'"
            assert 0.0 <= float(clause["confidence"]) <= 1.0, "'confidence' not between 0 and 1"

        # After upload, verify /dashboard endpoint shows detected clauses with links
        dashboard_response = requests.get(DASHBOARD_ENDPOINT, timeout=TIMEOUT)
        assert dashboard_response.status_code == 200, f"/dashboard returned {dashboard_response.status_code} instead of 200"

        dashboard_content = dashboard_response.text
        # Validate the dashboard contains at least the names of detected clauses as links (anchors)
        for clause in json_resp["clauses"]:
            clause_name = clause["name"]
            # Check if clause name string appears as a link in dashboard HTML: e.g. <a ...>clause_name</a>
            assert f">{clause_name}<" in dashboard_content or clause_name in dashboard_content, (
                f"Dashboard does not contain clause name '{clause_name}' with links"
            )

    finally:
        # No API for deletion; no cleanup needed for upload resource assumed
        pass


test_post_upload_with_clause_detection()