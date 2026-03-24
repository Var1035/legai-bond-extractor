# TestSprite AI Testing Report (Frontend fallback to API)

---

## 1️⃣ Document Metadata
- **Project Name:** bond-extractor
- **Date:** 2026-03-24
- **Prepared by:** TestSprite AI & Antigravity

---

## 2️⃣ Requirement Validation Summary

### Requirement: Document Upload Endpoints & Edge Cases

#### Test TC001 post upload with valid pdf or image file
- **Test Code:** [TC001_post_upload_with_valid_pdf_or_image_file.py](./TC001_post_upload_with_valid_pdf_or_image_file.py)
- **Status:** ❌ Failed
- **Analysis / Findings:** The test expects a `job_id` returned from the API, but the endpoint synchronously returns the extracted text and risk assessment directly instead of an async job ID.

#### Test TC002 post upload with unsupported file type
- **Test Code:** [TC002_post_upload_with_unsupported_file_type.py](./TC002_post_upload_with_unsupported_file_type.py)
- **Status:** ❌ Failed
- **Analysis / Findings:** The application throws an HTTP 400 error via FastAPI's generic format (`{"detail": "..."}`), whereas the test strictly looks for an `error` key in the JSON payload.

#### Test TC003 post upload with very large file exceeding limits
- **Test Code:** [TC003_post_upload_with_very_large_file_exceeding_limits.py](./TC003_post_upload_with_very_large_file_exceeding_limits.py)
- **Status:** ❌ Failed
- **Analysis / Findings:** Network timeout. The connection was aborted while uploading a massive file across the test tunnel.

#### Test TC004 post upload with scanned pdf for ocr extraction
- **Test Code:** [TC004_post_upload_with_scanned_pdf_for_ocr_extraction.py](./TC004_post_upload_with_scanned_pdf_for_ocr_extraction.py)
- **Status:** ❌ Failed
- **Analysis / Findings:** The test expects the API response to include an explicit `ocr_engine` key. The current API responds with `"method_used": "hybrid"` and a list under `hybrid_details` but no root `ocr_engine` key.

#### Test TC005 post upload with low quality scan
- **Test Code:** [TC005_post_upload_with_low_quality_scan.py](./TC005_post_upload_with_low_quality_scan.py)
- **Status:** ❌ Failed
- **Analysis / Findings:** The generated test asserts that `ocr_warnings` is a list, but this key does not exist in the API specification.


### Requirement: Advanced Parsing & Classification

#### Test TC006 post upload with document type classification
- **Test Code:** [TC006_post_upload_with_document_type_classification.py](./TC006_post_upload_with_document_type_classification.py)
- **Status:** ❌ Failed
- **Analysis / Findings:** `FileNotFoundError`. The test script attempts to read `employment_agreement_sample.pdf`, which doesn't exist in the local directory.

#### Test TC007 post upload with clause detection
- **Test Code:** [TC007_post_upload_with_clause_detection.py](./TC007_post_upload_with_clause_detection.py)
- **Status:** ❌ Failed
- **Analysis / Findings:** The test explicitly looks for a `clauses` key in the response JSON. The AI agent doesn't extract clauses into an array, rather it extracts overall purpose, parties, and summary.

#### Test TC008 post upload with weighted compliance scoring
- **Test Code:** [TC008_post_upload_with_weighted_compliance_scoring.py](./TC008_post_upload_with_weighted_compliance_scoring.py)
- **Status:** ❌ Failed
- **Analysis / Findings:** `FileNotFoundError`. Missing mock document `test_docs/multi_clause_contract.pdf` required by the test.

---

## 3️⃣ Coverage & Matching Metrics

- **0.00%** of tests passed

| Requirement | Total Tests | ✅ Passed | ❌ Failed |
| --- | --- | --- | --- |
| Document Upload Endpoints & Edge Cases | 5 | 0 | 5 |
| Advanced Parsing & Classification | 3 | 0 | 3 |
| Health Checks | 2 | 0 | 2 |
| **Total** | 10 | 0 | 10 |

---

## 4️⃣ Key Gaps / Risks
1. **Frontend Testing Limitation**: Although the "frontend test plan" tool was triggered, TestSprite detected the project as a Python monolithic backend (based on the previous config logic) and opted to test the static HTTP paths and REST endpoints instead of running a headless browser. To properly test frontend logic strictly on static HTML files, Playwright/Cypress end-to-end tests are necessary.
2. **Schema Assumptions**: The generated API tests assume incorrect response structures (`job_id`, `ocr_engine`, `clauses`, `ocr_warnings`) that don't match the current application contract.
3. **Missing Fixtures**: Test files like `employment_agreement_sample.pdf` and `test_docs/multi_clause_contract.pdf` are missing from the repository, causing automatic test suite failures before requests are even sent.
