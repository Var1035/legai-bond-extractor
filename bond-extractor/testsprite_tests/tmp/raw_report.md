
# TestSprite AI Testing Report(MCP)

---

## 1️⃣ Document Metadata
- **Project Name:** bond-extractor
- **Date:** 2026-03-24
- **Prepared by:** TestSprite AI Team

---

## 2️⃣ Requirement Validation Summary

#### Test TC001 post upload with valid pdf or image file
- **Test Code:** [TC001_post_upload_with_valid_pdf_or_image_file.py](./TC001_post_upload_with_valid_pdf_or_image_file.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 49, in <module>
  File "<string>", line 31, in test_post_upload_with_valid_pdf_or_image_file
AssertionError: Response JSON missing 'job_id'

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/bd9de1ed-4ade-427c-8220-b319c9c58cd2
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC002 post upload with unsupported file type
- **Test Code:** [TC002_post_upload_with_unsupported_file_type.py](./TC002_post_upload_with_unsupported_file_type.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 30, in <module>
  File "<string>", line 25, in test_post_upload_with_unsupported_file_type
AssertionError: Response JSON missing 'error' key

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/644e97d5-f6cb-4ff8-a088-c338ae187d8b
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC003 post upload with very large file exceeding limits
- **Test Code:** [TC003_post_upload_with_very_large_file_exceeding_limits.py](./TC003_post_upload_with_very_large_file_exceeding_limits.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/urllib3/connectionpool.py", line 787, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "/var/task/urllib3/connectionpool.py", line 493, in _make_request
    conn.request(
  File "/var/task/urllib3/connection.py", line 508, in request
    self.send(chunk)
  File "/var/lang/lib/python3.12/http/client.py", line 1057, in send
    self.sock.sendall(data)
TimeoutError: timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/task/requests/adapters.py", line 667, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "/var/task/urllib3/connectionpool.py", line 841, in urlopen
    retries = retries.increment(
              ^^^^^^^^^^^^^^^^^^
  File "/var/task/urllib3/util/retry.py", line 474, in increment
    raise reraise(type(error), error, _stacktrace)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/var/task/urllib3/util/util.py", line 38, in reraise
    raise value.with_traceback(tb)
  File "/var/task/urllib3/connectionpool.py", line 787, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "/var/task/urllib3/connectionpool.py", line 493, in _make_request
    conn.request(
  File "/var/task/urllib3/connection.py", line 508, in request
    self.send(chunk)
  File "/var/lang/lib/python3.12/http/client.py", line 1057, in send
    self.sock.sendall(data)
urllib3.exceptions.ProtocolError: ('Connection aborted.', TimeoutError('timed out'))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 11, in test_post_upload_with_very_large_file_exceeding_limits
  File "/var/task/requests/api.py", line 115, in post
    return request("post", url, data=data, json=json, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/var/task/requests/api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/var/task/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/var/task/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/var/task/requests/adapters.py", line 682, in send
    raise ConnectionError(err, request=request)
requests.exceptions.ConnectionError: ('Connection aborted.', TimeoutError('timed out'))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 36, in <module>
  File "<string>", line 13, in test_post_upload_with_very_large_file_exceeding_limits
AssertionError: Request failed: ('Connection aborted.', TimeoutError('timed out'))

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/3648c4cb-7c8d-4704-8bb7-ea8acdfd31a5
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC004 post upload with scanned pdf for ocr extraction
- **Test Code:** [TC004_post_upload_with_scanned_pdf_for_ocr_extraction.py](./TC004_post_upload_with_scanned_pdf_for_ocr_extraction.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 57, in <module>
  File "<string>", line 36, in test_post_upload_scanned_pdf_ocr_extraction
AssertionError: Response missing 'ocr_engine'

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/af448b8d-71b4-4300-8102-624a10db1468
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC005 post upload with low quality scan
- **Test Code:** [TC005_post_upload_with_low_quality_scan.py](./TC005_post_upload_with_low_quality_scan.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 46, in <module>
  File "<string>", line 35, in test_post_upload_with_low_quality_scan
AssertionError: ocr_warnings should be a list, got: <class 'NoneType'>

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/100421cc-95fd-4beb-a3d5-a654f804eca9
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC006 post upload with document type classification
- **Test Code:** [TC006_post_upload_with_document_type_classification.py](./TC006_post_upload_with_document_type_classification.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 39, in <module>
  File "<string>", line 13, in test_post_upload_with_document_type_classification
FileNotFoundError: [Errno 2] No such file or directory: 'employment_agreement_sample.pdf'

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/f79fbb40-dbb6-4788-a81a-8c73fa6b9684
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC007 post upload with clause detection
- **Test Code:** [TC007_post_upload_with_clause_detection.py](./TC007_post_upload_with_clause_detection.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 67, in <module>
  File "<string>", line 34, in test_post_upload_with_clause_detection
AssertionError: Response JSON missing 'clauses' key

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/ee746fd5-7ede-4827-a6a0-656f905dc9f3
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC008 post upload with weighted compliance scoring
- **Test Code:** [TC008_post_upload_with_weighted_compliance_scoring.py](./TC008_post_upload_with_weighted_compliance_scoring.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 50, in <module>
  File "<string>", line 16, in test_post_upload_with_weighted_compliance_scoring
FileNotFoundError: Test file not found: test_docs/multi_clause_contract.pdf

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/1703b3d5-56ba-47a7-909e-c304ecdf7fdb
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC009 get health check endpoint
- **Test Code:** [TC009_get_health_check_endpoint.py](./TC009_get_health_check_endpoint.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 30, in <module>
  File "<string>", line 23, in test_health_check_endpoint
AssertionError: Response JSON missing 'tesseract'

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/64716258-9ed7-4c68-8b1c-80bbed812956
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC010 get health check when tesseract not installed
- **Test Code:** [TC010_get_health_check_when_tesseract_not_installed.py](./TC010_get_health_check_when_tesseract_not_installed.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 24, in <module>
  File "<string>", line 11, in test_get_health_check_when_tesseract_not_installed
AssertionError: Expected status code 500, got 200

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/6ddd39dd-a6e4-43c4-9517-492b17dd0236/222c5bc5-97da-4cd5-b0fb-3a8bf071a9f9
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---


## 3️⃣ Coverage & Matching Metrics

- **0.00** of tests passed

| Requirement        | Total Tests | ✅ Passed | ❌ Failed  |
|--------------------|-------------|-----------|------------|
| ...                | ...         | ...       | ...        |
---


## 4️⃣ Key Gaps / Risks
{AI_GNERATED_KET_GAPS_AND_RISKS}
---