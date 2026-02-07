Bond Extractor quick notes:
1) Dependencies (Windows):
   - Poppler required for PDF support: https://blog.alivate.com.au/poppler-windows/
   - Install UB-Mannheim Tesseract for Windows: https://github.com/UB-Mannheim/tesseract/wiki
2) Python packages:
   - The agent will ensure requirements.txt contains pdf2image, python-dateutil, regex, requests (appends if missing).
3) Run server:
   .venv\Scripts\Activate.ps1
   python -m uvicorn app:app --reload --port 8000
4) Frontend:
   http://127.0.0.1:8000/
