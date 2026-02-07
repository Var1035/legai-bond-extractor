# app.py — Bond Extractor (OCR + Hybrid OCR: Tesseract → PaddleOCR → TrOCR) + local ML + optional Gemini/HF
import os
import io
import re
import json
import time
from dotenv import load_dotenv # New import

# Load environment variables from .env file immediately
load_dotenv()

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
from dateutil import parser as dateparser
import numpy as np # New import
from mistralai import Mistral # New import
from multilingual import normalize_query, translate_answer # Multilingual support

# --- configure Tesseract path (update if yours differs) ---
# If Tesseract is installed but not in PATH, set this to the absolute path to tesseract.exe
# Example: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# If left as None, we'll try shutil.which and fallback to default behaviour.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # <-- adjust if needed
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# PDF support (via pypdfium2, no system poppler needed)
try:
    import pypdfium2 as pdfium
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("pypdfium2 not installed. PDFs will not be accepted.")

# Optional external API libs
import requests

# ML libraries (lazy load later)
_ml_loaded = False

# ---------- Hybrid OCR adapters (lazy imported) ----------
_trocr_available = None
_trocr_pipeline = None

_paddle_available = None
_paddle_ocr = None

def ensure_paddleocr():
    """Lazy import/instantiate PaddleOCR. Returns True if available."""
    global _paddle_available, _paddle_ocr
    if _paddle_available is not None:
        return _paddle_available
    try:
        # import inside function to avoid heavy import at startup
        from paddleocr import PaddleOCR
        # instantiate with safe defaults — many paddleocr releases accept these args
        _paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
        _paddle_available = True
    except Exception as e:
        print("PaddleOCR not available:", e)
        _paddle_available = False
    return _paddle_available

def ocr_paddle(img: Image.Image) -> Dict[str, Any]:
    """Run PaddleOCR and return {'text': str, 'confidence': float}."""
    ok = ensure_paddleocr()
    if not ok:
        return {"text": "", "confidence": 0.0}
    try:
        import numpy as np
        arr = np.array(img.convert("RGB"))
        result = _paddle_ocr.ocr(arr, cls=False)
        texts = []
        confs = []
        # result can be nested lists; handle common shapes
        for line in result:
            if isinstance(line, list):
                for block in line:
                    # block usually like [box, (text, score)]
                    try:
                        candidate = block[1]
                        txt = candidate[0] if isinstance(candidate, (list, tuple)) else str(candidate)
                        score = float(candidate[1]) if isinstance(candidate, (list, tuple)) and len(candidate) > 1 else 0.0
                    except Exception:
                        try:
                            txt = str(block[1])
                            score = 0.0
                        except Exception:
                            txt = ""
                            score = 0.0
                    if txt:
                        texts.append(txt)
                        # normalize to percentage style
                        confs.append(score * 100.0 if score <= 1.0 else score)
            elif isinstance(line, tuple) and len(line) >= 2:
                try:
                    txt = line[1][0]
                    score = float(line[1][1])
                except Exception:
                    txt = str(line[1])
                    score = 0.0
                texts.append(txt)
                confs.append(score * 100.0 if score <= 1.0 else score)
        joined = "\n".join([t.strip() for t in texts if t.strip()])
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return {"text": joined, "confidence": round(avg_conf, 2)}
    except Exception as e:
        print("PaddleOCR error:", e)
        return {"text": "", "confidence": 0.0}

def ensure_trocr():
    """Lazy load TrOCR pipeline (image-to-text)."""
    global _trocr_available, _trocr_pipeline
    if _trocr_available is not None:
        return _trocr_available
    try:
        from transformers import pipeline
        # Use CPU (-1). If you have GPU change device accordingly.
        _trocr_pipeline = pipeline("image-to-text", model="microsoft/trocr-base-handwritten", device=-1)
        _trocr_available = True
    except Exception as e:
        print("TrOCR pipeline not available:", e)
        _trocr_available = False
    return _trocr_available

def ocr_trocr(img: Image.Image) -> Dict[str, Any]:
    """Run TrOCR (Vision->Text) pipeline. Returns dict with text+confidence heuristic."""
    ok = ensure_trocr()
    if not ok:
        return {"text": "", "confidence": 0.0}
    try:
        # Some pipeline versions accept lists, some accept images; pass PIL image directly
        res = _trocr_pipeline(img)
        if isinstance(res, list) and res:
            # typical return: [{"generated_text": "...", ...}] or list of strings
            if isinstance(res[0], dict):
                txts = [r.get("generated_text", "") for r in res]
            else:
                txts = [str(r) for r in res]
            joined = " ".join([t.strip() for t in txts if t.strip()])
            # There's no confidence in many TrOCR outputs — use heuristic
            return {"text": joined, "confidence": 60.0}
        # fallback
        return {"text": str(res), "confidence": 60.0}
    except Exception as e:
        print("TrOCR error:", e)
        return {"text": "", "confidence": 0.0}

def extract_text_from_pdf_raw(data: bytes) -> str:
    """
    Attempt to extract text directly from a digital PDF using pypdfium2.
    Returns empty string if failed or no text found.
    """
    if not PDF_AVAILABLE:
        return ""
    try:
        pdf = pdfium.PdfDocument(data)
        full_text = []
        for i in range(len(pdf)):
            page = pdf[i]
            text_page = page.get_textpage()
            text = text_page.get_text_range()
            if text:
                full_text.append(text)
        return "\n\n".join(full_text)
    except Exception as e:
        print(f"Digital PDF extraction failed: {e}")
        return ""

# ----------------- Utilities -----------------
def convert_file_to_images(data: bytes) -> List[Image.Image]:
    """Return list of PIL images for given file bytes (image or PDF pages)."""
    # 1. Try opening as standard image
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return [img]
    except Exception:
        pass

    # 2. Try opening as PDF with pypdfium2
    if PDF_AVAILABLE:
        try:
            pdf = pdfium.PdfDocument(data)
            images = []
            # We'll render at scale=2 (approx 144 DPI) or scale=3 (216 DPI). 
            # 300/72 ~= 4.16. Let's start with scale=3 for good OCR.
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = page.render(scale=3)
                pil_image = bitmap.to_pil().convert("RGB")
                images.append(pil_image)
            return images
        except Exception as e:
            # Not a valid PDF or error
            # If it was meant to be a PDF, this exception is relevant
            pass
            # We can re-raise if we are sure it's not some other binary garbage
            # But let's follow the pattern: checks specific headers? 
            # pypdfium2 raises exception if not PDF.
            
            # If the user uploaded a broken file, we should warn them.
            # But for now, let's assume if it failed image open, it might be PDF.
            if data[:4] == b'%PDF':
                 raise HTTPException(400, f"Could not convert PDF: {e}")

    raise HTTPException(400, "Unsupported file type or corrupt PDF/Image.")

def ocr_tesseract(img: Image.Image) -> Dict[str, Any]:
    """Run Tesseract OCR and return text + naive confidence (0-100)."""
    try:
        txt = pytesseract.image_to_string(img) or ""
        # pytesseract doesn't return confidence easily; use mean of confidences from hOCR if needed.
        # For simplicity we set heuristic confidence if text length is small/large
        conf = 70.0 if len(txt.strip()) > 10 else 30.0
        return {"text": txt, "confidence": conf}
    except Exception as e:
        print("pytesseract error:", e)
        return {"text": "", "confidence": 0.0}

def hybrid_ocr_image(img: Image.Image, engines: List[str] = None) -> Dict[str, Any]:
    """
    Try multiple OCR engines in order and return best result plus details.
    engines: list like ['tesseract','paddle','trocr'] - default hybrid order.
    Returns:
      {
        'text': str, 'confidence': float,
        'used': 'paddle', 'details': [ {engine, text, confidence, time_ms}, ... ]
      }
    """
    if engines is None:
        engines = ['tesseract', 'paddle', 'trocr']

    details = []
    best_text = ""
    best_conf = -1.0
    used_engine = None

    for eng in engines:
        start = time.time()
        try:
            if eng == 'tesseract':
                out = ocr_tesseract(img)
            elif eng == 'paddle':
                out = ocr_paddle(img)
            elif eng == 'trocr':
                out = ocr_trocr(img)
            else:
                out = {"text": "", "confidence": 0.0}

        except Exception as e:
            # Catch DLL errors (WinError 126) and others
            err_str = str(e)
            print(f"Engine {eng} failed: {err_str}")
            out = {"text": "", "confidence": 0.0}
            if "module could not be found" in err_str or "DLL load failed" in err_str:
                out["error"] = "Missing System Dependency (VC++ Redist?)"
        elapsed = int((time.time() - start) * 1000)
        details.append({
            "engine": eng,
            "text_snippet": (out.get("text") or "")[:300],
            "full_text": out.get("text") or "",
            "confidence": float(out.get("confidence") or 0.0),
            "time_ms": elapsed,
            "error": out.get("error")  # propagate error info
        })
        # choose best by confidence then by length heuristic
        conf = float(out.get("confidence") or 0.0)
        txt = (out.get("text") or "").strip()
        score = conf + min(len(txt), 100) * 0.1  # small boost for longer text
        if score > best_conf:
            best_conf = score
            best_text = txt
            used_engine = eng

        # quick accept if high confidence and non-empty
        if conf >= 85.0 and txt:
            used_engine = eng
            break

    return {
        "text": best_text,
        "confidence": round(best_conf, 2) if best_conf >= 0 else 0.0,
        "used": used_engine,
        "details": details
    }

def simple_summary(text: str, max_chars: int = 300) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
    out = []
    total = 0
    for s in sentences:
        s2 = s.strip()
        if s2:
            out.append(s2)
            total += len(s2)
            if total > max_chars:
                break
    return " ".join(out).strip()

# ----------------- Heuristics fields -----------------
def extract_duration(text: str) -> Optional[str]:
    patterns = [
        r'(\d{1,2}\s*(?:years|year|yrs|yr))',
        r'period of\s+(\d{1,2}\s*(?:years|year|yrs|yr))',
        r'for\s+(\d{1,2}\s*(?:years|year|yrs|yr))'
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def extract_dates(text: str) -> List[str]:
    dates = []
    date_like = re.findall(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s*\d{4})\b', text)
    for d in date_like:
        try:
            parsed = dateparser.parse(d, fuzzy=True)
            dates.append(parsed.strftime("%Y-%m-%d"))
        except Exception:
            pass
    chunks = re.split(r'[\n\r]+', text)[:10]
    for chunk in chunks:
        try:
            parsed = dateparser.parse(chunk, fuzzy=True)
            if parsed:
                dt = parsed.strftime("%Y-%m-%d")
                if dt not in dates:
                    dates.append(dt)
        except Exception:
            pass
    return dates

def extract_parties(text: str) -> List[str]:
    # 1. Cleaning: Skip potential header noise (dates, "Google Gemini", "Extracted Text", etc.)
    lines = text.split('\n')
    start_idx = 0
    for i in range(min(15, len(lines))):
        line_clean = lines[i].strip()
        if not line_clean:
            continue
        
        # Check for noise markers
        is_noise = False
        lc = line_clean.lower()
        if "extracted text" in lc or "lease agreement" in lc or "first page" in lc or "google" in lc or "gemini" in lc or "genin" in lc:
            is_noise = True
        elif len(line_clean) < 60 and (re.search(r'\d{1,2}/\d{1,2}', line_clean) or "PM" in line_clean or "AM" in line_clean):
            is_noise = True
            
        if is_noise:
            start_idx = i + 1
        else:
            # If we see a very short line that isn't clearly noise, it might still be garbage/header spacing
            if len(line_clean) < 10:
                continue
            # Found a substantial line that isn't noise -> stop skipping
            break
    
    t = "\n".join(lines[start_idx:])[:3000]

    # 2. Strict "Between ... And ..." Regex (High Confidence)
    patterns = [
        r'(?:between|bw)\s+([\w\s\.\(\),]+?)\s+(?:and|&)\s+([\w\s\.\(\),]+?)(?:\.|,|\n|which)',
        r'by and between\s+([\w\s\.\(\),]+?)\s+(?:and|&)\s+([\w\s\.\(\),]+?)(?:\.|,|\n|which)',
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            clean_a = re.sub(r'\(.*?\)', '', m.group(1)) # remove (hereinafter...)
            clean_b = re.sub(r'\(.*?\)', '', m.group(2))
            a = re.sub(r'\s+', ' ', clean_a).strip(" .,")
            b = re.sub(r'\s+', ' ', clean_b).strip(" .,")
            if len(a) > 2 and len(b) > 2:
                return [a, b]

    # 3. Fallback: Named Entity handling (Capitalized Words)
    # We ignore standard stopwords for contracts to avoid "This Rental Agreement" being a party
    # Also ignore UI artifacts if the user uploads a screenshot of the tool itself (meta-case)
    ignore = {
        'this', 'agreement', 'the', 'whereas', 'rental', 'lease', 'sale', 'deed', 'contract', 'between', 'and', 'extracted', 'text', 'ocr',
        'lease agreement', 'rental agreement', 'sale agreement', 'extracted text', 'purpose', 'duration', 'parties'
    }
    # 3. Fallback: Named Entity handling (Capitalized Words or ALL CAPS)
    # Match Title Case (Ram Kumar) or ALL CAPS (RAM KUMAR)
    # We ignore standard stopwords for contracts to avoid "This Rental Agreement" being a party
    # Also ignore UI artifacts if the user uploads a screenshot of the tool itself (meta-case)
    ignore = {
        'this', 'agreement', 'the', 'whereas', 'rental', 'lease', 'sale', 'deed', 'contract', 'between', 'and', 'extracted', 'text', 'ocr',
        'lease agreement', 'rental agreement', 'sale agreement', 'extracted text', 'purpose', 'duration', 'parties'
    }
    # Regex: [A-Z][a-z]+ (Title) OR [A-Z]{2,} (ALL CAPS)
    # Combined with spaces.
    # Simplified: \b[A-Z][a-zA-Z\.]+(?:\s+[A-Z][A-Za-z\.]+){0,4}\b
    # But be careful of "THIS IS A".
    # Let's try two passes or a flexible one.
    names = re.findall(r'\b([A-Z][A-Za-z\.]+(?:\s+[A-Z][A-Za-z\.]+){0,4})\b', t)
    unique = []
    for n in names:
        n_lower = n.lower()
        # check full phrase and individual words
        if len(n) > 2 and n_lower not in ignore: 
             # extra check: shouldn't be a month name
             if n_lower in {'january','february','march','april','may','june','july','august','september','october','november','december'}:
                 continue
             
             # check if it contains ignored generic terms like "Agreement" as the HEAD
             # e.g. "Rental Agreement" -> ignored. "Ram Kumar" -> kept.
             if "agreement" in n_lower or "deed" in n_lower:
                if n_lower not in ignore: 
                    # heuristic: purely generic agreements are ignored, but "Service Agreement" might be relevant? 
                    # For now, let's just ignore if it's exactly "Rental Agreement" (handled by set)
                    # or if starts with generic word
                    pass

             if n not in unique:
                unique.append(n)
        if len(unique) >= 2:
            break
    return unique

def extract_money(text: str) -> List[str]:
    monies = re.findall(r'(?:Rs\.?|INR|₹)\s*[,\d]+(?:\.\d+)?|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(?:rupees|Rs)\b', text, flags=re.IGNORECASE)
    return list({m.strip() for m in monies})

def detect_purpose(text: str) -> Optional[str]:
    mapping = {
        'rental': ['rental agreement', 'rent agreement', 'tenancy', 'renial'], # renial = common OCR typo
        'lease': ['lease', 'let', 'tenant', 'landlord'],
        'sale': ['sale', 'sold', 'purchase', 'buyer', 'seller'],
        'mortgage': ['mortgage', 'security', 'mortgagee', 'mortgagor'],
        'power_of_attorney': ['power of attorney', 'attorney'],
        'agreement': ['agreement', 'whereas', 'party of the first part'],
    }
    text_l = text.lower()
    scores = {}
    for k, kws in mapping.items():
        scores[k] = sum(1 for w in kws if w in text_l)
    
    # Priority Heuristic: If "Rental Agreement" or "Renial Agreement" is found explicitly, boost 'rental'
    if 'rental agreement' in text_l or 'renial agreement' in text_l:
        scores['rental'] += 5

    best = max(scores, key=lambda x: scores[x])
    if scores[best] > 0:
        labels = {
            'rental': 'Rental Agreement',
            'lease': 'Lease Agreement',
            'sale': 'Sale Agreement',
            'mortgage': 'Mortgage/Loan',
            'power_of_attorney': 'Power of Attorney',
            'agreement': 'Agreement/Other'
        }
        return labels.get(best, best)
    return None

def extract_fields_from_text(text: str) -> Dict[str, Any]:
    purpose = detect_purpose(text) or "Unknown"
    duration = extract_duration(text)
    dates = extract_dates(text)
    parties = extract_parties(text)
    money = extract_money(text)
    summary = simple_summary(text)
    return {
        "purpose": purpose,
        "duration": duration,
        "dates": dates,
        "parties": parties,
        "amounts": money,
        "summary": summary,
        "raw_text_length": len(text)
    }

# ----------------- ML lazy-loading (summarizer, NER, embeddings) -----------------
def load_ml_models_once():
    global _ml_loaded
    if _ml_loaded:
        return

    # 1. Load the core Phase 2 models (InLegalBERT, LawIPC, FLAN-T5)
    load_legai_models()

    # 2. Load Embeddings (all-MiniLM-L6-v2) - Request: "Keywords & embeddings"
    print("Loading embeddings (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        app.state.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("Embeddings loaded.")
    except Exception as e:
        print("Embeddings load failed:", e)
        app.state.embed_model = None

    # 3. Load Summarizer (Optional: distilbart)
    # Skipped to safe memory / fix strict loading issues in this env
    app.state.summarizer = None

    _ml_loaded = True
    print("Legacy ML lazy-load complete (merged with LEGAI).")

def ml_summary(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    load_ml_models_once()
    summ = None
    try:
        if getattr(app.state, "summarizer", None):
            # summarizer may throw warnings if input is tiny; that's fine
            out = app.state.summarizer(text[:4000], max_length=130, min_length=30, do_sample=False)
            if isinstance(out, list) and out:
                summ = out[0].get("summary_text")
    except Exception as e:
        print("Local summarizer error:", e)
        summ = None
    if not summ:
        return simple_summary(text, max_chars=400)
    return summ

def ml_entities(text: str) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []
    load_ml_models_once()
    try:
        # Use InLegalBERT (app.state.legal_ner) instead of old ner_pipe
        ner = getattr(app.state, "legal_ner", None)
        if ner:
            out = ner(text[:2000])
            cleaned = []
            for ent in out:
                # InLegalBERT: {entity_group, score, word, start, end}
                cat = ent.get("entity_group") or ent.get("entity") or "UNKNOWN"
                val = ent.get("word") or ent.get("entity") or ""
                cleaned.append({
                    "entity_group": cat,
                    "word": val,
                    "type": cat,
                    "value": val,
                    "score": float(ent.get("score", 0.0))
                })
            return cleaned
    except Exception as e:
        print("Local NER error:", e)
    return []

def ml_purpose_scores(text: str) -> Dict[str, Any]:
    labels = ["Rental Agreement", "Lease Agreement", "Sale Agreement", "Mortgage/Loan", "Power of Attorney", "Agreement/Other", "Unknown"]
    keywords = {
        "Rental Agreement": ["rental", "tenancy", "rent agreement", "renial"],
        "Lease Agreement": ["lease", "tenant", "landlord", "lessor", "lessee"],
        "Sale Agreement": ["sale", "buyer", "seller", "sold", "purchase", "vendor"],
        "Mortgage/Loan": ["mortgage", "loan", "security", "mortgagor", "mortgagee"],
        "Power of Attorney": ["power of attorney", "attorney", "agent"],
        "Agreement/Other": ["agreement", "whereas", "party of the first part"]
    }
    t = text.lower()
    scores = {k: 0 for k in labels}
    for k, kws in keywords.items():
        scores[k] = sum(1 for kw in kws if kw in t)
    maxv = max(scores.values()) if scores else 0
    if maxv == 0:
        scores["Unknown"] = 1
    else:
        for k in scores:
            scores[k] = round(scores[k] / (maxv if maxv > 0 else 1), 3)
    purpose = max(scores, key=lambda x: scores[x])
    return {"purpose": purpose, "purpose_scores": scores}

# ----------------- Gemini (external) placeholder -----------------
def call_gemini_external(text: str) -> Optional[Dict[str, Any]]:
    if os.environ.get("GEMINI_ENABLED") != "1":
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Gemini enabled but GEMINI_API_KEY missing.")
        return None
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-mini:generateContent?key=" + api_key
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Extract a JSON summary (purpose, parties, amounts, dates, summary, risks) from the contract text:\n\n{text[:8000]}"
            }]
        }]
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        data = r.json()
        cand = data.get("candidates", [])
        if cand and isinstance(cand, list):
            text_out = cand[0].get("content", {}).get("parts", [{}])[0].get("text")
            return {"gemini_output": text_out}
    except Exception as e:
        print("Gemini call failed:", e)
    return None

# ---- FastAPI app ----
app = FastAPI(title="Bond Extractor (OCR + ML + Hybrid)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("PDF Available (pypdfium2):", PDF_AVAILABLE)
print("Bond Extractor starting... (Hybrid OCR will try Tesseract → PaddleOCR → TrOCR)")

# ----------------- Endpoints -----------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload image or PDF. Returns OCR text + heuristics + ML + optional Gemini output."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file uploaded.")

    # save original upload
    fp = os.path.join(UPLOAD_DIR, file.filename)
    with open(fp, "wb") as f:
        f.write(data)

    # convert to images (first page or pages)
    pages = convert_file_to_images(data)

    # For each page, run hybrid OCR
    full_text = ""
    
    # 1. Try Digital PDF extraction first (FAST & ROBUST)
    digital_text = extract_text_from_pdf_raw(data)
    if len(digital_text.strip()) > 50:
         print("Digital text found! Using it primarily.")
         full_text = digital_text
         hybrid_details_per_page = [{"method": "digital_pdf", "text": digital_text}]
    else:
        # 2. Fallback to OCR if digital text is empty/short (Scanned PDF)
        hybrid_details_per_page = []
        for idx, p in enumerate(pages, start=1):
            res = hybrid_ocr_image(p, engines=['tesseract', 'paddle', 'trocr'])
            hybrid_details_per_page.append(res)
            # append only if we didn't get digital text (or mix/match? usually one is enough)
            full_text += (res.get("text") or "") + "\n\n"

    # heuristics
    fields = extract_fields_from_text(full_text)

    # local ML enrichments (lazy)
    try:
        ml_sum = ml_summary(full_text)
        ml_ents = ml_entities(full_text)
        ml_ipc = identify_ipc_sections(full_text) # Phase 1: IPC ID
        ml_purpose = ml_purpose_scores(full_text)
    except Exception as e:
        print("ML enrichment failed:", e)
        ml_sum = None
        ml_ents = []
        ml_ipc = []
        ml_purpose = {"purpose": None, "purpose_scores": {}}

    fields["ml"] = {
        "summary": ml_sum,
        "entities": ml_ents,
        "ipc_sections": ml_ipc,
        "purpose": ml_purpose.get("purpose"),
        "purpose_scores": ml_purpose.get("purpose_scores")
    }

    # optional Gemini external enrichment
    gem = call_gemini_external(full_text)
    if gem:
        fields["gemini"] = gem

    out = {
        "method_used": "hybrid",  # hybrid pipeline used
        "pages_ocr": len(pages),
        "extracted_text": full_text,
        "fields": fields,
        "hybrid_details": hybrid_details_per_page
    }
    return JSONResponse(out)

@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join("static", "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h2>UI not found: static/index.html</h2>", status_code=404)
    with open(path, "r", encoding="utf8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
def health():
    # quick Tesseract check
    import shutil
    tpath = shutil.which("tesseract")
    t_ok = bool(tpath)
    return {"status": "ok", "tesseract_in_path": t_ok, "tesseract_cmd_configured": pytesseract.pytesseract.tesseract_cmd}
    # =========================================================
# ===================== LEGAI MODULE (DYNAMIC AI) ======================
# ======================================================================
# Phase 2: Dynamic Chat with FLAN-T5 + Legal ML Models
# ======================================================================

_legai_loaded = False
IPC_BNS_MAPPING = {}

def load_ipc_bns_mapping():
    """Load IPC to BNS mapping from JSON file."""
    global IPC_BNS_MAPPING
    try:
        if os.path.exists("ipc_to_bns.json"):
            with open("ipc_to_bns.json", "r", encoding="utf-8") as f:
                IPC_BNS_MAPPING = json.load(f)
            print(f"Loaded {len(IPC_BNS_MAPPING)} IPC-BNS mappings.")
        else:
            print("Warning: ipc_to_bns.json not found.")
    except Exception as e:
        print(f"Error loading IPC mapping: {e}")

def load_legai_models():
    """
    Load the required ML models for the Legal AI Chat.
    """
    global _legai_loaded
    if _legai_loaded:
        return

    print("--- Loading LEGAI Phase 2 Models (This may take memory) ---")
    app.state.legai_error = None  # Global error tracker
    
    load_ipc_bns_mapping()

    try:
        import sentencepiece
    except ImportError:
        msg = "Missing dependency: 'sentencepiece'. Please run: pip install sentencepiece"
        print(msg)
        app.state.legai_error = msg
        return

    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    # 1. Legal NER (InLegalBERT)
    print("Loading InLegalBERT (NER)...")
    try:
        app.state.legal_ner = pipeline(
            "ner", 
            model="law-ai/InLegalBERT", 
            aggregation_strategy="simple",
            device=-1 # CPU
        )
        print("InLegalBERT loaded.")
    except Exception as e:
        print(f"Failed to load InLegalBERT: {e}")
        app.state.legal_ner = None

    # 2. IPC Identification (lawipc-ft)
    print("Loading LawIPC model...")
    try:
        # Pipeline failed with "Unrecognized model", so we load manually.
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        
        # sentencepiece should now be installed, so we can try default (fast) or slow
        ipc_tokenizer = AutoTokenizer.from_pretrained("shreyas-dev/lawipc-ft")
        ipc_model = AutoModelForSequenceClassification.from_pretrained("shreyas-dev/lawipc-ft")
        
        app.state.ipc_model = TextClassificationPipeline(
            model=ipc_model, 
            tokenizer=ipc_tokenizer, 
            return_all_scores=True,
            device=-1
        )
        print("LawIPC loaded (Direct Pipeline).")
    except Exception as e:
        print(f"Failed to load LawIPC: {e}")
        app.state.ipc_model = None

    # 3. Mistral API Check
    print("Checking Mistral API Key...")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        msg = "CRITICAL: MISTRAL_API_KEY not found in environment. Answer generation will fail."
        print(msg)
        app.state.legai_error = msg
    else:
        try:
            app.state.mistral_client = Mistral(api_key=api_key)
            print("Mistral Client initialized.")
        except Exception as e:
            print(f"Mistral Client Init Error: {e}")
            app.state.mistral_client = None

    _legai_loaded = True
    print("--- LEGAI Models Loaded (Mistral Ready) ---")

def identify_ipc_sections(text: str) -> List[Dict[str, Any]]:
    """
    Run LawIPC model to identify IPC sections and map them to BNS.
    """
    if not hasattr(app.state, "ipc_model") or not app.state.ipc_model:
        return []
    
    results = []
    try:
        # LawIPC: shreyas-dev/lawipc-ft is likely a text-classification model
        # We need to run it on segments if the text is long, but for now we look at the first chunk.
        preds = app.state.ipc_model(text[:1000])
        
        # Handle different return formats
        if preds and isinstance(preds[0], list):
             preds = preds[0]
        
        # Filter high confidence predictions
        detected_labels = [p['label'] for p in preds if p['score'] > 0.4]
        
        # Also map strictly based on regex if the model misses obvious ones 
        # (Hybrid approach is best for "Real World", but User emphasizes ML Insights)
        regex_ipc = re.findall(r'\bIPC\s*(\d+[A-Z]?)\b', text, re.IGNORECASE)
        for r in regex_ipc:
            lbl = f"IPC {r.upper()}"
            # Normalize for matching
            detected_labels.append(lbl)

        # Map to BNS
        # Load mapping if not loaded
        if not IPC_BNS_MAPPING:
             load_ipc_bns_mapping()

        unique_codes = set()
        
        for label in detected_labels:
            # Normalize label
            # label could be "LABEL_1" or "IPC 420" depending on model
            # For this specific model `shreyas-dev/lawipc-ft`, we assume it outputs class names or we need to map.
            # Without knowing exact label map, we rely on the regex fallback for robustness + model output usage.
            # If model outputs "IPC 420", we use it.
            
            clean_label = label.upper().replace("_", " ").strip()
            if "IPC" not in clean_label and clean_label.replace(" ", "").isdigit():
                clean_label = f"IPC {clean_label}"
            
            # Extract number
            match = re.search(r'(\d+[A-Z]?)', clean_label)
            if match:
                code_num = match.group(1)
                ipc_key = f"IPC {code_num}"
                unique_codes.add(ipc_key)

        for ipc_key in unique_codes:
            bns_val = IPC_BNS_MAPPING.get(ipc_key, "Mapping not found")
            results.append({"ipc": ipc_key, "bns": bns_val})
            
        return results

    except Exception as e:
        print(f"IPC Identification Error: {e}")
        return []

def extract_legal_entities(text: str) -> List[Dict[str, str]]:
    """
    Use InLegalBERT to extract Party, Role, Court, etc.
    Returns a list of dicts: [{"label": "ORG", "word": "Google"}, ...]
    """
    if not hasattr(app.state, "legal_ner") or not app.state.legal_ner:
        return []
    
    try:
        # InLegalBERT (NER)
        entities = app.state.legal_ner(text[:2000])
        valid_ents = []
        seen = set()
        
        for ent in entities:
             label = ent.get('entity_group') or ent.get('entity')
             word = ent.get('word')
             if label and word:
                 # Clean up word
                 word = word.replace("##", "")
                 # Create unique key for deduplication
                 key = f"{label}:{word}"
                 if key not in seen:
                     seen.add(key)
                     valid_ents.append({"label": label, "word": word})
        return valid_ents
    except Exception as e:
        print(f"Legal NER Error: {e}")
        return []

def semantic_retrieval(question: str, doc_text: str, top_k: int = 3) -> str:
    """
    Retrieve most relevant chunks from document using MiniLM embeddings.
    """
    if not hasattr(app.state, "embed_model") or not app.state.embed_model:
        return doc_text[:3000] # Fallback to first 3k chars if no embedding model

    # 1. Simple Chunking (Sentence-wise or fixed size)
    # We'll use a simple sliding window of words for robustness
    words = doc_text.split()
    chunk_size = 300
    overlap = 50
    chunks = []
    
    if len(words) <= chunk_size:
        chunks = [doc_text]
    else:
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)

    if not chunks: 
        return ""

    try:
        # 2. Embed
        chunk_embeddings = app.state.embed_model.encode(chunks) # [N, 384]
        q_embedding = app.state.embed_model.encode([question])  # [1, 384]

        # 3. Similarity (Dot product since normalized)
        # Handle numpy shapes
        scores = np.dot(chunk_embeddings, q_embedding.T).flatten()
        
        # 4. Top K
        top_indices = scores.argsort()[-top_k:][::-1]
        
        relevant_chunks = [chunks[i] for i in top_indices]
        return "\n...\n".join(relevant_chunks)
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return doc_text[:3000] # Fallback

def generate_legal_answer(question: str, doc_text: str, entities: List[Dict[str, str]], ipc_data: List[Dict[str, str]]) -> str:
    """
    Generate answer using Mistral API with retrieved context.
    Handles specific greetings and identity questions locally.
    """
    # --- 1. Static Responses (Heuristics) ---
    q_lower = question.lower().strip()
    
    # Greeting
    if q_lower in ["hi", "hello", "hi!", "hello!", "hey"]:
        return "Hello! I am your LEGAI Assistant. How can I help you with this document?"

    # Identity / Tasks
    identity_triggers = [
        "who are you?", "who are you", 
        "what are the task done by you?", "what are the tasks done by you?", 
        "what do you do?", "what is your purpose?"
    ]
    if any(trig in q_lower for trig in identity_triggers):
        return "I am an Document Legal Assistant which was developed by team 09 CSM KITS students"

    # --- 2. Dynamic Generation (Mistral) ---

    # Check for specific loading errors
    if hasattr(app.state, "legai_error") and app.state.legai_error:
        return f"System Error: {app.state.legai_error}"

    client = getattr(app.state, "mistral_client", None)
    if not client:
        return "Legal AI System Error: Mistral API client not initialized (Check MISTRAL_API_KEY)."

    # Retrieve Relevant Context
    context_text = semantic_retrieval(question, doc_text)

    # Construct Grounding Data
    # Format entities for the prompt (List[Dict] -> String)
    if entities:
        ent_str = ", ".join([f"{e['label']}: {e['word']}" for e in entities])
    else:
        ent_str = "None detected"

    law_str = "\n".join([f"- {x['ipc']} -> {x['bns']}" for x in ipc_data]) if ipc_data else "None detected"

    # 3. Strict Prompt (Updated for Formatting)
    system_prompt = """You are an expert Legal AI assistant. 
Your goal is to answer the user's question strictly based on the provided Document Context and Legal Data.

Formatting Rules:
- Use **bold** for key legal terms, parties, or headers.
- Use *italics* for emphasizing specific clauses or citations.
- Use bullet points for listing facts or conditions.
- Structure your answer clearly with paragraphs.

Content Rules:
1. Answer ONLY using the information provided below.
2. If the answer is not in the context, say "The answer is not present in the provided document or legal datasets."
3. Do not hallucinate or use outside knowledge.
4. Cite the IPC/BNS sections if relevant.
"""

    user_prompt = f"""
Document Context:
{context_text}

Extracted Legal Entities:
{ent_str}

Relevant Laws (IPC mapped to BNS):
{law_str}

User Question: {question}
"""

    try:
        msg = f"Calling Mistral for: {question}"
        print(msg)
        
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        print(f"Mistral Generation Error: {e}")
        return f"Error interacting with Legal AI API: {e}"

# ----------------- LEGAI Endpoints -----------------

@app.post("/chat-legai")
async def chat_legai(payload: Dict[str, Any]):
    """
    Dynamic LEGAI Chat Endpoint with Multilingual Support.
    """
    load_legai_models()
    
    doc_text = payload.get("text", "")
    user_question = payload.get("question", "")
    target_lang = payload.get("language", "en") # Default to english

    if not doc_text or not user_question:
        raise HTTPException(status_code=400, detail="Missing text or question.")

    client = getattr(app.state, "mistral_client", None)
    
    # 1. Normalize Query (Always convert to English per plan)
    final_question = user_question
    if client:
         # We normalize even if language is 'en', to handle Hinglish/Teluglish inputs robustly
         # The plan states: "Regardless of input language... ALWAYS convert user question to clear English FIRST"
         final_question = normalize_query(user_question, client)
         if final_question != user_question:
             print(f"Normalized query: '{user_question}' -> '{final_question}'")

    # 2. ML Insights (Local models)
    entities_list = extract_legal_entities(doc_text)
    ipc_data = identify_ipc_sections(doc_text)
    
    # 3. Answer Generation (English RAG)
    answer = generate_legal_answer(final_question, doc_text, entities_list, ipc_data)
    
    # 4. Translate Answer (if target is not English)
    final_answer = answer
    if client and target_lang in ["te", "hi"]:
        print(f"Translating answer to {target_lang}...")
        final_answer = translate_answer(answer, target_lang, client)
    
    return JSONResponse({
        "answer": final_answer,
        "identified_bns_sections": ipc_data or [],
        "legal_entities": entities_list or [],
        "debug_normalized_question": final_question
    })

@app.get("/chat-legai")
def chat_legai_get():
    return JSONResponse({"message": "POST strictly required."})

@app.get("/upload")
def upload_get():
    return JSONResponse({"message": "POST strictly required."})

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow external access if needed, or localhost
    # reload=True is good for dev, but might cause double-loading of heavy models.
    # To prevent double-loading issues in Dev, we can set reload=False or live with it.
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)