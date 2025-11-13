# app.py
# HealthExplain AI — single-column clean UI (Gemini-ready)
# Usage:
# 1) Activate venv: venv\Scripts\activate
# 2) (optional) Install Gemini SDK: pip install --upgrade --force-reinstall google-generativeai
# 3) (optional) Set API key in same CMD session: set GOOGLE_API_KEY=YOUR_KEY
# 4) Run: streamlit run app.py

import os
import re
import json
import tempfile
from io import BytesIO

import streamlit as st

# Optional imports (safe)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# Try to import the Gemini SDK (google.generativeai). Try common package names.
GEMINI_AVAILABLE = False
genai = None
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    try:
        import google_generativeai as genai
        GEMINI_AVAILABLE = True
    except Exception:
        GEMINI_AVAILABLE = False
        genai = None

# Read API key from environment (support multiple env var names)
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEN_AI_KEY")

# Configure genai if available and key provided
if GEMINI_AVAILABLE and API_KEY:
    try:
        if hasattr(genai, "configure"):
            genai.configure(api_key=API_KEY)
        elif hasattr(genai, "init"):
            genai.init(api_key=API_KEY)
    except Exception:
        # ignore now; errors surfaced later when calling
        pass

st.set_page_config(page_title="HealthExplain AI — Clean", layout="wide")

# Simple CSS for a clean single-column UI
st.markdown(
    """
    <style>
      body { background: #0b1220; color: #e6eef6; }
      .main { max-width: 1100px; margin: auto; padding-top: 24px; padding-bottom: 40px; }
      .card { background: #0f1724; padding:20px; border-radius:12px; box-shadow: 0 8px 30px rgba(2,6,23,0.6); }
      .bubble-user { background: linear-gradient(90deg,#60a5fa,#34d399); color:white; padding:10px 14px; border-radius:14px; display:inline-block; max-width:78%; }
      .bubble-bot { background:#e6eef6; color:#06121a; padding:10px 14px; border-radius:14px; display:inline-block; max-width:78%; }
      .small { font-size:13px; color:#9fb0c7; }
      .title { font-size:28px; font-weight:700; color:#ffffff; margin-bottom:6px; }
      .subtitle { color:#9fb0c7; margin-bottom:18px; }
      .logo { width:46px; height:46px; border-radius:10px; background: linear-gradient(135deg,#7b61ff,#00c6ff); color:white; display:inline-flex; align-items:center; justify-content:center; font-weight:800; margin-right:12px; }
      .header-row { display:flex; align-items:center; gap:12px; margin-bottom:18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Session state initialization (before widgets)
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if "pasted_text" not in st.session_state:
    st.session_state.pasted_text = ""
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "English"

# -------------------------
# Gemini helper (tries modern model names)
# -------------------------
def safe_gemini_generate_text(prompt: str, model_names=None, temperature: float = 0.2, max_output_tokens: int = 512):
    """
    Attempts to call the installed genai SDK with a list of model names (in order).
    Returns generated text or raises an exception describing the failure.
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini SDK not installed in this environment.")
    if not API_KEY:
        raise RuntimeError("Gemini API key not set in environment.")
    # default to working model names (updated)
    if model_names is None:
        model_names = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
    last_exc = None
    for model_name in model_names:
        try:
            # Try common modern function: genai.generate_text
            if hasattr(genai, "generate_text"):
                resp = genai.generate_text(model=model_name, prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
                # resp might be string or object
                if isinstance(resp, str):
                    return resp
                text = getattr(resp, "text", None) or getattr(resp, "result", None) or str(resp)
                return text
            # Try genai.models.generate pattern
            if hasattr(genai, "models") and hasattr(genai.models, "generate"):
                resp = genai.models.generate(model=model_name, content=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
                # attempt common fields
                text = getattr(resp, "text", None)
                if text:
                    return text
                try:
                    if isinstance(resp, dict):
                        # check candidates/content
                        if "candidates" in resp and resp["candidates"]:
                            cand = resp["candidates"][0]
                            if isinstance(cand, dict) and "content" in cand:
                                return cand["content"]
                            return str(cand)
                except Exception:
                    pass
                return str(resp)
            # Try older class patterns
            if hasattr(genai, "GenerativeModel"):
                model = genai.GenerativeModel(model_name)
                if hasattr(model, "generate_content"):
                    resp = model.generate_content(prompt)
                    text = getattr(resp, "text", None) or str(resp)
                    return text
            raise RuntimeError("No supported genai invocation pattern found in installed SDK.")
        except Exception as e:
            last_exc = e
            continue
    # If none worked, raise the last exception
    raise last_exc or RuntimeError("Gemini call failed for unknown reason.")

# -------------------------
# High-level assistant functions
# -------------------------
def ask_gemini(question: str, language: str = "English"):
    if not question or not question.strip():
        return "Please type your question."
    system = (
        "You are HealthExplain — a friendly assistant that explains medical symptoms and lab reports in simple, non-diagnostic language. "
        "Always encourage users to consult a physician for diagnosis and urgent care when needed. Keep answers concise."
    )
    prompt = f"SYSTEM: {system}\n\nUSER (language:{language}): {question}\n\nASSISTANT:"
    try:
        return safe_gemini_generate_text(prompt, temperature=0.15, max_output_tokens=512)
    except Exception as e:
        return f"Gemini not available or error: {e}\nFallback: Please upload a report or ask keywords like 'fever', 'headache', 'HbA1c'."

def summarize_report_with_gemini(report_text: str, language: str = "English"):
    system = (
        "You are a medical report summarizer. Read the lab report and return three labeled sections:\n"
        "1) Short summary (2-4 sentences)\n"
        "2) Key numeric values with one-line interpretations\n"
        "3) Simple non-diagnostic next steps\n"
        "Return plain, readable text."
    )
    prompt = f"SYSTEM: {system}\n\nREPORT:\n{report_text}\n\nRespond in {language}."
    try:
        return safe_gemini_generate_text(prompt, temperature=0.0, max_output_tokens=700)
    except Exception as e:
        return f"Gemini summarization failed: {e}\n\nFallback: (first 600 chars)\n\n{report_text[:600]}"

def simple_local_summary(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text)
    keywords = ["hba1c","glucose","ldl","hdl","creatinine","alt","ast","wbc","hb","platelet"]
    scored = []
    for s in sents:
        score = len(re.findall(r'\d', s)) * 2 + sum(1 for k in keywords if k in s.lower())
        scored.append((score, s))
    best = [s for _, s in sorted(scored, reverse=True)[:3]]
    return " ".join(best) if best else (text[:500] + ("..." if len(text) > 500 else ""))

def extract_text_from_image_bytes(image_bytes: bytes):
    if not (TESSERACT_AVAILABLE and PIL_AVAILABLE):
        raise RuntimeError("OCR not available: install pytesseract + pillow and the Tesseract engine.")
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    txt = pytesseract.image_to_string(img)
    return txt

# -------------------------
# Callbacks (safe modifications to session_state)
# -------------------------
def send_callback():
    txt = st.session_state.get("user_input", "").strip()
    if not txt:
        return
    st.session_state.chat_history.append(("You", txt))
    reply = ask_gemini(txt, st.session_state.get("ui_language", "English"))
    st.session_state.chat_history.append(("HealthExplain", reply))
    st.session_state.last_summary = reply
    # clear input inside callback (allowed)
    st.session_state["user_input"] = ""

def clear_chat_callback():
    st.session_state.chat_history = []
    st.session_state.last_summary = ""

def on_upload_change():
    uploaded = st.session_state.get("uploader_file", None)
    if uploaded is None:
        st.session_state.uploaded_file_bytes = None
    else:
        st.session_state.uploaded_file_bytes = uploaded.read()

def analyze_callback():
    uploaded_bytes = st.session_state.get("uploaded_file_bytes", None)
    pasted = st.session_state.get("pasted_text", "").strip()
    report_text = ""
    if uploaded_bytes:
        try:
            ocr_text = extract_text_from_image_bytes(uploaded_bytes)
            report_text = ocr_text.strip()
            if not report_text:
                st.session_state.chat_history.append(("HealthExplain", "OCR ran but could not extract clear text. Paste text instead."))
        except Exception as e:
            st.session_state.chat_history.append(("HealthExplain", f"OCR not available or failed: {e}"))
    if pasted and not report_text:
        report_text = pasted
    if not report_text:
        st.session_state.chat_history.append(("HealthExplain", "No report text found. Upload an image or paste the report text."))
        return
    if GEMINI_AVAILABLE and API_KEY:
        summary = summarize_report_with_gemini(report_text, st.session_state.get("ui_language", "English"))
        st.session_state.chat_history.append(("HealthExplain (Report)", summary))
        st.session_state.last_summary = summary
    else:
        summary = simple_local_summary(report_text)
        st.session_state.chat_history.append(("HealthExplain (Report)", summary))
        st.session_state.last_summary = summary

def play_tts_callback():
    txt = st.session_state.get("last_summary", "")
    if not txt:
        st.session_state.chat_history.append(("HealthExplain", "Nothing to play. Generate a summary or ask a question first."))
        return
    if not TTS_AVAILABLE:
        st.session_state.chat_history.append(("HealthExplain", "TTS not installed (pip install gTTS)."))
        return
    try:
        tts = gTTS(text=txt, lang="en")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        # streaming audio from callback may not always trigger UI audio; still attempt
        st.audio(tmp.name)
    except Exception as e:
        st.session_state.chat_history.append(("HealthExplain", f"TTS failed: {e}"))

# -------------------------
# Single-column UI
# -------------------------
st.markdown('<div class="main">', unsafe_allow_html=True)

# header
st.markdown('<div class="header-row"><div class="logo">H</div><div><div class="title">HealthExplain AI</div><div class="subtitle">Ask health questions or upload lab reports for simple explanations.</div></div></div>', unsafe_allow_html=True)

# main card
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Ask a question or describe symptoms")
st.text_input("Your question or symptoms", key="user_input", placeholder="e.g., I have a headache and fever")
st.selectbox("Language", ["English", "Hindi", "Telugu", "Tamil", "Gujarati"], key="ui_language")

c1, c2 = st.columns([0.18, 0.18])
c1.button("Send", on_click=send_callback)
c2.button("Clear chat", on_click=clear_chat_callback)

st.markdown("---")
st.subheader("Chat")
for who, msg in st.session_state.chat_history[-40:]:
    if who == "You":
        st.markdown(f"<div style='text-align:right; margin:8px'><span class='bubble-user'>{msg}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; margin:8px'><span class='bubble-bot'><b>{who}:</b> {msg}</span></div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Upload report (image) or paste text")
st.file_uploader("Upload image (png/jpg)", type=["png", "jpg", "jpeg"], key="uploader_file", on_change=on_upload_change)
st.text_area("Or paste report text here", key="pasted_text", placeholder="Paste lab report text...")

a1, a2 = st.columns([0.3, 0.3])
a1.button("Analyze", on_click=analyze_callback)
a2.button("Play last summary (TTS)", on_click=play_tts_callback)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# small footer
st.markdown('<div style="text-align:center; color:#9fb0c7; margin-top:12px">HealthExplain — Informational only. Not medical advice.</div>', unsafe_allow_html=True)
