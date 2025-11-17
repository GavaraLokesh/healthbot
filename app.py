# app.py
# HealthExplain AI — UI with Login/Register + Chat + Voice mic + Gemini-ready (proxy-first)
# Usage:
# Terminal 1 (proxy): venv\Scripts\activate ; $env:GOOGLE_API_KEY="YOUR_KEY" ; python -m uvicorn voice_proxy:app --port 8765
# Terminal 2 (frontend): venv\Scripts\activate ; streamlit run app.py


import os
import json
import re
import tempfile
from io import BytesIO


import streamlit as st
import streamlit.components.v1 as components
import requests


# Optional libs
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


# Gemini SDK detection (support both package names)
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


# API key (read from env or st.secrets)
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEN_AI_KEY")


# If genai SDK exists and API_KEY present, try configure (harmless if fails)
if GEMINI_AVAILABLE and API_KEY:
    try:
        if hasattr(genai, "configure"):
            genai.configure(api_key=API_KEY)
        elif hasattr(genai, "init"):
            genai.init(api_key=API_KEY)
    except Exception:
        pass


# ---- users file helpers ----
USERS_FILE = "users.json"


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                out = {}
                for i, v in enumerate(data):
                    out[str(i)] = {"password": str(v)}
                return out
    except Exception:
        return {}
    return {}


def save_users(users):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        return True
    except Exception:
        return False


# ---- session defaults ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "nav" not in st.session_state:
    st.session_state.nav = "Home"
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "English"
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if "pasted_text" not in st.session_state:
    st.session_state.pasted_text = ""
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""


st.set_page_config(page_title="HealthExplain AI", layout="wide")


# ---- styling ----
st.markdown("""
<style>
  body { background: #0b1220; color: #e6eef6; }
  .main { max-width: 1200px; margin: auto; padding-top: 18px; padding-bottom: 40px; }
  .sidebar-box { padding:14px; background:#0a0f16; border-radius:10px; }
  .title-big { font-size:36px; font-weight:800; color:#d7ffe6; text-shadow: 0 0 12px rgba(0,255,150,0.08); }
  .card { background: #0f1724; padding:18px; border-radius:12px; box-shadow: 0 8px 30px rgba(2,6,23,0.6); }
  .bubble-user { background: linear-gradient(90deg,#60a5fa,#34d399); color:white; padding:10px 14px; border-radius:14px; display:inline-block; max-width:78%; }
  .bubble-bot { background:#e6eef6; color:#06121a; padding:10px 14px; border-radius:14px; display:inline-block; max-width:78%; }
  .big-mic { width:110px; height:110px; border-radius:50%; background: radial-gradient(circle at 30% 20%, #ffd36b, #ff7bd6); display:flex; align-items:center; justify-content:center; font-size:42px; color:#07121a; cursor:pointer; border: 4px solid rgba(255,255,255,0.08); }
  .mic-on { box-shadow: 0 0 36px rgba(255,123,214,0.45) !important; transform: scale(1.02); }
  .small { font-size:13px; color:#9fb0c7; }
  .nav-item { font-weight:600; padding:6px 8px; color:#e6eef6; }
  .nav-item-selected { background: rgba(255,255,255,0.03); border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# -----------------------
# PROXY-FIRST Gemini wrapper replacement
# -----------------------
VOICE_PROXY_URL = os.getenv("VOICE_PROXY_URL", "http://127.0.0.1:8765").rstrip("/")


def shorten_reply(text: str, max_sentences: int = 2):
    """
    Ensure replies are concise: keep first max_sentences sentences (2 by default).
    Trim extra whitespace. If text has newline blocks, take first 2 meaningful lines.
    """
    if not text or not text.strip():
        return text
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', text).strip()
    # Split on sentence enders
    sents = re.split(r'(?<=[.!?])\s+', t)
    # Fallback: split on newline if few sentence enders
    if len(sents) < max_sentences:
        parts = [p.strip() for p in text.splitlines() if p.strip()]
        if parts:
            joined = " ".join(parts)
            sents = re.split(r'(?<=[.!?])\s+', joined)
    sel = sents[:max_sentences]
    out = " ".join(sel).strip()
    # Safety: if still long, truncate chars to ~360
    if len(out) > 360:
        out = out[:357].rsplit(' ', 1)[0] + "..."
    return out


def safe_gemini_generate_text(prompt: str, temperature: float = 0.2, max_output_tokens: int = 512):
    """
    Proxy-first approach:
    1) Try local voice_proxy /generate (expected to return {"reply": "..."} )
    2) Fallback to genai SDK only if available and configured
    Returns a plain string reply (shortened to 2 sentences).
    """
    # 1) Try local proxy
    proxy_err = ""
    try:
        resp = requests.post(f"{VOICE_PROXY_URL}/generate", json={"prompt": prompt}, timeout=30)
        try:
            j = resp.json()
        except Exception:
            j = None
        if resp.status_code == 200 and isinstance(j, dict) and "reply" in j:
            return shorten_reply(j.get("reply") or "")
        # handle some alternate structures
        if resp.status_code == 200 and isinstance(j, dict):
            if "reply" in j:
                return shorten_reply(j.get("reply") or "")
            if "text" in j:
                return shorten_reply(j.get("text") or "")
            if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
                try:
                    cand = j["candidates"][0]
                    # Gemini shape
                    if isinstance(cand, dict) and "content" in cand:
                        parts = cand["content"].get("parts") if isinstance(cand["content"], dict) else None
                        if parts and isinstance(parts, list) and len(parts) > 0:
                            return shorten_reply(parts[0].get("text") or str(parts[0]))
                    # Try several fallbacks
                    if isinstance(cand, dict):
                        for k in ("text","output","content"):
                            if k in cand:
                                return shorten_reply(str(cand[k]))
                    return shorten_reply(str(cand))
                except Exception:
                    pass
        if resp.status_code == 200 and isinstance(j, str):
            return shorten_reply(j)
        if resp.status_code == 200 and resp.text:
            return shorten_reply(resp.text)
        proxy_err = f"Proxy status {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        proxy_err = f"Proxy call failed: {e}"


    # 2) Optional: try genai SDK if installed & configured (legacy)
    sdk_err = "SDK not available or not configured."
    if GEMINI_AVAILABLE and API_KEY:
        try:
            # try several genai usage patterns
            if hasattr(genai, "generate_text"):
                try:
                    out = genai.generate_text(model="models/gemini-2.5-flash", prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
                except TypeError:
                    out = genai.generate_text(model="models/gemini-2.5-flash", prompt=prompt, temperature=temperature)
                if isinstance(out, str):
                    return shorten_reply(out)
                # attempt to extract textual content
                text = getattr(out, "text", None) or getattr(out, "result", None) or str(out)
                if text:
                    return shorten_reply(text)
            if hasattr(genai, "models") and hasattr(genai.models, "generate"):
                try:
                    out = genai.models.generate(model="models/gemini-2.5-flash", content=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
                except TypeError:
                    out = genai.models.generate(model="models/gemini-2.5-flash", content=prompt, temperature=temperature)
                if isinstance(out, dict) and "candidates" in out and out["candidates"]:
                    cand = out["candidates"][0]
                    if isinstance(cand, dict) and "content" in cand:
                        try:
                            parts = cand["content"].get("parts")
                            if parts and isinstance(parts, list):
                                return shorten_reply(parts[0].get("text") or str(parts[0]))
                        except Exception:
                            pass
                    return shorten_reply(str(cand))
                return shorten_reply(str(out))
            sdk_err = "SDK call returned no usable text."
        except Exception as e:
            sdk_err = f"SDK error: {e}"


    # If we reach here both proxy and SDK failed — return informative fallback string
    details = " | ".join([p for p in (locals().get("proxy_err",""), sdk_err) if p])[:800]
    return shorten_reply(f"Gemini call failed for unknown reason. Details: {details}")


def ask_gemini(question: str, language: str = "English"):
    if not question or not question.strip():
        return "Please type your question."
    system = ("You are HealthExplain — a friendly assistant that explains medical symptoms and lab reports in simple, non-diagnostic language. "
              "Always encourage users to consult a physician for diagnosis and urgent care when needed. Keep answers concise.")
    prompt = f"SYSTEM: {system}\n\nUSER (language:{language}): {question}\n\nASSISTANT:"
    try:
        return safe_gemini_generate_text(prompt, temperature=0.15, max_output_tokens=180)
    except Exception as e:
        return f"(gemini error) {e} — fallback: Please upload a report or ask keywords like 'fever', 'headache'."


# ---- assistant helper functions (unchanged) ----
def summarize_report_with_gemini(report_text: str, language: str = "English"):
    system = ("You are a medical report summarizer. Return three labeled sections:\n"
              "1) Short summary (2-4 sentences)\n"
              "2) Key numeric values with one-line interpretations\n"
              "3) Simple non-diagnostic next steps\n"
              "Return plain, readable text.")
    prompt = f"SYSTEM: {system}\n\nREPORT:\n{report_text}\n\nRespond in {language}."
    try:
        return safe_gemini_generate_text(prompt, temperature=0.0, max_output_tokens=700)
    except Exception as e:
        return f"(gemini error) {e}\n\nFallback: (first 600 chars)\n\n{report_text[:600]}"


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


# ---- callbacks ----
def safe_rerun():
    if hasattr(st, "rerun"):
        try:
            st.rerun()
        except Exception:
            pass


def show_login_register():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    cols = st.columns([0.5, 0.5])
    with cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Login")
        with st.form("login_form"):
            login_email = st.text_input("Email or username", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Login"):
                users = load_users()
                if login_email in users and users[login_email].get("password") == login_password:
                    st.session_state.logged_in = True
                    st.session_state.username = login_email
                    st.success("Logged in")
                    safe_rerun()
                else:
                    st.error("Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Create account")
        with st.expander("Create new account"):
            with st.form("register_form"):
                reg_email = st.text_input("New email", key="reg_email")
                reg_pw = st.text_input("New password", type="password", key="reg_pw")
                reg_confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
                if st.form_submit_button("Create account"):
                    if not reg_email or not reg_pw:
                        st.error("Please fill all fields")
                    elif reg_pw != reg_confirm:
                        st.error("Passwords do not match")
                    else:
                        users = load_users()
                        if reg_email in users:
                            st.error("User already exists")
                        else:
                            users[reg_email] = {"password": reg_pw}
                            ok = save_users(users)
                            if ok:
                                st.success("Account created. Please login.")
                            else:
                                st.error("Error saving account (maybe permission issue).")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---- main pages ----
def show_home():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Welcome to HealthExplain</h3><p class="small">Multilingual health assistant. Use Chat Assistant for voice or text Q&A, upload lab reports for summaries, or try the diabetes quick-check.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def handle_send():
    q = st.session_state.get("user_input", "").strip()
    if not q:
        return
    ans = ask_gemini(q, st.session_state.get("ui_language", "English"))
    st.session_state.chat_history.append(("You", q))
    st.session_state.chat_history.append(("HealthExplain", ans))
    st.session_state["user_input"] = ""


def chat_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Multilingual Chatbot")
    input_col, mic_col = st.columns([0.78, 0.22])
    with input_col:
        st.text_input("Ask a question:", key="user_input", placeholder="e.g., I have a headache and fever")
        st.selectbox("Language", ["English", "Hindi", "Telugu", "Tamil", "Gujarati"], key="ui_language")
        c1, c2, c3 = st.columns([0.18, 0.18, 0.18])
        c1.button("Send", on_click=handle_send)
        c2.button("Speak last answer")  # placeholder
        c3.button("Clear chat", on_click=lambda: st.session_state.update({"chat_history": []}))
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)


    st.markdown('<hr />', unsafe_allow_html=True)
    st.subheader("Chat")
    for who, msg in st.session_state.chat_history[-80:]:
        if who == "You":
            st.markdown(f"<div style='text-align:right; margin:8px'><span class='bubble-user'>{msg}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; margin:8px'><span class='bubble-bot'><b>{who}:</b> {msg}</span></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def image_analysis_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Upload and Analyze Medical Image")
    uploaded = st.file_uploader("Drag & drop image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="img_upload")
    if st.button("Analyze image"):
        if not uploaded:
            st.error("Please upload an image first")
        else:
            st.success("Image received. Analysis not implemented in demo.")
    st.markdown('</div>', unsafe_allow_html=True)


def diabetes_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Check Your Diabetes Risk")
    age = st.number_input("Age", min_value=1, max_value=120, value=25, key="age")
    glucose = st.number_input("Glucometer reading (mg/dL)", min_value=0, max_value=1000, value=90, key="glucose")
    if st.button("Predict"):
        risk = "Low"
        if glucose >= 200 or glucose >= 126:
            risk = "High"
        elif glucose >= 140:
            risk = "Moderate"
        st.success(f"Estimated risk: {risk} (not diagnostic)")
    st.markdown('</div>', unsafe_allow_html=True)


# ---- route ----
if not st.session_state.logged_in:
    show_login_register()
    st.stop()


left, main = st.columns([0.24, 0.76])
with left:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown(f"<div style='margin-bottom:8px'><b>👋 Logged in as</b><div class='small'>{st.session_state.username}</div></div>", unsafe_allow_html=True)
    st.markdown("<hr />", unsafe_allow_html=True)
    def nav_change():
        st.session_state.nav = st.session_state.get("nav_select", "Home")
    st.radio("Select:", ["Home", "Chat Assistant", "Image Analysis", "Diabetes Prediction"], index=["Home","Chat Assistant","Image Analysis","Diabetes Prediction"].index(st.session_state.nav) if st.session_state.nav in ["Home","Chat Assistant","Image Analysis","Diabetes Prediction"] else 0, key="nav_select", on_change=nav_change)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.update({"logged_in": False, "username": "", "chat_history": []})
        safe_rerun()


with main:
    st.markdown('<div class="title-big">💚 HealthExplain AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    page = st.session_state.nav
    if page == "Home":
        show_home()
    elif page == "Chat Assistant":
        chat_page()
    elif page == "Image Analysis":
        image_analysis_page()
    elif page == "Diabetes Prediction":
        diabetes_page()
    else:
        show_home()


# ---- Voice widget injection (self-contained iframe with robust mic JS) ----
# Render the voice widget only when Chat Assistant page is active
VOICE_PROXY = os.getenv("VOICE_PROXY_URL", "http://127.0.0.1:8765").rstrip("/")


_voice_widget = """
<div style="font-family: Inter, Arial, sans-serif; color: #e6eef6;">
  <style>
    .va-card { background:#041024; padding:12px; border-radius:10px; }
    .va-row { display:flex; align-items:center; gap:12px; }
    .va-mic { width:110px; height:110px; border-radius:50%; background: radial-gradient(circle at 30% 20%, #ffd36b, #ff7bd6); display:flex; align-items:center; justify-content:center; font-size:42px; color:#07121a; cursor:pointer; border: 4px solid rgba(255,255,255,0.08); }
    .va-micon { box-shadow: 0 0 36px rgba(255,123,214,0.45); transform: scale(1.02); }
    .va-small { font-size:13px; color:#9fb0c7; margin-top:8px; }
    .va-log { margin-top:10px; max-height:240px; overflow:auto; }
    .va-bubble-user { background: linear-gradient(90deg,#60a5fa,#34d399); color:white; padding:8px 12px; border-radius:12px; display:inline-block; max-width:85%; }
    .va-bubble-bot { background:#e6eef6; color:#06121a; padding:8px 12px; border-radius:12px; display:inline-block; max-width:85%; }
    .va-lang { background:#0b1220; color:#e6eef6; padding:6px 10px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }
  </style>


  <div class="va-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div style="font-weight:700; color:#dfffe6;">Voice Assistant</div>
      <div id="vaStatus" style="font-size:12px; color:#9fb0c7;">Idle</div>
    </div>


    <div class="va-row" style="margin-top:12px;">
      <div id="vaMic" class="va-mic" title="Click to talk">🎙️</div>


      <div style="flex:1;">
        <div style="display:flex; gap:8px; align-items:center;">
          <select id="vaLang" class="va-lang">
            <option>English</option>
            <option>Hindi</option>
            <option>Telugu</option>
            <option>Tamil</option>
            <option>Gujarati</option>
          </select>
          <div style="flex:1;"></div>
          <button id="vaClear" style="background:#0f1724;color:#e6eef6;padding:6px 10px;border-radius:8px;border:1px solid rgba(255,255,255,0.03);cursor:pointer;">Clear</button>
        </div>
        <div id="vaLog" class="va-log"></div>
      </div>
    </div>
  </div>

<script>
(function(){
  const proxyBase = "__VOICE_PROXY__";
  const LOG = (m)=>{ try{ console.log("[voice-assistant]", m); }catch(e){} };

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  LOG("voice widget loaded; SpeechRecognition=" + (SpeechRecognition ? "yes" : "no") + "; proxy=" + proxyBase);

  // Play a short beep on mic toggle for better feedback
  async function playBeep() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.connect(g);
      g.connect(ctx.destination);
      o.type = "sine";
      o.frequency.value = 500;
      g.gain.setValueAtTime(0.1, ctx.currentTime);
      o.start();
      o.stop(ctx.currentTime + 0.1);
      return new Promise(r => o.onended = r);
    } catch(e) {
      LOG("playBeep error "+e);
    }
  }

  function escapeHtml(s){
    if(!s) return '';
    return s.replace(/[&<"'>]/g, function(m){ return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'})[m]; });
  }

  function appendLog(who, text){
    try{
      const logEl = document.getElementById('vaLog');
      if(!logEl) return;
      const d = document.createElement('div');
      d.style.margin = '8px 0';
      if(who === 'user'){
        d.innerHTML = `<div style="text-align:right"><span class="va-bubble-user">${escapeHtml(text)}</span></div>`;
      } else {
        d.innerHTML = `<div style="text-align:left"><span class="va-bubble-bot">${escapeHtml(text)}</span></div>`;
      }
      logEl.appendChild(d);
      logEl.scrollTop = logEl.scrollHeight;
    }catch(e){ LOG("appendLog error " + e); }
  }

  async function fetchJson(url, opts){
    try{ const r = await fetch(url, opts); return await r.json(); } catch(e){ LOG("fetchJson " + e); return null; }
  }

  async function playBase64(b64){
    if(!b64) return false;
    try{
      const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
      const blob = new Blob([bytes], { type: 'audio/mp3' });
      const url = URL.createObjectURL(blob);
      const a = new Audio(url);
      await a.play();
      return true;
    }catch(e){ LOG("playBase64 " + e); return false; }
  }

  function speakBrowser(text, lang){
    return new Promise(res => {
      try{
        if(!window.speechSynthesis){ res(); return; }
        const ut = new SpeechSynthesisUtterance(text);
        ut.lang = (lang && lang.startsWith('Hindi')) ? 'hi-IN' : 'en-US';
        ut.onend = ()=> res();
        ut.onerror = ()=> res();
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(ut);
      }catch(e){ LOG("speakBrowser " + e); res(); }
    });
  }

  async function assistantFlow(text, lang){
    if(!text) return;
    appendLog('user', text);
    const langVal = lang || 'English';

    // try /voice proxy first
    try{
      const res = await fetchJson(proxyBase + '/voice', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ text: text, lang: langVal })
      });
      const reply = (res && (res.reply || res.text)) || '(no reply)';
      const audio_b64 = res && (res.audio_b64 || res.audio || res.audioContent || null);
      appendLog('assistant', reply);

      if(audio_b64){
        await playBase64(audio_b64);
      } else {
        await speakBrowser(reply, langVal);
      }

      try {
        await fetch(proxyBase + '/append_history', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ user_text: text, assistant_text: reply })
        });
      } catch(e){}

      return;
    }catch(e){
      LOG("voice endpoint failed, falling back: " + e);
    }

    // fallback generate -> tts
    try{
      const gen = await fetchJson(proxyBase + '/generate', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ prompt: text, lang: lang })
      });
      const reply = (gen && (gen.text || gen.reply)) || '(no reply)';
      const tts = await fetchJson(proxyBase + '/tts', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ text: reply })
      });
      const audio_b64 = tts && (tts.audioContent || tts.audio || tts.audio_b64 || null);
      appendLog('assistant', reply);

      if(audio_b64){
        await playBase64(audio_b64);
      } else {
        await speakBrowser(reply, lang);
      }
      try {
        await fetch(proxyBase + '/append_history', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ user_text: text, assistant_text: reply })
        });
      } catch(e){}

      return;
    }catch(e){
      LOG("generate/tts fallback failed " + e);
      await speakBrowser("(assistant error)", 'English');
    }
  }

  async function fallbackRecordAndUpload(langLabel){
    if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
      alert('Microphone not available in this browser.');
      return;
    }
    try{
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const chunks = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const fd = new FormData();
        fd.append('audio', blob, 'rec.webm');
        try{
          const r = await fetch(proxyBase + '/stt', { method: 'POST', body: fd });
          const j = await r.json();
          const text = j.transcript || j.text || '';
          if(text) await assistantFlow(text, langLabel);
          else await speakBrowser('(no speech detected)', langLabel || 'English');
        }catch(e){ LOG("STT upload failed " + e); alert('STT failed: ' + e); }
        stream.getTracks().forEach(t => t.stop());
      };
      // NO auto-stop: start and wait for user to stop (6s auto-stop removed)
      mediaRecorder.start();
      // store recorder so stop can be called elsewhere if needed - but here we just stop when user toggles mic off via UI
      window._fallbackRecorder = { mediaRecorder, stream };
      // indicate recording
      alert('Recording started (fallback). Click mic to stop when done.');
    }catch(e){ LOG("fallbackRecordAndUpload error " + e); alert('Recording failed: ' + e); }
  }

  // bind events
  try{
    const mic = document.getElementById('vaMic');
    const langSel = document.getElementById('vaLang');
    const statusEl = document.getElementById('vaStatus');
    const clearBtn = document.getElementById('vaClear');

    function setStatus(t){ try{ statusEl.innerText = t; }catch(e){} }

    let recog = null;
    let interimTranscript = '';
    let finalTranscript = '';
    let isListening = false;

    mic.addEventListener('click', async () => {
      mic.classList.toggle('va-micon');
      await playBeep();
      isListening = mic.classList.contains('va-micon');
      LOG("mic clicked; listening=" + isListening);
      if(isListening){
        setStatus('Listening… (click mic again to stop)');
        // Use browser SpeechRecognition if available, continuous mode
        if(!SpeechRecognition){
          LOG("SpeechRecognition not available; using fallbackRecordAndUpload");
          await fallbackRecordAndUpload(langSel.value || 'English');
          return;
        }
        try{
          finalTranscript = '';
          interimTranscript = '';
          const code = (langSel.value === 'Hindi' ? 'hi-IN' : (langSel.value === 'Telugu' ? 'te-IN' : (langSel.value === 'Tamil' ? 'ta-IN' : 'en-US')));

          recog = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
          recog.lang = code;
          recog.interimResults = true;
          recog.maxAlternatives = 1;
          recog.continuous = true; // important: keep listening until stopped manually

          recog.onresult = (ev) => {
            try{
              // accumulate results but do NOT stop on first result
              let interim = '';
              let final = '';
              for (let i = ev.resultIndex; i < ev.results.length; ++i) {
                const r = ev.results[i];
                if (r.isFinal) final += r[0].transcript;
                else interim += r[0].transcript;
              }
              if(final && final.trim()) {
                finalTranscript += (finalTranscript ? ' ' : '') + final.trim();
              }
              interimTranscript = interim;
              // update a visible little interim in log
              try {
                // remove previous interim indicator if present
                const prev = document.getElementById('vaInterim');
                if(prev) prev.remove();
                const logEl = document.getElementById('vaLog');
                const d = document.createElement('div');
                d.id = 'vaInterim';
                d.style.margin = '8px 0';
                d.innerHTML = `<div style="text-align:right"><span class="va-bubble-user">${escapeHtml((finalTranscript + ' ' + interimTranscript).trim())}</span></div>`;
                logEl.appendChild(d);
                logEl.scrollTop = logEl.scrollHeight;
              } catch(e){}
            } catch(e){
              LOG('onresult error ' + e);
            }
          };

          recog.onerror = (e) => { LOG("recog error " + JSON.stringify(e)); setStatus('Error'); };
          recog.onend = () => {
            LOG('recog ended (will not auto-restart). Waiting for stop toggle to send transcript.');
            // don't restart automatically; user will click mic to stop and trigger send
          };
          recog.start();
        }catch(e){
          LOG("start recog exception " + e);
          mic.classList.remove('va-micon');
          setStatus('Idle');
          await fallbackRecordAndUpload(langSel.value || 'English');
        }
      } else {
        // turning mic off: stop recognition and send accumulated text
        try{
          setStatus('Processing...');
          // if browser recog was used:
          if(recog){
            try{ recog.stop(); } catch(e){}
          }
          // if fallback recorder in use, stop and upload
          if(window._fallbackRecorder && window._fallbackRecorder.mediaRecorder){
            try{
              window._fallbackRecorder.mediaRecorder.onstop = async () => {
                const chunks = window._fallbackRecorder._chunks || [];
                const blob = new Blob(chunks, { type: 'audio/webm' });
                const fd = new FormData();
                fd.append('audio', blob, 'rec.webm');
                try{
                  const r = await fetch(proxyBase + '/stt', { method: 'POST', body: fd });
                  const j = await r.json();
                  const text = j.transcript || j.text || '';
                  if(text) await assistantFlow(text, langSel.value || 'English');
                  else await speakBrowser('(no speech detected)', langSel.value || 'English');
                }catch(e){ LOG("STT upload failed " + e); alert('STT failed: ' + e); }
                window._fallbackRecorder.stream.getTracks().forEach(t => t.stop());
                window._fallbackRecorder = null;
              };
              try{ window._fallbackRecorder.mediaRecorder.stop(); } catch(e){}
            }catch(e){ LOG('fallback stop error '+e);}
          }

          // Determine text from finalTranscript + interimTranscript
          let sendText = (finalTranscript + ' ' + interimTranscript).trim();
          // if empty, try to grab any logged interim element text (fallback)
          if(!sendText){
            try{
              const prev = document.getElementById('vaInterim');
              if(prev) {
                sendText = prev.innerText || '';
              }
            }catch(e){}
          }
          if(sendText && sendText.trim()){
            await assistantFlow(sendText, langSel.value || 'English');
          } else {
            await speakBrowser('(no speech detected)', langSel.value || 'English');
          }

          // cleanup interim display
          try {
            const prev = document.getElementById('vaInterim');
            if(prev) prev.remove();
          } catch(e){}

          setStatus('Idle');
        }catch(e){
          LOG("stop handling error " + e);
          setStatus('Idle');
        }
      }
    });

    clearBtn.addEventListener('click', ()=>{ try{ document.getElementById('vaLog').innerHTML=''; }catch(e){} });

    setStatus('Idle');
  }catch(e){
    LOG("voice widget binding error " + e);
  }

})();
</script>
</div>
"""

# Render voice widget when Chat Assistant active
if st.session_state.get("nav", "Home") == "Chat Assistant":
    widget_html = _voice_widget.replace("__VOICE_PROXY__", VOICE_PROXY)
    components.html(widget_html, height=380)
