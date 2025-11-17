# voice_proxy.py
# Multilingual voice proxy: Gemini text generation (short replies) + Google Cloud Text-to-Speech MP3 audio
# Supports: English, Hindi, Telugu, Tamil, Gujarati
# Endpoints:
#  - POST /voice   -> accepts {"text": "...", "lang": "Telugu"}  -> returns {"reply":"...", "audio_b64":"..."} or {"reply":"__STOP__"}
#  - POST /generate (optional)
#  - GET  /health
#
# Requirements:
#  - python-httpx
#  - FASTAPI & uvicorn
#  - Environment:
#      GOOGLE_API_KEY  -> key for Generative API (and/or Text-to-Speech). If you use separate keys, set
#      TTS_API_KEY     -> optional override specifically for Text-to-Speech
#
# Run:
#   uvicorn voice_proxy:app --port 8765

import os
import logging
import base64
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

logger = logging.getLogger("voice-proxy")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="voice-proxy-multilingual-tts")

# CORS (local dev, tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keys & model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # used for Gemini generateContent
TTS_API_KEY = os.getenv("TTS_API_KEY") or GOOGLE_API_KEY  # allow override for TTS
MODEL_NAME = os.getenv("VOICE_PROXY_MODEL", "models/gemini-2.5-flash")

# Short system instruction to force 2-3 line replies
SYSTEM_INSTRUCTION = (
    "You are a medical assistant. Always reply in 2–3 short lines. "
    "No long paragraphs. Be simple, clear, actionable (non-diagnostic)."
)

# Stop phrases for each language (exact-match trimmed). Add more phrases if you want fuzzy matching.
STOP_COMMANDS = {
    "en": ["stop", "stop speaking", "ok stop", "stop now", "please stop", "that's enough"],
    "hi": ["रुको", "बोलना बंद करो", "रुकिए", "बंद करो"],
    "te": ["ఆపు", "మాట్లాడటం ఆపు", "ఆపండి", "ఆపు ఇప్పుడు"],
    "ta": ["நிறுத்து", "பேசுவதை நிறுத்து", "நிறுத்துங்கள்"],
    "gu": ["રોકો", "બોલવું બંધ કરો", "બંધ કરો"]
}

# Map frontend language labels -> language codes (for TTS) + TTS voice selection hints
# language_code follows IETF BCP-47 (e.g., 'en-US', 'hi-IN', 'te-IN', 'ta-IN', 'gu-IN')
LANG_CONFIG = {
    "English": {"code": "en-US", "tts_voice": "en-US-Wavenet-D"},
    "Hindi": {"code": "hi-IN", "tts_voice": "hi-IN-Wavenet-A"},
    "Telugu": {"code": "te-IN", "tts_voice": "te-IN-Wavenet-A"},
    "Tamil": {"code": "ta-IN", "tts_voice": "ta-IN-Wavenet-A"},
    "Gujarati": {"code": "gu-IN", "tts_voice": "gu-IN-Wavenet-A"},
}

# Utility: trim / shorten text (extra safety)
def shorten_text_to_sentences(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    import re
    t = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+", t)
    if len(sents) >= max_sentences:
        out = " ".join(sents[:max_sentences]).strip()
    else:
        # fallback: use first ~300 chars
        out = t if len(t) <= 300 else t[:297].rsplit(" ", 1)[0] + "..."
    return out

# --- Call Gemini generateContent (text generation) ---
async def call_gemini_generate(prompt_text: str, timeout: float = 25.0) -> Tuple[int, str]:
    """
    Calls the Generative Language REST generateContent endpoint for Gemini-like models.
    Tries to extract a textual reply from the typical Gemini response structure.
    Returns (status_code, text_or_error)
    """
    if not GOOGLE_API_KEY:
        return 400, "Missing GOOGLE_API_KEY"

    base_v1 = "https://generativelanguage.googleapis.com/v1"
    url = f"{base_v1}/{MODEL_NAME}:generateContent?key={GOOGLE_API_KEY}"

    # Build a controlled prompt with the system instruction included
    full_prompt = f"{SYSTEM_INSTRUCTION}\n\nUser: {prompt_text}\nAssistant:"
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=payload)
        except Exception as e:
            logger.exception("Gemini generate error")
            return 502, f"Upstream generate request failed: {e}"

        if resp.status_code != 200:
            # return body for debugging in local dev
            return resp.status_code, resp.text or resp.reason_phrase

        try:
            data = resp.json()
        except Exception:
            return 200, resp.text

        # Try common extraction patterns for Gemini responses
        try:
            # candidates[0].content.parts[0].text
            cand = data.get("candidates")
            if cand and isinstance(cand, list) and len(cand) > 0:
                first = cand[0]
                content = first.get("content") or {}
                parts = content.get("parts") if isinstance(content, dict) else None
                if parts and isinstance(parts, list) and len(parts) > 0:
                    text = parts[0].get("text") or str(parts[0])
                    return 200, str(text)
                # fallback: maybe first has 'text' or 'output'
                for k in ("text", "output", "content"):
                    if k in first:
                        return 200, str(first[k])
            # sometimes top-level 'content' exists
            if "content" in data:
                return 200, str(data["content"])
            # fallback: try 'text' or full JSON
            for k in ("text", "output", "response"):
                if k in data:
                    return 200, str(data[k])
            # last fallback: stringified JSON
            return 200, str(data)
        except Exception as e:
            logger.exception("Error extracting gemini text")
            return 200, str(data)

# --- Call Google Cloud Text-to-Speech REST API to synthesize MP3 ---
async def tts_synthesize_mp3(text: str, language_code: str = "en-US", voice_name: Optional[str] = None, tts_api_key: Optional[str] = None) -> Tuple[int, Optional[str]]:
    """
    Uses Google Cloud Text-to-Speech REST API to synthesize MP3 and returns base64 audio.
    Returns (status_code, base64_audio_or_error)
    NOTE: You must enable 'Cloud Text-to-Speech' in your Google Cloud project if using a project key.
    """
    key = tts_api_key or TTS_API_KEY
    if not key:
        return 400, "Missing TTS API key (TTS_API_KEY or GOOGLE_API_KEY)"

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={key}"

    # Choose a voice; voice_name optional. If not provided, request languageCode only.
    voice = {"languageCode": language_code}
    if voice_name:
        voice["name"] = voice_name

    payload = {
        "input": {"text": text},
        "voice": voice,
        # audioConfig options: MP3 recommended for browser play
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.0, "pitch": 0.0}
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        try:
            resp = await client.post(url, json=payload)
        except Exception as e:
            logger.exception("TTS request failed")
            return 502, f"TTS request failed: {e}"

        if resp.status_code != 200:
            return resp.status_code, resp.text

        try:
            data = resp.json()
            audio_content = data.get("audioContent")
            if not audio_content:
                return 500, "No audioContent returned"
            # audioContent is already base64-encoded string per API
            return 200, audio_content
        except Exception as e:
            logger.exception("TTS parse error")
            return 500, resp.text

# --- Helpers ---
def detect_stop_phrase(text: str, lang_label: str) -> bool:
    if not text:
        return False
    label = lang_label or "English"
    cfg = LANG_CODE_FROM_LABEL(label)
    stop_list = STOP_COMMANDS.get(cfg, [])
    txt = text.strip().lower()
    return txt in [s.lower() for s in stop_list]

def LANG_CODE_FROM_LABEL(label: str) -> str:
    # Map front-end label to our short codes
    if not label:
        return "en"
    lookup = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
        "Gujarati": "gu",
        "en": "en",
        "hi": "hi",
        "te": "te",
        "ta": "ta",
        "gu": "gu"
    }
    return lookup.get(label, "en")

def TTS_LANG_CONFIG_FROM_LABEL(label: str):
    cfg = {
        "English": ("en-US", "en-US-Wavenet-D"),
        "Hindi": ("hi-IN", "hi-IN-Standard-A"),
        "Telugu": ("te-IN", "te-IN-Standard-A"),
        "Tamil": ("ta-IN", "ta-IN-Standard-A"),
        "Gujarati": ("gu-IN", "gu-IN-Standard-A")
    }
    return cfg.get(label, ("en-US", "en-US-Wavenet-D"))

# --- Endpoints ---

@app.post("/voice")
async def voice_endpoint(payload: dict):
    """
    Payload expected: {"text": "...", "lang": "Telugu"}  (lang optional, defaults to English)
    Returns: {"reply":"short text", "audio":"<base64 mp3>"} or {"reply":"__STOP__"}
    """
    if not payload:
        raise HTTPException(status_code=400, detail="Missing JSON body")
    text = str(payload.get("text") or payload.get("prompt") or "").strip()
    lang_label = payload.get("lang") or payload.get("language") or "English"

    # If user sent an empty text, return empty reply
    if text == "":
        return {"reply": ""}

    # Stop detection — exact-match; you can extend with fuzzy match if needed
    if detect_stop_phrase(text, lang_label):
        return {"reply": "__STOP__"}

    # Use Gemini (generateContent) to get short, multilingual reply
    # We instruct Gemini to reply in the requested language explicitly
    lang_code = LANG_CODE_FROM_LABEL(lang_label)
    language_instruction = {
        "en": "Reply in English.",
        "hi": "Reply in Hindi.",
        "te": "Reply in Telugu.",
        "ta": "Reply in Tamil.",
        "gu": "Reply in Gujarati."
    }.get(lang_code, "Reply in English.")

    final_prompt = f"{SYSTEM_INSTRUCTION}\n{language_instruction}\n\nUser: {text}\nAssistant:"

    status, gen_text = await call_gemini_generate(final_prompt)
    if status != 200:
        # Return empty in production; include debug details for local dev
        logger.warning("Gemini generate returned non-200: %s %s", status, gen_text)
        return {"reply": "", "error": f"Gemini upstream {status}", "detail": gen_text}

    # Safety: shorten again on proxy side
    short_reply = shorten_text_to_sentences(gen_text, max_sentences=2)

    # Create TTS audio for the selected language
    tts_lang_code, tts_voice_name = TTS_LANG_CONFIG_FROM_LABEL(lang_label)
    tts_status, audio_b64_or_err = await tts_synthesize_mp3(short_reply, language_code=tts_lang_code, voice_name=tts_voice_name)
    if tts_status != 200:
        logger.warning("TTS failed: %s %s", tts_status, audio_b64_or_err)
        # Return text-only reply as fallback
        return {"reply": short_reply, "audio": None, "error": f"TTS {tts_status}", "detail": audio_b64_or_err}

    # audio_b64_or_err is base64-encoded MP3 payload (string)
    return {"reply": short_reply, "audio": audio_b64_or_err}


@app.post("/generate")
async def generate_endpoint(payload: dict):
    prompt = payload.get("prompt") or payload.get("text") or ""
    status, result = await call_gemini_generate(prompt)
    if status == 200:
        return {"reply": result}
    return {"reply": "", "error": f"Upstream {status}", "detail": result}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}
