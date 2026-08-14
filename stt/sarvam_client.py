"""
Sarvam AI Saaras v3 Speech-to-Text (STT) Client.

Features:
- Saaras v3 batch transcription (`model="saaras:v3"`, `mode="transcribe"`, `with_timestamps=True`).
- Streaming / WebSocket transcription path for low latency.
- Automatic fallback from streaming to batch if connection drops.
- Extensible language routing reading from `config.SUPPORTED_LANGUAGE_REGISTRY`.
- Safe offline fallback mock when API key is not supplied in test environments.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import config

logger = logging.getLogger(__name__)

# Map internal ISO language codes (hi, ta, en, etc.) to Sarvam language codes
def get_sarvam_language_code(lang: str) -> str:
    if not lang or str(lang).lower() in ["auto", "unknown", "none", ""]:
        return "unknown"
    info = config.get_language_info(str(lang).lower())
    code = info.get("sarvam_code", "unknown")
    return code if code else "unknown"


def normalize_audio_to_wav(input_path: str) -> str:
    """
    Converts any incoming browser/system audio format (WebM, Opus, Ogg, MP3, AAC, etc.)
    into a clean 16kHz 16-bit mono PCM WAV file acceptable to Sarvam AI.
    """
    import subprocess
    import tempfile
    
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out_path = out_file.name
    out_file.close()
    
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        out_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=5)
        if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception as e:
        logger.warning(f"Audio normalization via ffmpeg failed ({e}), using original file: {input_path}")
        
    return input_path


class SarvamSTTClient:
    """
    Client for Sarvam Saaras v3 Speech-to-Text with batch and streaming fallback.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.SARVAM_API_KEY
        self.client = None
        
        if self.api_key and self.api_key.strip():
            try:
                from sarvamai import SarvamAI
                self.client = SarvamAI(api_subscription_key=self.api_key.strip())
                logger.info("Initialized SarvamAI STT client successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize SarvamAI client: {e}")
                self.client = None
        else:
            logger.info("No SARVAM_API_KEY provided. STT running in mock/fallback mode.")

    def transcribe(self, audio_path: str, language_code: str = "auto") -> Dict[str, Any]:
        """
        Batch transcription using Sarvam Saaras v3.
        """
        sarvam_lang = get_sarvam_language_code(language_code)
        
        if not self.client:
            # Offline test fallback: if audio_path has an accompanying .txt or filename hint
            logger.warning(f"SARVAM_API_KEY missing or client not initialized. Simulating transcription for {audio_path}")
            stem_text = Path(audio_path).stem.replace("_", " ")
            return {
                "transcript": f"Mock transcription for {stem_text}",
                "language_code": language_code,
                "sarvam_code": sarvam_lang,
                "confidence": 0.95,
                "is_fallback": True,
                "raw": None,
            }
            
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        normalized_path = normalize_audio_to_wav(audio_path)
        try:
            with open(normalized_path, "rb") as f:
                response = self.client.speech_to_text.transcribe(
                    file=f,
                    model=config.SARVAM_MODEL,
                    language_code=sarvam_lang,
                    mode=config.SARVAM_MODE,
                    with_timestamps=True,
                )
                
            transcript = getattr(response, "transcript", "")
            if not transcript and isinstance(response, dict):
                transcript = response.get("transcript", "")
                
            detected_lang = getattr(response, "language_code", sarvam_lang)
            if isinstance(response, dict):
                detected_lang = response.get("language_code", detected_lang)
                
            return {
                "transcript": transcript,
                "language_code": detected_lang or language_code,
                "sarvam_code": sarvam_lang,
                "confidence": 1.0,
                "is_fallback": False,
                "raw": response,
            }
        except Exception as e:
            logger.error(f"Sarvam batch transcription failed: {e}")
            raise RuntimeError(f"Sarvam STT failed: {e}")
        finally:
            if normalized_path != audio_path and os.path.exists(normalized_path):
                try:
                    os.unlink(normalized_path)
                except Exception:
                    pass

    async def transcribe_streaming(
        self, audio_path: str, language_code: str = "hi"
    ) -> Dict[str, Any]:
        """
        Streaming WebSocket transcription for lower time-to-first-token with automatic batch fallback.
        """
        if not self.client:
            return self.transcribe(audio_path, language_code)
            
        sarvam_lang = get_sarvam_language_code(language_code)
        try:
            # Sarvam streaming transcription if supported by SDK
            if hasattr(self.client.speech_to_text, "transcribe_stream"):
                stream_res = await self.client.speech_to_text.transcribe_stream(
                    file=audio_path,
                    model=config.SARVAM_MODEL,
                    language_code=sarvam_lang,
                )
                return {
                    "transcript": stream_res.transcript,
                    "language_code": language_code,
                    "sarvam_code": sarvam_lang,
                    "confidence": 1.0,
                    "is_fallback": False,
                    "mode": "streaming",
                }
            else:
                # Fallback to batch if streaming method is unavailable
                return self.transcribe(audio_path, language_code)
        except Exception as e:
            logger.warning(f"Streaming STT failed ({e}). Falling back to batch transcription.")
            return self.transcribe(audio_path, language_code)


_STT_CLIENT: Optional[SarvamSTTClient] = None


def get_stt_client() -> SarvamSTTClient:
    """Singleton getter for SarvamSTTClient."""
    global _STT_CLIENT
    if _STT_CLIENT is None:
        _STT_CLIENT = SarvamSTTClient()
    return _STT_CLIENT
