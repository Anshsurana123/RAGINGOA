"""
Hugging Face Space Application for Hacker House Goa 2026: Voice-Enabled Indic RAG.
Features:
- Complete Retro-Tropical Command Center Web UI at / (matching localhost:8000)
- Full FastAPI backend for Voice / Text RAG (/query, /health, /languages, /)
- Gradio fallback interface mounted at /gradio
- ZeroGPU compatible
"""

import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Compatibility shim for older packages importing HfFolder from huggingface_hub
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class DummyHfFolder:
            @staticmethod
            def get_token():
                import os
                return os.environ.get("HF_TOKEN") or None
            @staticmethod
            def save_token(token):
                pass
            @staticmethod
            def delete_token():
                pass
        huggingface_hub.HfFolder = DummyHfFolder
except Exception:
    pass

import gradio as gr
import uvicorn
from fastapi.responses import HTMLResponse

import config
from api.main import app as fastapi_app
from pipeline.orchestrator import get_orchestrator
from pipeline.schemas import QueryRequest, QueryResponse

# ZeroGPU decorator shim
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func


@spaces.GPU
def run_gradio_query(
    audio_path: Optional[str],
    text_query: Optional[str],
    language_hint: str,
    cross_lingual: bool,
) -> Tuple[str, str, str, str]:
    """Fallback Gradio interface handler."""
    if not audio_path and (not text_query or not text_query.strip()):
        return "⚠️ Please provide either a spoken voice recording or text query.", "", "N/A", "0.0 ms"
    
    hint = language_hint.strip() if language_hint and language_hint != "auto" else None
    req = QueryRequest(
        audio_path=audio_path if audio_path else None,
        text=text_query.strip() if text_query and text_query.strip() else None,
        language_hint=hint,
        cross_lingual=cross_lingual,
    )
    
    orchestrator = get_orchestrator()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    import nest_asyncio
    nest_asyncio.apply()
    res: QueryResponse = loop.run_until_complete(orchestrator.execute(req))
    return res.answer, res.query or res.transcript, res.language_detected.upper(), f"{res.total_ms:.1f} ms"


# Read the full custom HTML Command Center UI
def get_custom_html() -> str:
    demo_file = config.BASE_DIR / "demo" / "index.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Hacker House Goa 2026 Command Center</h1>"


# Minimal Gradio demo for ZeroGPU registration
with gr.Blocks(title="🌴 Hacker House Goa 2026 - Voice Indic RAG", theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    gr.HTML(get_custom_html())

# Mount Gradio app into FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"🌴 Starting Hacker House Goa Command Center UI on http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port)
