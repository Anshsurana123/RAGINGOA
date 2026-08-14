"""
Hugging Face Space Application for Hacker House Goa 2026: Voice-Enabled Indic RAG.
Renders the full retro-tropical Command Center UI and exposes FastAPI endpoints.
ZeroGPU compatible.
"""

import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
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
from fastapi import File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

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

import config
from pipeline.orchestrator import get_orchestrator
from pipeline.schemas import QueryRequest, QueryResponse
from retrieval.embed import get_embedder
from retrieval.index_faiss import get_index_manager


# Read the full custom HTML Command Center UI
def get_custom_html() -> str:
    demo_file = config.BASE_DIR / "demo" / "index.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Hacker House Goa 2026 Command Center</h1>"


@spaces.GPU
def _dummy_zerogpu():
    """ZeroGPU requirement: at least one function registered to event scan."""
    return True


# Build Gradio Blocks UI embedding the exact Command Center frontend with total CSS isolation
CUSTOM_CSS = """
body, html, .gradio-container, gradio-app {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
    height: 100% !important;
    background-color: #FEF8EA !important;
    color-scheme: light !important;
    overflow: auto !important;
}
footer, .svelte-10ymbgw, .built-with {
    display: none !important;
}
.contain {
    max-width: 100% !important;
    padding: 0 !important;
}
iframe#commandCenterFrame {
    width: 100% !important;
    min-height: 100vh !important;
    height: 100vh !important;
    border: none !important;
    display: block !important;
}
"""

with gr.Blocks(title="🌴 Hacker House Goa 2026 — Voice Indic RAG", css=CUSTOM_CSS) as demo:
    gr.HTML('<iframe id="commandCenterFrame" src="/demo_ui" style="width:100vw; height:100vh; border:none; position:fixed; top:0; left:0; z-index:99999; background:#FEF8EA;"></iframe>')
    # Hidden dummy button to ensure ZeroGPU handler registration
    dummy_btn = gr.Button("zero_gpu_anchor", visible=False)
    dummy_btn.click(fn=_dummy_zerogpu)


# Preload embedding model and index manager at startup
print("[Space Startup] Preloading embedding model and FAISS vector indexes...")
get_embedder()
get_index_manager()
print("[Space Startup] Models and FAISS indexes preloaded successfully.")


# Attach FastAPI endpoints directly to demo.app
app = demo.app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/demo_ui", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def serve_isolated_demo_ui():
    """Serves the isolated clean Command Center frontend."""
    return HTMLResponse(content=get_custom_html(), status_code=200)


@app.get("/health", response_class=JSONResponse)
async def health_check() -> Dict[str, Any]:
    """Health check reporting system and index readiness."""
    index_mgr = get_index_manager()
    index_stats = {
        name: idx.index.ntotal for name, idx in index_mgr.indexes.items()
    }
    return {
        "status": "healthy",
        "configured_languages": config.LANGUAGES,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "indexes_loaded": index_stats,
        "centroids_available": list(index_mgr.centroids.keys()),
        "sarvam_stt_configured": bool(config.SARVAM_API_KEY),
        "llm_fallback_configured": bool(config.LLM_API_KEY),
    }


@app.get("/languages", response_class=JSONResponse)
async def get_supported_languages() -> Dict[str, Any]:
    """Returns metadata for all currently configured active languages."""
    lang_details = [
        {"code": l, **config.get_language_info(l)} for l in config.LANGUAGES
    ]
    return {
        "active_languages": config.LANGUAGES,
        "language_details": lang_details,
    }


@app.post("/query", response_model=QueryResponse)
async def query_pipeline(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language_hint: Optional[str] = Form(None),
    cross_lingual: Optional[bool] = Form(True),
    request_body: Optional[QueryRequest] = None,
) -> QueryResponse:
    """
    Execute end-to-end Voice RAG query for the Command Center UI.
    """
    orchestrator = get_orchestrator()
    temp_audio_path = None
    
    try:
        if request_body and (request_body.text or request_body.audio_path):
            return await orchestrator.execute(request_body)
            
        if file and file.filename:
            suffix = Path(file.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_audio_path = tmp.name
                
            req = QueryRequest(
                audio_path=temp_audio_path,
                language_hint=language_hint,
                cross_lingual=True if cross_lingual is None else cross_lingual,
            )
            return await orchestrator.execute(req)
            
        if text and text.strip():
            req = QueryRequest(
                text=text.strip(),
                language_hint=language_hint,
                cross_lingual=True if cross_lingual is None else cross_lingual,
            )
            return await orchestrator.execute(req)
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file' audio upload or 'text' query must be provided.",
        )
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"🌴 Starting Hacker House Goa Command Center UI on http://{host}:{port}")
    demo.queue().launch(server_name=host, server_port=port)
