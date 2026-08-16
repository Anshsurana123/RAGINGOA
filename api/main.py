"""
FastAPI Serving Layer for Voice-Enabled Indic RAG System.

Endpoints:
- POST /query: Accepts audio file upload (multipart/form-data) OR JSON text bypass. Returns QueryResponse.
- GET /health: Healthcheck reporting active languages, index vector counts, and model status.
- GET /languages: Returns active languages strictly derived from config.LANGUAGES.
- GET /: Web Demo UI.
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline.orchestrator import get_orchestrator, RAGPipelineOrchestrator
from pipeline.schemas import QueryRequest, QueryResponse
from retrieval.embed import get_embedder
from retrieval.index_faiss import get_index_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler: Preload embedding models and FAISS indexes ONCE at startup.
    Never re-loads per request. Self-heals if indexes are missing or empty.
    """
    print("[API Lifespan] Initializing and pre-loading embedding model...")
    embedder = get_embedder()
    
    print("[API Lifespan] Loading FAISS HNSW indexes and corpus centroids into memory...")
    index_mgr = get_index_manager()
    
    # Ensure indexes are populated
    for name, idx in index_mgr.indexes.items():
        print(f"[API Lifespan] Index '{name}': {idx.index.ntotal} vectors loaded.")
        
    print(f"[API Lifespan] Initialized successfully. Active languages: {config.LANGUAGES}")
    print(f"[API Lifespan] Allow Network Calls Switch: {config.ALLOW_NETWORK_CALLS_IN_PIPELINE}")
    
    yield
    
    print("[API Lifespan] Shutting down RAG service.")


app = FastAPI(
    title="Voice-Enabled Indic RAG API",
    description="Instrumented low-latency Voice RAG pipeline for Indic languages (Hindi, Tamil, English)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for demo interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "allow_network_calls": config.ALLOW_NETWORK_CALLS_IN_PIPELINE,
        "sarvam_stt_configured": bool(config.SARVAM_API_KEY),
        "llm_fallback_configured": bool(config.LLM_API_KEY and config.ALLOW_NETWORK_CALLS_IN_PIPELINE),
        "semantic_answer_cache_configured": config.SEMANTIC_ANSWER_CACHE_ENABLED,
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
    cross_lingual: Optional[bool] = Form(False),
    request_body: Optional[QueryRequest] = None,
) -> QueryResponse:
    """
    Execute end-to-end Voice RAG query.
    Accepts:
    1. Multipart file upload ('file') with optional 'language_hint' and 'cross_lingual'
    2. Multipart form text ('text') with optional 'language_hint' and 'cross_lingual'
    3. JSON body with text / audio_path / language_hint / cross_lingual
    """
    orchestrator = get_orchestrator()
    temp_audio_path = None
    
    try:
        # 1. Handle JSON request body
        if request_body and (request_body.text or request_body.audio_path):
            return await orchestrator.execute(request_body)
            
        # 2. Handle Multipart audio upload
        if file and file.filename:
            suffix = Path(file.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_audio_path = tmp.name
                
            req = QueryRequest(
                audio_path=temp_audio_path,
                language_hint=language_hint,
                cross_lingual=False if cross_lingual is None else cross_lingual,
            )
            response = await orchestrator.execute(req)
            return response
            
        # 3. Handle Multipart Form text bypass
        if text and text.strip():
            req = QueryRequest(
                text=text.strip(),
                language_hint=language_hint,
                cross_lingual=False if cross_lingual is None else cross_lingual,
            )
            return await orchestrator.execute(req)
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file' audio upload or 'text' query must be provided.",
        )
    finally:
        # Clean up temporary audio file if created
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


@app.get("/", response_class=HTMLResponse)
async def serve_demo_ui():
    """Serves the interactive voice-enabled web demo."""
    demo_file = config.BASE_DIR / "demo" / "index.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Voice-Enabled Indic RAG Service</h1><p>API is running. Visit <a href='/docs'>/docs</a> for API specification.</p>"
