"""
Entry point for Hugging Face Spaces & local execution.
Runs the FastAPI Voice-Enabled Indic RAG application.
"""
import os
import uvicorn
from api.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Voice Indic RAG Server on http://{host}:{port}")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
