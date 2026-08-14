"""
Hugging Face Space Application for Hacker House Goa 2026: Voice-Enabled Indic RAG.
Features:
- Multilingual Voice Input (Sarvam Saaras v3 STT + ffmpeg normalization)
- Federated Dense FAISS Retrieval (multilingual-e5-small) & BM25-Hybrid Fusion
- Grounded Cross-Lingual LLM Synthesis (Groq Llama 3.3 70B / Extractive Fallback)
- Multi-tier Guardrails & Latency Waterfall Instrumentation
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

# Try importing spaces for HF ZeroGPU compatibility
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


# Initialize orchestrator singleton at startup
_orchestrator = None

def get_pipeline():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = get_orchestrator()
    return _orchestrator


@spaces.GPU
def run_query(
    audio_path: Optional[str],
    text_query: Optional[str],
    language_hint: str,
    cross_lingual: bool,
) -> Tuple[str, str, str, str, str, str]:
    """
    Executes the Voice / Text RAG pipeline.
    """
    if not audio_path and (not text_query or not text_query.strip()):
        return (
            "⚠️ Please provide either a spoken voice recording or text query.",
            "N/A",
            "N/A",
            "N/A",
            "{}",
            "[]"
        )
    
    # Clean language hint
    hint = language_hint.strip() if language_hint and language_hint != "auto" else None
    
    req = QueryRequest(
        audio_path=audio_path if audio_path else None,
        text=text_query.strip() if text_query and text_query.strip() else None,
        language_hint=hint,
        cross_lingual=cross_lingual,
    )
    
    orchestrator = get_pipeline()
    
    # Run async pipeline synchronously
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        response: QueryResponse = loop.run_until_complete(orchestrator.execute(req))
    else:
        response: QueryResponse = loop.run_until_complete(orchestrator.execute(req))
        
    # Format Answer Markdown
    answer_md = f"### 💡 Answer\n\n{response.answer}\n\n"
    if response.answer_source == "declined":
        answer_md += f"> ⚠️ **Notice**: Request was declined by safety guardrail or insufficient corpus context."
    elif response.answer_source == "cross_lingual_synthesis":
        answer_md += f"> 🌐 *Synthesized via Cross-Lingual Federation across multiple Indic languages.*"
    elif response.answer_source == "generated":
        answer_md += f"> 🤖 *Generated via Grounded LLM Synthesis.*"
    else:
        answer_md += f"> 📄 *Extracted directly from top-ranked corpus passages.*"

    # Format Retrieved Passages Markdown
    if response.retrieved_chunks:
        chunks_md = "### 📚 Retrieved Grounding Passages\n\n"
        for i, chunk in enumerate(response.retrieved_chunks, 1):
            chunks_md += f"#### Source #{i} [{chunk.source_lang.upper()}] — Score: `{chunk.final_score:.4f}` (Strategy: `{chunk.chunk_strategy}`)\n"
            chunks_md += f"> {chunk.text}\n\n"
    else:
        chunks_md = "*No grounding passages retrieved or query was declined before retrieval.*"

    # Format Timing Breakdown Markdown
    timing_md = f"### ⏱️ Latency Telemetry\n\n"
    timing_md += f"- **Retrieval Latency**: `{response.retrieval_ms:.2f} ms`\n"
    timing_md += f"- **Full Pipeline Latency**: `{response.total_ms:.2f} ms`\n\n"
    timing_md += "| Stage | Latency (ms) | Status | Details |\n"
    timing_md += "| :--- | :--- | :--- | :--- |\n"
    for st in response.stage_timings:
        status_icon = "✅" if st.success else "❌"
        timing_md += f"| `{st.stage}` | **{st.ms:.2f}** | {status_icon} | {st.details or ''} |\n"

    # Guardrails JSON
    guardrails_json = json.dumps(response.guardrail_flags, indent=2, ensure_ascii=False)
    
    # Raw JSON response
    raw_json = response.model_dump_json(indent=2)

    return (
        answer_md,
        response.query or response.transcript or (text_query or ""),
        response.language_detected.upper(),
        f"{response.total_ms:.1f} ms",
        chunks_md,
        timing_md,
    )


# ==========================================
# CUSTOM CSS & THEME
# ==========================================
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1200px !important;
    margin: auto !important;
}
.header-badge {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #064E3B;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #374151;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
"""

with gr.Blocks(title="🌴 Hacker House Goa 2026 - Voice Indic RAG", css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <span class="header-badge">🌴 HACKER HOUSE GOA 2026</span>
        <h1 class="hero-title">Voice-Enabled Multilingual Indic RAG</h1>
        <p class="hero-subtitle">
            Zero-code extensible cross-lingual RAG for <strong>Hindi</strong>, <strong>Tamil</strong>, and <strong>English</strong> 
            powered by Sarvam Saaras v3 STT, FAISS HNSW, multilingual-e5 embeddings & Groq Llama 3.3.
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.TabItem("🎙️ Voice & Text Assistant"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 📥 Input Query")
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎙️ Spoken Voice Input (Hindi / Tamil / English)",
                    )
                    text_input = gr.Textbox(
                        lines=3,
                        placeholder="Or enter text query in Hindi (e.g. 'कंप्यूटर क्या है?'), Tamil (e.g. 'கணினி என்றால் என்ன?'), or English...",
                        label="⌨️ Text Bypass Query",
                    )
                    
                    with gr.Row():
                        lang_select = gr.Dropdown(
                            choices=["auto", "hi", "ta", "en"],
                            value="auto",
                            label="🌐 Language Hint / Mode",
                        )
                        cross_lingual_check = gr.Checkbox(
                            value=True,
                            label="Cross-Lingual Federation",
                            info="Retrieve facts across all 3 languages",
                        )
                        
                    submit_btn = gr.Button("🚀 Execute Voice RAG", variant="primary", size="lg")
                    clear_btn = gr.ClearButton([audio_input, text_input])
                    
                    gr.Markdown("#### 💡 Example Queries")
                    gr.Examples(
                        examples=[
                            [None, "कंप्यूटर क्या है और यह कैसे काम करता है?", "hi", True],
                            [None, "கணினி என்றால் என்ன? அது எவ்வாறு செயல்படுகிறது?", "ta", True],
                            [None, "What is a computer and how does it store information?", "en", True],
                            [None, "भारत की राजधानी क्या है?", "hi", True],
                            [None, "சென்னை எந்த மாநிலத்தில் உள்ளது?", "ta", True],
                        ],
                        inputs=[audio_input, text_input, lang_select, cross_lingual_check],
                    )

                with gr.Column(scale=6):
                    gr.Markdown("### 📤 Pipeline Output")
                    
                    with gr.Row():
                        detected_lang_out = gr.Textbox(label="Detected Language", scale=1)
                        total_time_out = gr.Textbox(label="Total Latency", scale=1)
                    
                    resolved_query_out = gr.Textbox(label="Processed Query / Transcript", lines=1)
                    answer_out = gr.Markdown("### 💡 Answer\n\n*Response will appear here...*")
                    
                    with gr.Accordion("📚 Retrieved Grounding Passages & Evidence", open=True):
                        sources_out = gr.Markdown("*Grounding passages will appear here.*")
                        
                    with gr.Accordion("⏱️ Latency Waterfall & Stage Telemetry", open=False):
                        timing_out = gr.Markdown("*Telemetry timings will appear here.*")

            # Bind execution event
            submit_btn.click(
                fn=run_query,
                inputs=[audio_input, text_input, lang_select, cross_lingual_check],
                outputs=[
                    answer_out,
                    resolved_query_out,
                    detected_lang_out,
                    total_time_out,
                    sources_out,
                    timing_out,
                ],
            )
            
        with gr.TabItem("🏛️ Architecture & Benchmarks"):
            gr.Markdown("""
            ### 🏛️ System Architecture & Engineering Rationales
            
            - **Multilingual Semantic Space**: `intfloat/multilingual-e5-small` maps English, Hindi, and Tamil to shared 384d vector space with mandatory `query: ` and `passage: ` prefixes.
            - **Sub-Millisecond Vector Search**: In-memory FAISS HNSW (`M=32`, `efConstruction=200`, `efSearch=64`) achieves **<35ms** search latencies.
            - **Hybrid Re-ranking**: Fast BM25 score fusion operates in <2ms on candidates.
            - **Zero-Crash Resilience**: Automated retries with exponential backoff for external APIs, deterministic extractive local fallback if LLM is unavailable.
            - **Multi-Tier Safety Guardrails**: Fast regex safety pass $\\rightarrow$ Centroid distance gatekeeper ($d > 0.78$) $\\rightarrow$ Neural safety classifier $\\rightarrow$ Grounding verification.

            ### ⚡ SLA Latency Quantiles (61 Query Test Benchmark)
            | Metric Scope | Target SLA | P50 (Median) | P70 | P100 (Max) | SLA Status |
            | :--- | :--- | :--- | :--- | :--- | :--- |
            | **Retrieval Stage (FAISS + BM25)** | **~200 ms** | **34.64 ms** | **38.82 ms** | **107.82 ms** | ✅ **PASS (< 200 ms)** |
            | **Full End-to-End Pipeline (Text)** | — | **2456.27 ms** | **2663.28 ms** | **3382.86 ms** | ✅ **PASS** |
            """)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
