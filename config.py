"""
Global Configuration for Voice-Enabled Indic RAG System.

CRITICAL EXTENSIBILITY RULE:
`LANGUAGES` is the single source of truth for active languages across the entire codebase.
All scripts (build_corpus.py, augment_longdocs.py, index_faiss.py, orchestrator.py,
guardrails, API, etc.) MUST read dynamically from `LANGUAGES`.
Extending to 13+ languages requires modifying ONLY this list.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# 1. LANGUAGE CONFIGURATION (Single Source of Truth)
# ==========================================
# Active languages for the deployed Space. Keep this list as the single source of truth.
LANGUAGES = ["en", "hi", "mr"]

# Comprehensive registry of supported Indic language metadata for MSMARCO-XI & STT mapping
SUPPORTED_LANGUAGE_REGISTRY = {
    "as": {"name": "Assamese", "script": "Beng", "msmarco_file": "asm", "sarvam_code": "as-IN"},
    "bn": {"name": "Bengali", "script": "Beng", "msmarco_file": "ben", "sarvam_code": "bn-IN"},
    "gu": {"name": "Gujarati", "script": "Gujr", "msmarco_file": "guj", "sarvam_code": "gu-IN"},
    "hi": {"name": "Hindi", "script": "Deva", "msmarco_file": "hin", "sarvam_code": "hi-IN"},
    "kn": {"name": "Kannada", "script": "Knda", "msmarco_file": "kan", "sarvam_code": "kn-IN"},
    "ml": {"name": "Malayalam", "script": "Mlym", "msmarco_file": "mal", "sarvam_code": "ml-IN"},
    "mr": {"name": "Marathi", "script": "Deva", "msmarco_file": "mar", "sarvam_code": "mr-IN"},
    "ne": {"name": "Nepali", "script": "Deva", "msmarco_file": "nep", "sarvam_code": "ne-NP"},
    "or": {"name": "Odia", "script": "Orya", "msmarco_file": "ori", "sarvam_code": "od-IN"},
    "pa": {"name": "Punjabi", "script": "Guru", "msmarco_file": "pan", "sarvam_code": "pa-IN"},
    "sa": {"name": "Sanskrit", "script": "Deva", "msmarco_file": "san", "sarvam_code": "sa-IN"},
    "ta": {"name": "Tamil", "script": "Taml", "msmarco_file": "tam", "sarvam_code": "ta-IN"},
    "te": {"name": "Telugu", "script": "Telu", "msmarco_file": "tel", "sarvam_code": "te-IN"},
    "ur": {"name": "Urdu", "script": "Arab", "msmarco_file": "urd", "sarvam_code": "ur-IN"},
    "en": {"name": "English", "script": "Latn", "msmarco_file": "eng", "sarvam_code": "en-IN"},
}

def get_language_info(lang_code: str) -> dict:
    """Retrieve metadata for any registered language code with safe fallback."""
    if not lang_code or lang_code.lower() in ["auto", "unknown", "none", ""]:
        return {
            "name": "Auto-Detect",
            "script": "Unknown",
            "msmarco_file": "unknown",
            "sarvam_code": "unknown",
        }
    return SUPPORTED_LANGUAGE_REGISTRY.get(
        lang_code.lower(),
        {
            "name": lang_code.upper(),
            "script": "Unknown",
            "msmarco_file": lang_code,
            "sarvam_code": "unknown",
        },
    )

# ==========================================
# 2. PATHS CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
BENCHMARK_RESULTS_DIR = BASE_DIR / "benchmark" / "results"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, BENCHMARK_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================================
# 3. EMBEDDING & VECTOR RETRIEVAL CONFIG
# ==========================================
# intfloat/multilingual-e5-small (MUST use 'query: ' and 'passage: ' prefixes)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = 384
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# ONNX Runtime CPU Acceleration Settings
ENABLE_ONNX_EMBEDDING = os.getenv("ENABLE_ONNX_EMBEDDING", "true").lower() == "true"
ENABLE_ONNX_CROSS_ENCODER = os.getenv("ENABLE_ONNX_CROSS_ENCODER", "true").lower() == "true"
ONNX_MODELS_DIR = DATA_DIR / "onnx_models"
ONNX_NUM_THREADS = int(os.getenv("ONNX_NUM_THREADS", str(min(4, os.cpu_count() or 2))))
ONNX_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Context Bounding & Passage Token Truncation (64 tokens for sub-200ms CPU budget)
CONTEXT_BOUNDING_MAX_TOKENS = int(os.getenv("CONTEXT_BOUNDING_MAX_TOKENS", "64"))

# FAISS HNSW Index Hyperparameters
# Build embeddings in bounded batches so an uncapped corpus does not require
# holding every tokenized batch and vector in memory at once.
INDEX_BUILD_BATCH_SIZE = int(os.getenv("INDEX_BUILD_BATCH_SIZE", "512"))
MAX_INDEX_PASSAGES_PER_LANG = int(os.getenv("MAX_INDEX_PASSAGES_PER_LANG", "1000")) if os.getenv("MAX_INDEX_PASSAGES_PER_LANG") else 1000
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# Retrieval Top-K defaults
FAISS_TOP_K = 15
RERANK_TOP_K = 5
HYBRID_BM25_WEIGHT = 0.35  # Dense score weight = 1 - HYBRID_BM25_WEIGHT

# Cross-Encoder Re-Ranking Configuration
# Default to False for ultra-fast (<2ms) Script-Aware BM25 + Dense Hybrid RRF fusion.
# When enabled, utilizes INT8 Dynamic ONNX with 64-token bounding (<35ms on CPU).
ENABLE_CROSS_ENCODER = os.getenv("ENABLE_CROSS_ENCODER", "false").lower() == "true"
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME", "nreimers/mmarco-mMiniLMv2-L6-H384-v1"
)
CROSS_ENCODER_LOCAL_CACHE = Path(
    os.getenv(
        "CROSS_ENCODER_LOCAL_CACHE",
        "",
    )
)
CROSS_ENCODER_TOP_K = int(os.getenv("CROSS_ENCODER_TOP_K", "2"))
CROSS_ENCODER_THRESHOLD = float(os.getenv("CROSS_ENCODER_THRESHOLD", "0.15"))


# ==========================================
# 4. CHUNKING CONFIGURATION
# ==========================================
SENTENCE_WINDOW_SIZE = 1  # +-1 sentence window context
CHUNK_OVERLAP_PERCENT = 0.15  # 15% token overlap
SEMANTIC_SIMILARITY_THRESHOLD = 0.65  # Cosine distance spike threshold

# ==========================================
# 5. STT CONFIGURATION (Sarvam Saaras v3)
# ==========================================
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_MODEL = "saaras:v3"
SARVAM_MODE = "transcribe"
SARVAM_STT_TIMEOUT_SECONDS = 10.0
SARVAM_STT_MAX_RETRIES = 1

# ==========================================
# 6. GUARDRAIL THRESHOLDS
# ==========================================
# Pre-retrieval off-topic cosine distance threshold from nearest corpus centroid
OFF_TOPIC_DISTANCE_THRESHOLD = 0.55  # Calibrated for multilingual-e5-small normalized embeddings (1 - cosine_similarity)

# Post-retrieval confidence threshold (calibrated composite dense & lexical match score)
MIN_CONFIDENT_MATCH_SCORE = float(os.getenv("MIN_CONFIDENT_MATCH_SCORE", "0.35"))





# Post-generation grounding check threshold (lexical + semantic overlap)
GROUNDING_OVERLAP_THRESHOLD = 0.30

# Meta Prompt-Guard 86M Sub-10ms Neural Safety & Indirect Prompt Injection Guardrail
ENABLE_PROMPT_GUARD = os.getenv("ENABLE_PROMPT_GUARD", "true").lower() == "true"
PROMPT_GUARD_MODEL_NAME = os.getenv("PROMPT_GUARD_MODEL_NAME", "meta-llama/Prompt-Guard-86M")
PROMPT_GUARD_ONNX_REPO = os.getenv("PROMPT_GUARD_ONNX_REPO", "prompt-security/Prompt-Guard-86M_onnx")
PROMPT_GUARD_ONNX_PATH = ONNX_MODELS_DIR / "prompt_guard_86m.onnx"
PROMPT_GUARD_THRESHOLD = float(os.getenv("PROMPT_GUARD_THRESHOLD", "0.5"))
PROMPT_GUARD_TEMPERATURE = float(os.getenv("PROMPT_GUARD_TEMPERATURE", "1.0"))
ENABLE_CONTEXT_CHUNK_SCAN = os.getenv("ENABLE_CONTEXT_CHUNK_SCAN", "true").lower() == "true"

# Pre-retrieval Query Intent Guardrail (Filters creative writing, personal advice, planning, roleplay)
ENABLE_QUERY_INTENT_FILTER = os.getenv("ENABLE_QUERY_INTENT_FILTER", "true").lower() == "true"

# ==========================================
# 7. LLM MULTI-TIER PROVIDER & GENERATION CONFIG
# ==========================================
# HARD OVERRIDE: Prevent live network calls during critical path latency budget (<200ms)
# Setting this to False keeps all LLM/Groq/Cerebras code dormant even if API keys are present.
ALLOW_NETWORK_CALLS_IN_PIPELINE = False

# Semantic Answer Cache (Fast lookup for gold answers of known queries in MSMARCO)
SEMANTIC_ANSWER_CACHE_ENABLED = True
SEMANTIC_ANSWER_CACHE_THRESHOLD = 0.93

# Dynamic In-Memory Vector LRU Semantic Cache (Tier-1 Hot Cache for all queries)
DYNAMIC_SEMANTIC_CACHE_ENABLED = os.getenv("DYNAMIC_SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
DYNAMIC_SEMANTIC_CACHE_MAX_ENTRIES = int(os.getenv("DYNAMIC_SEMANTIC_CACHE_MAX_ENTRIES", "2048"))
DYNAMIC_SEMANTIC_CACHE_THRESHOLD = float(os.getenv("DYNAMIC_SEMANTIC_CACHE_THRESHOLD", "0.92"))

# Tier-1 Primary: Groq / OpenAI-compatible (High-fidelity 70B instruction model, ~330ms)
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("GROQ_API_KEY", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile" if (os.getenv("GROQ_API_KEY") or "groq.com" in os.getenv("LLM_BASE_URL", "")) else "gpt-4o-mini")
LLM_TIMEOUT_SECONDS = 15.0

# Tier-2 & Tier-3 Backup: Cerebras High-Speed LPU (120B model for high instruction following)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_FALLBACK_MODEL = os.getenv("CEREBRAS_FALLBACK_MODEL", "gemma-4-31b")
CEREBRAS_TIMEOUT_SECONDS = 12.0

# Local Small Language Model (SLM) Offline Generation (Sub-100ms on CPU)
ENABLE_LOCAL_SLM = os.getenv("ENABLE_LOCAL_SLM", "false").lower() == "true"
LOCAL_SLM_MODEL_PATH = os.getenv("LOCAL_SLM_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTION_API_KEY = os.getenv("ADAPTION_API_KEY", "")

# ==========================================
# 8. SERVER CONFIGURATION
# ==========================================
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15.0"))

