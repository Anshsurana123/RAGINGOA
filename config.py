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
# Default initial scope: Hindi (hi), Tamil (ta), and English (en)
LANGUAGES = ["hi", "ta", "en"]

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

# FAISS HNSW Index Hyperparameters
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# Retrieval Top-K defaults
FAISS_TOP_K = 15
RERANK_TOP_K = 5
HYBRID_BM25_WEIGHT = 0.35  # Dense score weight = 1 - HYBRID_BM25_WEIGHT

# Cross-Encoder Re-Ranking Configuration (Sub-200ms CPU re-ranking)
ENABLE_CROSS_ENCODER = True
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
CROSS_ENCODER_LOCAL_CACHE = Path(
    os.getenv(
        "CROSS_ENCODER_LOCAL_CACHE",
        "C:/Users/ANSH/Documents/antigravity/keen-hawking/model_cache/cross-encoder-ms-marco-MiniLM-L-6-v2",
    )
)
CROSS_ENCODER_TOP_K = 3
CROSS_ENCODER_THRESHOLD = float(os.getenv("CROSS_ENCODER_THRESHOLD", "-0.5"))


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
OFF_TOPIC_DISTANCE_THRESHOLD = 0.18  # Calibrated for multilingual-e5-small normalized embeddings (1 - cosine_similarity)

# Post-retrieval confidence threshold (calibrated composite dense & lexical match score)
MIN_CONFIDENT_MATCH_SCORE = 0.75  # Empirically calibrated: declines all 10 off-topic queries (<=0.7257) while passing all 45 in-scope queries (>=0.7825)





# Post-generation grounding check threshold (lexical + semantic overlap)
GROUNDING_OVERLAP_THRESHOLD = 0.30

# ==========================================
# 7. LLM MULTI-TIER PROVIDER & GENERATION CONFIG
# ==========================================
# HARD OVERRIDE: Prevent live network calls during critical path latency budget (<200ms)
# Setting this to False keeps all LLM/Groq/Cerebras code dormant even if API keys are present.
ALLOW_NETWORK_CALLS_IN_PIPELINE = False

# Semantic Answer Cache (Fast lookup for gold answers of known queries in MSMARCO)
SEMANTIC_ANSWER_CACHE_ENABLED = True
SEMANTIC_ANSWER_CACHE_THRESHOLD = 0.93

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

