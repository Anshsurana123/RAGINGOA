"""
Retriever module providing fast warmup and retrieval latency evaluation.
Combines ONNX CPU embedding vectorization with FAISS HNSW index search.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure root workspace directory is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config
from retrieval.embed import get_embedder
from retrieval.index_faiss import get_index_manager

logger = logging.getLogger(__name__)


@dataclass
class SearchResponse:
    """Standard search response object containing stage latencies and retrieved candidates."""
    total_ms: float
    embed_ms: float
    search_ms: float
    results: List[Dict[str, Any]] = field(default_factory=list)
    query: str = ""


_EMBEDDER = None
_INDEX_MANAGER = None


def get_retriever_components():
    """Lazily load and cache embedder and FAISS index manager singletons."""
    global _EMBEDDER, _INDEX_MANAGER
    if _EMBEDDER is None:
        _EMBEDDER = get_embedder()
    if _INDEX_MANAGER is None:
        _INDEX_MANAGER = get_index_manager()
    return _EMBEDDER, _INDEX_MANAGER


def warmup():
    """
    Warms up embedding ONNX runtime graph and loads FAISS indexes into memory.
    Executes a dry-run inference to eliminate cold-start JIT overhead.
    """
    embedder, index_manager = get_retriever_components()
    
    # Warm up embedding graph
    dummy_vec = embedder.encode_queries("warmup query")
    
    # Warm up FAISS index search
    if "passage_native" in index_manager.indexes:
        _ = index_manager.indexes["passage_native"].search(dummy_vec, top_k=5)
    elif index_manager.indexes:
        first_idx = next(iter(index_manager.indexes.values()))
        _ = first_idx.search(dummy_vec, top_k=5)


def search(query: str, top_k: int = 5, target_lang: Optional[str] = None) -> SearchResponse:
    """
    Executes end-to-end vector search:
    1. Encodes query using ONNX Runtime with 'query: ' prefix.
    2. Queries FAISS HNSW index.
    3. Measures precise millisecond timing for both stages.
    """
    embedder, index_manager = get_retriever_components()
    
    # Stage 1: Query Vectorization
    t0 = time.perf_counter()
    query_vector = embedder.encode_queries(query)
    embed_ms = (time.perf_counter() - t0) * 1000.0
    
    # Stage 2: FAISS Index Search
    t1 = time.perf_counter()
    results = []
    if "passage_native" in index_manager.indexes:
        results = index_manager.indexes["passage_native"].search(
            query_vec=query_vector,
            target_lang=target_lang,
            top_k=top_k,
        )
    elif index_manager.indexes:
        first_idx = next(iter(index_manager.indexes.values()))
        results = first_idx.search(
            query_vec=query_vector,
            target_lang=target_lang,
            top_k=top_k,
        )
    search_ms = (time.perf_counter() - t1) * 1000.0
    
    total_ms = embed_ms + search_ms
    
    return SearchResponse(
        total_ms=round(total_ms, 2),
        embed_ms=round(embed_ms, 2),
        search_ms=round(search_ms, 2),
        results=results,
        query=query,
    )
