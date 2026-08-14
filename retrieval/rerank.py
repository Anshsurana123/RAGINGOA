"""
Lightweight BM25-Hybrid Re-ranking.

Re-ranks only the merged top-k candidates (not the whole corpus) using BM25Okapi score fusion.
Fuses dense vector similarity scores with lexical BM25 scores to produce balanced ranking in <2ms.
"""

import re
from typing import Any, Dict, List
import numpy as np
from rank_bm25 import BM25Okapi
import config


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Multilingual word-level tokenization for BM25 scoring.
    Splits on whitespace and non-alphanumeric punctuation.
    """
    if not text:
        return []
    # Clean and split into lowercased tokens
    tokens = re.findall(r'\w+', text.lower(), re.UNICODE)
    return tokens if tokens else text.lower().split()


def rerank_bm25_hybrid(
    query_text: str,
    candidates: List[Dict[str, Any]],
    bm25_weight: float = config.HYBRID_BM25_WEIGHT,
    top_k: int = config.RERANK_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Hybrid re-ranking over candidate list combining Dense vector score with BM25 score.
    
    Formula:
        FinalScore = (1 - bm25_weight) * NormalizedDenseScore + bm25_weight * NormalizedBM25Score
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        c = candidates[0].copy()
        c["final_score"] = float(c.get("score", c.get("dense_score", 1.0)))
        c["bm25_score"] = 1.0
        return [c]
        
    query_tokens = tokenize_for_bm25(query_text)
    if not query_tokens:
        query_tokens = query_text.lower().split()
        
    # Build BM25 corpus from candidate texts
    corpus_tokens = [tokenize_for_bm25(c.get("text", "")) for c in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    raw_bm25_scores = bm25.get_scores(query_tokens)
    
    # Normalize BM25 scores to [0, 1]
    max_bm25 = float(np.max(raw_bm25_scores)) if len(raw_bm25_scores) > 0 else 0.0
    min_bm25 = float(np.min(raw_bm25_scores)) if len(raw_bm25_scores) > 0 else 0.0
    bm25_range = max_bm25 - min_bm25
    
    # Extract dense scores
    raw_dense_scores = [float(c.get("score", c.get("dense_score", 0.0))) for c in candidates]
    max_dense = max(raw_dense_scores) if raw_dense_scores else 1.0
    min_dense = min(raw_dense_scores) if raw_dense_scores else 0.0
    dense_range = max_dense - min_dense
    
    reranked = []
    for idx, cand in enumerate(candidates):
        # Normalized BM25
        if bm25_range > 1e-6:
            norm_bm25 = (raw_bm25_scores[idx] - min_bm25) / bm25_range
        else:
            norm_bm25 = 1.0 if max_bm25 > 0 else 0.0
            
        # Normalized Dense
        if dense_range > 1e-6:
            norm_dense = (raw_dense_scores[idx] - min_dense) / dense_range
        else:
            norm_dense = max(0.0, min(1.0, raw_dense_scores[idx]))
            
        # Linear hybrid combination
        final_score = (1.0 - bm25_weight) * norm_dense + bm25_weight * norm_bm25
        
        item = cand.copy()
        item["dense_score"] = float(raw_dense_scores[idx])
        item["bm25_score"] = float(raw_bm25_scores[idx])
        item["final_score"] = float(final_score)
        reranked.append(item)
        
    # Sort by final_score descending
    reranked = sorted(reranked, key=lambda x: x["final_score"], reverse=True)
    return reranked[:top_k]
