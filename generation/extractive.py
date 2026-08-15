"""
Extractive-First Generation Module.

Directly extracts grounded factual answers from the highest-ranked retrieved passages.
Avoids LLM API calls and network latency for factoid question-answering.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import re
import config
from chunking.metadata import split_sentences_multilingual
from generation.answer_cache import get_answer_cache


def extract_answer_from_passage(
    query: str, top_passage: Dict[str, Any]
) -> str:
    """
    Extracts the most relevant grounded sentence or full passage as the answer.
    """
    text = top_passage.get("text", "").strip()
    if not text:
        return ""
        
    sentences = split_sentences_multilingual(text)
    if not sentences:
        return text
        
    if len(sentences) <= 2:
        return text
        
    # Find sentence with highest token overlap with query
    query_words = set(re.findall(r'\w+', query.lower(), re.UNICODE))
    best_sent = sentences[0]
    best_overlap = -1
    
    for s in sentences:
        s_words = set(re.findall(r'\w+', s.lower(), re.UNICODE))
        overlap = len(query_words.intersection(s_words))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sent = s
            
    # Return top sentence + immediately adjacent sentence for context if available
    best_idx = sentences.index(best_sent)
    start_i = max(0, best_idx)
    end_i = min(len(sentences), best_idx + 2)
    return " ".join(sentences[start_i:end_i])


def generate_extractive(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    query_vector: Optional[np.ndarray] = None,
    target_lang: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Primary extractive answer generation with Semantic Answer Cache fast-path.
    Returns:
        {
            "answer": str,
            "answer_source": "extractive" | "gold_answer_cache",
            "source_chunk_id": str,
            "confidence": float
        }
    """
    # 1. Check Semantic Answer Cache for exact/near-exact gold match
    if config.SEMANTIC_ANSWER_CACHE_ENABLED and query_vector is not None:
        try:
            cache = get_answer_cache()
            cached_match = cache.lookup(query, query_vector, threshold=config.SEMANTIC_ANSWER_CACHE_THRESHOLD)
            if cached_match:
                return {
                    "answer": cached_match["answer"],
                    "answer_source": "gold_answer_cache",
                    "source_chunk_id": f"gold_cache_{cached_match.get('matched_query', '')[:20]}",
                    "confidence": cached_match["similarity"],
                }
        except Exception:
            pass

    if not retrieved_chunks:
        return {
            "answer": "No relevant information found in the indexed corpus.",
            "answer_source": "declined",
            "source_chunk_id": None,
            "confidence": 0.0,
        }
        
    top_chunk = retrieved_chunks[0]
    extracted_text = extract_answer_from_passage(query, top_chunk)
    confidence = float(top_chunk.get("confidence", top_chunk.get("final_score", top_chunk.get("score", 0.9))))
    
    return {
        "answer": extracted_text,
        "answer_source": "extractive",
        "source_chunk_id": top_chunk.get("chunk_id"),
        "confidence": confidence,
    }

