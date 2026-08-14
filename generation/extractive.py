"""
Extractive-First Generation Module.

Directly extracts grounded factual answers from the highest-ranked retrieved passages.
Avoids LLM API calls and network latency for factoid question-answering.
"""

from typing import Any, Dict, List, Optional
import re
from chunking.metadata import split_sentences_multilingual


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
    query: str, retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Primary extractive answer generation.
    Returns:
        {
            "answer": str,
            "answer_source": "extractive",
            "source_chunk_id": str,
            "confidence": float
        }
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant information found in the indexed corpus.",
            "answer_source": "declined",
            "source_chunk_id": None,
            "confidence": 0.0,
        }
        
    top_chunk = retrieved_chunks[0]
    extracted_text = extract_answer_from_passage(query, top_chunk)
    confidence = float(top_chunk.get("final_score", top_chunk.get("score", 0.9)))
    
    return {
        "answer": extracted_text,
        "answer_source": "extractive",
        "source_chunk_id": top_chunk.get("chunk_id"),
        "confidence": confidence,
    }
