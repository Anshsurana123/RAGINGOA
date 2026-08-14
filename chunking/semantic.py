"""
Semantic Chunking Strategy.

For long documents:
1. Splits text into sentences.
2. Computes consecutive sentence embedding cosine distances using the embedding model.
3. Splits at distance spikes (topic transitions) rather than arbitrary token counts.
4. Applies 10-20% token overlap across semantic chunk boundaries.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import config
from chunking.metadata import (
    Chunk,
    split_sentences_multilingual,
    calculate_overlap_tokens,
    estimate_token_count,
)


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine distance between two 1D vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return float(max(0.0, min(2.0, 1.0 - similarity)))


def chunk_document_semantic(
    doc_dict: Dict[str, Any],
    embedder_func=None,
    distance_threshold: float = config.SEMANTIC_SIMILARITY_THRESHOLD,
) -> List[Chunk]:
    """
    Split a document based on embedding cosine distance spikes across consecutive sentences.
    """
    full_text = doc_dict.get("text", "")
    doc_id = doc_dict.get("doc_id", "doc_unknown")
    source_lang = doc_dict.get("source_lang", "en")
    title = doc_dict.get("title", "")
    
    sentences = split_sentences_multilingual(full_text)
    if not sentences:
        return []
    if len(sentences) == 1:
        text = sentences[0]
        return [
            Chunk(
                chunk_id=f"{doc_id}_sem_0000",
                text=text,
                embed_text=text,
                chunk_strategy="semantic",
                source_lang=source_lang,
                token_count=estimate_token_count(text),
                doc_id=doc_id,
                metadata={"title": title, "cluster_id": 0},
            )
        ]
        
    # If embedder_func is provided, compute true sentence embeddings;
    # otherwise fallback to token-based lexical similarity
    sentence_vectors = None
    if embedder_func is not None:
        try:
            # embedder_func accepts a list of texts and returns numpy array (N, D)
            sentence_vectors = embedder_func(sentences)
        except Exception:
            sentence_vectors = None
            
    # Identify split boundaries
    split_indices = [0]
    
    if sentence_vectors is not None and len(sentence_vectors) == len(sentences):
        distances = []
        for i in range(len(sentences) - 1):
            d = cosine_distance(sentence_vectors[i], sentence_vectors[i + 1])
            distances.append(d)
            
        # Calculate dynamic threshold if distances exist
        if distances:
            mean_d = float(np.mean(distances))
            std_d = float(np.std(distances))
            dynamic_thresh = max(distance_threshold, mean_d + 0.5 * std_d)
            
            for i, d in enumerate(distances):
                if d >= dynamic_thresh:
                    split_indices.append(i + 1)
    else:
        # Fallback heuristic: paragraph or length-based boundary detection
        cur_len = 0
        for i, s in enumerate(sentences):
            cur_len += len(s.split())
            if cur_len >= 80 and i > 0:
                split_indices.append(i)
                cur_len = 0
                
    if split_indices[-1] != len(sentences):
        split_indices.append(len(sentences))
        
    # Group sentences into semantic chunks and attach token overlap
    chunks: List[Chunk] = []
    prev_tail_overlap = ""
    
    for idx in range(len(split_indices) - 1):
        start_i = split_indices[idx]
        end_i = split_indices[idx + 1]
        
        group_sentences = sentences[start_i:end_i]
        raw_group_text = " ".join(group_sentences)
        
        if prev_tail_overlap:
            chunk_text = f"{prev_tail_overlap} {raw_group_text}"
        else:
            chunk_text = raw_group_text
            
        chunk_id = f"{doc_id}_sem_{idx:04d}"
        
        chunk = Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            embed_text=raw_group_text,
            chunk_strategy="semantic",
            source_lang=source_lang,
            token_count=estimate_token_count(chunk_text),
            doc_id=doc_id,
            metadata={
                "cluster_index": idx,
                "title": title,
                "sentence_range": [start_i, end_i],
            },
        )
        chunks.append(chunk)
        
        prev_tail_overlap = calculate_overlap_tokens(
            raw_group_text, overlap_percent=config.CHUNK_OVERLAP_PERCENT
        )
        
    return chunks


def process_longdocs_semantic(
    longdocs: List[Dict[str, Any]], embedder_func=None
) -> List[Chunk]:
    """
    Process long documents using semantic boundary splitting.
    """
    all_chunks = []
    for doc in longdocs:
        all_chunks.extend(chunk_document_semantic(doc, embedder_func=embedder_func))
    return all_chunks
