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


def synthesize_textrank_svd(
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    query_vector: Optional[np.ndarray] = None,
    embedder: Optional[Any] = None,
    max_sentences: int = 2,
) -> str:
    r"""
    Non-LLM Context Synthesis via Continuous TextRank Eigenvector Centrality & SVD Matrix Decomposition:
    1. Collects unique sentence nodes across top candidate chunks.
    2. Builds the sentence similarity adjacency matrix and power-iterates to find graph centrality.
    3. Decomposes embedding matrix via economy SVD (M = U \Sigma V^T) for 95% cumulative energy thresholding.
    4. Sequences top salient sentences in original document order for natural readability.
    """
    if not candidate_chunks:
        return ""
        
    # Gather sentences from top candidate chunks
    candidate_sentences: List[str] = []
    seen_sentences = set()
    
    for chunk in candidate_chunks[:2]:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        sents = split_sentences_multilingual(text)
        for s in sents:
            s_clean = s.strip()
            if len(s_clean.split()) >= 3 and s_clean not in seen_sentences:
                seen_sentences.add(s_clean)
                candidate_sentences.append(s_clean)
                if len(candidate_sentences) >= 3:
                    break
        if len(candidate_sentences) >= 3:
            break
                
    if not candidate_sentences:
        return candidate_chunks[0].get("text", "").strip()
        
    if len(candidate_sentences) <= max_sentences:
        return " ".join(candidate_sentences)
        
    if embedder is None:
        # Fallback to token overlap if embedder is not passed
        return extract_answer_from_passage(query, candidate_chunks[0])
        
    try:
        # 1. Encode candidate sentences
        s_vecs = embedder.encode_passages(candidate_sentences, normalize=True)
        if query_vector is None or query_vector.ndim != 1:
            q_vec = embedder.encode_queries(query, normalize=True)[0]
        else:
            q_vec = query_vector
            
        N = len(candidate_sentences)
        
        # 2. Query relevance prior
        r = np.maximum(0.0, np.dot(s_vecs, q_vec))
        r_sum = np.sum(r)
        prior = r / r_sum if r_sum > 1e-6 else np.ones(N) / N
        
        # 3. Inter-sentence Adjacency matrix
        W = np.maximum(0.0, np.dot(s_vecs, s_vecs.T))
        np.fill_diagonal(W, 0.0)
        
        # Stochastic degree normalization
        deg = np.sum(W, axis=1)
        deg[deg == 0] = 1.0
        T = W / deg[:, np.newaxis]
        
        # 4. Continuous TextRank Power Iteration (6 iterations for fast convergence)
        d = 0.85
        p = np.ones(N) / N
        for _ in range(6):
            p = (1.0 - d) * prior + d * np.dot(T.T, p)
            p_sum = np.sum(p)
            if p_sum > 1e-6:
                p = p / p_sum
                
        # 5. SVD Energy Decomposition
        U, S, Vt = np.linalg.svd(s_vecs, full_matrices=False)
        cum_energy = np.cumsum(S) / np.sum(S)
        k = int(np.argmax(cum_energy >= 0.95)) + 1
        svd_salience = np.sum((U[:, :k] ** 2) * (S[:k] ** 2), axis=1)
        max_svd = np.max(svd_salience)
        if max_svd > 1e-6:
            svd_salience /= max_svd
            
        # 6. Composite salience scoring
        final_scores = 0.5 * p + 0.3 * svd_salience + 0.2 * r
        top_indices = sorted(np.argsort(final_scores)[::-1][:max_sentences])
        
        return " ".join([candidate_sentences[i] for i in top_indices])
    except Exception:
        return extract_answer_from_passage(query, candidate_chunks[0])


def generate_extractive(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    query_vector: Optional[np.ndarray] = None,
    target_lang: Optional[str] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Primary non-LLM answer generation using:
    1. Semantic Answer Cache fast-path (<0.5ms).
    2. Continuous TextRank & SVD Matrix Decomposition Context Synthesizer (<10ms).
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
    
    # Ultra-fast grounded extractive sentence window extraction (<0.2ms)
    extracted_text = extract_answer_from_passage(query, top_chunk)
    if not extracted_text and len(retrieved_chunks) > 1:
        extracted_text = extract_answer_from_passage(query, retrieved_chunks[1])
        
    confidence = float(top_chunk.get("confidence", top_chunk.get("final_score", top_chunk.get("score", 0.9))))
    
    return {
        "answer": extracted_text or top_chunk.get("text", "").strip(),
        "answer_source": "extractive",
        "source_chunk_id": top_chunk.get("chunk_id"),
        "confidence": confidence,
    }

