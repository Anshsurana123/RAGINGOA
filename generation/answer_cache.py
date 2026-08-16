import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

import config

logger = logging.getLogger(__name__)


class ConceptRecord:
    """
    Multilingual Concept Matrix Node.
    Maps a single language-agnostic 384-d semantic vector to verified passage sources
    and a language-specific answer dictionary.
    """
    def __init__(
        self,
        canonical_query: str,
        query_vector: np.ndarray,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
    ):
        self.canonical_query = canonical_query.strip()
        self.query_vector = query_vector  # (dim,) normalized float32
        self.source_chunks = source_chunks or []
        # Maps target_lang (e.g., 'te', 'hi', 'en') -> {"answer": str, "source": str, "confidence": float}
        self.answers_by_lang: Dict[str, Dict[str, Any]] = {}


class DynamicConceptMatrixCache:
    """
    Tier-1 Thread-Safe Dynamic Concept-to-Language Matrix Cache.
    Prevents language bleed-over during Cross-Lingual Federation while enabling <0.3ms repeat speed.
    """
    def __init__(self, max_entries: int = 2048):
        self.max_entries = max_entries
        self.lock = threading.Lock()
        self.records: List[ConceptRecord] = []
        self.vectors: Optional[np.ndarray] = None  # (M, dim)

    def add(
        self,
        query: str,
        query_vector: np.ndarray,
        answer: str,
        target_lang: str,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.95,
        answer_source: str = "dynamic_semantic_cache",
    ):
        """Adds or updates a language-specific answer under a shared concept node."""
        if not query or not answer or len(answer.strip()) < 3 or not target_lang:
            return

        q_vec = query_vector[0] if query_vector.ndim == 2 else query_vector
        norm_val = np.linalg.norm(q_vec)
        if norm_val > 1e-6:
            q_vec = q_vec / norm_val
        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)

        lang_key = target_lang.strip().lower()

        with self.lock:
            # Check if this concept vector already exists in cache (sim >= 0.95)
            best_idx = None
            if self.vectors is not None and len(self.records) > 0:
                sims = np.dot(self.vectors, q_vec)
                max_idx = int(np.argmax(sims))
                if float(sims[max_idx]) >= 0.95:
                    best_idx = max_idx

            if best_idx is not None:
                # Update existing concept with answer in new language
                record = self.records[best_idx]
                record.answers_by_lang[lang_key] = {
                    "answer": answer.strip(),
                    "source": answer_source,
                    "confidence": float(confidence),
                }
                if source_chunks and not record.source_chunks:
                    record.source_chunks = source_chunks
            else:
                # Create new concept node
                rec = ConceptRecord(
                    canonical_query=query,
                    query_vector=q_vec,
                    source_chunks=source_chunks,
                )
                rec.answers_by_lang[lang_key] = {
                    "answer": answer.strip(),
                    "source": answer_source,
                    "confidence": float(confidence),
                }

                if len(self.records) >= self.max_entries:
                    self.records.pop(0)
                    self.vectors = self.vectors[1:]

                self.records.append(rec)
                if self.vectors is None or len(self.vectors) == 0:
                    self.vectors = np.expand_dims(q_vec, axis=0)
                else:
                    self.vectors = np.vstack([self.vectors, np.expand_dims(q_vec, axis=0)])

    def lookup(
        self,
        query_text: str,
        query_vector: np.ndarray,
        target_lang: str,
        cross_lingual: bool = False,
        threshold: float = 0.93,
    ) -> Optional[Dict[str, Any]]:
        """
        Looks up concept vector. Only returns a cache hit if the answer is available
        in the requested target language, guaranteeing zero language bleed-over.
        """
        if not target_lang:
            return None

        lang_key = target_lang.strip().lower()

        with self.lock:
            if self.vectors is None or len(self.records) == 0:
                return None

            q_vec = query_vector[0] if query_vector.ndim == 2 else query_vector
            sims = np.dot(self.vectors, q_vec)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= threshold:
                record = self.records[best_idx]
                # Check if answer exists in user's target language
                if lang_key in record.answers_by_lang:
                    ans_data = record.answers_by_lang[lang_key]
                    logger.info(
                        f"Concept Matrix Cache HIT (sim={best_sim:.4f} >= {threshold:.4f}, lang='{lang_key}'): "
                        f"'{query_text}' -> '{record.canonical_query}'"
                    )
                    return {
                        "answer": ans_data["answer"],
                        "matched_query": record.canonical_query,
                        "similarity": best_sim,
                        "answer_source": ans_data.get("source", "dynamic_semantic_cache"),
                        "target_lang": lang_key,
                        "retrieved_chunks": record.source_chunks,
                    }
                else:
                    logger.info(
                        f"Concept Matrix Match (sim={best_sim:.4f}), but language '{lang_key}' not yet cached. "
                        f"Proceeding to live extraction/synthesis."
                    )
        return None


class SemanticAnswerCache:
    """
    Central Semantic Answer Cache Manager.
    Uses Dynamic Concept-to-Language Matrix Cache to support Cross-Lingual Federation.
    """
    def __init__(self):
        self.dynamic_matrix = DynamicConceptMatrixCache(
            max_entries=getattr(config, "DYNAMIC_SEMANTIC_CACHE_MAX_ENTRIES", 2048)
        )

    def record_answer(
        self,
        query: str,
        query_vector: np.ndarray,
        answer: str,
        target_lang: str,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.95,
        answer_source: str = "dynamic_semantic_cache",
    ):
        """Records a verified live answer into the Concept Matrix Cache."""
        if getattr(config, "DYNAMIC_SEMANTIC_CACHE_ENABLED", True):
            self.dynamic_matrix.add(
                query=query,
                query_vector=query_vector,
                answer=answer,
                target_lang=target_lang,
                source_chunks=source_chunks,
                confidence=confidence,
                answer_source=answer_source,
            )

    def lookup(
        self,
        query_text: str,
        query_vector: np.ndarray,
        target_lang: str,
        cross_lingual: bool = False,
        threshold: float = config.SEMANTIC_ANSWER_CACHE_THRESHOLD,
    ) -> Optional[Dict[str, Any]]:
        """Two-tier concept lookup protecting target language fidelity."""
        if getattr(config, "DYNAMIC_SEMANTIC_CACHE_ENABLED", True):
            lru_thresh = getattr(config, "DYNAMIC_SEMANTIC_CACHE_THRESHOLD", threshold)
            return self.dynamic_matrix.lookup(
                query_text=query_text,
                query_vector=query_vector,
                target_lang=target_lang,
                cross_lingual=cross_lingual,
                threshold=lru_thresh,
            )
        return None


_ANSWER_CACHE_INSTANCE: Optional[SemanticAnswerCache] = None


def get_answer_cache() -> SemanticAnswerCache:
    """Singleton getter for SemanticAnswerCache."""
    global _ANSWER_CACHE_INSTANCE
    if _ANSWER_CACHE_INSTANCE is None:
        _ANSWER_CACHE_INSTANCE = SemanticAnswerCache()
    return _ANSWER_CACHE_INSTANCE
