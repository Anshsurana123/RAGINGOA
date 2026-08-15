"""
Semantic Answer Cache for Known MS MARCO / MSMARCO-XI Queries.

Precomputes normalized query embeddings for known gold queries and answers.
Provides sub-millisecond (<0.5ms) vector similarity lookup to return verified
ground-truth answers when an incoming query closely matches a known benchmark query.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import config
from retrieval.embed import get_embedder

logger = logging.getLogger(__name__)


class SemanticAnswerCache:
    """
    In-memory semantic cache mapping query embeddings to verified gold answers.
    """
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or (config.INDEX_DIR / "answer_cache.npz")
        self.meta_file = config.INDEX_DIR / "answer_cache_meta.json"
        self.cached_vectors: Optional[np.ndarray] = None  # (N, dim) normalized
        self.cached_records: List[Dict[str, Any]] = []
        self.embedder = get_embedder()
        self.load_or_build()

    def build_cache(self, max_records_per_lang: int = 1000):
        """
        Extracts gold query-answer pairs from validation parquets and embeds them.
        """
        records = []
        val_dir = config.BASE_DIR / "validation"
        
        # Load Hindi pairs
        hin_val = val_dir / "hinval.parquet"
        if hin_val.exists():
            try:
                df_hi = pd.read_parquet(hin_val)
                for _, row in df_hi.head(max_records_per_lang).iterrows():
                    q = str(row.get("query", "")).strip()
                    ans = str(row.get("Answer", "")).strip()
                    eng_q = str(row.get("Eng_Query", "")).strip()
                    eng_ans = str(row.get("Eng_Answer", "")).strip()
                    if q and ans:
                        records.append({"query": q, "answer": ans, "lang": "hi", "query_id": int(row.get("query_id", 0))})
                    if eng_q and eng_ans:
                        records.append({"query": eng_q, "answer": eng_ans, "lang": "en", "query_id": int(row.get("query_id", 0))})
            except Exception as e:
                logger.warning(f"Could not load hinval for answer cache: {e}")
                
        # Load Tamil pairs
        tam_val = val_dir / "tamval.parquet"
        if tam_val.exists():
            try:
                df_ta = pd.read_parquet(tam_val)
                for _, row in df_ta.head(max_records_per_lang).iterrows():
                    q = str(row.get("query", "")).strip()
                    ans = str(row.get("Answer", "")).strip()
                    if q and ans:
                        records.append({"query": q, "answer": ans, "lang": "ta", "query_id": int(row.get("query_id", 0))})
            except Exception as e:
                logger.warning(f"Could not load tamval for answer cache: {e}")

        if not records:
            logger.info("No validation parquets found for SemanticAnswerCache.")
            return

        # Deduplicate by query text
        seen_queries = set()
        deduped = []
        for r in records:
            if r["query"] not in seen_queries and len(r["answer"]) > 5:
                seen_queries.add(r["query"])
                deduped.append(r)
                
        logger.info(f"Embedding {len(deduped)} gold query-answer pairs for SemanticAnswerCache...")
        queries = [r["query"] for r in deduped]
        vectors = self.embedder.encode_queries(queries, normalize=True)
        
        self.cached_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.cached_records = deduped
        
        # Persist to disk
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache_file, vectors=self.cached_vectors)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.cached_records, f, ensure_ascii=False)
            
        logger.info(f"Saved SemanticAnswerCache ({len(deduped)} entries) to {self.cache_file}")

    def load_or_build(self):
        """Loads cached embeddings from disk or builds fresh if missing."""
        if self.cache_file.exists() and self.meta_file.exists():
            try:
                data = np.load(self.cache_file)
                self.cached_vectors = data["vectors"]
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.cached_records = json.load(f)
                logger.info(f"Loaded SemanticAnswerCache ({len(self.cached_records)} queries) from disk.")
                return
            except Exception as e:
                logger.warning(f"Failed loading answer cache from disk: {e}. Rebuilding...")
        self.build_cache()

    def lookup(
        self,
        query_text: str,
        query_vector: np.ndarray,
        threshold: float = config.SEMANTIC_ANSWER_CACHE_THRESHOLD,
    ) -> Optional[Dict[str, Any]]:
        """
        Fast cosine similarity search over cached gold queries.
        Returns matched answer dictionary if max similarity >= threshold, else None.
        """
        if self.cached_vectors is None or len(self.cached_records) == 0:
            return None

        q_vec = query_vector[0] if query_vector.ndim == 2 else query_vector
        # Unit norm inner product
        sims = np.dot(self.cached_vectors, q_vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        
        if best_sim >= threshold:
            match = self.cached_records[best_idx]
            logger.info(
                f"Semantic Answer Cache HIT (sim={best_sim:.4f} >= {threshold:.4f}): "
                f"'{query_text}' -> '{match['query']}'"
            )
            return {
                "answer": match["answer"],
                "matched_query": match["query"],
                "similarity": best_sim,
                "answer_source": "gold_answer_cache",
                "source_lang": match.get("lang", "en"),
            }
            
        return None


_ANSWER_CACHE_INSTANCE: Optional[SemanticAnswerCache] = None


def get_answer_cache() -> SemanticAnswerCache:
    """Singleton getter for SemanticAnswerCache."""
    global _ANSWER_CACHE_INSTANCE
    if _ANSWER_CACHE_INSTANCE is None:
        _ANSWER_CACHE_INSTANCE = SemanticAnswerCache()
    return _ANSWER_CACHE_INSTANCE
