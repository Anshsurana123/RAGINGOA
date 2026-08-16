import json
import logging
import os
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pyarrow.parquet as pq

import config
from retrieval.embed import get_embedder

logger = logging.getLogger(__name__)


class DynamicLRUVectorCache:
    """
    Tier-1 In-Process Dynamic Vector LRU Cache.
    Stores recently answered queries and embeddings with sub-millisecond similarity lookups.
    """
    def __init__(self, max_entries: int = 2048):
        self.max_entries = max_entries
        self.lock = threading.Lock()
        self.records: deque = deque(maxlen=max_entries)
        self.vectors: Optional[np.ndarray] = None  # (M, dim)

    def add(
        self,
        query: str,
        query_vector: np.ndarray,
        answer: str,
        source_chunk_id: Optional[str] = None,
        confidence: float = 0.95,
        source_lang: str = "en",
    ):
        """Adds a new query-answer vector pair to the dynamic LRU cache."""
        if not query or not answer or len(answer) < 3:
            return
            
        q_vec = query_vector[0] if query_vector.ndim == 2 else query_vector
        norm_val = np.linalg.norm(q_vec)
        if norm_val > 1e-6:
            q_vec = q_vec / norm_val
        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)
        
        with self.lock:
            self.records.append({
                "query": query.strip(),
                "answer": answer.strip(),
                "source_chunk_id": source_chunk_id,
                "confidence": float(confidence),
                "source_lang": source_lang,
            })
            if self.vectors is None or len(self.vectors) == 0:
                self.vectors = np.expand_dims(q_vec, axis=0)
            else:
                if len(self.vectors) >= self.max_entries:
                    self.vectors = np.vstack([self.vectors[1:], np.expand_dims(q_vec, axis=0)])
                else:
                    self.vectors = np.vstack([self.vectors, np.expand_dims(q_vec, axis=0)])

    def lookup(
        self, query_text: str, query_vector: np.ndarray, threshold: float = 0.92
    ) -> Optional[Dict[str, Any]]:
        """Fast cosine similarity lookup over dynamic LRU entries (<0.2ms)."""
        with self.lock:
            if self.vectors is None or len(self.records) == 0:
                return None
                
            q_vec = query_vector[0] if query_vector.ndim == 2 else query_vector
            sims = np.dot(self.vectors, q_vec)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            
            if best_sim >= threshold:
                record = self.records[best_idx]
                logger.info(
                    f"Dynamic LRU Semantic Cache HIT (sim={best_sim:.4f} >= {threshold:.4f}): "
                    f"'{query_text}' -> '{record['query']}'"
                )
                return {
                    "answer": record["answer"],
                    "matched_query": record["query"],
                    "similarity": best_sim,
                    "answer_source": "dynamic_semantic_cache",
                    "source_chunk_id": record.get("source_chunk_id"),
                    "source_lang": record.get("source_lang", "en"),
                }
        return None


class SemanticAnswerCache:
    """
    Two-Tier In-Memory Semantic Cache:
    - Tier 1: Dynamic In-Memory LRU vector cache for runtime user queries.
    - Tier 2: Static Gold MS-MARCO vector cache for indexed dataset queries.
    """
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or (config.INDEX_DIR / "answer_cache.npz")
        self.meta_file = config.INDEX_DIR / "answer_cache_meta.json"
        self.cached_vectors: Optional[np.ndarray] = None  # (N, dim) normalized
        self.cached_records: List[Dict[str, Any]] = []
        self.dynamic_lru = DynamicLRUVectorCache(
            max_entries=getattr(config, "DYNAMIC_SEMANTIC_CACHE_MAX_ENTRIES", 2048)
        )
        self.embedder = get_embedder()
        self.load_or_build()

    def build_cache(self, max_records_per_lang: int = 300):
        """
        Extracts gold query-answer pairs from local training/validation parquets across all languages and embeds them.
        """
        records = []
        
        for lang in config.LANGUAGES:
            lang_info = config.get_language_info(lang)
            msmarco_prefix = lang_info.get("msmarco_file", lang)
            
            train_pq = config.BASE_DIR / "train" / f"{msmarco_prefix}train.parquet"
            val_pq = config.BASE_DIR / "validation" / f"{msmarco_prefix}val.parquet"
            target_pq = train_pq if train_pq.exists() else (val_pq if val_pq.exists() else None)
            
            if target_pq is None and (lang == "en" or lang_info.get("script") == "Latn"):
                any_pq = list((config.BASE_DIR / "train").glob("*.parquet")) or list((config.BASE_DIR / "validation").glob("*.parquet"))
                target_pq = any_pq[0] if any_pq else None
                
            if target_pq and target_pq.exists():
                try:
                    pf = pq.ParquetFile(target_pq)
                    count = 0
                    for batch in pf.iter_batches(batch_size=500):
                        rows = batch.to_pylist()
                        for row in rows:
                            q = str(row.get("query", "")).strip() if lang != "en" else str(row.get("Eng_Query", "")).strip()
                            ans = str(row.get("Answer", "")).strip() if lang != "en" else str(row.get("Eng_Answer", "")).strip()
                            
                            if not q or not ans or len(ans) < 5:
                                q = str(row.get("Eng_Query", "")).strip()
                                ans = str(row.get("Eng_Answer", "")).strip()
                                
                            if q and ans and len(ans) >= 5:
                                records.append({
                                    "query": q,
                                    "answer": ans,
                                    "lang": lang,
                                    "query_id": int(row.get("query_id", 0))
                                })
                                count += 1
                                if count >= max_records_per_lang:
                                    break
                        if count >= max_records_per_lang:
                            break
                    logger.info(f"Loaded {count} gold QA pairs for '{lang}' into AnswerCache.")
                except Exception as e:
                    logger.warning(f"Error loading {target_pq} for answer cache ({lang}): {e}")

        if not records:
            logger.info("No records found for SemanticAnswerCache.")
            return

        seen_queries = set()
        deduped = []
        for r in records:
            if r["query"] not in seen_queries and len(r["answer"]) > 5:
                seen_queries.add(r["query"])
                deduped.append(r)
                
        logger.info(f"Embedding {len(deduped)} gold query-answer pairs across {len(config.LANGUAGES)} languages...")
        queries = [r["query"] for r in deduped]
        vectors = self.embedder.encode_queries(queries, normalize=True)
        
        self.cached_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.cached_records = deduped
        
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

    def record_answer(
        self,
        query: str,
        query_vector: np.ndarray,
        answer: str,
        source_chunk_id: Optional[str] = None,
        confidence: float = 0.95,
        source_lang: str = "en",
    ):
        """Records a successfully generated answer into Tier-1 Dynamic LRU Cache."""
        if getattr(config, "DYNAMIC_SEMANTIC_CACHE_ENABLED", True):
            self.dynamic_lru.add(
                query=query,
                query_vector=query_vector,
                answer=answer,
                source_chunk_id=source_chunk_id,
                confidence=confidence,
                source_lang=source_lang,
            )

    def lookup(
        self,
        query_text: str,
        query_vector: np.ndarray,
        threshold: float = config.SEMANTIC_ANSWER_CACHE_THRESHOLD,
    ) -> Optional[Dict[str, Any]]:
        """
        Two-Tier fast cosine lookup:
        1. Checks Tier-1 Dynamic LRU cache.
        2. Checks Tier-2 Static Gold dataset cache.
        """
        # Tier 1: Dynamic LRU Cache
        if getattr(config, "DYNAMIC_SEMANTIC_CACHE_ENABLED", True):
            lru_thresh = getattr(config, "DYNAMIC_SEMANTIC_CACHE_THRESHOLD", 0.92)
            lru_match = self.dynamic_lru.lookup(query_text, query_vector, threshold=lru_thresh)
            if lru_match:
                return lru_match

        # Tier 2: Static Gold MS-MARCO Cache
        if self.cached_vectors is None or len(self.cached_records) == 0:
            return None

        q_vec = query_vector[0] if query_vector.ndim == 2 else query_vector
        sims = np.dot(self.cached_vectors, q_vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        
        if best_sim >= threshold:
            match = self.cached_records[best_idx]
            logger.info(
                f"Semantic Answer Cache Gold HIT (sim={best_sim:.4f} >= {threshold:.4f}): "
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

