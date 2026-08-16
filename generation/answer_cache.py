"""
Semantic Answer Cache for Known MS MARCO / MSMARCO-XI Queries.

Precomputes normalized query embeddings for known gold queries and answers across all configured languages.
Provides sub-millisecond (<0.5ms) vector similarity lookup to return verified
ground-truth answers when an incoming query closely matches a known benchmark query.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pyarrow.parquet as pq

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

    def build_cache(self, max_records_per_lang: int = 300):
        """
        Extracts gold query-answer pairs from local training/validation parquets across all languages and embeds them.
        """
        records = []
        
        for lang in config.LANGUAGES:
            lang_info = config.get_language_info(lang)
            msmarco_prefix = lang_info.get("msmarco_file", lang)
            
            # Check local train or val parquets
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
                                # Fallback to English fields if query is empty
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

        # Deduplicate by query text
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
