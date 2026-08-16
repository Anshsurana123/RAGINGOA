"""
In-Memory FAISS HNSW Vector Indexing and Search.

Features:
- In-Memory HNSW (IndexHNSWFlat) with cosine similarity (METRIC_INNER_PRODUCT on normalized vectors).
- M=32, efConstruction=200, efSearch=64.
- Single combined index per strategy spanning all configured languages (config.LANGUAGES).
- Metadata-based language pre-filtering.
- Corpus centroid computation and persistence for pre-retrieval off-topic guardrails.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from chunking.metadata import Chunk, filter_chunks_by_language
from chunking.passage_native import process_corpus_passage_native
from chunking.sentence_window import process_longdocs_sentence_window
from chunking.semantic import process_longdocs_semantic
from retrieval.embed import get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)


class StrategyVectorIndex:
    """
    Manages an in-memory FAISS HNSW index and aligned metadata for a specific chunking strategy.
    """
    def __init__(
        self,
        strategy_name: str,
        dim: int = config.EMBEDDING_DIM,
        m: int = config.HNSW_M,
        ef_construction: int = config.HNSW_EF_CONSTRUCTION,
        ef_search: int = config.HNSW_EF_SEARCH,
    ):
        self.strategy_name = strategy_name
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # FAISS HNSW with inner product (cosine similarity on unit-normalized vectors)
        self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        self.chunks: List[Chunk] = []
        # Precomputed index mappings for ultra-fast language filtering
        self.lang_to_indices: Dict[str, List[int]] = {}

    @property
    def size(self) -> int:
        """Returns the number of indexed chunks."""
        return len(self.chunks)

    def add_chunks(self, chunks: List[Chunk], vectors: np.ndarray):
        """Add chunks and precomputed vectors to index."""
        if len(chunks) == 0:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunk count ({len(chunks)}) != vector count ({len(vectors)})")
            
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        
        # Build language index mapping
        for idx_offset, c in enumerate(chunks):
            global_idx = start_idx + idx_offset
            lang = c.source_lang.lower()
            if lang not in self.lang_to_indices:
                self.lang_to_indices[lang] = []
            self.lang_to_indices[lang].append(global_idx)
            
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        logger.info(
            f"Added {len(chunks)} chunks to '{self.strategy_name}' index. "
            f"Total index size: {self.index.ntotal}."
        )

    def search(
        self, query_vec: np.ndarray, target_lang: Optional[str] = None, top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Search the index for query vector with optional language filtering.
        """
        if self.index.ntotal == 0:
            return []
            
        # Ensure query_vec is 2D (1, dim)
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)
            
        # Query FAISS HNSW (optimized candidate search for sub-1ms traversal)
        search_k = min(self.index.ntotal, max(400, top_k * 25) if target_lang else max(60, top_k * 3))
        scores, indices = self.index.search(query_vec, search_k)

        
        results: List[Dict[str, Any]] = []
        target_lang_clean = target_lang.lower().strip() if target_lang else None
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            
            # Apply language metadata pre-filter
            if target_lang_clean and chunk.source_lang.lower() != target_lang_clean:
                continue
                
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": float(score),
                "source_lang": chunk.source_lang,
                "chunk_strategy": chunk.chunk_strategy,
                "source_query_ids": chunk.source_query_ids,
                "doc_id": chunk.doc_id,
                "context_window": chunk.context_window,
                "metadata": chunk.metadata,
            })
            
            if len(results) >= top_k:
                break
                
        return results

    def save(self, directory: Path):
        """Save FAISS index and chunk metadata to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        index_file = directory / f"{self.strategy_name}.faiss"
        meta_file = directory / f"{self.strategy_name}_meta.json"
        
        faiss.write_index(self.index, str(index_file))
        
        # Serialize chunk metadata
        meta_data = {
            "strategy_name": self.strategy_name,
            "dim": self.dim,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "chunks": [c.model_dump() for c in self.chunks],
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False)
        logger.info(f"Saved index and metadata for '{self.strategy_name}' to {directory}")

    @classmethod
    def load(cls, directory: Path, strategy_name: str) -> "StrategyVectorIndex":
        """Load FAISS index and chunk metadata from disk."""
        index_file = directory / f"{strategy_name}.faiss"
        meta_file = directory / f"{strategy_name}_meta.json"
        
        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Index or metadata file missing in {directory} for '{strategy_name}'")
            
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            
        inst = cls(
            strategy_name=meta_data["strategy_name"],
            dim=meta_data.get("dim", config.EMBEDDING_DIM),
            m=meta_data.get("m", config.HNSW_M),
            ef_construction=meta_data.get("ef_construction", config.HNSW_EF_CONSTRUCTION),
            ef_search=meta_data.get("ef_search", config.HNSW_EF_SEARCH),
        )
        inst.index = faiss.read_index(str(index_file))
        inst.index.hnsw.efSearch = inst.ef_search
        
        # Reconstruct chunks and language mapping
        inst.chunks = [Chunk(**c) for c in meta_data["chunks"]]
        inst.lang_to_indices = {}
        for idx, c in enumerate(inst.chunks):
            lang = c.source_lang.lower()
            if lang not in inst.lang_to_indices:
                inst.lang_to_indices[lang] = []
            inst.lang_to_indices[lang].append(idx)
            
        logger.info(f"Loaded index '{strategy_name}' ({inst.index.ntotal} vectors) from {directory}")
        return inst


class IndexManager:
    """
    Manages all strategy indexes and off-topic centroid models for the pipeline.
    """
    def __init__(self, index_dir: Path = config.INDEX_DIR):
        self.index_dir = index_dir
        self.indexes: Dict[str, StrategyVectorIndex] = {}
        self.centroids: Dict[str, np.ndarray] = {}  # lang -> centroid vector
        self.global_centroid: Optional[np.ndarray] = None
        self.embedder = get_embedder()

    @staticmethod
    def _iter_jsonl(path: Path, limit: Optional[int] = None):
        """Yield JSONL records without loading an entire corpus into memory."""
        yielded = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if limit is not None and yielded >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL record %s:%s: %s", path, line_number, exc)
                    continue
                yielded += 1
                yield record

    def _add_embedded_chunks(
        self,
        index: StrategyVectorIndex,
        chunks: List[Chunk],
        centroid_sums: Dict[str, np.ndarray],
        centroid_counts: Dict[str, int],
    ) -> int:
        """Embed and add one bounded batch, updating centroid statistics online."""
        if not chunks:
            return 0
        texts = [chunk.embed_text for chunk in chunks]
        vectors = self.embedder.encode_passages(
            texts,
            batch_size=min(config.INDEX_BUILD_BATCH_SIZE, 128),
        )
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        index.add_chunks(chunks, vectors)
        for chunk, vector in zip(chunks, vectors):
            lang = chunk.source_lang.lower()
            if lang not in centroid_sums:
                centroid_sums[lang] = np.zeros(index.dim, dtype=np.float64)
                centroid_counts[lang] = 0
            centroid_sums[lang] += vector.astype(np.float64, copy=False)
            centroid_counts[lang] += 1
        return len(chunks)

    def build_all_indexes(self, max_passages_per_lang: Optional[int] = None):
        """
        Build complete combined indexes for all configured languages.

        ``None`` means no per-language cap. Records are streamed from JSONL and
        embedded in bounded batches, so the full corpus is indexed without the
        previous implicit 700-record truncation or an all-corpus memory spike.
        """
        if max_passages_per_lang is not None and max_passages_per_lang <= 0:
            raise ValueError("max_passages_per_lang must be positive or None")
        batch_size = max(1, int(getattr(config, "INDEX_BUILD_BATCH_SIZE", 512)))
        logger.info(
            "Building complete indexes for languages=%s (passage_limit=%s, batch_size=%s)",
            config.LANGUAGES,
            max_passages_per_lang if max_passages_per_lang is not None else "unlimited",
            batch_size,
        )

        self.indexes = {}
        self.centroids = {}
        self.global_centroid = None
        centroid_sums: Dict[str, np.ndarray] = {}
        centroid_counts: Dict[str, int] = {}
        passage_counts: Dict[str, int] = {}
        longdoc_counts: Dict[str, int] = {}

        # Passage-native index: stream each language corpus and add bounded batches.
        passage_index = StrategyVectorIndex("passage_native")
        passage_total = 0
        for lang in config.LANGUAGES:
            corpus_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
            if not corpus_file.exists():
                logger.warning("Corpus file %s not found; skipping language '%s'.", corpus_file, lang)
                continue
            batch_records: List[Dict[str, Any]] = []
            lang_total = 0
            for record in self._iter_jsonl(corpus_file, max_passages_per_lang):
                batch_records.append(record)
                if len(batch_records) >= batch_size:
                    chunks = process_corpus_passage_native(batch_records)
                    added = self._add_embedded_chunks(
                        passage_index, chunks, centroid_sums, centroid_counts
                    )
                    lang_total += added
                    logger.info("Language '%s': embedded %d chunks (lang total: %d)...", lang, added, lang_total)
                    batch_records.clear()
            if batch_records:
                chunks = process_corpus_passage_native(batch_records)
                added = self._add_embedded_chunks(
                    passage_index, chunks, centroid_sums, centroid_counts
                )
                lang_total += added
                logger.info("Language '%s': embedded final %d chunks (lang total: %d)...", lang, added, lang_total)
            passage_total += lang_total
            passage_counts[lang] = lang_total
            logger.info("Finished indexing %s passage-native chunks for '%s'.", lang_total, lang)

        if passage_total:
            passage_index.save(self.index_dir)
            self.indexes["passage_native"] = passage_index
            self._save_centroids(centroid_sums, centroid_counts)
        else:
            logger.warning("No passage corpus records were indexed.")

        # Long-document index: process one document at a time and embed bounded batches.
        longdoc_index = StrategyVectorIndex("semantic_longdoc")
        longdoc_total = 0
        for lang in config.LANGUAGES:
            longdoc_file = config.PROCESSED_DATA_DIR / f"{lang}_longdocs.jsonl"
            if not longdoc_file.exists():
                logger.info("Long-document file %s not found; skipping.", longdoc_file)
                continue
            pending_chunks: List[Chunk] = []
            lang_total = 0
            for record in self._iter_jsonl(longdoc_file):
                pending_chunks.extend(process_longdocs_sentence_window([record]))
                pending_chunks.extend(process_longdocs_semantic([record]))
                if len(pending_chunks) >= batch_size:
                    lang_total += self._add_embedded_chunks(
                        longdoc_index, pending_chunks, {}, {}
                    )
                    pending_chunks.clear()
            if pending_chunks:
                lang_total += self._add_embedded_chunks(
                    longdoc_index, pending_chunks, {}, {}
                )
            longdoc_total += lang_total
            longdoc_counts[lang] = lang_total
            logger.info("Indexed %s long-document chunks for '%s'.", lang_total, lang)

        if longdoc_total:
            longdoc_index.save(self.index_dir)
            self.indexes["semantic_longdoc"] = longdoc_index
        else:
            logger.warning("No long-document records were indexed.")

        manifest = {
            "languages": list(config.LANGUAGES),
            "full_corpus": max_passages_per_lang is None,
            "passage_counts": passage_counts,
            "longdoc_counts": longdoc_counts,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
        }
        with open(self.index_dir / "index_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

        logger.info(
            "Completed full indexing: %s passage-native and %s long-document chunks.",
            passage_total,
            longdoc_total,
        )

    def _save_centroids(
        self, centroid_sums: Dict[str, np.ndarray], centroid_counts: Dict[str, int]
    ):
        """Normalize and persist online per-language and global centroid statistics."""
        if not centroid_counts:
            return
        centroids_dict: Dict[str, List[float]] = {}
        total_sum = np.zeros(config.EMBEDDING_DIM, dtype=np.float64)
        total_count = 0
        self.centroids = {}
        for lang in config.LANGUAGES:
            if lang not in centroid_counts:
                continue
            mean_vec = centroid_sums[lang] / centroid_counts[lang]
            norm_vec = (mean_vec / (np.linalg.norm(mean_vec) + 1e-12)).astype(np.float32)
            centroids_dict[lang] = norm_vec.tolist()
            self.centroids[lang] = norm_vec
            total_sum += centroid_sums[lang]
            total_count += centroid_counts[lang]
        global_mean = total_sum / max(total_count, 1)
        global_norm = (global_mean / (np.linalg.norm(global_mean) + 1e-12)).astype(np.float32)
        centroids_dict["global"] = global_norm.tolist()
        self.global_centroid = global_norm
        centroid_file = self.index_dir / "centroids.json"
        with open(centroid_file, "w", encoding="utf-8") as handle:
            json.dump(centroids_dict, handle, ensure_ascii=False)
        logger.info("Saved corpus centroids to %s", centroid_file)

    def _compute_and_save_centroids(self, chunks: List[Chunk], vectors: np.ndarray):
        """Compatibility helper for callers that already hold one vector batch."""
        sums: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}
        for chunk, vector in zip(chunks, vectors):
            lang = chunk.source_lang.lower()
            sums.setdefault(lang, np.zeros(vectors.shape[1], dtype=np.float64))
            sums[lang] += vector.astype(np.float64, copy=False)
            counts[lang] = counts.get(lang, 0) + 1
        self._save_centroids(sums, counts)

    def load_all_indexes(self):
        """Loads all existing strategy indexes and centroids from disk into memory with auto-rebuild fallback."""
        loaded_ok = True
        for strategy in ["passage_native", "semantic_longdoc"]:
            try:
                idx = StrategyVectorIndex.load(self.index_dir, strategy)
                if idx.index.ntotal == 0:
                    raise ValueError(f"Index '{strategy}' has 0 vectors.")
                self.indexes[strategy] = idx
            except Exception as e:
                logger.warning(f"Could not load index '{strategy}': {e}")
                loaded_ok = False
                
        # Load centroids
        centroid_file = self.index_dir / "centroids.json"
        if centroid_file.exists():
            try:
                with open(centroid_file, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                for k, v in c_data.items():
                    arr = np.array(v, dtype=np.float32)
                    if k == "global":
                        self.global_centroid = arr
                    else:
                        self.centroids[k] = arr
                logger.info(f"Loaded centroids for: {list(self.centroids.keys())}")
            except Exception as e:
                logger.warning(f"Failed loading centroids: {e}")
                loaded_ok = False
        else:
            loaded_ok = False

        # Rebuild stale artifacts produced by old languages or missing manifest.
        manifest_file = self.index_dir / "index_manifest.json"
        manifest_ok = False
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                manifest_ok = set(manifest.get("languages", [])) == set(config.LANGUAGES)
            except Exception as exc:
                logger.warning("Failed loading index manifest: %s", exc)
        else:
            logger.info("Index manifest is missing; treating existing indexes as stale.")

        if not loaded_ok or not manifest_ok or "passage_native" not in self.indexes or self.indexes["passage_native"].index.ntotal == 0:
            logger.info("[IndexManager] Auto-recovering: building complete indexes from corpus...")
            limit = getattr(config, "MAX_INDEX_PASSAGES_PER_LANG", None)
            self.build_all_indexes(max_passages_per_lang=limit)



_INDEX_MANAGER: Optional[IndexManager] = None


def get_index_manager() -> IndexManager:
    """Singleton getter for IndexManager."""
    global _INDEX_MANAGER
    if _INDEX_MANAGER is None:
        _INDEX_MANAGER = IndexManager()
        _INDEX_MANAGER.load_all_indexes()
    return _INDEX_MANAGER


if __name__ == "__main__":
    manager = IndexManager()
    manager.build_all_indexes(max_passages_per_lang=None)
