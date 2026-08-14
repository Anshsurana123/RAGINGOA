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
            
        # Query FAISS HNSW (request more candidates if language filter is active)
        search_k = min(self.index.ntotal, top_k * 5 if target_lang else top_k)
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

    def build_all_indexes(self, max_passages_per_lang: int = 10000):
        """
        Builds combined FAISS indexes for:
        1. 'passage_native' (MS MARCO passages across all configured languages)
        2. 'semantic_longdoc' (Sentence-window + Semantic chunks across all configured languages)
        """
        logger.info(f"Building all indexes for configured languages: {config.LANGUAGES}")
        
        # --- 1. Passage Native Index ---
        passage_index = StrategyVectorIndex("passage_native")
        all_passage_chunks: List[Chunk] = []
        
        for lang in config.LANGUAGES:
            corpus_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
            if not corpus_file.exists():
                logger.warning(f"Corpus file {corpus_file} not found. Skipping.")
                continue
                
            lang_records = []
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lang_records.append(json.loads(line.strip()))
                        if len(lang_records) >= max_passages_per_lang:
                            break
                            
            lang_chunks = process_corpus_passage_native(lang_records)
            all_passage_chunks.extend(lang_chunks)
            logger.info(f"Loaded {len(lang_chunks)} passage-native chunks for '{lang}'.")
            
        if all_passage_chunks:
            logger.info(f"Embedding {len(all_passage_chunks)} total passage chunks with 'passage: ' prefix...")
            texts_to_embed = [c.embed_text for c in all_passage_chunks]
            passage_vectors = self.embedder.encode_passages(texts_to_embed, batch_size=32)
            passage_index.add_chunks(all_passage_chunks, passage_vectors)
            passage_index.save(self.index_dir)
            self.indexes["passage_native"] = passage_index
            
            # Compute centroids for off-topic guardrail
            self._compute_and_save_centroids(all_passage_chunks, passage_vectors)
            
        # --- 2. Semantic & Sentence-Window Long-Doc Index ---
        longdoc_index = StrategyVectorIndex("semantic_longdoc")
        all_longdoc_chunks: List[Chunk] = []
        
        for lang in config.LANGUAGES:
            longdoc_file = config.PROCESSED_DATA_DIR / f"{lang}_longdocs.jsonl"
            if not longdoc_file.exists():
                continue
                
            longdoc_records = []
            with open(longdoc_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        longdoc_records.append(json.loads(line.strip()))
                        
            # Create sentence-window chunks
            sw_chunks = process_longdocs_sentence_window(longdoc_records)
            # Create semantic chunks
            sem_chunks = process_longdocs_semantic(longdoc_records)
            
            all_longdoc_chunks.extend(sw_chunks)
            all_longdoc_chunks.extend(sem_chunks)
            logger.info(f"Created {len(sw_chunks)} sentence-window and {len(sem_chunks)} semantic chunks for '{lang}'.")
            
        if all_longdoc_chunks:
            logger.info(f"Embedding {len(all_longdoc_chunks)} longdoc chunks...")
            longdoc_texts = [c.embed_text for c in all_longdoc_chunks]
            longdoc_vectors = self.embedder.encode_passages(longdoc_texts, batch_size=32)
            longdoc_index.add_chunks(all_longdoc_chunks, longdoc_vectors)
            longdoc_index.save(self.index_dir)
            self.indexes["semantic_longdoc"] = longdoc_index
            
        logger.info("All strategy indexes successfully built and persisted!")

    def _compute_and_save_centroids(self, chunks: List[Chunk], vectors: np.ndarray):
        """Compute mean centroid vectors per language and globally for off-topic check."""
        lang_vectors: Dict[str, List[np.ndarray]] = {}
        for c, v in zip(chunks, vectors):
            lang = c.source_lang.lower()
            if lang not in lang_vectors:
                lang_vectors[lang] = []
            lang_vectors[lang].append(v)
            
        centroids_dict = {}
        for lang, vecs in lang_vectors.items():
            arr = np.array(vecs)
            mean_vec = np.mean(arr, axis=0)
            norm_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
            centroids_dict[lang] = norm_vec.tolist()
            self.centroids[lang] = norm_vec
            
        # Global centroid
        global_mean = np.mean(vectors, axis=0)
        global_norm = global_mean / (np.linalg.norm(global_mean) + 1e-9)
        centroids_dict["global"] = global_norm.tolist()
        self.global_centroid = global_norm
        
        centroid_file = self.index_dir / "centroids.json"
        with open(centroid_file, "w", encoding="utf-8") as f:
            json.dump(centroids_dict, f)
        logger.info(f"Saved corpus centroids to {centroid_file}")

    def load_all_indexes(self):
        """Loads all existing strategy indexes and centroids from disk into memory."""
        for strategy in ["passage_native", "semantic_longdoc"]:
            try:
                idx = StrategyVectorIndex.load(self.index_dir, strategy)
                self.indexes[strategy] = idx
            except Exception as e:
                logger.warning(f"Could not load index '{strategy}': {e}")
                
        # Load centroids
        centroid_file = self.index_dir / "centroids.json"
        if centroid_file.exists():
            with open(centroid_file, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            for k, v in c_data.items():
                arr = np.array(v, dtype=np.float32)
                if k == "global":
                    self.global_centroid = arr
                else:
                    self.centroids[k] = arr
            logger.info(f"Loaded centroids for: {list(self.centroids.keys())}")


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
    manager.build_all_indexes(max_passages_per_lang=700)
