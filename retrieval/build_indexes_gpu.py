"""
High-Speed GPU Corpus Indexer for Multilingual RAG.
Optimized for Google Colab (T4/V100/A100 GPU) or any NVIDIA CUDA machine.
Runs batch FP16 PyTorch inference to index 150,000+ passages in ~60-90 seconds.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.chunker import (
    process_corpus_passage_native,
    process_longdocs_sentence_window,
    process_longdocs_semantic,
    Chunk,
)
from retrieval.index_faiss import StrategyVectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)


def build_indexes_gpu(batch_size: int = 512, output_dir: Path = None):
    output_dir = output_dir or config.INDEX_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    from sentence_transformers import SentenceTransformer
    logger.info("Loading '%s' in FP16 on %s...", config.EMBEDDING_MODEL_NAME, device)
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
    if device == "cuda":
        model = model.half()  # FP16 for 3x throughput on modern Tensor Cores

    # 1. Index Passage-Native Corpus
    passage_index = StrategyVectorIndex("passage_native")
    centroid_sums: Dict[str, np.ndarray] = {}
    centroid_counts: Dict[str, int] = {}
    passage_counts: Dict[str, int] = {}
    total_start = time.time()

    for lang in config.LANGUAGES:
        corpus_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
        if not corpus_file.exists():
            logger.warning("Corpus file %s not found; skipping '%s'.", corpus_file, lang)
            continue
        
        logger.info("Processing language '%s' from %s...", lang, corpus_file.name)
        lang_start = time.time()
        batch_records: List[Dict[str, Any]] = []
        lang_total = 0

        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch_records.append(json.loads(line))
                if len(batch_records) >= batch_size:
                    chunks = process_corpus_passage_native(batch_records)
                    texts = [f"{config.PASSAGE_PREFIX}{c.embed_text.strip()}" for c in chunks]
                    with torch.inference_mode():
                        embeddings = model.encode(
                            texts,
                            batch_size=batch_size,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                        )
                    passage_index.add_chunks(chunks, embeddings)
                    for chunk, vec in zip(chunks, embeddings):
                        c_lang = chunk.source_lang.lower()
                        if c_lang not in centroid_sums:
                            centroid_sums[c_lang] = np.zeros(passage_index.dim, dtype=np.float64)
                            centroid_counts[c_lang] = 0
                        centroid_sums[c_lang] += vec.astype(np.float64, copy=False)
                        centroid_counts[c_lang] += 1
                    
                    lang_total += len(chunks)
                    if lang_total % 5120 == 0 or lang_total == len(chunks):
                        elapsed = time.time() - lang_start
                        rate = lang_total / max(0.1, elapsed)
                        logger.info("Language '%s': %d passages indexed (%.1f passages/sec)...", lang, lang_total, rate)
                    batch_records.clear()

            if batch_records:
                chunks = process_corpus_passage_native(batch_records)
                texts = [f"{config.PASSAGE_PREFIX}{c.embed_text.strip()}" for c in chunks]
                with torch.inference_mode():
                    embeddings = model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                passage_index.add_chunks(chunks, embeddings)
                for chunk, vec in zip(chunks, embeddings):
                    c_lang = chunk.source_lang.lower()
                    if c_lang not in centroid_sums:
                        centroid_sums[c_lang] = np.zeros(passage_index.dim, dtype=np.float64)
                        centroid_counts[c_lang] = 0
                    centroid_sums[c_lang] += vec.astype(np.float64, copy=False)
                    centroid_counts[c_lang] += 1
                lang_total += len(chunks)
                batch_records.clear()

        passage_counts[lang] = lang_total
        elapsed = time.time() - lang_start
        logger.info("Finished language '%s': %d passages in %.1fs (%.1f passages/sec).", lang, lang_total, elapsed, lang_total / max(0.1, elapsed))

    passage_index.save(output_dir)
    logger.info("Saved 'passage_native' index to %s (total size: %d)", output_dir, passage_index.size)

    # Save Centroids
    centroids = {}
    for lang, count in centroid_counts.items():
        if count > 0:
            c_vec = centroid_sums[lang] / count
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                c_vec = c_vec / c_norm
            centroids[lang] = c_vec.tolist()
    with open(output_dir / "centroids.json", "w", encoding="utf-8") as f:
        json.dump(centroids, f, ensure_ascii=False, indent=2)
    logger.info("Saved corpus centroids to %s", output_dir / "centroids.json")

    # 2. Index Long-Document Corpus
    longdoc_index = StrategyVectorIndex("semantic_longdoc")
    longdoc_counts: Dict[str, int] = {}
    for lang in config.LANGUAGES:
        longdoc_file = config.PROCESSED_DATA_DIR / f"{lang}_longdocs.jsonl"
        if not longdoc_file.exists():
            continue
        pending_chunks: List[Chunk] = []
        with open(longdoc_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line.strip())
                pending_chunks.extend(process_longdocs_sentence_window([record]))
                pending_chunks.extend(process_longdocs_semantic([record]))
        
        if pending_chunks:
            texts = [f"{config.PASSAGE_PREFIX}{c.embed_text.strip()}" for c in pending_chunks]
            with torch.inference_mode():
                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            longdoc_index.add_chunks(pending_chunks, embeddings)
            longdoc_counts[lang] = len(pending_chunks)
            logger.info("Indexed %d longdoc chunks for '%s'.", len(pending_chunks), lang)

    longdoc_index.save(output_dir)
    logger.info("Saved 'semantic_longdoc' index to %s (total size: %d)", output_dir, longdoc_index.size)

    # 3. Write Manifest
    manifest = {
        "languages": list(config.LANGUAGES),
        "passage_counts": passage_counts,
        "longdoc_counts": longdoc_counts,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "index_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - total_start
    logger.info("=== FULL GPU INDEXING COMPLETED in %.1f seconds (~%.1f minutes) ===", total_elapsed, total_elapsed / 60)


if __name__ == "__main__":
    build_indexes_gpu(batch_size=512)
