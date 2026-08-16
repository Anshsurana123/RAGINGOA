"""
High-Speed GPU Corpus Indexer for Multilingual RAG.
Optimized for Google Colab (T4/V100/A100 GPU) or any NVIDIA CUDA machine.
Runs batch FP16 PyTorch inference to index 148,500+ passages in ~60-90 seconds.
"""

import json
import logging
import os
import subprocess
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


def ensure_lfs_pulled():
    """Check if any corpus files are Git LFS pointers and pull them if needed."""
    for lang in config.LANGUAGES:
        corpus_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
        if corpus_file.exists():
            with open(corpus_file, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                if first_line.startswith("version https://git-lfs"):
                    logger.warning("Detected Git LFS pointer in %s. Pulling real data via 'git lfs pull'...", corpus_file.name)
                    try:
                        subprocess.run(["git", "lfs", "install"], check=True)
                        subprocess.run(["git", "lfs", "pull"], check=True)
                        logger.info("Git LFS pull completed successfully.")
                    except Exception as e:
                        logger.error("Failed to run 'git lfs pull': %s", e)
                    break


def build_indexes_gpu(batch_size: int = 512, output_dir: Path = None):
    ensure_lfs_pulled()

    output_dir = output_dir or config.INDEX_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    logger.info("Using device: %s (%s)", device, device_name)

    from sentence_transformers import SentenceTransformer
    logger.info("Loading '%s' in FP16 on %s...", config.EMBEDDING_MODEL_NAME, device)
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
    if device == "cuda":
        model = model.half()  # FP16 for 3x Tensor Core throughput

    # 1. Index Passage-Native Corpus
    passage_index = StrategyVectorIndex("passage_native")
    centroid_sums: Dict[str, np.ndarray] = {}
    centroid_counts: Dict[str, int] = {}
    passage_counts: Dict[str, int] = {}
    total_start = time.time()
    grand_total = 0

    for lang in config.LANGUAGES:
        corpus_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
        if not corpus_file.exists():
            logger.warning("Corpus file %s not found; skipping '%s'.", corpus_file, lang)
            continue
        
        logger.info(">>> Starting language '%s' from %s...", lang, corpus_file.name)
        lang_start = time.time()
        batch_records: List[Dict[str, Any]] = []
        lang_total = 0

        with open(corpus_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if line.startswith("version https://git-lfs"):
                        raise RuntimeError(
                            f"Corpus file {corpus_file} is a Git LFS pointer! Please run 'git lfs pull' first."
                        )
                    continue
                batch_records.append(record)
                
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
        grand_total += lang_total
        elapsed = time.time() - lang_start
        logger.info("Finished language '%s': %d passages in %.1fs (%.1f passages/sec).", lang, lang_total, elapsed, lang_total / max(0.1, elapsed))

    passage_index.save(output_dir)
    logger.info("Saved 'passage_native' index to %s (total passages: %d, size on disk: ~%.1f MB)", 
                output_dir, passage_index.size, (passage_index.size * 384 * 4) / (1024 * 1024))

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
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
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
        "full_corpus": True,
        "passage_counts": passage_counts,
        "longdoc_counts": longdoc_counts,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "index_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - total_start
    logger.info("================================================================================")
    logger.info("=== FULL GPU INDEXING COMPLETED: %d passages + %d longdocs in %.1fs (%.1f min) ===", 
                grand_total, longdoc_index.size, total_elapsed, total_elapsed / 60)
    logger.info("================================================================================")


if __name__ == "__main__":
    build_indexes_gpu(batch_size=512)
