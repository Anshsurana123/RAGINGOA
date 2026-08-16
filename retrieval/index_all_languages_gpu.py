"""
GPU Indexer for ALL 15 Indic Languages + English.
Supports indexing:
  'as', 'bn', 'en', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'sa', 'ta', 'te', 'ur'
Runs FP16 PyTorch inference on CUDA/GPU for maximum throughput (~1,500 - 2,000 passages/sec).
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from chunking.metadata import Chunk
from chunking.passage_native import process_corpus_passage_native
from chunking.sentence_window import process_longdocs_sentence_window
from chunking.semantic import process_longdocs_semantic
from retrieval.index_faiss import StrategyVectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

ALL_15_LANGUAGES = [
    "as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"
]


def ensure_corpus_exists(languages: List[str]):
    """Ensure processed jsonl corpus exists for each language, building from raw data if needed."""
    from data.build_corpus import load_raw_dataset_for_lang, extract_and_deduplicate_passages
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        corpus_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
        if not corpus_file.exists() or corpus_file.stat().st_size < 100:
            logger.info("Building missing corpus file for '%s' from data/raw/%s/...", lang, lang)
            try:
                raw_records = load_raw_dataset_for_lang(lang, max_queries=6000)
                passages = extract_and_deduplicate_passages(lang, raw_records)
                with open(corpus_file, "w", encoding="utf-8") as f:
                    for p in passages:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                logger.info("Saved %d passages for '%s' to %s", len(passages), corpus_file.name)
            except Exception as e:
                logger.warning("Could not auto-build corpus for '%s': %s", lang, e)


def build_all_languages_gpu(
    languages: Optional[List[str]] = None,
    batch_size: int = 512,
    output_dir: Optional[Path] = None,
):
    languages = languages or ALL_15_LANGUAGES
    output_dir = output_dir or config.INDEX_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting GPU Indexing for %d languages: %s", len(languages), languages)
    ensure_corpus_exists(languages)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    logger.info("Hardware Accelerator: %s (%s)", device, device_name)

    from sentence_transformers import SentenceTransformer
    logger.info("Loading '%s' in FP16 on %s...", config.EMBEDDING_MODEL_NAME, device)
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
    if device == "cuda":
        model = model.half()  # FP16 Tensor Core acceleration

    total_start = time.time()
    passage_index = StrategyVectorIndex("passage_native")
    centroid_sums: Dict[str, np.ndarray] = {}
    centroid_counts: Dict[str, int] = {}
    passage_counts: Dict[str, int] = {}
    grand_total = 0

    for lang in languages:
        corpus_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
        if not corpus_file.exists():
            logger.warning("Corpus file %s not found; skipping '%s'.", corpus_file, lang)
            continue

        logger.info("================================================================")
        logger.info(">>> Indexing language '%s' (%s) from %s...", 
                    lang, config.get_language_info(lang).get("name", lang), corpus_file.name)
        logger.info("================================================================")
        
        lang_start = time.time()
        batch_records: List[Dict[str, Any]] = []
        lang_total = 0

        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
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
        logger.info("Finished language '%s': %d passages in %.1fs (%.1f passages/sec).", 
                    lang, lang_total, elapsed, lang_total / max(0.1, elapsed))

    passage_index.save(output_dir)
    num_passages = len(passage_index.chunks)
    logger.info("Saved 'passage_native' index to %s (total passages: %d, size: ~%.1f MB)", 
                output_dir, num_passages, (num_passages * 384 * 4) / (1024 * 1024))

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
    logger.info("Saved centroids for %d languages to %s", len(centroids), output_dir / "centroids.json")

    # 2. Index Long-Document Corpus
    logger.info(">>> Indexing Long Documents...")
    longdoc_index = StrategyVectorIndex("semantic_longdoc")
    longdoc_counts: Dict[str, int] = {}
    for lang in languages:
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
    num_longdocs = len(longdoc_index.chunks)
    logger.info("Saved 'semantic_longdoc' index to %s (total size: %d chunks)", output_dir, num_longdocs)

    # 3. Write Manifest
    manifest = {
        "languages": list(languages),
        "full_corpus": True,
        "passage_counts": passage_counts,
        "longdoc_counts": longdoc_counts,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "index_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Saved index manifest to %s", output_dir / "index_manifest.json")

    total_elapsed = time.time() - total_start
    logger.info("================================================================================")
    logger.info("=== FULL ALL-LANGUAGE GPU INDEXING COMPLETE: %d passages in %.1fs (%.1f min) ===", 
                grand_total, total_elapsed, total_elapsed / 60)
    logger.info("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Indexer for all Indic languages + English")
    parser.add_argument("--languages", nargs="+", default=ALL_15_LANGUAGES, help="Languages to index")
    parser.add_argument("--batch-size", type=int, default=512, help="Inference batch size")
    args = parser.parse_args()
    build_all_languages_gpu(languages=args.languages, batch_size=args.batch_size)
