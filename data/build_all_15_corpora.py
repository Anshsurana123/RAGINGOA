"""
Builds deduplicated passage corpora for ALL 15 registered Indic languages + English.
Reads from local cache in data/raw/<lang>/raw_queries.json or Hugging Face.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.build_corpus import load_raw_dataset_for_lang, extract_and_deduplicate_passages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

ALL_15_LANGUAGES = [
    "as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"
]


def build_all_15_corpora(max_queries_per_lang: int = 6000) -> Dict[str, int]:
    results = {}
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Building corpora for ALL 15 languages: %s", ALL_15_LANGUAGES)
    for lang in ALL_15_LANGUAGES:
        output_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
        logger.info(">>> Processing language '%s' (%s)...", lang, config.get_language_info(lang).get("name", lang))
        
        try:
            raw_records = load_raw_dataset_for_lang(lang, max_queries=max_queries_per_lang)
            corpus = extract_and_deduplicate_passages(lang, raw_records)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in corpus:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
            logger.info("Successfully saved %d passages to %s", len(corpus), output_file.name)
            results[lang] = len(corpus)
        except Exception as e:
            logger.error("Failed to build corpus for '%s': %s", lang, e)
            
    logger.info("=== SUMMARY OF ALL 15 CORPORA ===")
    for lang, count in results.items():
        logger.info("  %s (%s): %d passages", lang, config.get_language_info(lang).get("name", lang), count)
    return results


if __name__ == "__main__":
    build_all_15_corpora()
