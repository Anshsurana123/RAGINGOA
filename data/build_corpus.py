"""
Builds deduplicated passage corpora for all configured languages.

Strict Extensibility Requirement:
This script iterates dynamically over `config.LANGUAGES`.
No language codes are hardcoded in this logic.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
import pyarrow.parquet as pq

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def stream_parquet_records(parquet_path: Path, max_records: int) -> List[Dict[str, Any]]:
    """
    Stream records safely from large Parquet files using PyArrow batches.
    Avoids nested conversion memory spikes and errors.
    """
    records = []
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=1000):
        rows = batch.to_pylist()
        records.extend(rows)
        if len(records) >= max_records:
            records = records[:max_records]
            break
    return records


def load_raw_dataset_for_lang(lang: str, max_queries: int = 6000) -> List[Dict[str, Any]]:
    """
    Load raw MS MARCO / MSMARCO-XI data for a given language.
    Checks local cache/files first, then falls back to Hugging Face datasets.
    """
    lang_info = config.get_language_info(lang)
    msmarco_prefix = lang_info.get("msmarco_file", lang)
    
    # 1. Check local cache in data/raw/<lang>/
    local_raw_dir = config.RAW_DATA_DIR / lang
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_json_cache = local_raw_dir / "raw_queries.json"
    
    if raw_json_cache.exists():
        logger.info(f"Loading cached raw data for '{lang}' from {raw_json_cache}")
        try:
            with open(raw_json_cache, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    # 2. Check local train / validation directory in workspace
    local_train_parquet = config.BASE_DIR / "train" / f"{msmarco_prefix}train.parquet"
    local_val_parquet = config.BASE_DIR / "validation" / f"{msmarco_prefix}val.parquet"
    
    records = []
    if local_train_parquet.exists():
        logger.info(f"Streaming local train parquet for '{lang}' from {local_train_parquet} (limit: {max_queries})...")
        records = stream_parquet_records(local_train_parquet, max_records=max_queries)
    elif local_val_parquet.exists():
        logger.info(f"Streaming local val parquet for '{lang}' from {local_val_parquet} (limit: {max_queries})...")
        records = stream_parquet_records(local_val_parquet, max_records=max_queries)
    elif lang == "en" or lang_info.get("script") == "Latn":
        # Stream English fields from any available training parquet
        available_parquets = list((config.BASE_DIR / "train").glob("*.parquet")) or list((config.BASE_DIR / "validation").glob("*.parquet"))
        if available_parquets:
            logger.info(f"Streaming English fields from {available_parquets[0]} for '{lang}'...")
            records = stream_parquet_records(available_parquets[0], max_records=max_queries)
    else:
        # Fallback: Pull from Hugging Face
        try:
            from datasets import load_dataset
            logger.info(f"Downloading dataset for language '{lang}' from Hugging Face...")
            if lang == "en":
                ds = load_dataset("ms_marco", "v1.1", split=f"validation[:{max_queries}]")
                for row in ds:
                    records.append({
                        "query": row["query"],
                        "Answer": row["answers"][0] if row.get("answers") else "",
                        "query_id": row["query_id"],
                        "query_type": row.get("query_type", "DESCRIPTION"),
                        "passages": {
                            "is_selected": row["passages"]["is_selected"],
                            "English_passages": row["passages"]["passage_text"],
                            "Translated_passages": row["passages"]["passage_text"],
                        },
                        "Eng_Query": row["query"],
                        "Eng_Answer": row["answers"][0] if row.get("answers") else "",
                    })
            else:
                ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split=f"validation[:{max_queries}]")
                records = list(ds)
        except Exception as e:
            logger.warning(f"Could not download directly from Hugging Face for '{lang}': {e}")
            raise RuntimeError(f"Unable to load data for language '{lang}'")

    if records:
        if len(records) > max_queries:
            records = records[:max_queries]
        try:
            with open(raw_json_cache, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)
            logger.info(f"Saved raw data cache ({len(records)} queries) to {raw_json_cache}")
        except Exception as e:
            logger.warning(f"Could not cache raw queries to {raw_json_cache}: {e}")
            
    return records


def extract_and_deduplicate_passages(
    lang: str, raw_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Flatten and deduplicate passages across queries into a clean corpus.
    Attaches passage_id, text, source_lang, source_query_ids, and is_selected.
    """
    lang_info = config.get_language_info(lang)
    is_english = (lang_info.get("script") == "Latn") or (lang == "en")
    
    passage_map: Dict[str, Dict[str, Any]] = {}
    
    for row in raw_records:
        qid = int(row.get("query_id", 0))
        passages_data = row.get("passages", {})
        
        if not isinstance(passages_data, dict):
            continue
            
        is_selected_list = passages_data.get("is_selected", [])
        
        if is_english:
            passages_list = passages_data.get("English_passages", [])
            if passages_list is None or len(passages_list) == 0:
                passages_list = passages_data.get("Translated_passages", [])
        else:
            passages_list = passages_data.get("Translated_passages", [])
            if passages_list is None or len(passages_list) == 0:
                passages_list = passages_data.get("English_passages", [])
        
        if passages_list is None or len(passages_list) == 0:
            continue
            
        passages_list = list(passages_list)
        if is_selected_list is not None:
            is_selected_list = list(is_selected_list)
        else:
            is_selected_list = []
            
        for idx, text in enumerate(passages_list):
            if not text or not isinstance(text, str):
                continue
            cleaned_text = text.strip()
            if len(cleaned_text) < 15:
                continue
                
            is_sel = 0
            if idx < len(is_selected_list):
                is_sel = int(is_selected_list[idx])
                
            if cleaned_text not in passage_map:
                p_id = f"{lang}_p_{len(passage_map):06d}"
                passage_map[cleaned_text] = {
                    "passage_id": p_id,
                    "text": cleaned_text,
                    "source_lang": lang,
                    "source_query_ids": [qid],
                    "is_selected": is_sel,
                }
            else:
                if qid not in passage_map[cleaned_text]["source_query_ids"]:
                    passage_map[cleaned_text]["source_query_ids"].append(qid)
                if is_sel == 1:
                    passage_map[cleaned_text]["is_selected"] = 1
                    
    deduped_passages = list(passage_map.values())
    logger.info(
        f"Extracted {len(deduped_passages)} unique deduplicated passages for '{lang}' "
        f"across {len(raw_records)} queries."
    )
    return deduped_passages


def build_all_corpora(max_queries_per_lang: int = 5000) -> Dict[str, int]:
    """
    Iterates dynamically over config.LANGUAGES and builds deduplicated passage corpora.
    Returns dictionary of language -> corpus passage count.
    """
    results = {}
    logger.info(f"Building corpora for configured languages: {config.LANGUAGES}")
    
    for lang in config.LANGUAGES:
        logger.info(f"Processing language: '{lang}' ...")
        raw_records = load_raw_dataset_for_lang(lang, max_queries=max_queries_per_lang)
        corpus = extract_and_deduplicate_passages(lang, raw_records)
        
        output_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in corpus:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        logger.info(f"Successfully saved {len(corpus)} passages to {output_file}")
        results[lang] = len(corpus)
        
    return results


if __name__ == "__main__":
    build_all_corpora()
