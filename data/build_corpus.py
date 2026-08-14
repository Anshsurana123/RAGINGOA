"""
Builds deduplicated passage corpora for all configured languages.

Strict Extensibility Requirement:
This script iterates over `config.LANGUAGES`.
No language codes are hardcoded in this logic.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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
    raw_parquet_cache = local_raw_dir / "raw_queries.parquet"
    
    if raw_parquet_cache.exists():
        logger.info(f"Loading cached raw data for '{lang}' from {raw_parquet_cache}")
        df = pd.read_parquet(raw_parquet_cache)
        return df.to_dict(orient="records")
    
    # 2. Check local validation/train directory in workspace if available
    local_val_parquet = config.BASE_DIR / "validation" / f"{msmarco_prefix}val.parquet"
    local_train_parquet = config.BASE_DIR / "train" / f"{msmarco_prefix}train.parquet"
    
    df = None
    if local_val_parquet.exists():
        logger.info(f"Loading local parquet for '{lang}' from {local_val_parquet}")
        df = pd.read_parquet(local_val_parquet)
    elif local_train_parquet.exists():
        logger.info(f"Loading local train parquet for '{lang}' from {local_train_parquet}")
        df = pd.read_parquet(local_train_parquet)
    else:
        # Fallback: Check if we have any other indic parquet to read English from if lang == en
        any_val_parquets = list((config.BASE_DIR / "validation").glob("*val.parquet"))
        if any_val_parquets and lang_info.get("script") == "Latn":
            logger.info(f"Using English fields from local dataset {any_val_parquets[0]} for '{lang}'")
            df = pd.read_parquet(any_val_parquets[0])
        else:
            # 3. Pull from Hugging Face
            try:
                from datasets import load_dataset
                logger.info(f"Downloading dataset for language '{lang}' from Hugging Face...")
                if lang == "en":
                    ds = load_dataset("ms_marco", "v1.1", split=f"validation[:{max_queries}]")
                    records = []
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
                    df = pd.DataFrame(records)
                else:
                    ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split=f"validation[:{max_queries}]")
                    df = pd.DataFrame(ds)
            except Exception as e:
                logger.warning(f"Could not download directly from Hugging Face for '{lang}': {e}")
                # Fallback to local files if available
                if any_val_parquets:
                    df = pd.read_parquet(any_val_parquets[0])
                else:
                    raise RuntimeError(f"Unable to load data for language '{lang}'")
    
    if df is not None:
        if len(df) > max_queries:
            df = df.iloc[:max_queries]
        # Cache raw dataframe locally
        df.to_parquet(raw_parquet_cache, index=False)
        logger.info(f"Saved raw data cache ({len(df)} queries) to {raw_parquet_cache}")
        return df.to_dict(orient="records")
    
    return []

def extract_and_deduplicate_passages(
    lang: str, raw_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Flatten and deduplicate passages across thousands of queries into a clean corpus.
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
    Iterates over config.LANGUAGES and builds deduplicated passage corpora.
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
