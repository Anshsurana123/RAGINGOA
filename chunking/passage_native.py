"""
Passage Native Chunking Strategy.

Treats each deduplicated MS MARCO passage as an atomic chunk.
Baseline, zero-loss index strategy preserving original passage boundaries and query associations.
"""

from typing import Any, Dict, List
from chunking.metadata import Chunk, estimate_token_count


def chunk_passage_native(passage_dict: Dict[str, Any]) -> Chunk:
    """
    Convert a deduplicated MS MARCO passage record into a standardized Chunk.
    """
    text = passage_dict.get("text", "").strip()
    source_lang = passage_dict.get("source_lang", "en")
    p_id = passage_dict.get("passage_id", f"{source_lang}_p_unknown")
    source_query_ids = passage_dict.get("source_query_ids", [])
    
    return Chunk(
        chunk_id=p_id,
        text=text,
        embed_text=text,
        chunk_strategy="passage_native",
        source_lang=source_lang,
        token_count=estimate_token_count(text),
        source_query_ids=source_query_ids,
        doc_id=p_id,
        context_window=None,
        metadata={
            "is_selected": passage_dict.get("is_selected", 0),
            "original_passage_id": p_id,
        },
    )


def process_corpus_passage_native(corpus: List[Dict[str, Any]]) -> List[Chunk]:
    """
    Process an entire corpus into passage-native chunks.
    """
    chunks = []
    for item in corpus:
        chunks.append(chunk_passage_native(item))
    return chunks
