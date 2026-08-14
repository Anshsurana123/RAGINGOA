"""
Chunk metadata schemas, tagging utilities, and language pre-filtering.

Strict Extensibility Requirement:
Language filtering operates dynamically against any language code in `config.LANGUAGES`.
"""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Standardized schema for all indexed chunks across chunking strategies and languages.
    """
    chunk_id: str
    text: str
    embed_text: str  # The exact text passed to the embedding model
    chunk_strategy: str  # 'passage_native', 'sentence_window', 'semantic'
    source_lang: str  # 'hi', 'ta', 'en', etc.
    token_count: int
    source_query_ids: List[int] = Field(default_factory=list)
    doc_id: Optional[str] = None
    context_window: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for multilingual text (word-based approximation with character factor).
    """
    if not text:
        return 0
    words = text.split()
    # For Indic scripts, characters/words roughly map to 1.3-1.5 subword tokens
    return max(len(words), int(len(text) / 4.0))


def filter_chunks_by_language(chunks: List[Chunk], target_lang: str) -> List[Chunk]:
    """
    Language-based pre-filter to restrict search or candidate selection to the query language.
    """
    if not target_lang:
        return chunks
    target_clean = target_lang.lower().strip()
    return [c for c in chunks if c.source_lang.lower() == target_clean]


def split_sentences_multilingual(text: str) -> List[str]:
    """
    Robust multilingual sentence splitter handling Indic punctuation (।, ॥) and standard (. ! ?).
    """
    if not text:
        return []
    # Split on Devanagari/Indic danda (।), double danda (॥), period, question mark, exclamation mark, newlines
    raw_sentences = re.split(r'(?<=[।॥.!?\n])\s+', text)
    cleaned = []
    for s in raw_sentences:
        s_str = s.strip()
        if len(s_str) > 3:
            cleaned.append(s_str)
    return cleaned if cleaned else [text.strip()]


def calculate_overlap_tokens(text: str, overlap_percent: float = 0.15) -> str:
    """
    Extract trailing overlap text (10-20% tokens) from the end of a chunk to prepend to the next.
    """
    words = text.split()
    if len(words) < 5:
        return ""
    num_overlap_words = max(1, int(len(words) * overlap_percent))
    return " ".join(words[-num_overlap_words:])
