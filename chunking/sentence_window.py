"""
Sentence Window Chunking Strategy.

For long documents:
1. Splits text into sentences.
2. For each sentence, attaches a ±1 sentence window of surrounding context.
3. Incorporates 10-20% token overlap across sentence boundaries so cross-boundary queries retain continuity.
"""

from typing import Any, Dict, List
import config
from chunking.metadata import (
    Chunk,
    split_sentences_multilingual,
    calculate_overlap_tokens,
    estimate_token_count,
)


def chunk_document_sentence_window(
    doc_dict: Dict[str, Any], window_size: int = config.SENTENCE_WINDOW_SIZE
) -> List[Chunk]:
    """
    Split a long document into sentence-window chunks with ±window_size surrounding context
    and 10-20% boundary overlap.
    """
    full_text = doc_dict.get("text", "")
    doc_id = doc_dict.get("doc_id", "doc_unknown")
    source_lang = doc_dict.get("source_lang", "en")
    title = doc_dict.get("title", "")
    
    sentences = split_sentences_multilingual(full_text)
    if not sentences:
        return []
        
    chunks: List[Chunk] = []
    prev_tail_overlap = ""
    
    for i, center_sent in enumerate(sentences):
        # Determine window boundaries
        start_idx = max(0, i - window_size)
        end_idx = min(len(sentences), i + window_size + 1)
        
        # Build window context
        window_sentences = sentences[start_idx:end_idx]
        raw_window_text = " ".join(window_sentences)
        
        # Prepend overlap from previous sentence window boundary if available
        if prev_tail_overlap:
            stitched_text = f"{prev_tail_overlap} {raw_window_text}"
        else:
            stitched_text = raw_window_text
            
        chunk_id = f"{doc_id}_sw_{i:04d}"
        
        chunk = Chunk(
            chunk_id=chunk_id,
            text=stitched_text,
            embed_text=center_sent,  # Focused central sentence for precise embedding
            chunk_strategy="sentence_window",
            source_lang=source_lang,
            token_count=estimate_token_count(stitched_text),
            doc_id=doc_id,
            context_window=raw_window_text,
            metadata={
                "sentence_index": i,
                "total_sentences": len(sentences),
                "title": title,
                "center_sentence": center_sent,
            },
        )
        chunks.append(chunk)
        
        # Calculate 15% overlap for next chunk
        prev_tail_overlap = calculate_overlap_tokens(
            raw_window_text, overlap_percent=config.CHUNK_OVERLAP_PERCENT
        )
        
    return chunks


def process_longdocs_sentence_window(longdocs: List[Dict[str, Any]]) -> List[Chunk]:
    """
    Process a collection of long documents using sentence-window chunking.
    """
    all_chunks = []
    for doc in longdocs:
        all_chunks.extend(chunk_document_sentence_window(doc))
    return all_chunks
