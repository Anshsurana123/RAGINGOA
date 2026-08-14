"""
Embedding Model Wrapper for multilingual-e5-small.

CRITICAL REQUIREMENT:
`intfloat/multilingual-e5-small` is a retrieval-trained model.
All query encodings MUST use the 'query: ' prefix.
All passage/document encodings MUST use the 'passage: ' prefix.
"""

import logging
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)

# Optimize CPU parallelism
try:
    torch.set_num_threads(max(1, torch.get_num_threads()))
except Exception:
    pass

_EMBEDDER_INSTANCE = None


class MultilingualE5Embedder:
    """
    Singleton wrapper for sentence-transformers multilingual-e5-small.
    Enforces required 'query: ' and 'passage: ' prefixes and normalized vectors.
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: '{model_name}'...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = config.EMBEDDING_DIM
        logger.info(f"Embedding model '{model_name}' loaded successfully (dim={self.dim}).")

    def encode_queries(
        self, queries: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """
        Encodes one or more queries with mandatory 'query: ' prefix.
        """
        if isinstance(queries, str):
            queries = [queries]
        prefixed = [f"{config.QUERY_PREFIX}{q.strip()}" for q in queries]
        vectors = self.model.encode(
            prefixed,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_passages(
        self, passages: Union[str, List[str]], batch_size: int = 64, normalize: bool = True
    ) -> np.ndarray:
        """
        Encodes one or more passages with mandatory 'passage: ' prefix.
        """
        if isinstance(passages, str):
            passages = [passages]
        prefixed = [f"{config.PASSAGE_PREFIX}{p.strip()}" for p in passages]
        vectors = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=(len(passages) > 200),
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encodes consecutive sentences for semantic distance analysis.
        """
        return self.encode_passages(sentences, normalize=True)


def get_embedder() -> MultilingualE5Embedder:
    """
    Get or initialize the global singleton embedder instance.
    """
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = MultilingualE5Embedder()
    return _EMBEDDER_INSTANCE
