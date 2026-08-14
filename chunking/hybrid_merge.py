"""
Hybrid Multi-Strategy Merge Coordinator.

Executes parallel retrieval across multiple strategy indexes (passage-native, semantic/longdoc)
and combines candidates using Reciprocal Rank Fusion (RRF) and score normalization.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
import numpy as np


class MergedCandidate:
    """Represents a merged candidate chunk retrieved from one or more strategies."""
    def __init__(
        self,
        chunk_id: str,
        text: str,
        source_lang: str,
        chunk_strategy: str,
        dense_score: float,
        rrf_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source_lang = source_lang
        self.chunk_strategy = chunk_strategy
        self.dense_score = dense_score
        self.rrf_score = rrf_score
        self.metadata = metadata or {}
        self.contributing_strategies: List[str] = [chunk_strategy]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_lang": self.source_lang,
            "chunk_strategy": self.chunk_strategy,
            "dense_score": float(self.dense_score),
            "rrf_score": float(self.rrf_score),
            "contributing_strategies": self.contributing_strategies,
            "metadata": self.metadata,
        }


def merge_and_fuse_candidates(
    strategy_results: Dict[str, List[Dict[str, Any]]],
    rrf_k: int = 60,
    strategy_weights: Optional[Dict[str, float]] = None,
) -> List[MergedCandidate]:
    """
    Combines results from multiple retrieval strategies using Reciprocal Rank Fusion (RRF)
    and deduplicates candidates.
    
    RRF Score: sum_i ( weight_i / (k + rank_i) )
    """
    if strategy_weights is None:
        strategy_weights = {
            "passage_native": 1.0,
            "semantic_longdoc": 0.85,
            "sentence_window": 0.85,
        }
        
    candidate_map: Dict[str, MergedCandidate] = {}
    
    for strategy_name, results in strategy_results.items():
        weight = strategy_weights.get(strategy_name, 1.0)
        
        for rank, item in enumerate(results):
            # Key candidate by chunk_id or text snippet hash to avoid duplicates
            cid = item.get("chunk_id", "")
            text = item.get("text", "")
            if not text:
                continue
                
            dedup_key = cid if cid else hash(text[:100])
            score = float(item.get("score", 0.0))
            rrf_addition = weight / (rrf_k + rank + 1)
            
            if dedup_key not in candidate_map:
                candidate_map[dedup_key] = MergedCandidate(
                    chunk_id=cid,
                    text=text,
                    source_lang=item.get("source_lang", ""),
                    chunk_strategy=item.get("chunk_strategy", strategy_name),
                    dense_score=score,
                    rrf_score=rrf_addition,
                    metadata=item.get("metadata", {}),
                )
            else:
                existing = candidate_map[dedup_key]
                existing.rrf_score += rrf_addition
                if score > existing.dense_score:
                    existing.dense_score = score
                if strategy_name not in existing.contributing_strategies:
                    existing.contributing_strategies.append(strategy_name)
                    
    # Sort candidates by combined RRF score descending
    sorted_candidates = sorted(candidate_map.values(), key=lambda c: c.rrf_score, reverse=True)
    return sorted_candidates


async def parallel_retrieve_and_merge(
    query_text: str,
    target_lang: str,
    retrievers: Dict[str, Callable[[str, str, int], List[Dict[str, Any]]]],
    top_k_per_strategy: int = 15,
    final_top_k: int = 15,
) -> List[MergedCandidate]:
    """
    Executes parallel retrieval across all active strategy indexes and merges candidates.
    """
    loop = asyncio.get_event_loop()
    strategy_results: Dict[str, List[Dict[str, Any]]] = {}
    
    with ThreadPoolExecutor(max_workers=len(retrievers) if retrievers else 1) as executor:
        futures = {}
        for strategy_name, retriever_fn in retrievers.items():
            futures[strategy_name] = loop.run_in_executor(
                executor, retriever_fn, query_text, target_lang, top_k_per_strategy
            )
            
        for strategy_name, fut in futures.items():
            try:
                res = await fut
                strategy_results[strategy_name] = res
            except Exception as e:
                strategy_results[strategy_name] = []
                
    merged = merge_and_fuse_candidates(strategy_results)
    return merged[:final_top_k]
