"""
Post-Generation Grounding Guardrail.

Performs lexical and semantic overlap scoring between candidate answer and retrieved context passages.
If overlap score falls below threshold, the response is declined with:
"I don't have enough grounded information to answer that."
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import config

logger = logging.getLogger(__name__)

DECLINED_RESPONSE_TEMPLATE = "I don't have enough grounded information to answer that."


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase multilingual words."""
    if not text:
        return []
    return [w for w in re.findall(r'\w+', text.lower(), re.UNICODE) if len(w) > 1]


def compute_lexical_grounding_score(answer: str, context_texts: List[str]) -> float:
    """
    Computes lexical containment / token overlap ratio of answer in context.
    Score = (Count of answer tokens present in context) / (Total answer tokens)
    """
    answer_tokens = set(tokenize_words(answer))
    if not answer_tokens:
        return 0.0
        
    combined_context = " ".join(context_texts).lower()
    context_tokens = set(tokenize_words(combined_context))
    
    if not context_tokens:
        return 0.0
        
    intersection = answer_tokens.intersection(context_tokens)
    lexical_score = len(intersection) / len(answer_tokens)
    return float(lexical_score)


def check_grounding(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    threshold: float = config.GROUNDING_OVERLAP_THRESHOLD,
    embedder=None,
) -> Tuple[bool, float, str, Optional[str]]:
    """
    Evaluates grounding quality of answer against retrieved contexts.
    
    Returns:
        (is_grounded, grounding_score, final_answer, reason)
    """
    if not answer or not answer.strip():
        reason = "Empty answer produced"
        return False, 0.0, DECLINED_RESPONSE_TEMPLATE, reason
        
    if not retrieved_chunks:
        reason = "No retrieved context available to ground answer"
        return False, 0.0, DECLINED_RESPONSE_TEMPLATE, reason
        
    context_texts = [c.get("text", "") for c in retrieved_chunks if c.get("text")]
    if not context_texts:
        reason = "Retrieved chunks contained empty text"
        return False, 0.0, DECLINED_RESPONSE_TEMPLATE, reason
        
    lexical_score = compute_lexical_grounding_score(answer, context_texts)
    
    # Optional semantic score if embedder is available
    semantic_score = 0.0
    if embedder is not None and lexical_score < threshold:
        try:
            ans_vec = embedder.encode_queries(answer)
            ctx_vec = embedder.encode_passages(" ".join(context_texts[:3]))
            sim = float(np.dot(ans_vec[0], ctx_vec[0]))
            semantic_score = max(0.0, min(1.0, sim))
        except Exception:
            semantic_score = 0.0
            
    combined_score = max(lexical_score, semantic_score * 0.8)
    
    if combined_score < threshold:
        reason = (
            f"Grounding check failed: overlap score ({combined_score:.4f}) "
            f"is below threshold ({threshold:.4f})"
        )
        logger.info(f"Grounding guardrail declined answer: {reason}")
        return False, combined_score, DECLINED_RESPONSE_TEMPLATE, reason
        
    reason = f"Grounding check passed (score={combined_score:.4f})"
    return True, combined_score, answer, reason
