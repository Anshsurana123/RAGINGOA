"""
Pre-Retrieval Guardrails:
1. Unsafe / Inappropriate Content Filter (Fast regex and keyword blocklist)
2. Off-Topic Query Filter (Embedding distance to corpus cluster centroids)

Decisions are logged with boolean flags and explicit reason strings.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import config
from retrieval.embed import get_embedder

logger = logging.getLogger(__name__)

# Comprehensive multilingual unsafe / inappropriate keyword and regex patterns
# Covers profanity, hate speech, self-harm, violent extremism, jailbreak attacks
UNSAFE_PATTERNS = [
    # Jailbreak / Prompt Injection patterns
    r"(?i)\b(ignore (all )?previous instructions|system prompt|override safety|bypass filter|DAN mode)\b",
    r"(?i)\b(jailbreak|prompt injection|developer mode enabled)\b",
    
    # Violence / Weapons / Dangerous materials
    r"(?i)\b(how to build a bomb|make a weapon|synthesize poison|create explosive|commit suicide|self harm)\b",
    r"(?i)(बम बनाने का तरीका|हथियार कैसे बनाएं|ज़हर कैसे बनाएं|आत्महत्या)",
    r"(?i)(குண்டு தயாரிப்பது எப்படி|ஆயுதம் செய்வது|தற்கொலை)",
    
    # Profanity / Extreme slurs (representative cross-lingual filter)
    r"(?i)\b(kill yourself|assassinate|terrorist attack|cyberattack government)\b",
    r"(?i)(हत्या करो|आतंकवादी हमला|देशद्रोह)",
    r"(?i)(கொலை செய்|பயங்கரவாத தாக்குதல்)",
]

COMPILED_UNSAFE_REGEXES = [re.compile(p, re.UNICODE) for p in UNSAFE_PATTERNS]


def check_unsafe_content(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check 1: Fast keyword and regex blocklist pass.
    Returns:
        (is_safe, reason)
    """
    if not text:
        return True, None
        
    cleaned = text.strip()
    for rx in COMPILED_UNSAFE_REGEXES:
        match = rx.search(cleaned)
        if match:
            matched_term = match.group(0)
            reason = f"Blocked: unsafe or inappropriate content detected ('{matched_term}')"
            logger.warning(f"Unsafe content guardrail triggered: {reason}")
            return False, reason
            
    return True, None


def check_off_topic_query(
    query_text: str,
    query_vector: np.ndarray,
    centroids: Dict[str, np.ndarray],
    global_centroid: Optional[np.ndarray] = None,
    language_hint: Optional[str] = None,
    threshold: float = config.OFF_TOPIC_DISTANCE_THRESHOLD,
) -> Tuple[bool, float, Optional[str]]:
    """
    Check 2: Computes cosine distance from query vector to corpus centroid.
    If minimum distance > threshold, classify query as off-topic and skip retrieval.
    
    Cosine distance = 1.0 - inner_product(query_vec_norm, centroid_norm)
    Returns:
        (is_on_topic, min_distance, reason)
    """
    if query_vector.ndim == 2:
        q_vec = query_vector[0]
    else:
        q_vec = query_vector
        
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    
    # Check distance to language-specific centroid if available
    distances = []
    
    if language_hint and language_hint.lower() in centroids:
        c_vec = centroids[language_hint.lower()]
        sim = float(np.dot(q_norm, c_vec))
        dist = max(0.0, 1.0 - sim)
        distances.append(dist)
        
    # Also check all language centroids
    for lang, c_vec in centroids.items():
        sim = float(np.dot(q_norm, c_vec))
        dist = max(0.0, 1.0 - sim)
        distances.append(dist)
        
    if global_centroid is not None:
        sim = float(np.dot(q_norm, global_centroid))
        dist = max(0.0, 1.0 - sim)
        distances.append(dist)
        
    if not distances:
        # If no centroids available, default to on-topic
        return True, 0.0, None
        
    min_dist = min(distances)
    
    if min_dist > threshold:
        reason = (
            f"Classified off-topic: query distance to corpus centroid ({min_dist:.4f}) "
            f"exceeds threshold ({threshold:.4f})"
        )
        logger.info(f"Off-topic guardrail triggered: {reason}")
        return False, min_dist, reason
        
    return True, min_dist, None
