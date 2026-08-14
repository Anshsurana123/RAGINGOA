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
# Covers profanity, hate speech, self-harm, violent extremism, weapons, and jailbreak attacks
UNSAFE_PATTERNS = [
    # Jailbreak / Prompt Injection patterns
    r"(?i)\b(ignore\s+(all\s+)?(previous\s+)?(instructions|rules|prompts|directions))\b",
    r"(?i)\b(system\s*prompt|override\s*safety|bypass\s*filter|DAN\s*mode|jailbreak|prompt\s*injection)\b",
    r"(?i)\b(developer\s*mode\s*enabled|unfiltered\s*mode|disregard\s+(all\s+)?guidelines)\b",
    r"(?i)\b(you\s*are\s*now\s*in\s*unrestricted\s*mode|act\s*as\s*an\s*unfiltered\s*ai)\b",
    
    # Violence / Weapons / Explosives / Dangerous materials (flexible phrase and root matching)
    r"(?i)\b(how\s+to\s+)?(make|build|create|craft|assemble|synthesize|manufacture|prepare|construct)\s+(a\s+)?(bomb|explosive|weapon|grenade|ied|molotov|poison|toxin|firearm|chemical\s+weapon|biological\s+weapon|gunpowder|detonator)\b",
    r"(?i)\b(bomb\s*making|explosive\s*recipe|pipe\s*bomb|suicide\s*vest|car\s*bomb|dirty\s*bomb)\b",
    r"(?i)\b(how\s+to\s+)?(kill|murder|attack|assassinate|stab|poison|torture|harm|abuse)\s+(someone|people|a\s+person|anybody|myself|yourself)\b",
    r"(?i)\b(commit\s+suicide|how\s+to\s+hang\s+myself|self[- ]harm|slit\s+(my\s+)?wrists|kill\s+yourself|ways\s+to\s+die)\b",
    
    # Cyberattacks / Illegal Exploits
    r"(?i)\b(how\s+to\s+)?(hack|ddos\s+attack|bypass\s+security|steal\s+passwords|malware\s+source\s+code|ransomware\s+attack|exploit\s+vulnerability)\b",
    
    # Indic Safety Patterns (Hindi / Devanagari)
    r"(?i)(बम\s*(बनाने|बनाना|तैयार)|विस्फोटक|हथियार\s*(बना|तैयार)|ज़हर\s*बना|आत्महत्या|फांसी\s*लगा|कत्ल\s*कर|जान\s*से\s*मार|आतंकवादी\s*हमला|देशद्रोह)",
    
    # Indic Safety Patterns (Tamil)
    r"(?i)(குண்டு\s*(தயாரி|செய்வது)|வெடிகுண்டு|ஆயுதம்\s*செய்|விஷம்\s*தயாரி|தற்கொலை|கொலை\s*செய்|பயங்கரவாத\s*தாக்குதல்)",
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
