"""Deterministic fail-safe classifier used when Prompt-Guard inference is unavailable."""
import re
from typing import Dict, Tuple

HIGH_RISK_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:make|build|assemble|create|deploy)\b.{0,70}\b(?:bomb|explosive|ied|pipe bomb)\b",
    r"\b(?:create|write|build|deploy|spread)\b.{0,60}\b(?:malware|ransomware|virus|worm|trojan)\b",
    r"\b(?:steal|harvest|obtain|copy)\b.{0,50}\b(?:credit[- ]?card|card number|cvv|bank account)\b",
    r"\b(?:make|use|mix|administer)\b.{0,60}\b(?:chemical|chloroform|poison|sedative)\b.{0,50}\b(?:unconscious|knock out|incapacitate)\b",
    r"\b(?:manufacture|make|build)\b.{0,50}\b(?:illegal|ghost)\s*(?:firearm|gun|weapon)\b",
    r"\b(?:contaminate|poison|sabotage)\b.{0,60}\b(?:public|municipal|city)?\s*(?:water|food)\s*(?:supply|system)\b",
))
INJECTION_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:ignore|disregard|override|bypass|forget)\b.{0,40}\b(?:instructions|rules|system prompt|safety)\b",
    r"\b(?:jailbreak|DAN mode|unrestricted mode|prompt injection)\b",
))

def evaluate_fail_safe(text: str, suspicious: bool = False) -> Tuple[bool, float, str, Dict[str, float], str]:
    """Classify conservatively after a neural-model failure."""
    value = str(text or "")
    high_risk = any(pattern.search(value) for pattern in HIGH_RISK_PATTERNS)
    injection = suspicious or any(pattern.search(value) for pattern in INJECTION_PATTERNS)
    blocked = high_risk or injection
    label = "JAILBREAK" if high_risk else "INJECTION" if injection else "BENIGN"
    risk = 1.0 if blocked else 0.0
    reason = (
        "Prompt-Guard inference failed; deterministic fail-closed heuristic blocked high-risk content" if high_risk else
        "Prompt-Guard inference failed; deterministic injection heuristic blocked the input" if injection else
        "Prompt-Guard inference failed; deterministic heuristic found no high-risk signature"
    )
    probabilities = {"BENIGN": 1.0 - risk, "INJECTION": risk if injection else 0.0, "JAILBREAK": risk if high_risk else 0.0}
    return not blocked, risk, label, probabilities, reason
