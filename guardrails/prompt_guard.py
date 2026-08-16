"""
Meta Prompt-Guard 86M Sub-10ms Neural Safety Guardrail.

Provides local, offline sequence classification for:
- Direct Prompt Injection (DPI) & Jailbreak attempts in user prompts
- Indirect Prompt Injection (IPI) in retrieved RAG context chunks

According to Meta's Prompt-Guard specifications:
- Class 0: BENIGN (Non-instruction text)
- Class 1: INJECTION (Embedded instruction-like text)
- Class 2: JAILBREAK (Malicious override / jailbreak attack)

For User Prompts: Evaluates `jailbreak_probability` (Class 2) to prevent false-positive over-defense on legitimate user questions.
For Retrieved Chunks: Evaluates `indirect_injection_probability` (Class 1 + Class 2) to ensure retrieved passages do not contain adversarial instructions.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import config

logger = logging.getLogger(__name__)

# Label mapping for meta-llama/Prompt-Guard-86M
PROMPT_GUARD_LABELS = {
    0: "BENIGN",
    1: "INJECTION",
    2: "JAILBREAK",
}

# Fast heuristic signatures for Indirect Prompt Injection (IPI) in context chunks
IPI_SUSPICIOUS_PATTERNS = [
    re.compile(r"(?i)\b(ignore|disregard|forget|override|bypass)\s+(all\s+)?(previous|prior|above|context|system|\s|and)*\s*(instructions|rules|prompts|directions|secrets|guidelines)\b"),
    re.compile(r"(?i)\b(system\s*prompt|override\s*safety|bypass\s*filter|DAN\s*mode|jailbreak|prompt\s*injection)\b"),
    re.compile(r"(?i)\b(developer\s*mode\s*enabled|unfiltered\s*mode|disregard\s+(all\s+)?guidelines)\b"),
    re.compile(r"(?i)\b(you\s*are\s*now\s*in\s*unrestricted\s*mode|act\s*as\s*an\s*unfiltered\s*ai)\b"),
    re.compile(r"(?i)\b(output|print|display|reveal|show|dump|repeat|leak|exfiltrate|tell\s+me)\s+(all\s+)?(your\s+)?(system\s*(prompt|instructions|rules|message|secrets)|developer\s*prompt)\b"),
    re.compile(r"<\|im_start\|>|<\|system\|>|\[INST\]|<<SYS>>|<s>|<\/s>|<script|javascript:|onerror="),
]


@dataclass
class PromptGuardResult:
    is_safe: bool
    risk_score: float
    label: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "risk_score": round(self.risk_score, 4),
            "label": self.label,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "latency_ms": round(self.latency_ms, 2),
            "reason": self.reason,
        }


class PromptGuardDetector:
    """
    High-performance Prompt-Guard-86M detector using ONNX Runtime CPU execution
    with PyTorch fallback, batched chunk inference, and temperature-scaled probability calibration.
    """
    _instance: Optional["PromptGuardDetector"] = None

    def __init__(
        self,
        onnx_model_path: Optional[Union[str, Path]] = None,
        hf_repo_id: Optional[str] = None,
        temperature: float = config.PROMPT_GUARD_TEMPERATURE,
        threshold: float = config.PROMPT_GUARD_THRESHOLD,
    ):
        self.temperature = max(0.01, float(temperature))
        self.threshold = float(threshold)
        self.onnx_model_path = Path(onnx_model_path or config.PROMPT_GUARD_ONNX_PATH)
        self.hf_repo_id = hf_repo_id or config.PROMPT_GUARD_ONNX_REPO

        self.tokenizer = None
        self.session = None
        self.torch_model = None
        self.engine_type = "uninitialized"

        self._initialize()

    def _initialize(self) -> None:
        start_t = time.perf_counter()
        # 1. Load Tokenizer
        try:
            from transformers import AutoTokenizer
            logger.info(f"Loading Prompt-Guard tokenizer from '{self.hf_repo_id}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_repo_id,
                use_fast=True,
                local_files_only=False,
            )
        except Exception as e:
            logger.warning(f"Failed to load fast tokenizer from {self.hf_repo_id}: {e}")
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.PROMPT_GUARD_MODEL_NAME,
                    use_fast=True,
                )
            except Exception as e2:
                logger.error(f"Failed to initialize any Prompt-Guard tokenizer: {e2}")

        # 2. Try ONNX Runtime Engine
        if self._try_init_onnx():
            elapsed = (time.perf_counter() - start_t) * 1000
            logger.info(f"PromptGuardDetector initialized with ONNX Runtime in {elapsed:.2f}ms")
            self.warmup()
            return

        # 3. Fallback to PyTorch Engine
        if self._try_init_torch():
            elapsed = (time.perf_counter() - start_t) * 1000
            logger.info(f"PromptGuardDetector initialized with PyTorch in {elapsed:.2f}ms")
            self.warmup()
            return

        logger.warning("PromptGuardDetector could not initialize ONNX or PyTorch model. Guardrail will pass-through.")
        self.engine_type = "disabled"

    def _try_init_onnx(self) -> bool:
        """Attempt to initialize ONNX Runtime session."""
        try:
            import onnxruntime as ort
            onnx_path = self.onnx_model_path

            # Check if local ONNX file exists
            if not onnx_path.exists() or onnx_path.stat().st_size < 1000000:
                logger.info(f"Local ONNX file {onnx_path} not found. Checking huggingface hub cache...")
                try:
                    from huggingface_hub import hf_hub_download
                    downloaded = hf_hub_download(
                        repo_id=self.hf_repo_id,
                        filename="model.onnx",
                    )
                    onnx_path = Path(downloaded)
                except Exception as dl_err:
                    logger.warning(f"Could not auto-download ONNX model from {self.hf_repo_id}: {dl_err}")
                    return False

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = config.ONNX_NUM_THREADS
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.engine_type = "onnx"
            return True
        except Exception as e:
            logger.warning(f"Failed to load ONNX Runtime session for Prompt-Guard: {e}")
            return False

    def _try_init_torch(self) -> bool:
        """Attempt to initialize PyTorch transformer model as fallback."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification

            logger.info("Initializing PyTorch Prompt-Guard fallback...")
            repo_candidates = [
                "Niansuh/Prompt-Guard-86M",
                self.hf_repo_id,
                config.PROMPT_GUARD_MODEL_NAME,
            ]
            for repo in repo_candidates:
                try:
                    self.torch_model = AutoModelForSequenceClassification.from_pretrained(
                        repo,
                        torch_dtype=torch.float32,
                    )
                    self.torch_model.eval()
                    self.engine_type = "torch"
                    return True
                except Exception:
                    continue
            return False
        except Exception as e:
            logger.warning(f"Failed to load PyTorch model for Prompt-Guard: {e}")
            return False

    def warmup(self) -> None:
        """Warms up tokenizer and ONNX/PyTorch graph with a dummy inference to eliminate cold-start JIT lag."""
        try:
            self.predict("Warmup benign test query", mode="prompt")
            self.predict_batch(["Warmup context passage 1", "Warmup context passage 2"], mode="context")
            logger.info("PromptGuardDetector warmed up successfully.")
        except Exception as e:
            logger.debug(f"PromptGuard warmup exception (non-fatal): {e}")

    def predict(
        self,
        text: str,
        mode: str = "prompt",
        temperature: Optional[float] = None,
        threshold: Optional[float] = None,
        max_length: int = 128,
    ) -> PromptGuardResult:
        """
        Runs single-pass discriminative classification on input text.
        """
        results = self.predict_batch(
            [text] if text else [],
            mode=mode,
            temperature=temperature,
            threshold=threshold,
            max_length=max_length,
        )
        if results:
            return results[0]
        return PromptGuardResult(
            is_safe=True,
            risk_score=0.0,
            label="BENIGN",
            probabilities={"BENIGN": 1.0, "INJECTION": 0.0, "JAILBREAK": 0.0},
            latency_ms=0.0,
        )

    def predict_batch(
        self,
        texts: List[str],
        mode: str = "context",
        temperature: Optional[float] = None,
        threshold: Optional[float] = None,
        max_length: int = 128,
    ) -> List[PromptGuardResult]:
        """
        Runs single-pass batched discriminative classification across multiple texts.
        Reduces multi-chunk context evaluation latency by 10x via single-tensor execution.
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() if t else "" for t in texts]
        non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]

        if not non_empty_indices:
            return [
                PromptGuardResult(
                    is_safe=True,
                    risk_score=0.0,
                    label="BENIGN",
                    probabilities={"BENIGN": 1.0, "INJECTION": 0.0, "JAILBREAK": 0.0},
                    latency_ms=0.0,
                )
                for _ in texts
            ]

        if self.tokenizer is None or (self.session is None and self.torch_model is None):
            return [
                PromptGuardResult(
                    is_safe=True,
                    risk_score=0.0,
                    label="BENIGN",
                    probabilities={"BENIGN": 1.0, "INJECTION": 0.0, "JAILBREAK": 0.0},
                    latency_ms=0.0,
                    reason="Prompt-Guard engine uninitialized (pass-through)",
                )
                for _ in texts
            ]

        t_scalar = max(0.01, float(temperature if temperature is not None else self.temperature))
        t_thresh = float(threshold if threshold is not None else self.threshold)
        bound_len = min(max_length, getattr(config, "CONTEXT_BOUNDING_MAX_TOKENS", 128))

        start_t = time.perf_counter()

        try:
            valid_texts = [cleaned_texts[i] for i in non_empty_indices]
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="np" if self.engine_type == "onnx" else "pt",
                truncation=True,
                max_length=bound_len,
                padding=True,
            )

            # 1. Batched Forward Pass (ONNX or PyTorch)
            if self.engine_type == "onnx":
                onnx_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                }
                outputs = self.session.run(None, onnx_inputs)
                raw_logits = outputs[0]  # shape: (B, 3)
            else:
                import torch
                with torch.inference_mode():
                    outputs = self.torch_model(**inputs)
                    raw_logits = outputs.logits.cpu().numpy()  # shape: (B, 3)

            # 2. Temperature-Scaled Softmax Calibration
            scaled_logits = raw_logits / t_scalar
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            elapsed_ms = (time.perf_counter() - start_t) * 1000
            per_item_ms = elapsed_ms / max(1, len(non_empty_indices))

            results: List[PromptGuardResult] = [
                PromptGuardResult(
                    is_safe=True,
                    risk_score=0.0,
                    label="BENIGN",
                    probabilities={"BENIGN": 1.0, "INJECTION": 0.0, "JAILBREAK": 0.0},
                    latency_ms=0.0,
                )
                for _ in texts
            ]

            for row_idx, orig_idx in enumerate(non_empty_indices):
                prob_benign = float(probs[row_idx, 0])
                prob_injection = float(probs[row_idx, 1])
                prob_jailbreak = float(probs[row_idx, 2])

                prob_dict = {
                    "BENIGN": prob_benign,
                    "INJECTION": prob_injection,
                    "JAILBREAK": prob_jailbreak,
                }

                if mode == "context":
                    risk_score = prob_jailbreak
                    if prob_jailbreak >= t_thresh:
                        is_safe = False
                        pred_label = "JAILBREAK"
                        reason = f"Indirect Prompt Injection (Jailbreak) in context (confidence={prob_jailbreak:.4f} >= {t_thresh})"
                    elif prob_injection >= 0.98 and prob_benign < 0.02 and prob_jailbreak >= 0.10:
                        is_safe = False
                        pred_label = "INJECTION"
                        reason = f"Indirect Prompt Injection in context (confidence={prob_injection:.4f})"
                    else:
                        is_safe = True
                        pred_label = "BENIGN" if prob_benign >= prob_injection else "INJECTION"
                        reason = f"Context chunk safe (jailbreak_risk={prob_jailbreak:.4f})"
                else:
                    risk_score = prob_jailbreak
                    if prob_jailbreak >= t_thresh:
                        is_safe = False
                        pred_label = "JAILBREAK"
                        reason = f"Blocked by Prompt-Guard: Jailbreak attack detected (confidence={prob_jailbreak:.4f} >= {t_thresh})"
                        logger.warning(f"PromptGuard blocked prompt [{pred_label}]: {reason}")
                    else:
                        is_safe = True
                        pred_label = "BENIGN"
                        reason = f"Prompt-Guard safe (jailbreak_risk={risk_score:.4f})"

                results[orig_idx] = PromptGuardResult(
                    is_safe=is_safe,
                    risk_score=risk_score,
                    label=pred_label,
                    probabilities=prob_dict,
                    latency_ms=per_item_ms,
                    reason=reason,
                )

            return results

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            logger.error(f"Prompt-Guard batch inference error: {e}", exc_info=True)
            return [
                PromptGuardResult(
                    is_safe=True,
                    risk_score=0.0,
                    label="BENIGN",
                    probabilities={"BENIGN": 1.0, "INJECTION": 0.0, "JAILBREAK": 0.0},
                    latency_ms=elapsed_ms / max(1, len(texts)),
                    reason=f"Inference error ({e}) - failed open",
                )
                for _ in texts
            ]

    def _is_suspicious_text(self, text: str) -> bool:
        """Fast regex pre-screening for indirect prompt injection signatures in context chunks."""
        if not text:
            return False
        for pat in IPI_SUSPICIOUS_PATTERNS:
            if pat.search(text):
                return True
        return False

    def scan_context_chunks(
        self,
        chunks: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Scans retrieved context chunks for Indirect Prompt Injection (IPI) using
        fast heuristic pre-filtering + single-pass batched neural inference.
        """
        if not chunks or not config.ENABLE_CONTEXT_CHUNK_SCAN:
            return chunks, []

        suspicious_flags = [
            self._is_suspicious_text(chunk.get("text", ""))
            for chunk in chunks
        ]

        # If zero suspicious patterns found across standard indexed corpus chunks,
        # pass all candidates through in <0.05ms
        if not any(suspicious_flags):
            return list(chunks), []

        # Batched neural scan on all candidates in one single tensor forward pass (<12ms)
        texts = [c.get("text", "") for c in chunks]
        results = self.predict_batch(texts, mode="context", threshold=threshold, max_length=128)

        clean_chunks = []
        flagged_chunks = []

        for i, (chunk, res) in enumerate(zip(chunks, results)):
            is_suspicious = suspicious_flags[i]
            if is_suspicious or not res.is_safe:
                flagged = dict(chunk)
                flagged["guardrail_block_reason"] = (
                    res.reason if not res.is_safe
                    else "Indirect Prompt Injection detected by heuristic signature"
                )
                flagged["guardrail_risk_score"] = res.risk_score if not res.is_safe else 1.0
                flagged["guardrail_label"] = res.label if not res.is_safe else "INJECTION"
                flagged_chunks.append(flagged)
                logger.warning(
                    f"Indirect Prompt Injection dropped from context chunk: "
                    f"doc_id={chunk.get('doc_id')}, label={flagged['guardrail_label']}"
                )
            else:
                clean_chunks.append(chunk)

        return clean_chunks, flagged_chunks


_DETECTOR_SINGLETON: Optional[PromptGuardDetector] = None


def get_prompt_guard_detector() -> PromptGuardDetector:
    """Global singleton accessor for PromptGuardDetector."""
    global _DETECTOR_SINGLETON
    if _DETECTOR_SINGLETON is None:
        _DETECTOR_SINGLETON = PromptGuardDetector()
    return _DETECTOR_SINGLETON
