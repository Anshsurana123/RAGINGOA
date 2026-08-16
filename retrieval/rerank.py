import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
import config

logger = logging.getLogger(__name__)

_CROSS_ENCODER_INSTANCE = None


class ONNXCrossEncoderRanker:
    """
    ONNX Runtime accelerated CrossEncoder for ms-marco-MiniLM-L-6-v2.
    Computes cross-attention scoring on (query, passage) pairs in sub-15ms on CPU.
    """
    def __init__(self, model_name: str = config.CROSS_ENCODER_MODEL_NAME):
        self.model_name = model_name
        self.onnx_dir = Path(getattr(config, "ONNX_MODELS_DIR", config.DATA_DIR / "onnx_models"))
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_path = self.onnx_dir / "ce_minilm.onnx"
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure ONNX model exists
        self._ensure_onnx_model()
        
        import onnxruntime as ort
        opts = ort.SessionOptions()
        num_threads = getattr(config, "ONNX_NUM_THREADS", 4)
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        logger.info(f"Loading ONNX CrossEncoder from: {self.onnx_path} (threads={num_threads})")
        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])
        
        # Warmup ONNX inference graph to avoid cold-start JIT latency
        try:
            dummy_in = self.tokenizer(["warmup query"], ["warmup passage"], padding=True, return_tensors="np")
            self.session.run(None, {
                "input_ids": dummy_in["input_ids"].astype(np.int64),
                "attention_mask": dummy_in["attention_mask"].astype(np.int64),
                "token_type_ids": dummy_in["token_type_ids"].astype(np.int64),
            })
        except Exception:
            pass
        logger.info("ONNX CrossEncoder session initialized and warmed up successfully.")

    def _ensure_onnx_model(self):
        """Auto-export CrossEncoder to ONNX if not already present."""
        if self.onnx_path.exists():
            return
            
        logger.info("Exporting CrossEncoder to ONNX format...")
        import torch.nn as nn
        from transformers import AutoModelForSequenceClassification
        from torch.export import Dim
        
        class CEWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, input_ids, attention_mask, token_type_ids):
                return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)[0]

        base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        base_model.eval()
        wrapper = CEWrapper(base_model)
        wrapper.eval()
        
        b = Dim("batch")
        s = Dim("seq")
        dummy = self.tokenizer(["query 1", "query 2"], ["passage 1", "passage 2"], padding=True, return_tensors="pt")
        
        try:
            torch.onnx.export(
                wrapper,
                (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
                str(self.onnx_path),
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["logits"],
                dynamic_shapes={
                    "input_ids": {0: b, 1: s},
                    "attention_mask": {0: b, 1: s},
                    "token_type_ids": {0: b, 1: s},
                },
                opset_version=18,
                do_constant_folding=True,
            )
            logger.info("Exported ONNX CrossEncoder model.")
        except Exception as e:
            logger.warning(f"ONNX CrossEncoder export failed: {e}. PyTorch fallback will be used.")

    def score_pairs(self, query: str, passages: List[str], max_length: int = 80) -> np.ndarray:
        """
        Scores (query, passage) pairs using ONNX Runtime with Context Bounding.
        """
        if not passages:
            return np.array([], dtype=np.float32)
            
        bound_len = min(max_length, getattr(config, "CONTEXT_BOUNDING_MAX_TOKENS", 80))
        pairs = [[query, p[:200]] for p in passages]
        
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=bound_len,
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64),
        }
        logits = self.session.run(None, ort_inputs)[0]
        return np.asarray(logits.flatten(), dtype=np.float32)


class PyTorchCrossEncoderRanker:
    """
    PyTorch fallback wrapper for sentence-transformers CrossEncoder.
    """
    def __init__(self, model_name: str = config.CROSS_ENCODER_MODEL_NAME):
        load_path = model_name
        local_cache = getattr(config, "CROSS_ENCODER_LOCAL_CACHE", None)
        if local_cache and Path(local_cache).exists():
            load_path = str(local_cache)
            logger.info(f"Loading PyTorch CrossEncoder from local cache: {load_path}")
        else:
            logger.info(f"Loading PyTorch CrossEncoder from model name: {load_path}")
            
        try:
            self.model = CrossEncoder(load_path)
            logger.info("PyTorch CrossEncoder loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder from '{load_path}': {e}. Falling back to default '{model_name}'.")
            self.model = CrossEncoder(model_name)

    def score_pairs(self, query: str, passages: List[str], max_length: int = 96) -> np.ndarray:
        if not passages:
            return np.array([], dtype=np.float32)
        bound_len = min(max_length, getattr(config, "CONTEXT_BOUNDING_MAX_TOKENS", 128))
        pairs = [[query, p[:220]] for p in passages]
        with torch.inference_mode():
            scores = self.model.predict(
                pairs,
                show_progress_bar=False,
                batch_size=len(pairs),
                max_length=bound_len,
            )
        return np.asarray(scores, dtype=np.float32)


def get_cross_encoder():
    """
    Get or initialize the global singleton cross-encoder instance with ONNX-first policy.
    """
    global _CROSS_ENCODER_INSTANCE
    if _CROSS_ENCODER_INSTANCE is None:
        if getattr(config, "ENABLE_ONNX_CROSS_ENCODER", True):
            try:
                _CROSS_ENCODER_INSTANCE = ONNXCrossEncoderRanker()
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX CrossEncoder: {e}. Falling back to PyTorch.")
                _CROSS_ENCODER_INSTANCE = PyTorchCrossEncoderRanker()
        else:
            _CROSS_ENCODER_INSTANCE = PyTorchCrossEncoderRanker()
    return _CROSS_ENCODER_INSTANCE


STOPWORDS = {
    "what", "is", "the", "of", "in", "and", "how", "do", "does", "are", "for", "to", "a", "an",
    "why", "who", "which", "can", "with", "from", "by", "on", "as", "between", "explain",
    "difference", "best", "make", "made", "rule", "rules", "basic", "basics",
    "way", "ways", "method", "methods", "tell", "give", "show", "know", "anyone", "someone",
    "recipe", "baking", "baker",
    "क्या", "है", "हैं", "के", "की", "का", "में", "और", "से", "होता", "होती", "होते",
    "कैसे", "क्यों", "किए", "किया", "जाता", "जाती", "गया", "गई", "को", "पर", "लिए", "एक", "या",
    "बारे", "नियम", "विधि", "तरीका", "बुनियादी", "आसान", "सबसे", "बनाने", "बताएं",
    "என்ன", "எவ்வாறு", "ஏன்", "மற்றும்", "ஒரு", "ஆகும்", "உள்ளது", "என்பது", "யாவை",
    "பற்றி", "செய்கிறது", "செய்யப்படுகிறது", "எப்படி", "செய்வது", "வழிமுறை", "வழிமுறைகள்",
    "அடிப்படை", "விதிகள்"
}

PUNCT_REGEX = re.compile(r'[\s!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥]+')


def detect_script(text: str) -> str:
    """Fast Unicode script identifier for adaptive multilingual routing."""
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "Deva"
        if 0x0980 <= code <= 0x09FF:
            return "Beng"
        if 0x0A00 <= code <= 0x0A7F:
            return "Guru"
        if 0x0A80 <= code <= 0x0AFF:
            return "Gujr"
        if 0x0B00 <= code <= 0x0B7F:
            return "Orya"
        if 0x0B80 <= code <= 0x0BFF:
            return "Taml"
        if 0x0C00 <= code <= 0x0C7F:
            return "Telu"
        if 0x0C80 <= code <= 0x0CFF:
            return "Knda"
        if 0x0D00 <= code <= 0x0D7F:
            return "Mlym"
        if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFF:
            return "Arab"
    if bool(re.search(r"[a-zA-Z]", text)):
        return "Latn"
    return "Latn"


def tokenize_indic(text: str) -> List[str]:
    """Clean and split multilingual and Indic text without breaking ligatures."""
    if not text:
        return []
    clean = PUNCT_REGEX.sub(' ', text.lower()).strip()
    return [w for w in clean.split() if len(w) > 1]


def tokenize_for_bm25(text: str) -> List[str]:
    """Multilingual tokenization for BM25 scoring."""
    return tokenize_indic(text)


def rerank_bm25_hybrid(
    query_text: str,
    candidates: List[Dict[str, Any]],
    bm25_weight: float = config.HYBRID_BM25_WEIGHT,
    top_k: int = config.RERANK_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Adaptive Script-Aware Hybrid re-ranking:
    - Same-Script (e.g. Hindi -> Hindi, English -> English): Blends BM25 lexical precision with dense vector score.
    - Cross-Script (e.g. English -> Hindi, Hindi -> English): Dynamically bypasses BM25 to prevent false lexical penalties.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        c = candidates[0].copy()
        c["final_score"] = float(c.get("score", c.get("dense_score", 1.0)))
        c["bm25_score"] = 1.0
        c["confidence"] = float(c.get("dense_score", 0.9))
        return [c]
        
    query_script = detect_script(query_text)
    query_tokens = tokenize_indic(query_text)
    if not query_tokens:
        query_tokens = query_text.lower().split()
        
    # Build BM25 corpus from candidate texts
    corpus_tokens = [tokenize_indic(c.get("text", "")) for c in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    raw_bm25_scores = bm25.get_scores(query_tokens)
    
    # Normalize BM25 scores to [0, 1]
    max_bm25 = float(np.max(raw_bm25_scores)) if len(raw_bm25_scores) > 0 else 0.0
    min_bm25 = float(np.min(raw_bm25_scores)) if len(raw_bm25_scores) > 0 else 0.0
    bm25_range = max_bm25 - min_bm25
    
    # Extract dense scores
    raw_dense_scores = [float(c.get("score", c.get("dense_score", 0.0))) for c in candidates]
    max_dense = max(raw_dense_scores) if raw_dense_scores else 1.0
    min_dense = min(raw_dense_scores) if raw_dense_scores else 0.0
    dense_range = max_dense - min_dense
    
    q_words = [w for w in query_tokens if w not in STOPWORDS and len(w) > 2]
    if not q_words:
        q_words = query_tokens
        
    reranked = []
    for idx, cand in enumerate(candidates):
        cand_text = cand.get("text", "")
        cand_script = detect_script(cand_text)
        is_cross_script = (query_script != cand_script)
        
        # Normalized BM25
        if bm25_range > 1e-6:
            norm_bm25 = (raw_bm25_scores[idx] - min_bm25) / bm25_range
        else:
            norm_bm25 = 1.0 if max_bm25 > 0 else 0.0
            
        # Normalized Dense
        if dense_range > 1e-6:
            norm_dense = (raw_dense_scores[idx] - min_dense) / dense_range
        else:
            norm_dense = max(0.0, min(1.0, raw_dense_scores[idx]))
            
        # Adaptive Hybrid Combination:
        # For cross-script matches, avoid penalizing with a 0 BM25 score.
        if is_cross_script:
            final_score = norm_dense
        else:
            final_score = (1.0 - bm25_weight) * norm_dense + bm25_weight * norm_bm25
            
        # Absolute confidence computation
        dense_val = float(raw_dense_scores[idx])
        
        if is_cross_script:
            # Cross-script candidate relies directly on dense semantic alignment
            confidence = dense_val
        else:
            p_tokens = set(tokenize_indic(cand_text))
            p_clean_text = " ".join(tokenize_indic(cand_text))
            
            matched = 0
            for qw in q_words:
                stem = qw[:5] if len(qw) > 5 else qw
                if qw in p_tokens or (len(qw) > 4 and stem in p_clean_text) or any(pt.startswith(stem) for pt in p_tokens if len(stem) > 3):
                    matched += 1
                    
            overlap = matched / len(q_words) if q_words else 0.0
            
            if matched > 0:
                confidence = (dense_val * 0.70) + (0.30 * min(1.0, overlap + 0.2))
            else:
                confidence = dense_val * 0.75
        
        item = cand.copy()
        item["dense_score"] = dense_val
        item["bm25_score"] = float(raw_bm25_scores[idx])
        item["final_score"] = float(final_score)
        item["confidence"] = round(float(confidence), 4)
        item["is_cross_script"] = is_cross_script
        reranked.append(item)
        
    # Sort by calibrated confidence descending, breaking ties with final_score
    reranked = sorted(reranked, key=lambda x: (x["confidence"], x["final_score"]), reverse=True)
    return reranked[:top_k]


def rerank_cross_encoder(
    query_text: str,
    candidates: List[Dict[str, Any]],
    top_k: int = config.CROSS_ENCODER_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Applies deep cross-attention re-ranking over candidate passages.
    Attaches `cross_encoder_score` and recalibrates ranking.
    """
    if not candidates:
        return []
        
    try:
        ranker = get_cross_encoder()
        passages = [c.get("text", "") for c in candidates[:top_k]]
        
        ce_scores = ranker.score_pairs(query_text, passages)
        
        scored_candidates = []
        for idx, cand in enumerate(candidates[:top_k]):
            c = cand.copy()
            raw_ce = float(ce_scores[idx]) if idx < len(ce_scores) else -10.0
            c["cross_encoder_score"] = round(raw_ce, 4)
            
            # Sigmoid normalize cross-encoder logits (typically -11 to +11 for ms-marco-MiniLM)
            sig_ce = 1.0 / (1.0 + np.exp(-raw_ce))
            c["ce_prob"] = round(float(sig_ce), 4)
            
            # Recalibrate composite confidence blending cross-encoder with dense score
            dense_val = float(c.get("dense_score", 0.5))
            c["confidence"] = round(0.70 * sig_ce + 0.30 * dense_val, 4)
            c["final_score"] = round(raw_ce, 4)
            scored_candidates.append(c)
            
        # Append remaining candidates beyond top_k if any
        if len(candidates) > top_k:
            for cand in candidates[top_k:]:
                c = cand.copy()
                c["cross_encoder_score"] = -10.0
                c["ce_prob"] = 0.0
                scored_candidates.append(c)
                
        # Sort by cross-encoder score descending
        scored_candidates = sorted(scored_candidates, key=lambda x: x.get("cross_encoder_score", -10.0), reverse=True)
        return scored_candidates
    except Exception as e:
        logger.warning(f"Cross-encoder reranking failed: {e}. Falling back to BM25-hybrid ranking.")
        return candidates

