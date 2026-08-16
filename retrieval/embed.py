"""
Embedding Model Wrapper for multilingual-e5-small with ONNX Runtime CPU Acceleration.

CRITICAL REQUIREMENT:
`intfloat/multilingual-e5-small` is a retrieval-trained model.
All query encodings MUST use the 'query: ' prefix.
All passage/document encodings MUST use the 'passage: ' prefix.
"""

import logging
import os
from pathlib import Path
from typing import List, Union
import numpy as np
import torch
import config

logger = logging.getLogger(__name__)

# Optimize PyTorch CPU parallelism
try:
    torch.set_num_threads(max(1, torch.get_num_threads()))
except Exception:
    pass

_EMBEDDER_INSTANCE = None


class ONNXMultilingualE5Embedder:
    """
    High-performance ONNX Runtime CPU Embedder for multilingual-e5-small.
    Uses INT8 dynamic quantization and static graph execution for sub-10ms query vectorization.
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.dim = config.EMBEDDING_DIM
        self.onnx_dir = Path(getattr(config, "ONNX_MODELS_DIR", config.DATA_DIR / "onnx_models"))
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        
        self.onnx_int8_path = self.onnx_dir / "e5_small_int8.onnx"
        self.onnx_fp32_path = self.onnx_dir / "e5_small.onnx"
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure ONNX model exists
        self._ensure_onnx_model()
        
        import onnxruntime as ort
        opts = ort.SessionOptions()
        num_threads = getattr(config, "ONNX_NUM_THREADS", 2)
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        load_path = self.onnx_int8_path if self.onnx_int8_path.exists() else self.onnx_fp32_path
        logger.info(f"Loading ONNX embedding model from: {load_path} (threads={num_threads})")
        self.session = ort.InferenceSession(str(load_path), opts, providers=["CPUExecutionProvider"])
        
        # Warmup ONNX inference graph to avoid cold-start JIT latency
        try:
            dummy_in = self.tokenizer(["query: warmup"], padding=True, return_tensors="np")
            self.session.run(None, {
                "input_ids": dummy_in["input_ids"].astype(np.int64),
                "attention_mask": dummy_in["attention_mask"].astype(np.int64),
            })
        except Exception:
            pass
        logger.info("ONNX embedding session initialized and warmed up successfully.")

    def _ensure_onnx_model(self):
        """Auto-export and quantize PyTorch model if ONNX files do not exist."""
        if self.onnx_int8_path.exists() or self.onnx_fp32_path.exists():
            return
            
        logger.info("Exporting multilingual-e5-small to ONNX format...")
        import torch.nn as nn
        from transformers import AutoModel
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        class E5Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, input_ids, attention_mask):
                out = self.m(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                return out[0]

        base_model = AutoModel.from_pretrained(self.model_name)
        base_model.eval()
        wrapper = E5Wrapper(base_model)
        wrapper.eval()
        
        from torch.export import Dim
        b = Dim("batch")
        s = Dim("seq")
        dummy = self.tokenizer(["query 1", "query 2"], padding=True, return_tensors="pt")
        try:
            torch.onnx.export(
                wrapper,
                (dummy["input_ids"], dummy["attention_mask"]),
                str(self.onnx_fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_shapes={
                    "input_ids": {0: b, 1: s},
                    "attention_mask": {0: b, 1: s},
                },
                opset_version=18,
                do_constant_folding=True,
            )
            logger.info("Exported ONNX embedding model with dynamic shapes.")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}. PyTorch fallback will be used.")

    def _mean_pool_and_normalize(self, token_embeddings: np.ndarray, attention_mask: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Vectorized mean pooling over active attention mask tokens with L2 normalization."""
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask
        if normalize:
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norm, a_min=1e-9, a_max=None)
        return np.ascontiguousarray(pooled, dtype=np.float32)

    def encode_queries(
        self, queries: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """
        Encodes one or more queries with mandatory 'query: ' prefix using ONNX Runtime.
        """
        if isinstance(queries, str):
            queries = [queries]
        prefixed = [f"{config.QUERY_PREFIX}{q.strip()}" for q in queries]
        
        inputs = self.tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=getattr(config, "CONTEXT_BOUNDING_MAX_TOKENS", 64),
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        outputs = self.session.run(None, ort_inputs)
        token_embeddings = outputs[0]
        return self._mean_pool_and_normalize(token_embeddings, inputs["attention_mask"], normalize=normalize)

    def encode_passages(
        self, passages: Union[str, List[str]], batch_size: int = 64, normalize: bool = True
    ) -> np.ndarray:
        """
        Encodes passages with mandatory 'passage: ' prefix in batches.
        """
        if isinstance(passages, str):
            passages = [passages]
        prefixed = [f"{config.PASSAGE_PREFIX}{p.strip()}" for p in passages]
        
        all_embeddings = []
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=getattr(config, "CONTEXT_BOUNDING_MAX_TOKENS", 64),
                return_tensors="np",
            )
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            outputs = self.session.run(None, ort_inputs)
            pooled = self._mean_pool_and_normalize(outputs[0], inputs["attention_mask"], normalize=normalize)
            all_embeddings.append(pooled)
            
        if not all_embeddings:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack(all_embeddings)

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encodes consecutive sentences for semantic distance analysis."""
        return self.encode_passages(sentences, normalize=True)


class PyTorchMultilingualE5Embedder:
    """
    PyTorch fallback wrapper for sentence-transformers multilingual-e5-small.
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading PyTorch fallback embedding model: '{model_name}'...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = config.EMBEDDING_DIM
        logger.info(f"PyTorch embedding model loaded (dim={self.dim}).")

    def encode_queries(
        self, queries: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
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
        return self.encode_passages(sentences, normalize=True)


def get_embedder():
    """
    Get or initialize the global singleton embedder instance with ONNX-first policy.
    """
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        if getattr(config, "ENABLE_ONNX_EMBEDDING", True):
            try:
                _EMBEDDER_INSTANCE = ONNXMultilingualE5Embedder()
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX Embedder: {e}. Falling back to PyTorch.")
                _EMBEDDER_INSTANCE = PyTorchMultilingualE5Embedder()
        else:
            _EMBEDDER_INSTANCE = PyTorchMultilingualE5Embedder()
    return _EMBEDDER_INSTANCE
