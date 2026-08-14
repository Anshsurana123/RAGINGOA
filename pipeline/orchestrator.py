"""
Hand-Rolled Async RAG Pipeline Orchestrator.

Strict Requirements:
- No LangChain, LlamaIndex, or external framework abstractions.
- Pydantic v2 schemas for every stage boundary.
- Full StageTiming latency breakdown and guardrail auditing on every request.
- Extractive-first generation with swappable LLM fallback adapter.
- Dynamic language routing adhering strictly to `config.LANGUAGES`.
"""

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional
import numpy as np

import config
from pipeline.schemas import (
    GuardrailFlags,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    StageTiming,
)
from retrieval.embed import get_embedder
from retrieval.index_faiss import get_index_manager
from retrieval.rerank import rerank_bm25_hybrid
from chunking.hybrid_merge import merge_and_fuse_candidates
from guardrails.pre_retrieval import check_unsafe_content, check_off_topic_query
from guardrails.post_generation import check_grounding, DECLINED_RESPONSE_TEMPLATE
from generation.extractive import generate_extractive
from generation.llm_fallback import get_llm_adapter
from stt.sarvam_client import get_stt_client

logger = logging.getLogger(__name__)


class RAGPipelineOrchestrator:
    """
    Asynchronous RAG Orchestrator managing STT, Guardrails, Multi-Strategy Retrieval,
    BM25 Re-ranking, Extractive/LLM Generation, and Post-Generation Grounding.
    """
    def __init__(self):
        self.embedder = get_embedder()
        self.index_manager = get_index_manager()
        self.stt_client = get_stt_client()
        self.llm_adapter = get_llm_adapter()

    async def execute(self, request: QueryRequest) -> QueryResponse:
        """
        Executes end-to-end stage graph with comprehensive timing instrumentation.
        """
        start_pipeline_t = time.perf_counter()
        timings: List[StageTiming] = []
        guardrails = GuardrailFlags()
        
        raw_query_text = ""
        language = request.language_hint or "hi"
        transcript = ""
        
        # -------------------------------------------------------------
        # STAGE 1: Speech-to-Text (STT) or Text Bypass
        # -------------------------------------------------------------
        stt_start_t = time.perf_counter()
        if request.audio_path:
            try:
                # STT with retry policy
                stt_res = None
                for attempt in range(config.SARVAM_STT_MAX_RETRIES + 1):
                    try:
                        stt_res = self.stt_client.transcribe(
                            audio_path=request.audio_path, language_code=language
                        )
                        break
                    except Exception as stt_err:
                        if attempt == config.SARVAM_STT_MAX_RETRIES:
                            raise stt_err
                        await asyncio.sleep(0.1)
                        
                transcript = stt_res.get("transcript", "")
                raw_query_text = transcript
                language = stt_res.get("language_code", language)
                fallback_used = stt_res.get("is_fallback", False)
                
                timings.append(StageTiming(
                    stage="stt_transcription",
                    ms=round((time.perf_counter() - stt_start_t) * 1000, 2),
                    success=bool(transcript),
                    fallback_used=fallback_used,
                    details=f"Transcribed via Sarvam Saaras v3 ({language})",
                ))
            except Exception as e:
                timings.append(StageTiming(
                    stage="stt_transcription",
                    ms=round((time.perf_counter() - stt_start_t) * 1000, 2),
                    success=False,
                    fallback_used=True,
                    details=f"STT Error: {str(e)}",
                ))
                return self._build_declined_response(
                    query=raw_query_text,
                    transcript=transcript,
                    language=language,
                    reason=f"STT processing failed: {e}",
                    guardrails=guardrails,
                    timings=timings,
                    start_t=start_pipeline_t,
                )
        else:
            raw_query_text = (request.text or "").strip()
            transcript = raw_query_text
            timings.append(StageTiming(
                stage="stt_transcription",
                ms=0.0,
                success=True,
                fallback_used=False,
                details="Text bypass utilized for benchmark/testing",
            ))
            
        if not raw_query_text:
            return self._build_declined_response(
                query="",
                transcript="",
                language=language,
                reason="Empty query received",
                guardrails=guardrails,
                timings=timings,
                start_t=start_pipeline_t,
            )
            
        # -------------------------------------------------------------
        # STAGE 2: Language Routing & Validation
        # -------------------------------------------------------------
        lang_start_t = time.perf_counter()
        target_lang = self._resolve_target_language(raw_query_text, language)
        timings.append(StageTiming(
            stage="language_routing",
            ms=round((time.perf_counter() - lang_start_t) * 1000, 2),
            success=True,
            details=f"Routed query to language '{target_lang}' from config.LANGUAGES",
        ))
        
        # -------------------------------------------------------------
        # STAGE 3: Pre-Retrieval Guardrail 1 - Unsafe Content Check
        # -------------------------------------------------------------
        unsafe_start_t = time.perf_counter()
        is_safe, unsafe_reason = check_unsafe_content(raw_query_text)
        if not is_safe:
            guardrails.unsafe_detected = True
            guardrails.unsafe_reason = unsafe_reason
            timings.append(StageTiming(
                stage="pre_retrieval_safety_guardrail",
                ms=round((time.perf_counter() - unsafe_start_t) * 1000, 2),
                success=False,
                details=unsafe_reason,
            ))
            return self._build_declined_response(
                query=raw_query_text,
                transcript=transcript,
                language=target_lang,
                reason=unsafe_reason or "Unsafe query blocked",
                guardrails=guardrails,
                timings=timings,
                start_t=start_pipeline_t,
            )
            
        timings.append(StageTiming(
            stage="pre_retrieval_safety_guardrail",
            ms=round((time.perf_counter() - unsafe_start_t) * 1000, 2),
            success=True,
            details="Passed keyword and regex blocklist check",
        ))
        
        # -------------------------------------------------------------
        # STAGE 4: Query Embedding & Pre-Retrieval Guardrail 2 - Off-Topic Check
        # -------------------------------------------------------------
        embed_start_t = time.perf_counter()
        # MUST use 'query: ' prefix for retrieval query embedding
        query_vector = self.embedder.encode_queries(raw_query_text)
        embed_ms = round((time.perf_counter() - embed_start_t) * 1000, 2)
        
        offtopic_start_t = time.perf_counter()
        is_on_topic, off_topic_dist, off_topic_reason = check_off_topic_query(
            query_text=raw_query_text,
            query_vector=query_vector,
            centroids=self.index_manager.centroids,
            global_centroid=self.index_manager.global_centroid,
            language_hint=target_lang,
        )
        
        guardrails.off_topic_distance = round(off_topic_dist, 4)
        if not is_on_topic:
            guardrails.off_topic_detected = True
            guardrails.off_topic_reason = off_topic_reason
            timings.append(StageTiming(
                stage="query_embedding",
                ms=embed_ms,
                success=True,
                details="Generated normalized query embedding with 'query: ' prefix",
            ))
            timings.append(StageTiming(
                stage="pre_retrieval_topic_guardrail",
                ms=round((time.perf_counter() - offtopic_start_t) * 1000, 2),
                success=False,
                details=off_topic_reason,
            ))
            return self._build_declined_response(
                query=raw_query_text,
                transcript=transcript,
                language=target_lang,
                reason="Query is off-topic relative to indexed knowledge corpus",
                guardrails=guardrails,
                timings=timings,
                start_t=start_pipeline_t,
            )
            
        timings.append(StageTiming(
            stage="query_embedding",
            ms=embed_ms,
            success=True,
            details="Generated normalized query embedding with 'query: ' prefix",
        ))
        timings.append(StageTiming(
            stage="pre_retrieval_topic_guardrail",
            ms=round((time.perf_counter() - offtopic_start_t) * 1000, 2),
            success=True,
            details=f"On-topic (centroid cosine distance: {off_topic_dist:.4f})",
        ))
        
        # -------------------------------------------------------------
        # STAGE 5: Multi-Strategy FAISS Retrieval & Cross-Lingual Federation
        # -------------------------------------------------------------
        retrieval_start_t = time.perf_counter()
        strategy_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # When cross_lingual is enabled, search across all indexed language partitions
        search_lang = None if request.cross_lingual else target_lang
        
        # Execute search over passage-native and semantic-longdoc indexes
        for strat_name, strat_idx in self.index_manager.indexes.items():
            results = strat_idx.search(
                query_vec=query_vector,
                target_lang=search_lang,
                top_k=config.FAISS_TOP_K,
            )
            strategy_results[strat_name] = results
            
        merged_candidates = merge_and_fuse_candidates(strategy_results)
        
        if not merged_candidates:
            timings.append(StageTiming(
                stage="vector_retrieval_and_merge",
                ms=round((time.perf_counter() - retrieval_start_t) * 1000, 2),
                success=False,
                fallback_used=True,
                details="No matching passages found in FAISS indexes",
            ))
            return self._build_declined_response(
                query=raw_query_text,
                transcript=transcript,
                language=target_lang,
                reason="No relevant information found in the indexed corpus.",
                guardrails=guardrails,
                timings=timings,
                start_t=start_pipeline_t,
            )
            
        timings.append(StageTiming(
            stage="vector_retrieval_and_merge",
            ms=round((time.perf_counter() - retrieval_start_t) * 1000, 2),
            success=True,
            details=f"Retrieved {len(merged_candidates)} candidate chunks (cross-lingual={request.cross_lingual})",
        ))
        
        # -------------------------------------------------------------
        # STAGE 6: BM25-Hybrid Re-ranking
        # -------------------------------------------------------------
        rerank_start_t = time.perf_counter()
        candidate_dicts = [c.to_dict() for c in merged_candidates]
        reranked_chunks = rerank_bm25_hybrid(
            query_text=raw_query_text,
            candidates=candidate_dicts,
            bm25_weight=config.HYBRID_BM25_WEIGHT,
            top_k=config.RERANK_TOP_K,
        )
        
        timings.append(StageTiming(
            stage="bm25_hybrid_reranking",
            ms=round((time.perf_counter() - rerank_start_t) * 1000, 2),
            success=True,
            details=f"BM25 hybrid score fusion on top-{len(reranked_chunks)} candidates",
        ))
        
        # Calculate isolated retrieval stage latency (embedding + FAISS search + rerank)
        retrieval_ms = round(
            (embed_ms + (time.perf_counter() - retrieval_start_t) * 1000), 2
        )
        
        # -------------------------------------------------------------
        # STAGE 7: Grounded Generation (Extractive / Cross-Lingual Synthesis)
        # -------------------------------------------------------------
        gen_start_t = time.perf_counter()
        
        # Check if retrieved evidence contains cross-lingual sources relative to target_lang
        top_languages = [c.get("source_lang", "").lower() for c in reranked_chunks[:3]]
        has_cross_lingual_evidence = any(l != target_lang.lower() for l in top_languages if l)
        
        if has_cross_lingual_evidence and config.LLM_API_KEY:
            # Multi-source cross-lingual compilation & translation
            context_blocks = []
            for i, c in enumerate(reranked_chunks[:5]):
                lang_code = c.get("source_lang", "UNK").upper()
                strat = c.get("chunk_strategy", "")
                context_blocks.append(f"[{lang_code} Source #{i+1} ({strat})]:\n{c.get('text', '')}")
            compiled_context = "\n\n".join(context_blocks)
            
            candidate_answer = self.llm_adapter.generate(
                prompt=raw_query_text,
                context=compiled_context,
                target_lang=target_lang,
            )
            answer_source = "cross_lingual_synthesis"
            gen_details = f"Cross-lingual multi-source synthesis into '{target_lang}' ({len(set(top_languages))} languages combined)"
        else:
            extractive_res = generate_extractive(raw_query_text, reranked_chunks)
            candidate_answer = extractive_res["answer"]
            answer_source = extractive_res["answer_source"]
            gen_details = "Extractive-first grounded passage extraction (zero LLM overhead)"
        
        timings.append(StageTiming(
            stage="extractive_generation",
            ms=round((time.perf_counter() - gen_start_t) * 1000, 2),
            success=True,
            details=gen_details,
        ))
        
        # -------------------------------------------------------------
        # STAGE 8: Post-Generation Grounding Guardrail
        # -------------------------------------------------------------
        ground_start_t = time.perf_counter()
        is_grounded, ground_score, final_answer, ground_reason = check_grounding(
            answer=candidate_answer,
            retrieved_chunks=reranked_chunks,
            threshold=config.GROUNDING_OVERLAP_THRESHOLD,
            embedder=self.embedder,
        )
        
        guardrails.grounding_passed = is_grounded
        guardrails.grounding_score = round(ground_score, 4)
        guardrails.grounding_reason = ground_reason
        
        if not is_grounded:
            answer_source = "declined"
            
        timings.append(StageTiming(
            stage="post_generation_grounding_guardrail",
            ms=round((time.perf_counter() - ground_start_t) * 1000, 2),
            success=is_grounded,
            details=ground_reason,
        ))
        
        total_ms = round((time.perf_counter() - start_pipeline_t) * 1000, 2)
        
        # Convert reranked chunks to schema
        schema_chunks = [
            RetrievedChunk(
                chunk_id=c.get("chunk_id", ""),
                text=c.get("text", ""),
                source_lang=c.get("source_lang", ""),
                chunk_strategy=c.get("chunk_strategy", ""),
                dense_score=round(float(c.get("dense_score", 0.0)), 4),
                bm25_score=round(float(c.get("bm25_score", 0.0)), 4) if c.get("bm25_score") is not None else None,
                final_score=round(float(c.get("final_score", 0.0)), 4),
                contributing_strategies=c.get("contributing_strategies", []),
                metadata=c.get("metadata", {}),
            )
            for c in reranked_chunks
        ]
        
        return QueryResponse(
            query=raw_query_text,
            transcript=transcript,
            language_detected=target_lang,
            answer=final_answer,
            answer_source=answer_source,
            retrieved_chunks=schema_chunks,
            guardrail_flags=guardrails.to_dict(),
            stage_timings=timings,
            retrieval_ms=retrieval_ms,
            total_ms=total_ms,
        )

    def _resolve_target_language(self, text: str, hint: Optional[str]) -> str:
        """
        Dynamically detects or routes language against config.LANGUAGES.
        1. Checks character script Unicode blocks (Devanagari, Tamil, Bengali, Telugu, etc.).
        2. Detects Latin script (English) when input consists of English letters.
        3. Uses valid hint when script is ambiguous or matching.
        """
        cleaned = text.strip() if text else ""
        
        # 1. First scan for native Indic scripts
        for char in cleaned:
            code = ord(char)
            # Devanagari (Hindi, Marathi, Sanskrit, Nepali)
            if 0x0900 <= code <= 0x097F:
                for cand in ["hi", "mr", "ne", "sa"]:
                    if cand in config.LANGUAGES:
                        return cand
            # Tamil
            elif 0x0B80 <= code <= 0x0BFF:
                if "ta" in config.LANGUAGES:
                    return "ta"
            # Bengali / Assamese
            elif 0x0980 <= code <= 0x09FF:
                for cand in ["bn", "as"]:
                    if cand in config.LANGUAGES:
                        return cand
            # Telugu
            elif 0x0C00 <= code <= 0x0C7F:
                if "te" in config.LANGUAGES:
                    return "te"
            # Kannada
            elif 0x0C80 <= code <= 0x0CFF:
                if "kn" in config.LANGUAGES:
                    return "kn"
            # Gujarati
            elif 0x0A80 <= code <= 0x0AFF:
                if "gu" in config.LANGUAGES:
                    return "gu"
                    
        # 2. Check if the text contains Latin letters (English)
        has_latin = bool(re.search(r"[a-zA-Z]", cleaned))
        if has_latin:
            if "en" in config.LANGUAGES:
                return "en"
                
        # 3. If no script was identifiable, rely on explicit hint if valid
        if hint and hint.lower() in config.LANGUAGES and hint.lower() != "auto":
            return hint.lower()
            
        # 4. Default fallback to 'en' or first language
        if "en" in config.LANGUAGES:
            return "en"
        return config.LANGUAGES[0]

    def _build_declined_response(
        self,
        query: str,
        transcript: str,
        language: str,
        reason: str,
        guardrails: GuardrailFlags,
        timings: List[StageTiming],
        start_t: float,
    ) -> QueryResponse:
        """Helper to construct standard declined response schema."""
        total_ms = round((time.perf_counter() - start_t) * 1000, 2)
        return QueryResponse(
            query=query,
            transcript=transcript,
            language_detected=language,
            answer=DECLINED_RESPONSE_TEMPLATE if "grounded" in reason.lower() else f"Declined: {reason}",
            answer_source="declined",
            retrieved_chunks=[],
            guardrail_flags=guardrails.to_dict(),
            stage_timings=timings,
            retrieval_ms=0.0,
            total_ms=total_ms,
        )


_ORCHESTRATOR_INSTANCE: Optional[RAGPipelineOrchestrator] = None


def get_orchestrator() -> RAGPipelineOrchestrator:
    """Singleton getter for RAGPipelineOrchestrator."""
    global _ORCHESTRATOR_INSTANCE
    if _ORCHESTRATOR_INSTANCE is None:
        _ORCHESTRATOR_INSTANCE = RAGPipelineOrchestrator()
    return _ORCHESTRATOR_INSTANCE
