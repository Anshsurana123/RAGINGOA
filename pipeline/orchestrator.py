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
from retrieval.rerank import rerank_bm25_hybrid, rerank_cross_encoder
from chunking.hybrid_merge import merge_and_fuse_candidates
from guardrails.pre_retrieval import check_unsafe_content, check_off_topic_query
from guardrails.post_generation import check_grounding, DECLINED_RESPONSE_TEMPLATE
from generation.extractive import generate_extractive
from generation.answer_cache import get_answer_cache
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
        self.answer_cache = get_answer_cache()

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
        # Fast regex blocklist + Groq Neural Safety Model (strictly gated on ALLOW_NETWORK_CALLS_IN_PIPELINE)
        enable_neural_safety = bool(config.ALLOW_NETWORK_CALLS_IN_PIPELINE and config.LLM_API_KEY)
        is_safe, unsafe_reason = check_unsafe_content(raw_query_text, enable_neural=enable_neural_safety)
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
            details="Passed keyword, regex, and safety checks",
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
        # STAGE 4.5: Two-Tier Semantic Answer Cache Lookup (<1ms Fast Path)
        # -------------------------------------------------------------
        cache_start_t = time.perf_counter()
        cached_result = self.answer_cache.lookup(
            raw_query_text,
            query_vector,
            threshold=config.SEMANTIC_ANSWER_CACHE_THRESHOLD,
        )
        if cached_result:
            cache_ms = round((time.perf_counter() - cache_start_t) * 1000, 2)
            timings.append(StageTiming(
                stage="semantic_answer_cache",
                ms=cache_ms,
                success=True,
                details=f"Cache HIT ({cached_result['answer_source']}, sim={cached_result['similarity']:.4f})",
            ))
            timings.append(StageTiming(
                stage="vector_retrieval_and_merge",
                ms=0.0,
                success=True,
                details="Bypassed due to semantic cache hit",
            ))
            timings.append(StageTiming(
                stage="reranking",
                ms=0.0,
                success=True,
                details="Bypassed due to semantic cache hit",
            ))
            timings.append(StageTiming(
                stage="generation",
                ms=0.0,
                success=True,
                details=f"Instant cached answer ({cached_result['answer_source']})",
            ))
            timings.append(StageTiming(
                stage="post_generation_grounding_guardrail",
                ms=0.0,
                success=True,
                details="Cached verified ground-truth",
            ))
            total_latency_ms = round((time.perf_counter() - start_pipeline_t) * 1000, 2)
            return QueryResponse(
                query=raw_query_text,
                transcript=transcript,
                language_detected=target_lang,
                answer=cached_result["answer"],
                answer_source=cached_result["answer_source"],
                retrieved_chunks=[],
                guardrail_flags=guardrails.to_dict(),
                stage_timings=timings,
                retrieval_ms=cache_ms,
                total_ms=total_latency_ms,
            )
        
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
        # STAGE 6: BM25-Hybrid & Cross-Encoder Re-ranking & Disqualification Guardrail
        # -------------------------------------------------------------
        rerank_start_t = time.perf_counter()
        candidate_dicts = [c.to_dict() for c in merged_candidates]
        bm25_reranked = rerank_bm25_hybrid(
            query_text=raw_query_text,
            candidates=candidate_dicts,
            bm25_weight=config.HYBRID_BM25_WEIGHT,
            top_k=config.RERANK_TOP_K,
        )
        
        # Cross-Encoder deep cross-attention re-ranking on top candidates
        if getattr(config, "ENABLE_CROSS_ENCODER", True):
            reranked_chunks = rerank_cross_encoder(
                query_text=raw_query_text,
                candidates=bm25_reranked,
                top_k=config.CROSS_ENCODER_TOP_K,
            )
        else:
            reranked_chunks = bm25_reranked
            
        # Post-retrieval confidence & cross-encoder relevance check
        top_chunk = reranked_chunks[0] if reranked_chunks else None
        top_conf = float(top_chunk.get("confidence", top_chunk.get("dense_score", 0.0))) if top_chunk else 0.0
        top_ce_score = float(top_chunk.get("cross_encoder_score", 0.0)) if top_chunk and "cross_encoder_score" in top_chunk else None
        
        is_disqualified = False
        disqualify_reason = ""
        
        if not reranked_chunks:
            is_disqualified = True
            disqualify_reason = "Declined: no candidate passages available"
        elif top_ce_score is not None and top_ce_score < getattr(config, "CROSS_ENCODER_THRESHOLD", -0.5):
            is_disqualified = True
            disqualify_reason = (
                f"Declined: top cross-encoder relevance ({top_ce_score:.4f}) below threshold "
                f"({config.CROSS_ENCODER_THRESHOLD:.4f})"
            )
        elif top_conf < config.MIN_CONFIDENT_MATCH_SCORE:
            is_disqualified = True
            disqualify_reason = (
                f"Declined: top retrieval confidence ({top_conf:.4f}) below calibrated "
                f"minimum threshold ({config.MIN_CONFIDENT_MATCH_SCORE:.4f})"
            )
        
        if is_disqualified:
            guardrails.off_topic_detected = True
            guardrails.off_topic_reason = disqualify_reason
            timings.append(StageTiming(
                stage="bm25_cross_encoder_reranking",
                ms=round((time.perf_counter() - rerank_start_t) * 1000, 2),
                success=False,
                details=guardrails.off_topic_reason,
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
            
        ce_info = f", CE={top_ce_score:.4f}" if top_ce_score is not None else ""
        timings.append(StageTiming(
            stage="bm25_cross_encoder_reranking",
            ms=round((time.perf_counter() - rerank_start_t) * 1000, 2),
            success=True,
            details=f"BM25 + Cross-Encoder hybrid ranking on top-{len(reranked_chunks)} candidates (confidence={top_conf:.4f}{ce_info})",
        ))

        
        # Calculate isolated retrieval stage latency (embedding + FAISS search + rerank)
        retrieval_ms = round(
            (embed_ms + (time.perf_counter() - retrieval_start_t) * 1000), 2
        )
        
        # -------------------------------------------------------------
        # STAGE 7: Grounded Generation (Extractive-First with LLM Fallback)
        # -------------------------------------------------------------
        gen_start_t = time.perf_counter()
        
        top_languages = [c.get("source_lang", "").lower() for c in reranked_chunks[:3]]
        has_cross_lingual_evidence = any(l != target_lang.lower() for l in top_languages if l)
        
        if config.ENABLE_LOCAL_SLM:
            # High-speed local offline SLM generation on CPU (< 60 ms)
            from generation.local_slm import get_local_slm_adapter
            context_blocks = []
            for i, c in enumerate(reranked_chunks[:3]):
                lang_code = c.get("source_lang", "UNK").upper()
                context_blocks.append(f"[{lang_code} Passage]: {c.get('text', '')}")
            compiled_context = "\n\n".join(context_blocks)
            
            candidate_answer = get_local_slm_adapter().generate(
                prompt=raw_query_text,
                context=compiled_context,
                target_lang=target_lang,
            )
            
            if "don't have enough grounded information" in candidate_answer.lower():
                answer_source = "declined"
                gen_details = "Declined: local SLM detected insufficient facts in retrieved context"
            else:
                answer_source = "local_slm_generated"
                gen_details = f"Local Offline SLM Synthesis ({config.LOCAL_SLM_MODEL_PATH})"
        elif config.ALLOW_NETWORK_CALLS_IN_PIPELINE and config.LLM_API_KEY and config.LLM_API_KEY.strip():
            # Multi-source compilation & grounded synthesis with Groq/Cerebras LLM
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
            
            if "don't have enough grounded information" in candidate_answer.lower():
                answer_source = "declined"
                gen_details = "Declined: retrieved passages lack sufficient facts to answer question"
            elif has_cross_lingual_evidence:
                answer_source = "cross_lingual_synthesis"
                gen_details = f"Cross-lingual multi-source synthesis into '{target_lang}' via Groq ({config.LLM_MODEL})"
            else:
                answer_source = "generated"
                gen_details = f"Grounded LLM synthesis via Groq ({config.LLM_MODEL})"
        else:
            # Deterministic local extractive selection & Semantic Answer Cache fast path
            extractive_res = generate_extractive(
                raw_query_text,
                reranked_chunks,
                query_vector=query_vector,
                target_lang=target_lang,
                embedder=self.embedder,
            )
            candidate_answer = extractive_res["answer"]
            answer_source = extractive_res["answer_source"]
            gen_details = (
                "Gold dataset answer returned via SemanticAnswerCache lookup (<0.5ms)"
                if answer_source == "gold_answer_cache"
                else "Non-LLM Context Synthesis via Continuous TextRank & SVD Matrix Decomposition"
            )
        
        timings.append(StageTiming(
            stage="generation",
            ms=round((time.perf_counter() - gen_start_t) * 1000, 2),
            success=True,
            details=gen_details,
        ))
        
        # -------------------------------------------------------------
        # STAGE 8: Post-Generation Grounding & LLM Safety Refusal Guardrail

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
        
        if ground_reason and "Blocked:" in ground_reason:
            guardrails.unsafe_detected = True
            guardrails.unsafe_reason = ground_reason
            answer_source = "declined"
        elif not is_grounded:
            answer_source = "declined"
            
        timings.append(StageTiming(
            stage="post_generation_grounding_guardrail",
            ms=round((time.perf_counter() - ground_start_t) * 1000, 2),
            success=is_grounded,
            details=ground_reason,
        ))
        
        # Record grounded answer into dynamic vector LRU cache for subsequent zero-latency hits
        if is_grounded and answer_source != "declined" and len(final_answer) >= 5:
            top_chunk_id = reranked_chunks[0].get("chunk_id") if reranked_chunks else None
            conf_val = float(reranked_chunks[0].get("confidence", 0.95)) if reranked_chunks else 0.95
            self.answer_cache.record_answer(
                query=raw_query_text,
                query_vector=query_vector,
                answer=final_answer,
                source_chunk_id=top_chunk_id,
                confidence=conf_val,
                source_lang=target_lang,
            )
        
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
        1. Explicit User Selection: If the user selected a valid language (not 'auto' or 'unknown'), respect it.
        2. Auto-detection: If 'auto' or no hint, inspect character script Unicode blocks (Devanagari, Tamil, Bengali, etc.).
        3. Latin script check: Route to English ('en') if Latin characters are present in auto mode.
        4. Safe default fallback to 'en' or first configured language.
        """
        # 1. First priority: Respect explicit user selection if provided (and not 'auto' / 'unknown')
        if hint and hint.lower().strip() not in ["auto", "unknown", "none", ""]:
            normalized_hint = hint.lower().strip()
            if normalized_hint in config.LANGUAGES:
                return normalized_hint
            # Check if hint matches a prefix like 'hi-IN' or registered language code
            for lang_code in config.LANGUAGES:
                if normalized_hint.startswith(lang_code):
                    return lang_code

        cleaned = text.strip() if text else ""
        
        # 2. Auto-detection from Indic native Unicode scripts
        # Check for specific Assamese characters first
        if any(c in cleaned for c in ["\u09f0", "\u09f1"]) and "as" in config.LANGUAGES:
            return "as"

        for char in cleaned:
            code = ord(char)
            # Devanagari (Hindi, Marathi, Sanskrit, Nepali)
            if 0x0900 <= code <= 0x097F:
                for cand in ["hi", "mr", "ne", "sa"]:
                    if cand in config.LANGUAGES:
                        return cand
            # Bengali / Assamese
            elif 0x0980 <= code <= 0x09FF:
                for cand in ["bn", "as"]:
                    if cand in config.LANGUAGES:
                        return cand
            # Gurmukhi / Punjabi
            elif 0x0A00 <= code <= 0x0A7F:
                if "pa" in config.LANGUAGES:
                    return "pa"
            # Gujarati
            elif 0x0A80 <= code <= 0x0AFF:
                if "gu" in config.LANGUAGES:
                    return "gu"
            # Odia
            elif 0x0B00 <= code <= 0x0B7F:
                if "or" in config.LANGUAGES:
                    return "or"
            # Tamil
            elif 0x0B80 <= code <= 0x0BFF:
                if "ta" in config.LANGUAGES:
                    return "ta"
            # Telugu
            elif 0x0C00 <= code <= 0x0C7F:
                if "te" in config.LANGUAGES:
                    return "te"
            # Kannada
            elif 0x0C80 <= code <= 0x0CFF:
                if "kn" in config.LANGUAGES:
                    return "kn"
            # Malayalam
            elif 0x0D00 <= code <= 0x0D7F:
                if "ml" in config.LANGUAGES:
                    return "ml"
            # Arabic / Urdu
            elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFF:
                if "ur" in config.LANGUAGES:
                    return "ur"
                    
        # 3. Check if the text contains Latin letters (English)
        has_latin = bool(re.search(r"[a-zA-Z]", cleaned))
        if has_latin:
            if "en" in config.LANGUAGES:
                return "en"
                
        # 4. Default fallback to 'en' or first configured language
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
