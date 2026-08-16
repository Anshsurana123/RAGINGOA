# ⚡ Voice-Enabled Indic RAG — Latency & Performance Report

**Benchmark Timestamp**: `2026-08-16T01:57:10Z`  
**Hardware Environment**: `8 vCPUs | 15.78 GB RAM | Windows 11 (AMD64)`  
**Active Languages**: `as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur, en`  
**Total Benchmark Queries**: `88` (`69` in-scope factoid queries)  

---

## 1. Key Latency Targets vs Measured Performance

> [!IMPORTANT]
> **Retrieval-Stage Latency** covers `Query Embedding (multilingual-e5-small) + In-Memory FAISS HNSW Search + BM25-Hybrid Re-ranking`.
> This core pipeline stage is held against the **~200ms latency target**.
> **End-to-End Latency** includes all pre-retrieval guardrails, extractive/LLM generation, and grounding verification.

| Metric Scope | Target SLA | P50 (Median) | P70 | P100 (Max) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Retrieval Stage (FAISS + BM25)** | **~200 ms** | **105.34 ms** | **118.28 ms** | **2146.31 ms** | ✅ PASS (<200ms) |
| **Full End-to-End Pipeline (Text Bypass)** | — | **191.04 ms** | **235.61 ms** | **2271.84 ms** | ✅ PASS |

---

## 2. Stage-by-Stage Latency Breakdown (P50 / P70 / P100)

| Pipeline Stage | P50 (ms) | P70 (ms) | P100 (ms) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1. STT Transcription (Sarvam) | 0.00 ms | 0.00 ms | 0.00 ms | Instrumented |
| 2. Language Routing & Dynamic Dispatch | 0.01 ms | 0.01 ms | 0.03 ms | Instrumented |
| 3. Pre-Retrieval Safety Regex Check | 0.13 ms | 0.15 ms | 1.43 ms | Instrumented |
| 4. Query Embedding ('query: ' prefix) | 17.80 ms | 19.02 ms | 51.45 ms | Instrumented |
| 5. Pre-Retrieval Centroid Off-Topic Check | 0.13 ms | 0.15 ms | 0.29 ms | Instrumented |
| 6. Parallel Multi-Strategy FAISS Search | 0.86 ms | 0.95 ms | 1.61 ms | Instrumented |
| bm25_cross_encoder_reranking | 99.32 ms | 108.14 ms | 2127.50 ms | Instrumented |
| generation | 104.36 ms | 116.28 ms | 225.53 ms | Instrumented |
| 9. Post-Generation Grounding Check | 0.64 ms | 0.90 ms | 1.83 ms | Instrumented |
| semantic_answer_cache | 0.26 ms | 0.29 ms | 0.67 ms | Instrumented |
| reranking | 0.00 ms | 0.00 ms | 0.00 ms | Instrumented |

---

## 3. Guardrail Enforcement Metrics

- **Unsafe Queries Blocked**: `9` test queries (100% precision on safety blocklist)
- **Off-Topic Queries Rejected**: `15` test queries (100% precision on centroid distance threshold)
- **Total Test Queries Processed**: `88` across Hindi, Tamil, and English
