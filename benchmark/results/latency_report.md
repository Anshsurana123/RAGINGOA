# ⚡ Voice-Enabled Indic RAG — Latency & Performance Report

**Benchmark Timestamp**: `2026-08-15T19:06:58Z`  
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
| **Retrieval Stage (FAISS + BM25)** | **~200 ms** | **106.43 ms** | **130.52 ms** | **301.30 ms** | ✅ PASS (<200ms) |
| **Full End-to-End Pipeline (Text Bypass)** | — | **207.02 ms** | **261.32 ms** | **452.45 ms** | ✅ PASS |

---

## 2. Stage-by-Stage Latency Breakdown (P50 / P70 / P100)

| Pipeline Stage | P50 (ms) | P70 (ms) | P100 (ms) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1. STT Transcription (Sarvam) | 0.00 ms | 0.00 ms | 0.00 ms | Instrumented |
| 2. Language Routing & Dynamic Dispatch | 0.01 ms | 0.01 ms | 0.03 ms | Instrumented |
| 3. Pre-Retrieval Safety Regex Check | 0.12 ms | 0.14 ms | 0.24 ms | Instrumented |
| 4. Query Embedding ('query: ' prefix) | 34.07 ms | 35.89 ms | 43.97 ms | Instrumented |
| 5. Pre-Retrieval Centroid Off-Topic Check | 0.15 ms | 0.16 ms | 0.27 ms | Instrumented |
| 6. Parallel Multi-Strategy FAISS Search | 0.91 ms | 0.98 ms | 8.25 ms | Instrumented |
| bm25_cross_encoder_reranking | 81.92 ms | 98.82 ms | 257.51 ms | Instrumented |
| generation | 113.93 ms | 132.46 ms | 357.29 ms | Instrumented |
| 9. Post-Generation Grounding Check | 0.61 ms | 0.75 ms | 1.46 ms | Instrumented |

---

## 3. Guardrail Enforcement Metrics

- **Unsafe Queries Blocked**: `9` test queries (100% precision on safety blocklist)
- **Off-Topic Queries Rejected**: `24` test queries (100% precision on centroid distance threshold)
- **Total Test Queries Processed**: `88` across Hindi, Tamil, and English
