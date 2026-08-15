# ⚡ Voice-Enabled Indic RAG — Latency & Performance Report

**Benchmark Timestamp**: `2026-08-15T13:54:44Z`  
**Hardware Environment**: `8 vCPUs | 15.78 GB RAM | Windows 11 (AMD64)`  
**Active Languages**: `hi, ta, en`  
**Total Benchmark Queries**: `61` (`45` in-scope factoid queries)  

---

## 1. Key Latency Targets vs Measured Performance

> [!IMPORTANT]
> **Retrieval-Stage Latency** covers `Query Embedding (multilingual-e5-small) + In-Memory FAISS HNSW Search + BM25-Hybrid Re-ranking`.
> This core pipeline stage is held against the **~200ms latency target**.
> **End-to-End Latency** includes all pre-retrieval guardrails, extractive/LLM generation, and grounding verification.

| Metric Scope | Target SLA | P50 (Median) | P70 | P100 (Max) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Retrieval Stage (FAISS + BM25)** | **~200 ms** | **52.12 ms** | **56.39 ms** | **107.24 ms** | ✅ PASS (<200ms) |
| **Full End-to-End Pipeline (Text Bypass)** | — | **54.26 ms** | **58.29 ms** | **135.10 ms** | ✅ PASS |

---

## 2. Stage-by-Stage Latency Breakdown (P50 / P70 / P100)

| Pipeline Stage | P50 (ms) | P70 (ms) | P100 (ms) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1. STT Transcription (Sarvam) | 0.00 ms | 0.00 ms | 0.00 ms | Instrumented |
| 2. Language Routing & Dynamic Dispatch | 0.01 ms | 0.04 ms | 0.19 ms | Instrumented |
| 3. Pre-Retrieval Safety Regex Check | 0.12 ms | 0.16 ms | 0.24 ms | Instrumented |
| 4. Query Embedding ('query: ' prefix) | 38.20 ms | 41.79 ms | 92.83 ms | Instrumented |
| 5. Pre-Retrieval Centroid Off-Topic Check | 0.11 ms | 0.12 ms | 0.18 ms | Instrumented |
| 6. Parallel Multi-Strategy FAISS Search | 0.92 ms | 0.96 ms | 1.19 ms | Instrumented |
| 7. BM25-Hybrid Re-ranking | 12.53 ms | 13.75 ms | 19.13 ms | Instrumented |
| generation | 0.69 ms | 0.77 ms | 54.10 ms | Instrumented |
| 9. Post-Generation Grounding Check | 0.88 ms | 0.97 ms | 1.54 ms | Instrumented |

---

## 3. Guardrail Enforcement Metrics

- **Unsafe Queries Blocked**: `6` test queries (100% precision on safety blocklist)
- **Off-Topic Queries Rejected**: `10` test queries (100% precision on centroid distance threshold)
- **Total Test Queries Processed**: `61` across Hindi, Tamil, and English
