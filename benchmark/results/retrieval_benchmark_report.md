# 🏎️ End-to-End Retrieval Latency Benchmark (`benchmark.py`)

**Benchmark Timestamp**: `2026-08-17T08:35:22Z`  
**Hardware Environment**: `8 vCPUs | 15.78 GB RAM | Windows 11 (AMD64) | 100% CPU Execution`  
**Active Loaded Corpus**: `148,545 native passage vectors (en, hi, mr) + 309 semantic longdoc vectors`  
**Target Latency Budget**: `50.0 ms` (defined in `app/config.py`)  
**Evaluation Script**: `python -m app.benchmark 50` / `python C:\Users\ANSH\Downloads\benchmark.py`  

---

## 1. Retrieval Latency Performance (50 Queries)

| Stage | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`embed` (Query Vectorization via `multilingual-e5-small` INT8 ONNX)** | 6.31 | 6.21 | 7.28 | 7.94 | ⚡ Sub-8ms |
| **`search` (In-Memory FAISS HNSW `IndexHNSWFlat` Graph Traversal)** | 0.73 | 0.71 | 0.93 | 1.16 | ⚡ Sub-1ms |
| **`total` (End-to-End Embed + Search Retrieval)** | **7.04** | **6.96** | **7.97** | **8.95** | ✅ **PASS (<50ms)** |

---

## 2. SLA Compliance

- **Latency Budget**: `50.0 ms`
- **Observed P95 Total Latency**: `7.97 ms`
- **Result**: **PASS: within budget (84.06% faster than the 50ms requirement)**

---

## 3. Query Suite
1. `What is FAISS used for?`
2. `How does HNSW indexing work?`
3. `What is retrieval augmented generation?`
4. `Which embedding model is fast on CPU?`
5. `How do you reduce RAG latency?`
6. `What does efSearch control?`
7. `Why normalize embeddings before indexing?`
8. `What are the stages of a RAG pipeline?`
