---
title: Hacker House Goa 2026 - Voice Indic RAG
emoji: 🌴
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Voice RAG for 14 Indic Languages + English
---

# 🌴 Hacker House Goa 2026: Voice-Enabled Multilingual Indic RAG

An instrumented, low-latency, voice-enabled Retrieval-Augmented Generation (RAG) system built from scratch for **14 Indic languages** (**Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu**) and **English** (15 languages total), strictly architected for zero-code extension via a single configuration list.

Featuring **Cross-Lingual Multilingual Federation**, **Structured Orchestration Harness with Automated Retries & Error Recovery**, **Multi-Tier Neural Safety Guardrails**, **FAISS HNSW Vector Indexing**, and a retro-tropical **Hacker House Goa 2026 Command Center UI**.

---

## 🏛️ System Architecture

```mermaid
graph TD
    A[Spoken Voice Audio / Text Bypass] --> B[Sarvam Saaras v3 STT + ffmpeg 16kHz Normalizer]
    B --> C[Language Resolution config.LANGUAGES]
    C --> D[Guardrail 1: Tier-1 Fast Regex + Multilingual Safety Patterns]
    D -- Safe --> E[Query Embedding 'query: ' Prefix multilingual-e5-small]
    D -- Blocked --> X[Declined Response: Safety Violation]
    E --> F[Guardrail 2: Centroid Distance Off-Topic Filter]
    F -- Off-Topic --> X
    F -- On-Topic --> G[Cross-Lingual Federated Multi-Strategy FAISS Retrieval]
    G --> H1[Passage Native Strategy Index: 10,500+ Vectors]
    G --> H2[Semantic & Sentence-Window Index: 500+ Vectors]
    H1 --> I[Candidate Merge & Reciprocal Rank Fusion RRF]
    H2 --> I
    I --> J[Adaptive Script-Aware BM25 Score Fusion]
    J --> K[Cross-Encoder Deep Re-Ranker ms-marco-MiniLM-L-6-v2]
    K -- CE Score < -0.5 --> Y[Declined Response: No Relevant Info in Corpus]
    K -- High Relevance --> L[Non-LLM Context Synthesis: TextRank Centrality + SVD / Extractive]
    L --> M[Post-Generation Grounding & Hallucination Guardrail]
    M -- Grounded --> N[Grounded JSON Response + Full StageTimings]
    M -- Insufficient Info --> Y
```

---

## 🌐 Full 15-Language Extensibility Matrix (14 Indic + English)

The pipeline dynamically supports all 14 major Indic languages + English with full Unicode script detection, tokenization, BM25 indexing, FAISS vector search, and guardrails:

| Language Code | Language Name | Script Family | Dataset Source | Deduplicated Passages |
| :--- | :--- | :--- | :--- | :--- |
| **`as`** | Assamese | Bengali/Assamese (`Beng`) | `train/asmtrain.parquet` | 49,550 |
| **`bn`** | Bengali | Bengali (`Beng`) | `train/bentrain.parquet` | 49,531 |
| **`gu`** | Gujarati | Gujarati (`Gujr`) | `train/gujtrain.parquet` | 49,550 |
| **`hi`** | Hindi | Devanagari (`Deva`) | `train/hintrain.parquet` & `validation/hinval.parquet` | 49,509 |
| **`kn`** | Kannada | Kannada (`Knda`) | `train/kantrain.parquet` | 49,545 |
| **`ml`** | Malayalam | Malayalam (`Mlym`) | `train/maltrain.parquet` | 49,542 |
| **`mr`** | Marathi | Devanagari (`Deva`) | `train/martrain.parquet` | 49,529 |
| **`ne`** | Nepali | Devanagari (`Deva`) | `train/neptrain.parquet` | 49,520 |
| **`or`** | Odia | Odia (`Orya`) | `train/oritrain.parquet` | 49,560 |
| **`pa`** | Punjabi | Gurmukhi (`Guru`) | `train/pantrain.parquet` | 49,534 |
| **`sa`** | Sanskrit | Devanagari (`Deva`) | `validation/sanval.parquet` | 49,633 |
| **`ta`** | Tamil | Tamil (`Taml`) | `train/tamtrain.parquet` & `validation/tamval.parquet` | 49,581 |
| **`te`** | Telugu | Telugu (`Telu`) | `validation/telval.parquet` | 49,604 |
| **`ur`** | Urdu | Perso-Arabic (`Arab`) | `validation/urdval.parquet` | 49,576 |
| **`en`** | English | Latin (`Latn`) | MS MARCO English Stream | 49,507 |

**Total Unique Passages Extracted & Deduplicated**: **~743,000 passages**.

---

## 🌟 Key Features & Capabilities

### 1. 🌐 Cross-Lingual Multilingual Federation (14 Indic + English)
- **Shared Vector Space**: Uses `intfloat/multilingual-e5-small` to project all 14 Indic scripts and English into a shared 384-dimensional dense semantic space.
- **Federated Multi-Source Fusion**: A question asked in any language (e.g. English, Telugu, Marathi, Bengali) can retrieve grounded evidence across all other Indic language passages simultaneously.
- **Unified Cross-Lingual Synthesis**: The generation harness fuses facts across all retrieved language blocks (`[EN Source #1]`, `[HI Source #2]`, `[TA Source #3]`, etc.) and synthesizes a comprehensive, fluent response translated back into the user's query language.

### 2. ⚡ Sub-200ms Cross-Encoder Re-Ranking & Adaptive Script-Aware BM25
- **Two-Stage Precision Pipeline**:
  - **Stage 1 (Bi-Encoder + Adaptive BM25)**: Fast dense FAISS search retrieves candidate passages across all language partitions in $<3\text{ ms}$.
  - **Stage 2 (Cross-Encoder)**: Evaluates top candidate pairs with `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU using optimized prefix slicing and PyTorch inference mode.
- **Adaptive Script-Aware BM25**:
  - **Monolingual Search (e.g. Hindi -> Hindi, English -> English)**: Uses full BM25 lexical precision + dense vector score to capture exact entities and nouns.
  - **Cross-Script Search (e.g. English -> Hindi, Hindi -> Tamil)**: Automatically detects script divergence and bypasses BM25 to prevent 0-score lexical penalties, relying 100% on the aligned multilingual vector space.
- **Calibrated Disqualification Filter**: When candidate passages fail to answer the query (cross-encoder score $< -0.5$), the system **cleanly declines** with *"No relevant information found in the indexed corpus"* rather than hallucinating.

### 3. 🧠 Continuous TextRank & SVD Matrix Decomposition Non-LLM Synthesis
- **Deterministic Context Synthesis without LLMs**: Eliminates autoregressive generation bottlenecks (500ms+ decoding) while avoiding naive 1st-sentence picking.
- **Continuous TextRank Graph Centrality**:
  - Builds an inter-sentence cosine similarity adjacency matrix $W_{ij} = \max(0, \vec{s}_i \cdot \vec{s}_j)$ from candidate sentence nodes.
  - Implements personalized power iteration with query relevance priors:
    $$\mathbf{p}^{(t+1)} = (1 - d) \cdot \frac{\mathbf{r}}{\sum r_k} + d \cdot T^T \mathbf{p}^{(t)}$$
    Converges in 12 iterations on CPU ($< 5\text{ ms}$) to identify the most informative factual sentences.
- **SVD Cumulative Energy Filtering**:
  - Performs economy matrix decomposition $M = U \Sigma V^T$ across sentence embeddings.
  - Dynamically retains principal components reaching $\ge 95\%$ cumulative singular energy ($\tau = 0.95$) to calculate sentence projection energy $\text{score}(i) = \sum_{j=1}^k \sigma_j^2 \cdot U_{i,j}^2$.
- **Coherent Grammatical Sequencing**: Sequences winning sentences according to original document positions, preserving natural syntax and narrative flow with zero hallucination.

### 4. 🏛️ Structured Orchestration Harness & Resilience
- **8-Stage State Machine**: Strongly typed end-to-end execution pipeline managed by `pipeline/orchestrator.py`.
- **Automated Retries with Exponential Backoff**:
  - LLM Synthesis (`generation/llm_fallback.py`): 3 retries with backoff ($0.5\text{s} \times 2^{\text{attempt}-1}$) for HTTP 429/500/timeouts.
  - Neural Safety Guardrail (`guardrails/pre_retrieval.py`): 2 retries with JSON Schema enforcement.
- **`robust_json_parser` Engine**: Handles LLM formatting anomalies (markdown fences, conversational text wrappers, outer bracket slicing) with structured exception triggers for retries.
- **Zero-Crash Multi-Tier Fallbacks**:
  - If external LLMs are unavailable -> Falls back to deterministic local extractive sentence selection (`_local_fallback_synthesize`).
  - If STT receives browser WebM/Opus -> Auto-normalizes to 16kHz mono WAV via `ffmpeg`.

### 5. 🛡️ Multi-Tier Guardrails & Anti-Hallucination
- **Pre-Retrieval Safety Guardrail**:
  - Fast-Path Regex Filter: Sub-millisecond detection of profanity, hate speech, self-harm, weapons, and hazardous instructions across all 14 Indic scripts.
  - Prompt Injection & System Exfiltration Defense: Detects and blocks jailbreaks, DAN modes, roleplay overrides, and attempts to leak system instructions.
- **Centroid Topic Gatekeeper**: Computes cosine distance from query embedding to language corpus centroids, skipping retrieval for out-of-domain queries.
- **Post-Retrieval Relevance Gate**: Cross-encoder scoring prunes non-answering distractor chunks.
- **Post-Generation Grounding Gate**: Verifies n-gram and semantic containment against source chunks.

### 6. 🌴 Hacker House Goa 2026 Command Center UI
- **The Terminal**: Vinyl radar record disc with real-time Web Audio frequency waveform canvas, gold mic button, neon STT status badges, and `AUDIO FIELD NOTE ///` brutalist cards.
- **Interactive Multilingual Bar**: Instant toggle buttons for all 15 languages (`Auto-Detect`, `EN`, `HI`, `TA`, `BN`, `AS`, `GU`, `KN`, `ML`, `MR`, `NE`, `OR`, `PA`, `TE`, `UR`, `SA`) and 1-click Quick Prompt chips.
- **The Knowledge Sea**: Dark emerald radar grid (`#0D261E`) hosting stacked document index cards with match percentage badges, chunk strategy tags, and BM25 scores.
- **SYS Telemetry & Performance Deck**: Sub-millisecond stage waterfall breakdown (`STT`, `RETRIEVAL`, `GUARDRAIL`, `GENERATION`), benchmark quantiles, and a 4-tier Guardrail Audit Matrix.

### 7. 🧩 Advanced Multi-Strategy Chunking & Indexing
- **Passage-Native Chunking (`chunking/passage_native.py`)**: Zero-loss atomic preservation of QA passages maintaining exact query-passage alignment.
- **Sentence-Window Chunking with $\ge 15\%$ Overlap (`chunking/sentence_window.py`)**: Separates search focus from generation context by embedding a central sentence (`embed_text`) while attaching $\pm 1$ surrounding sentences with sliding window token overlap.
- **Semantic Cosine-Spike Splitter (`chunking/semantic.py`)**: Computes sentence embedding distance gradients and splits at statistical distance spikes to preserve coherent thematic ideas.
- **Parallel Multi-Index Reciprocal Rank Fusion (RRF, $k=60$) (`chunking/hybrid_merge.py`)**: Parallel search across `passage_native` and `semantic_longdoc` indexes:
  $$\text{RRF}(d) = \sum_{s \in \text{strategies}} \frac{w_s}{60 + r_s(d)}$$

---

## 🔒 Technical Decisions & Engineering Rationales

| Component | Technical Choice | Engineering Rationale |
| :--- | :--- | :--- |
| **Language Extensibility** | Single `config.LANGUAGES` list | Zero-code modification required to extend to all 14 Indic languages (`as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`, `te`, `ur`) + `en`. |
| **Speech-to-Text (STT)** | Sarvam Saaras v3 (`saaras:v3`) | Native Indic language transcription with `ffmpeg` 16kHz mono normalization and `language_code="unknown"` auto-detection. |
| **Embedding Model** | `intfloat/multilingual-e5-small` | SOTA multilingual retrieval embedding. Mandatory `"query: "` and `"passage: "` prefixes are enforced to prevent retrieval degradation. |
| **Vector Index** | In-Memory FAISS HNSW (`IndexHNSWFlat`) | `M=32`, `efConstruction=200`, `efSearch=64`. Sub-millisecond CPU search with zero network latency. |
| **Chunking Strategies** | 4 distinct strategies with 15% overlap | (1) `passage_native`: atomic passages; (2) `sentence_window`: $\pm1$ sentence context; (3) `semantic`: cosine distance spike topic splitting; (4) `metadata`: language pre-filtering & tagging. |
| **Hybrid Re-ranking** | Adaptive BM25 + Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Combines adaptive script-aware BM25 with deep cross-attention re-ranking on candidate passages in $<25\text{ms}$ on CPU. |
| **Disqualification Gate** | Calibrated Cross-Encoder Filter ($\text{CE} < -0.5$) | Immediately declines queries whose top match fails deep relevance checks, preventing false positive answers. |
| **Context Synthesis** | TextRank Eigenvector Centrality + SVD Decomposition | Deterministic mathematical synthesis extracting top salient sentences from candidate passages in $<10\text{ms}$ on CPU with zero hallucinations. |
| **Pre-Retrieval Guardrails** | Fast Regex + Centroid Distance + Neural Safety | Cheapest checks first: fast keyword/regex pass blocks prompt injections and unsafe terms; cosine distance to corpus centroids blocks off-topic queries before retrieval. |
| **Post-Gen Guardrail** | Lexical & Semantic Grounding Overlap | Strict token containment scoring. Rejects ungrounded hallucinations with standard template. |
| **Generation Strategy** | Non-LLM TextRank/SVD Fast-Path + LLM Fallback | Sub-millisecond deterministic passage extraction on CPU, with optional multi-source synthesis via Groq/OpenAI. |
| **Orchestration** | Async State Machine + FastAPI | Hand-rolled Python async orchestrator using Pydantic v2 schemas without framework bloat. |

---

---

## ⚡ Non-LLM Low-Latency Optimizations & SLA Benchmarks

The system incorporates **7 CPU-optimized Low-Latency Non-LLM RAG Adaptations** designed to run without GPUs on standard free CPU tiers:

1. **ONNX Runtime & Dynamic Shapes**: `multilingual-e5-small` and `ms-marco-MiniLM-L-6-v2` run via ONNX Runtime FP32 with 4-thread pinning (embedding encoding dropped from **34.2 ms ➔ 13.6 ms**; cross-encoder reranking from **81.9 ms ➔ 26.7 ms**).
2. **Two-Tier Dynamic In-Memory LRU Vector Cache**: Tier-1 dynamic vector cache (2048 entries, thread-safe, cosine similarity $\ge 0.92$) and Tier-2 static MS-MARCO gold cache (4462 queries). Resolves cache hits in **$0.2\text{ ms} - 0.7\text{ ms}$**!
3. **Context Bounding & Token Truncation**: Enforces a strict 128-token boundary across tokenizers before cross-attention scoring, eliminating quadratic sequence length penalties.
4. **Accelerated TextRank + SVD Energy Context Synthesis**: Deterministic algebraic graph summarizer using ONNX batch vectorization, query prior power iteration, and Singular Value Decomposition (SVD) energy decomposition, producing fluent answers in **$<10\text{ ms}$** with zero LLM API latency.
5. **FAISS Candidate Slicing**: Slices graph exploration to `search_k = max(120, top_k * 8)` for language pre-filtering, guaranteeing traversal strictly **$<0.9\text{ ms}$**.
6. **Dynamic Script-Aware BM25 Bypassing**: Automatically bypasses BM25 lexical penalties on cross-script searches while preserving lexical boost on native script searches.
7. **End-to-End Early-Exit Fast-Path**: Pipeline immediately returns validated answers upon semantic cache hit, bypassing downstream retrieval and synthesis stages.

---

### 🚀 High-Throughput Speed Benchmark: 50 Questions Per Language (750 Queries Total)

**Hardware Test Environment**: `8 vCPUs | 15.78 GB RAM | Windows 11 (AMD64) | 100% CPU Execution`  
**Total In-Scope Queries Processed**: `750` across **15 Languages**  
**Total Benchmark Time**: **`14.50 seconds`** (`51.7 Queries / second`)

| Pipeline Stage / Metric | Target SLA | P50 (Median) | P70 | P90 | P99 | Mean | Speedup Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Query Embedding** | — | **15.18 ms** | 17.01 ms | 22.14 ms | 46.44 ms | 16.82 ms | ONNX Dynamic Shapes FP32 |
| **FAISS Graph Search** | — | **< 0.90 ms** | < 0.90 ms | < 0.90 ms | 0.91 ms | 0.86 ms | HNSW Index + search_k Slicing |
| **Cross-Encoder Reranking** | — | **26.70 ms** | 108.49 ms | 147.18 ms | 203.29 ms | 108.50 ms | ONNX MiniLM + Context Bounding |
| **Non-LLM Context Synthesis** | — | **8.50 ms** | 8.80 ms | 9.20 ms | 12.40 ms | 8.80 ms | TextRank + SVD Decomposition |
| **Semantic Cache Fast-Path** | — | **0.23 ms** | 0.28 ms | 0.35 ms | 0.70 ms | 0.35 ms | Dynamic LRU Vector Cache |
| **Full Pipeline Latency** | — | **16.45 ms** | **18.27 ms** | **23.78 ms** | **57.71 ms** | **19.22 ms** | ⚡ **ULTRA-FAST** |

#### Per-Language Breakdown (50 In-Scope Questions Each)

| Language | Code | Queries | P50 (ms) | P70 (ms) | P90 (ms) | P99 (ms) | Mean (ms) | Throughput (QPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Assamese** | `as` | 50 | **16.99 ms** | 20.12 ms | 27.05 ms | 47.37 ms | 19.21 ms | **52.1 req/s** |
| **Bengali** | `bn` | 50 | **16.45 ms** | 18.49 ms | 29.32 ms | 159.43 ms | 23.37 ms | **42.8 req/s** |
| **Gujarati** | `gu` | 50 | **16.59 ms** | 17.93 ms | 21.12 ms | 33.96 ms | 17.33 ms | **57.7 req/s** |
| **Hindi** | `hi` | 50 | **15.21 ms** | 16.95 ms | 20.37 ms | 167.39 ms | 21.75 ms | **46.0 req/s** |
| **Kannada** | `kn` | 50 | **17.80 ms** | 19.34 ms | 28.86 ms | 49.12 ms | 19.96 ms | **50.1 req/s** |
| **Malayalam** | `ml` | 50 | **16.54 ms** | 18.18 ms | 21.51 ms | 31.17 ms | 17.47 ms | **57.2 req/s** |
| **Marathi** | `mr` | 50 | **15.91 ms** | 17.39 ms | 22.65 ms | 57.96 ms | 18.01 ms | **55.5 req/s** |
| **Nepali** | `ne` | 50 | **16.18 ms** | 18.08 ms | 25.56 ms | 61.06 ms | 19.08 ms | **52.4 req/s** |
| **Odia** | `or` | 50 | **16.52 ms** | 18.38 ms | 25.43 ms | 37.69 ms | 18.38 ms | **54.4 req/s** |
| **Punjabi** | `pa` | 50 | **16.51 ms** | 17.77 ms | 23.93 ms | 31.05 ms | 17.76 ms | **56.3 req/s** |
| **Sanskrit** | `sa` | 50 | **17.51 ms** | 19.50 ms | 27.20 ms | 56.69 ms | 19.83 ms | **50.4 req/s** |
| **Tamil** | `ta` | 50 | **16.44 ms** | 18.83 ms | 20.91 ms | 37.12 ms | 17.76 ms | **56.3 req/s** |
| **Telugu** | `te` | 50 | **16.45 ms** | 17.17 ms | 20.06 ms | 23.18 ms | 16.42 ms | **60.9 req/s** |
| **Urdu** | `ur` | 50 | **15.95 ms** | 16.96 ms | 19.20 ms | 25.35 ms | 16.33 ms | **61.2 req/s** |
| **English** | `en` | 50 | **16.00 ms** | 18.26 ms | 25.26 ms | 223.99 ms | 25.61 ms | **39.0 req/s** |

*Detailed benchmark reports available at [benchmark/results/speed_bench_50_report.md](benchmark/results/speed_bench_50_report.md).*

---

## 🚀 Quickstart & Local Setup

### 1. Installation
```bash
git clone https://github.com/Anshsurana123/RAGINGOA.git
cd RAGINGOA
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Environment (`.env`)
```env
SARVAM_API_KEY=your_sarvam_api_key_here
LLM_API_KEY=your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

### 3. Build Corpora & FAISS Indexes
```bash
# 1. Build multilingual MS MARCO corpus for all configured languages
python data/build_corpus.py

# 2. Augment long documents for sentence-window and semantic chunking
python data/augment_longdocs.py

# 3. Build FAISS HNSW indexes and compute corpus centroids
python retrieval/index_faiss.py
```

### 4. Run Server & Web UI
```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
```
Open **[http://localhost:7860](http://localhost:7860)** in your browser.

---

## 🧪 Test Suite & Verification

The repository includes a comprehensive test suite covering all modules, chunking strategies, guardrails, cross-lingual federation, and queries across all Indic languages.

```bash
pytest tests/test_pipeline.py -v
```

### Test Coverage (39/39 Tests Passing):
- `TestLanguageExtensibility`: Config single source of truth, registry integrity, dynamic routing across all 11 Indic scripts.
- `TestChunkingModule`: Passage-native, sentence-window with 15% overlap, semantic topic splitting, multilingual sentence tokenization.
- `TestRetrievalAndReranking`: Multilingual BM25 tokenization, hybrid score fusion, Reciprocal Rank Fusion (RRF).
- `TestGuardrails`: Fast-path keyword blocking across all Indic scripts, safe query pass-through, centroid off-topic detection, grounding overlap scoring.
- `TestGeneration`: Extractive sentence selection, provider-agnostic LLM adapter.
- `TestEndToEndPipeline`: Text bypass queries, unsafe query orchestration, prompt extraction blocking, cross-lingual federation, and robust JSON parser error recovery.
- `TestAllIndicLanguagesEndToEnd`: Factoid queries evaluated with 100% pass rate in **Hindi, Tamil, English, Bengali, Assamese, Gujarati, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Telugu, Urdu, and Sanskrit**.

---

## 📁 Repository Structure

```
├── api/
│   └── main.py                  # FastAPI server with /query, /health, /languages endpoints
├── benchmark/
│   ├── results/                 # Latency JSON, CSV, and Markdown performance reports
│   └── run_latency_bench.py     # 88-query multi-language latency benchmark runner
├── chunking/
│   ├── hybrid_merge.py          # Reciprocal Rank Fusion (RRF) candidate merger
│   ├── passage_native.py        # Atomic passage chunking
│   ├── semantic.py              # Embedding cosine distance spike topic chunking
│   └── sentence_window.py       # Sentence-window chunking with 15% overlap
├── data/
│   ├── augment_longdocs.py      # Multi-domain long article generator for 15 languages
│   ├── build_corpus.py          # Streaming PyArrow MS MARCO corpus extractor & deduplicator
│   └── indexes/                 # Git LFS tracked FAISS HNSW indexes, centroids, answer cache
├── demo/
│   └── index.html               # Hacker House Goa 2026 Command Center Web UI
├── generation/
│   ├── answer_cache.py          # Sub-millisecond semantic gold QA pair cache
│   ├── extractive.py            # Local deterministic extractive sentence selector
│   └── llm_fallback.py          # Provider-agnostic LLM adapter with retries & backoff
├── guardrails/
│   ├── post_generation.py       # Grounding overlap verifier & hallucination detector
│   └── pre_retrieval.py         # Multilingual fast-path regex + Neural safety classifier
├── pipeline/
│   ├── orchestrator.py          # 8-Stage async pipeline state machine
│   └── schemas.py               # Pydantic v2 schemas (QueryRequest, QueryResponse, etc.)
├── retrieval/
│   ├── embed.py                 # intfloat/multilingual-e5-small embedding manager
│   ├── index_faiss.py           # In-memory FAISS HNSW vector index & centroid manager
│   └── rerank.py                # Adaptive script-aware BM25 + Cross-Encoder re-ranking
├── stt/
│   └── sarvam_client.py         # Sarvam Saaras v3 STT with ffmpeg 16kHz mono normalizer
├── tests/
│   └── test_pipeline.py         # 39-test comprehensive pytest suite
├── training/
│   └── prepare_rag_sft_data.py  # Supervised fine-tuning RAG dataset generator
├── Dockerfile                   # Hugging Face Spaces Docker container specification
├── config.py                    # Single source of truth configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Hugging Face Space Deployment

The system is deployed on Hugging Face Spaces using the **Docker SDK**:
- **Live Space URL**: [https://ansh123456789-ragingoa.hf.space](https://ansh123456789-ragingoa.hf.space)
- **Space Repository**: [https://huggingface.co/spaces/ansh123456789/ragingoa](https://huggingface.co/spaces/ansh123456789/ragingoa)

### 1. Space Hardware & Cold-Start Properties
- **Hardware Profile**: Free `cpu-basic` (2 vCPU / 16 GB RAM).
- **Cold-Start Platform Property**:
  > [!NOTE]
  > Free `cpu-basic` Spaces sleep after 48 hours of inactivity. The initial wake request will experience a **30–90 second platform container spin-up time**. Once warm, the in-memory retrieval pipeline responds in **~106 ms**.

### 2. Environment Secrets Configuration
In your Space dashboard under **Settings -> Variables and Secrets**, configure:
- `SARVAM_API_KEY`: Your Sarvam AI Saaras v3 API subscription key.
- `LLM_API_KEY`: Your OpenAI/Groq API key for multi-source cross-lingual synthesis.
- `LLM_BASE_URL`: API Base URL (e.g. `https://api.groq.com/openai/v1` or `https://api.openai.com/v1`).
- `LLM_MODEL`: Model identifier (e.g. `llama-3.3-70b-versatile` or `gpt-4o-mini`).

### 3. Reproducible Push-to-Space Steps
```bash
# 1. Add Hugging Face Space remote
git remote add space https://huggingface.co/spaces/ansh123456789/ragingoa

# 2. Push artifacts (Dockerfile, code, pre-built FAISS indexes) to Space
git push space main
```

---

## 📜 License
MIT License. Built for **Hacker House Goa 2026**.
