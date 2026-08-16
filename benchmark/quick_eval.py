import asyncio
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from pipeline.orchestrator import get_orchestrator
from pipeline.schemas import QueryRequest

async def main():
    print(f"--- RAG Pipeline Latency Benchmark ---")
    print(f"ENABLE_CROSS_ENCODER = {config.ENABLE_CROSS_ENCODER}")
    print(f"ONNX_NUM_THREADS = {config.ONNX_NUM_THREADS}")
    print(f"CONTEXT_BOUNDING_MAX_TOKENS = {config.CONTEXT_BOUNDING_MAX_TOKENS}")
    
    orchestrator = get_orchestrator()
    
    queries = [
        ("mr", "मॅनहॅटन प्रकल्पाचा मुख्य उद्देश काय होता?"),
        ("hi", "मैनहट्टन परियोजना का मुख्य उद्देश्य क्या था?"),
        ("en", "What was the purpose of the Manhattan Project?"),
    ]
    
    # Run each query 3 times and measure latency
    for lang, text in queries:
        print(f"\nEvaluating [{lang.upper()}]: '{text}'")
        for iteration in range(3):
            req = QueryRequest(
                text=text,
                language_hint=lang,
                cross_lingual=False,
                bypass_cache=(iteration == 0),  # Test cold path on iteration 0, cache on 1, 2
            )
            t0 = time.perf_counter()
            resp = await orchestrator.execute(req)
            wall_ms = (time.perf_counter() - t0) * 1000
            
            print(f"  [Iter {iteration+1}] Total: {resp.total_ms:.2f}ms (Wall: {wall_ms:.2f}ms) | Retrieval: {resp.retrieval_ms:.2f}ms | Source: {resp.answer_source}")
            for st in resp.stage_timings:
                print(f"    - {st.stage}: {st.ms:.2f}ms ({st.details[:50]}...)")
            print(f"    Answer: {resp.answer[:80]}...")

if __name__ == "__main__":
    asyncio.run(main())
