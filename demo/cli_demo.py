"""
Command-Line Interface (CLI) Demo for Voice-Enabled Indic RAG.

Usage:
1. Interactive text mode:
   python demo/cli_demo.py --text "हृदय के चार कक्ष कौन से हैं?" --lang hi
2. Audio file mode:
   python demo/cli_demo.py --audio sample.wav --lang ta
3. Interactive Shell mode:
   python demo/cli_demo.py --interactive
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Reconfigure stdout for Windows unicode support
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from pipeline.orchestrator import get_orchestrator
from pipeline.schemas import QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cli_demo")


def print_response_card(resp: QueryResponse):
    """Prints formatted response card in terminal."""
    print("\n" + "=" * 70)
    print(f"🎤 QUERY: {resp.query}")
    if resp.transcript and resp.transcript != resp.query:
        print(f"📝 TRANSCRIPT: {resp.transcript}")
    print(f"🌐 LANGUAGE: {resp.language_detected.upper()}")
    print("-" * 70)
    print(f"💬 ANSWER [{resp.answer_source.upper()}]:")
    print(f"   {resp.answer}")
    print("-" * 70)
    print("🛡️  GUARDRAILS:")
    for k, v in resp.guardrail_flags.items():
        if v is not None and v is not False:
            print(f"   - {k}: {v}")
    print("-" * 70)
    print("⏱️  STAGE LATENCY BREAKDOWN:")
    for st in resp.stage_timings:
        status_icon = "✓" if st.success else "✗"
        fb_text = " (fallback)" if st.fallback_used else ""
        print(f"   [{status_icon}] {st.stage:<38}: {st.ms:>7.2f} ms{fb_text}")
    print("-" * 70)
    print(f"⚡ RETRIEVAL LATENCY (FAISS + BM25) : {resp.retrieval_ms:>7.2f} ms  (Target: ~200ms)")
    print(f"🚀 TOTAL END-TO-END LATENCY        : {resp.total_ms:>7.2f} ms")
    print("=" * 70 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Voice-Enabled Indic RAG CLI Demo")
    parser.add_argument("--audio", type=str, help="Path to input audio file (.wav, .mp3)")
    parser.add_argument("--text", type=str, help="Direct text query for text-bypass mode")
    parser.add_argument("--lang", type=str, default="hi", choices=config.LANGUAGES, help="Language code")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI shell")
    args = parser.parse_args()

    orchestrator = get_orchestrator()

    if args.interactive:
        print("\n=== Voice-Enabled Indic RAG Interactive CLI ===")
        print(f"Supported Languages: {config.LANGUAGES}")
        print("Type 'exit' or 'quit' to end session.\n")
        
        while True:
            try:
                lang = input(f"Select Language ({'/'.join(config.LANGUAGES)}) [default: hi]: ").strip().lower()
                if not lang:
                    lang = "hi"
                if lang in ["exit", "quit"]:
                    break
                if lang not in config.LANGUAGES:
                    print(f"Invalid language '{lang}'. Choose from {config.LANGUAGES}")
                    continue
                    
                query_text = input("Enter Question: ").strip()
                if not query_text:
                    continue
                if query_text.lower() in ["exit", "quit"]:
                    break
                    
                req = QueryRequest(text=query_text, language_hint=lang)
                resp = await orchestrator.execute(req)
                print_response_card(resp)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        req = QueryRequest(
            audio_path=args.audio,
            text=args.text,
            language_hint=args.lang,
        )
        if not req.audio_path and not req.text:
            req.text = "हृदय के चार कक्ष कौन से हैं?"
            req.language_hint = "hi"
            print(f"No query supplied. Running default Hindi test query: '{req.text}'")
            
        resp = await orchestrator.execute(req)
        print_response_card(resp)


if __name__ == "__main__":
    asyncio.run(main())
