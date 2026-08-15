"""
Builds Multilingual RAG SFT Dataset for Fine-Tuning Qwen2.5-0.5B on Google Colab.
Extracts grounded triplets (Question, Context, Grounded Answer) across Hindi, Tamil, and English.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any
import config

DATA_DIR = Path(config.DATA_DIR)
PROCESSED_DIR = Path(getattr(config, "PROCESSED_DATA_DIR", DATA_DIR / "processed"))
OUTPUT_FILE = PROCESSED_DIR / "rag_sft_dataset.jsonl"


def extract_sft_examples() -> List[Dict[str, Any]]:
    """
    Extracts high-quality (Question, Context, Answer) triplets from processed files.
    """
    examples = []
    
    # 1. Load native passages from processed corpus files
    for lang in config.LANGUAGES:
        passages_file = PROCESSED_DIR / f"{lang}_corpus.jsonl"
        if not passages_file.exists():
            passages_file = PROCESSED_DIR / f"{lang}_passages.jsonl"
        if not passages_file.exists():
            continue
            
        lang_name = config.get_language_info(lang)["name"]
        print(f"Loading {lang_name} ({lang}) passages from {passages_file}...")
        
        count = 0
        with open(passages_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    p = json.loads(line)
                    text = p.get("text", "").strip()
                    if len(text) < 40:
                        continue
                    
                    # Generate natural synthetic queries and grounded answers from passage sentences
                    sentences = [s.strip() for s in text.replace("।", ".").split(".") if len(s.strip()) > 15]
                    if len(sentences) >= 2:
                        # Factoid Q/A pair
                        target_fact = sentences[0]
                        context = text
                        
                        # Format for Qwen2.5 Chat Template
                        prompt_text = (
                            f"Context:\n{context}\n\n"
                            f"Question: Explain the key facts mentioned in the context.\n\n"
                            f"Respond strictly in {lang_name} based on the context:"
                        )
                        answer_text = target_fact
                        
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert multilingual RAG assistant. "
                                    f"Synthesize accurate, grounded answers strictly in {lang_name} based only on the provided context."
                                )
                            },
                            {"role": "user", "content": prompt_text},
                            {"role": "assistant", "content": answer_text}
                        ]
                        
                        examples.append({"messages": messages, "lang": lang})
                        count += 1
                        if count >= 1500:  # 1,500 balanced pairs per language (4,500 total)
                            break
                except Exception:
                    continue
        print(f"Extracted {count} SFT examples for {lang_name}.")

    # 2. Add adversarial / negative unanswerable refusal examples (Teaches model when to decline)
    refusal_templates = {
        "en": ("Who won the 1994 football world cup?", "The cardiovascular system circulates blood throughout the body.", "I don't have enough grounded information to answer that."),
        "hi": ("1994 का फुटबॉल विश्व कप किसने जीता था?", "मानव हृदय चार कक्षों वाला एक पेशीय अंग है जो शरीर में रक्त का संचार करता है।", "मेरे पास इसका उत्तर देने के लिए पर्याप्त प्रामाणिक जानकारी नहीं है।"),
        "ta": ("1994 உலகக் கோப்பை கால்பந்து போட்டியில் யார் வென்றது?", "மனித இதயம் உடலில் இரத்தத்தை செலுத்தும் நான்கு அறைகளைக் கொண்ட ஒரு தசை உறுப்பாகும்.", "பதிலளிக்க போதுமான ஆதாரபூர்வமான தகவல்கள் என்னிடம் இல்லை.")
    }
    
    for lang, (q, ctx, ans) in refusal_templates.items():
        lang_name = config.get_language_info(lang)["name"]
        for _ in range(50):
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are an expert multilingual RAG assistant. Synthesize accurate, grounded answers strictly in {lang_name} based only on the provided context."
                    },
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}\n\nRespond strictly in {lang_name} based on the context:"},
                    {"role": "assistant", "content": ans}
                ],
                "lang": lang
            })

    return examples


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    examples = extract_sft_examples()
    print(f"Total SFT training examples: {len(examples)}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
    print(f"Saved dataset to {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
