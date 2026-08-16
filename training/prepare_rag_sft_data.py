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

from typing import List, Dict, Any, Optional
import config

DATA_DIR = Path(config.DATA_DIR)
PROCESSED_DIR = Path(getattr(config, "PROCESSED_DATA_DIR", DATA_DIR / "processed"))
OUTPUT_FILE = PROCESSED_DIR / "rag_sft_dataset.jsonl"


def extract_sft_examples(limit_per_lang: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Extracts high-quality (Question, Context, Answer) triplets from processed files.
    If limit_per_lang is None, extracts 100% of all passages in the corpus.
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
                        if limit_per_lang and count >= limit_per_lang:
                            break
                except Exception:
                    continue
        print(f"Extracted {count} SFT examples for {lang_name}.")

    # 2. Add adversarial / negative unanswerable refusal examples (Teaches model when to decline)
    refusal_templates = {
        "en": ("Who won the 1994 football world cup?", "The cardiovascular system circulates blood throughout the body.", "I don't have enough grounded information to answer that."),
        "hi": ("1994 का फुटबॉल विश्व कप किसने जीता था?", "मानव हृदय चार कक्षों वाला एक पेशीय अंग है जो शरीर में रक्त का संचार करता है।", "मेरे पास इसका उत्तर देने के लिए पर्याप्त प्रामाणिक जानकारी नहीं है।"),
        "ta": ("1994 உலகக் கோப்பை கால்பந்து போட்டியில் யார் வென்றது?", "மனித இதயம் உடலில் இரத்தத்தை செலுத்தும் நான்கு அறைகளைக் கொண்ட ஒரு தசை உறுப்பாகும்.", "பதிலளிக்க போதுமான ஆதாரபூர்வமான தகவல்கள் என்னிடம் இல்லை."),
        "bn": ("১৯৯৪ ফুটবল বিশ্বকাপ কে জিতেছিল?", "মানব হৃদপিণ্ড রক্ত সংবহনতন্ত্রের প্রধান পেশীবহুল অঙ্গ।", "আমার কাছে উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য নেই।"),
        "as": ("১৯৯৪ চনৰ ফুটবল বিশ্বকাপ কোনে জিকিছিল?", "মানৱ হৃদযন্ত্ৰ চাৰিটা কোঠালীৰে গঠিত।", "মোৰ ওচৰত উত্তৰ দিবলৈ পৰ্যাপ্ত তথ্য নাই।"),
        "gu": ("1994 ફૂટબોલ વર્લ્ડ કપ કોણે જીત્યો હતો?", "માનવ હૃદય ચાર ખંડો ધરાવતું એક સ્નાયુબદ્ધ અંગ છે.", "મારી પાસે આનો જવાબ આપવા માટે પૂરતી માહિતી નથી."),
        "kn": ("1994 ರ ಫುಟ್‌ಬಾಲ್ ವಿಶ್ವಕಪ್ ಅನ್ನು ಯಾರು ಗೆದ್ದರು?", "ಮಾನವ ಹೃದಯವು ದೇಹದಾದ್ಯಂತ ರಕ್ತವನ್ನು ಪಂಪ್ ಮಾಡುತ್ತದೆ.", "ಉತ್ತರಿಸಲು ನನ್ನ ಬಳಿ ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ."),
        "ml": ("1994 ലെ ഫുട്ബോൾ ലോകകപ്പ് ആരാണ് നേടിയത്?", "മനുഷ്യ ഹൃദയം ശരീരത്തിലുടനീളം രക്തം എത്തിക്കുന്നു.", "ഉത്തരം നൽകാൻ ആവശ്യമായ വിവരങ്ങൾ എന്റെ പക്കലില്ല."),
        "mr": ("1994 चा फुटबॉल विश्वचषक कोणी जिंकला?", "मानवी हृदय संपूर्ण शरीरात रक्ताचा पुरवठा करते.", "माझ्याकडे पुरेसा संदर्भ उपलब्ध नाही."),
        "ne": ("१९९४ को फुटबल विश्वकप कसले जितेको थियो?", "मानव मुटुले शरीरभरि रगत पम्प गर्दछ।", "मसँग उत्तर दिनको लागि पर्याप्त जानकारी छैन।"),
        "or": ("୧୯୯୪ ଫୁଟବଲ ବିଶ୍ୱକପ କିଏ ଜିତିଥିଲା?", "ମାନବ ହୃଦୟ ଶରୀରରେ ରକ୍ତ ସଞ୍ଚାଳନ କରେ।", "ମୋ ପାଖରେ ଉତ୍ତର ଦେବା ପାଇଁ ପର୍ଯ୍ୟାପ୍ତ ତଥ୍ୟ ନାହିଁ।"),
        "pa": ("1994 ਦਾ ਫੁੱਟਬਾਲ ਵਿਸ਼ਵ ਕੱਪ ਕਿਸਨੇ ਜਿੱਤਿਆ ਸੀ?", "ਮਨੁੱਖੀ ਦਿਲ ਪੂਰੇ ਸਰੀਰ ਵਿੱਚ ਖੂਨ ਦਾ ਸੰਚਾਰ ਕਰਦਾ ਹੈ।", "ਮੇਰੇ ਕੋਲ ਇਸਦਾ ਜਵਾਬ ਦੇਣ ਲਈ ਲੋੜੀਂਦੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।"),
        "te": ("1994 ఫుట్‌బాల్ ప్రపంచ కప్‌ను ఎవరు గెలుచుకున్నారు?", "మానవ గుండె శరీరమంతటా రక్తాన్ని సరఫరా చేస్తుంది.", "సమాధానం ఇవ్వడానికి నా వద్ద తగినంత సమాచారం లేదు."),
        "ur": ("1994 کا فٹ بال ورلڈ کپ کس نے جیتا تھا؟", "انسانی دل پورے جسم میں خون پمپ کرتا ہے۔", "میرے پاس جواب دینے کے لیے کافی معلومات نہیں ہیں۔"),
        "sa": ("१९९४ वर्षे पादकन्दुक-विश्वकपम् कः अजपत्?", "मानवहृदयं सम्पूर्णशरीरे रक्तं सञ्चारयति।", "मम समीपे अस्य उत्तरार्थं पर्याप्तं प्रमाणं नास्ति।"),
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
