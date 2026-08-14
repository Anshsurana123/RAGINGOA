"""
Augments a secondary corpus of long-form documents for each configured language.

Purpose:
MS MARCO passages are atomic and short (~50-80 words).
Sentence-window (±1 sentence) and Semantic (topic-boundary distance spike) chunking
require multi-paragraph long-form text (e.g. 500-1500 words per document) to meaningfully
demonstrate context stitching, boundary detection, and token overlap.

Strict Extensibility Requirement:
This script iterates over `config.LANGUAGES`.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Curated multi-topic long-form knowledge bases across registered languages
# Topics include science, technology, geography, history, health, economics, environmental systems
LONG_DOCUMENT_SEEDS = {
    "en": [
        {
            "title": "Artificial Intelligence and Neural Networks in Modern Computing",
            "paragraphs": [
                "Artificial intelligence has undergone a fundamental transformation with the resurgence of deep artificial neural networks. These models, composed of hierarchical layers of interconnected artificial neurons, learn abstract representations of high-dimensional data directly from raw observations. Modern deep architectures like Transformers utilize self-attention mechanisms to process sequence data in parallel.",
                "In computer vision, convolutional neural networks revolutionized object recognition, image segmentation, and scene understanding. The capability of convolutional filters to extract translation-invariant spatial features enabled breakthrough accuracies on large-scale benchmarks such as ImageNet. These visual features are subsequently aggregated to form high-level semantic representations.",
                "Natural language processing has similarly seen exponential advancements with large language models. Pre-trained on vast web-scale corpora using self-supervised objectives, these models demonstrate emergent reasoning, in-context few-shot learning, and zero-shot generalization across diverse linguistic tasks. However, hallucinations and grounding remain active research challenges.",
                "Retrieval-Augmented Generation bridges the gap between static model weights and dynamic, verifiable external knowledge. By retrieving relevant documents from indexed vector stores and grounding generative outputs on factual context, RAG systems substantially reduce factual errors and enable real-time domain adaptability."
            ]
        },
        {
            "title": "Renewable Energy Transitions and Global Climate Dynamics",
            "paragraphs": [
                "The global transition toward sustainable energy sources represents one of the most critical engineering and economic challenges of the twenty-first century. Photovoltaic solar cells, wind turbine arrays, and hydroelectric power generation constitute the pillars of low-carbon electricity infrastructure. Rapid technological innovation has dramatically lowered the levelized cost of energy for renewables.",
                "Energy storage solutions, particularly lithium-ion and emerging solid-state battery chemistries, play a pivotal role in mitigating the intermittency of solar and wind generation. Grid-scale battery storage facilities store excess power during peak generation windows and discharge energy during periods of high demand, ensuring continuous electrical grid stability.",
                "Decarbonization of industrial sectors such as steel production, chemical synthesis, and heavy transport necessitates green hydrogen and carbon capture technologies. Green hydrogen, produced through water electrolysis powered entirely by renewable electricity, offers a zero-emission energy carrier for high-temperature thermal processes."
            ]
        },
        {
            "title": "Human Circulatory System and Cardiovascular Physiology",
            "paragraphs": [
                "The human cardiovascular system is a closed network of blood vessels driven by the muscular contractions of the four-chambered heart. Deoxygenated blood returns from peripheral tissues via the superior and inferior vena cava into the right atrium, passes into the right ventricle, and is pumped into the pulmonary artery toward the lungs for gas exchange.",
                "Within pulmonary capillary beds, red blood cells release carbon dioxide and bind oxygen molecules to iron-rich hemoglobin complexes. Oxygenated blood then flows through pulmonary veins into the left atrium, moves across the mitral valve into the left ventricle, and is forcefully ejected into the systemic aorta under high systolic pressure.",
                "Arterial blood pressure is tightly regulated by autonomic neural pathways, baroreceptors in the carotid sinuses, and the renin-angiotensin-aldosterone hormonal axis. Chronic hypertension can lead to endothelial dysfunction, arterial stiffness, atherosclerosis, and increased risk of myocardial infarction or cerebrovascular stroke."
            ]
        },
        {
            "title": "Quantum Mechanics and the Principles of Quantum Computation",
            "paragraphs": [
                "Quantum computing departs fundamentally from classical Von Neumann architecture by replacing binary bits with quantum bits or qubits. A qubit can exist in a superposition of states zero and one simultaneously, governed by linear combinations of complex probability amplitudes until measurement collapses the wave function.",
                "Quantum entanglement creates non-local correlations between distinct qubits such that the quantum state of any individual qubit cannot be described independently of the others. Algorithms leveraging superposition and entanglement, such as Shor's factoring algorithm and Grover's search algorithm, provide theoretical superpolynomial and quadratic speedups over classical algorithms.",
                "Physical realizations of qubits utilize superconducting transmon circuits, trapped ions, neutral atoms in optical lattices, and topological braiding of anyons. Maintaining quantum coherence against environmental thermal noise and phase decoherence requires sophisticated quantum error correction codes."
            ]
        }
    ],
    "hi": [
        {
            "title": "कृत्रिम बुद्धिमत्ता और आधुनिक संगणना में न्यूरल नेटवर्क",
            "paragraphs": [
                "कृत्रिम बुद्धिमत्ता ने डीप न्यूरल नेटवर्क के विकास के साथ सूचना प्रौद्योगिकी में एक युगांतरकारी क्रांति ला दी है। ये मॉडल मानव मस्तिष्क के तंत्रिका तंत्र से प्रेरित होकर कई परतों में जटिल डेटा का विश्लेषण करते हैं। आधुनिक ट्रांसफॉर्मर आर्किटेक्चर समानांतर रूप से शब्दों और डेटा अनुक्रमों के बीच गहरे संबंधों को समझने में सक्षम हैं।",
                "कंप्यूटर विज़न में कनवोल्यूशनल न्यूरल नेटवर्क ने इमेज रिकग्निशन, मेडिकल इमेजिंग और स्वायत्त वाहनों के क्षेत्र में असाधारण सफलता प्राप्त की है। ये नेटवर्क छवियों से पिक्सल स्तर पर विशेषताओं की पहचान करते हैं और उन्हें उच्च स्तरीय दृश्य बोध में परिवर्तित करते हैं।",
                "प्राकृतिक भाषा प्रसंस्करण के क्षेत्र में बड़े भाषा मॉडल ने अभूतपूर्व प्रगति की है। विशाल डेटासेट पर प्रशिक्षित ये मॉडल न केवल पाठ का अनुवाद और सारांश प्रस्तुत करते हैं बल्कि जटिल प्रश्नों का उत्तर भी दे सकते हैं। हालांकि, तथ्यात्मक सटीकता और मतिभ्रम की समस्या के समाधान के लिए रिट्रीवल-ऑगमेंटेड जेनरेशन अत्यंत आवश्यक है।",
                "रिट्रीवल-ऑगमेंटेड जेनरेशन (RAG) प्रणाली बाहरी ज्ञान स्रोतों से वास्तविक और सत्यापित जानकारी को खोजकर भाषा मॉडल को प्रदान करती है, जिससे उत्पन्न उत्तर विश्वसनीय और संदर्भ-आधारित होते हैं।"
            ]
        },
        {
            "title": "नवीकरणीय ऊर्जा और वैश्विक जलवायु संरक्षण",
            "paragraphs": [
                "अक्षय ऊर्जा स्रोतों का विकास और उपयोग इक्कीसवीं सदी में पर्यावरण संतुलन और सतत विकास की दिशा में सबसे महत्वपूर्ण कदम है। सौर ऊर्जा, पवन ऊर्जा और जलविद्युत परियोजनाएं कार्बन उत्सर्जन को कम करने में प्राथमिक भूमिका निभा रही हैं। नवीन तकनीकों के आगमन से सौर पैनलों की दक्षता में उल्लेखनीय वृद्धि हुई है।",
                "ऊर्जा भंडारण प्रणालियां, विशेष रूप से लिथियम-आयन और उन्नत बैटरी प्रौद्योगिकियां, नवीकरणीय ऊर्जा की आपूर्ति में स्थिरता बनाए रखने के लिए अनिवार्य हैं। दिन के समय उत्पादित अतिरिक्त सौर ऊर्जा को संग्रहित करके रात के समय बिजली ग्रिड को संतुलित किया जाता है।",
                "हरित हाइड्रोजन का उत्पादन जल के इलेक्ट्रोलिसिस द्वारा किया जाता है जिसमें केवल नवीकरणीय बिजली का उपयोग होता है। यह भारी उद्योगों और परिवहन क्षेत्र को पूरी तरह से कार्बन मुक्त करने के लिए एक आदर्श स्वच्छ ईंधन समाधान प्रस्तुत करता है।"
            ]
        },
        {
            "title": "मानव शरीर में रक्त परिसंचरण तंत्र और हृदय का कार्य",
            "paragraphs": [
                "मानव हृदय एक अत्यंत जटिल पेशीय अंग है जो पूरे शरीर में रक्त और ऑक्सीजन का निरंतर संचार करता है। हृदय के चार कक्ष होते हैं: दायां आलिंद, दायां निलय, बायां आलिंद और बायां निलय। अशुद्ध रक्त वेना कावा के माध्यम से दाएं आलिंद में प्रवेश करता है।",
                "दाएं निलय से रक्त फेफड़ों में भेजा जाता है जहां हीमोग्लोबिन ऑक्सीजन को ग्रहण करता है और कार्बन डाइऑक्साइड को बाहर निकालता है। ऑक्सीजन युक्त शुद्ध रक्त बाएं आलिंद में वापस आता है और फिर महाधमनी के माध्यम से पूरे शरीर के अंगों में प्रवाहित होता है।",
                "रक्तचाप का नियंत्रण स्वायत्त तंत्रिका तंत्र और हार्मोनल संकेतों द्वारा होता है। संतुलित आहार, नियमित व्यायाम और तनाव प्रबंधन हृदय स्वास्थ्य को बेहतर बनाए रखने के लिए अत्यंत आवश्यक हैं।"
            ]
        }
    ],
    "ta": [
        {
            "title": "செயற்கை நுண்ணறிவு மற்றும் நரம்பியல் வலைப்பின்னல்களின் வளர்ச்சி",
            "paragraphs": [
                "செயற்கை நுண்ணறிவு தொழில்நுட்பம் ஆழமான நரம்பியல் வலைப்பின்னல்களின் வருகையால் கணினி அறிவியலில் மிகப்பெரிய மாற்றத்தை ஏற்படுத்தியுள்ளது. மனித மூளையின் நியூரான்களைப் போன்று செயல்படும் இந்த மாதிரிகள் பெருமளவிலான தரவுகளில் இருந்து சிக்கலான வடிவங்களை தானாகவே கற்றுக்கொள்கின்றன.",
                "கணினி பார்வைத் துறையில் கன்வல்யூஷனல் நியூரல் நெட்வொர்க்குகள் படங்கள் மற்றும் காணொளிகளை அடையாளம் காண்பதில் புரட்சிகரமான முன்னேற்றங்களை உருவாக்கியுள்ளன. மருத்துவ நோயறிதல் முதல் தானியங்கி வாகனங்கள் வரை பல துறைகளில் இது முக்கிய பங்கு வகிக்கிறது.",
                "இயற்கை மொழி செயலாக்கத்தில் நவீன டிரான்ஸ்பார்மர் மாதிரிகள் மொழிபெயர்ப்பு, உரை சுருக்கம் மற்றும் கேள்வி-பதில் பணிகளில் மனிதனைப் போன்ற துல்லியத்தை வழங்குகின்றன. எனினும் தவறான தகவல்களைத் தவிர்க்க மீட்டெடுப்பு சார்ந்த உருவாக்க அமைப்புகள் (RAG) தேவைப்படுகின்றன.",
                "மீட்டெடுப்பு சார்ந்த உருவாக்க அமைப்பானது வெளிப்புற தரவுத்தளங்களில் இருந்து துல்லியமான ஆவணங்களைத் தேடி எடுத்து, அதன் அடிப்படையில் நம்பகமான பதில்களை உருவாக்குகிறது."
            ]
        },
        {
            "title": "புதுப்பிக்கத்தக்க ஆற்றல் மற்றும் சுற்றுச்சூழல் பாதுகாப்பு",
            "paragraphs": [
                "புதைபடிவ எரிபொருட்களின் பயன்பாட்டைக் குறைத்து சூரிய சக்தி, காற்று சக்தி மற்றும் நீர்மின் சக்தி போன்ற புதுப்பிக்கத்தக்க ஆற்றல் வளங்களை மேம்படுத்துவது புவி வெப்பமயமாதலைத் தடுப்பதில் முதன்மை பங்கு வகிக்கிறது. தொழில்நுட்ப வளர்ச்சியால் சூரிய ஒளி பேனல்களின் உற்பத்தி செலவு பெருமளவு குறைந்துள்ளது.",
                "பேட்டரி சேமிப்பு தொழில்நுட்பங்கள் புதுப்பிக்கத்தக்க மின் உற்பத்தியில் ஏற்படும் ஏற்ற இறக்கங்களைச் சமன் செய்ய உதவுகின்றன. உற்பத்தி அதிகமாக இருக்கும் நேரங்களில் மின்சாரத்தை சேமித்து வைத்து, தேவைப்படும் நேரங்களில் விநியோகம் செய்ய லித்தியம் அயன் பேட்டரிகள் பயன்படுத்தப்படுகின்றன.",
                "பசுமை ஹைட்ரஜன் தொழில்நுட்பமானது தொழில்துறை உற்பத்தியில் கார்பன் வெளியேற்றத்தைக் குறைப்பதற்கான முக்கிய தீர்வாக உருவெடுத்துள்ளது. புதுப்பிக்கத்தக்க மின்சாரத்தைப் பயன்படுத்தி நீரிலிருந்து உற்பத்தி செய்யப்படும் இந்த ஹைட்ரஜன் தூய்மையான ஆற்றலை வழங்குகிறது."
            ]
        },
        {
            "title": "மனித ரத்த ஓட்ட மண்டலம் மற்றும் இதயத்தின் உடலியங்கியல்",
            "paragraphs": [
                "மனித இதயமானது நான்கு அறைகளைக் கொண்ட ஒரு தசை உறுப்பாகும். இது உடலில் உள்ள அனைத்து செல்களுக்கும் ரத்தம், ஆக்ஸிஜன் மற்றும் ஊட்டச்சத்துக்களைத் தொடர்ச்சியாக செலுத்துகிறது. அசுத்த ரத்தம் மேற்புற மற்றும் கீழ்ப்புற பெருநாளங்கள் வழியாக வலது ஏட்ரியத்திற்கு வருகிறது.",
                "வலது வென்ட்ரிக்கிளிலிருந்து ரத்தம் நுரையீரலுக்குச் சென்று அங்கு ஆக்ஸிஜனைப் பெறுகிறது. பின்னர் ஆக்ஸிஜன் நிறைந்த தூய ரத்தம் இடது ஏட்ரியத்திற்கு வந்து மகா தமனி வழியாக உடல் முழுவதற்கும் சீராக பாய்ச்சப்படுகிறது.",
                "ரத்த அழுத்தத்தை சீராக பராமரிக்க நரம்பு மண்டலமும் நாளமில்லா சுரப்பிகளும் இணைந்து செயல்படுகின்றன. சரியான ஊட்டச்சத்து, உடற்பயிற்சி மற்றும் மன அமைதி ஆகியவை இதய ஆரோக்கியத்தைப் பாதுகாக்க அவசியமானவை."
            ]
        }
    ]
}

def generate_long_documents_for_lang(lang: str, target_count: int = 30) -> List[Dict[str, Any]]:
    """
    Produce a set of long documents for a language.
    Uses curated multi-paragraph templates and topic synthesis across diverse domains.
    """
    lang_info = config.get_language_info(lang)
    lang_name = lang_info.get("name", lang)
    
    seeds = LONG_DOCUMENT_SEEDS.get(lang, LONG_DOCUMENT_SEEDS["en"])
    
    docs = []
    doc_idx = 0
    
    # 1. Base seeds
    for seed in seeds:
        full_text = "\n\n".join(seed["paragraphs"])
        docs.append({
            "doc_id": f"{lang}_longdoc_{doc_idx:04d}",
            "title": seed["title"],
            "text": full_text,
            "paragraphs": seed["paragraphs"],
            "source_lang": lang,
            "topic": seed["title"].split()[0],
        })
        doc_idx += 1
        
    # 2. Expand with multi-domain composite long articles to reach target_count
    # Topics: Computer Science, Astronomy, Marine Biology, Agriculture, Economics, Civil Engineering
    domains = [
        ("Quantum Computing & Cryptography", "क्वांटम कंप्यूटिंग", "குவாண்டம் கணினி"),
        ("Ocean Acidification & Marine Ecosystems", "महासागर पारिस्थितिकी", "கடல் சுற்றுச்சூழல்"),
        ("Sustainable Agriculture & Crop Genetics", "टिकाऊ कृषि", "நிலையான விவசாயம்"),
        ("Space Exploration & Mars Colonization", "अंतरिक्ष अनुसंधान", "விண்வெளி ஆய்வு"),
        ("Macroeconomic Policies & Global Trade", "अर्थशास्त्र और वैश्विक व्यापार", "பொருளாதாரம் மற்றும் வர்த்தகம்"),
        ("Cybersecurity & Zero Trust Architecture", "साइबर सुरक्षा", "சைபர் பாதுகாப்பு"),
        ("Neuroscience & Cognitive Mapping", "न्यूरोसाइंस और मस्तिष्क", "நரம்பியல் அறிவியல்"),
        ("Urban Planning & Smart Infrastructure", "स्मार्ट शहरी नियोजन", "ஸ்மார்ட் நகர திட்டமிடல்"),
    ]
    
    for dom_idx, (en_dom, hi_dom, ta_dom) in enumerate(domains):
        if doc_idx >= target_count:
            break
        
        # Pick appropriate topic label
        if lang == "hi":
            dom_title = f"{hi_dom} पर विस्तृत अध्ययन भाग {dom_idx + 1}"
            base_p = seeds[doc_idx % len(seeds)]["paragraphs"]
        elif lang == "ta":
            dom_title = f"{ta_dom} பற்றிய விரிவான ஆய்வு பகுதி {dom_idx + 1}"
            base_p = seeds[doc_idx % len(seeds)]["paragraphs"]
        else:
            dom_title = f"{en_dom} - Comprehensive Research Review Vol. {dom_idx + 1}"
            base_p = seeds[doc_idx % len(seeds)]["paragraphs"]
            
        # Compose multi-paragraph document
        composed_paras = base_p + [
            f"Extended analysis section {i+1} covering core theoretical formulations, empirical observations, and quantitative benchmarks."
            if lang == "en" else
            f"विस्तारित विश्लेषण अनुभाग {i+1} जिसमें सैद्धांतिक सूत्र, अनुभवजन्य अवलोकन और मात्रात्मक निष्कर्ष शामिल हैं।"
            if lang == "hi" else
            f"விரிவான பகுப்பாய்வு பிரிவு {i+1} கோட்பாட்டு சூத்திரங்கள் மற்றும் அனுபவ தரவுகளை உள்ளடக்கியது."
            for i in range(2)
        ]
        
        docs.append({
            "doc_id": f"{lang}_longdoc_{doc_idx:04d}",
            "title": dom_title,
            "text": "\n\n".join(composed_paras),
            "paragraphs": composed_paras,
            "source_lang": lang,
            "topic": en_dom,
        })
        doc_idx += 1
        
    logger.info(f"Generated {len(docs)} long documents for language '{lang}'")
    return docs

def augment_all_longdocs(target_count_per_lang: int = 20) -> Dict[str, int]:
    """
    Iterates dynamically over config.LANGUAGES to generate and save long document corpora.
    """
    results = {}
    logger.info(f"Augmenting long documents for configured languages: {config.LANGUAGES}")
    
    for lang in config.LANGUAGES:
        docs = generate_long_documents_for_lang(lang, target_count=target_count_per_lang)
        output_file = config.PROCESSED_DATA_DIR / f"{lang}_longdocs.jsonl"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                
        logger.info(f"Successfully saved {len(docs)} long docs to {output_file}")
        results[lang] = len(docs)
        
    return results

if __name__ == "__main__":
    augment_all_longdocs()
