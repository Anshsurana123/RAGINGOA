---
title: RAGINGOA - Voice-Enabled Indic RAG
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# MS MARCO Translations Dataset

## Dataset Description

This dataset contains the MS MARCO dataset translated into various Indic languages. The original MS MARCO dataset is a collection of queries, passages, and answers for machine reading comprehension and question answering tasks. Each example includes both the original English content and the translated content, along with translation metadata.

## Supported Languages

| Language Code | Language Name | Train File | Validation File |
|---------------|---------------|------------|-----------------|
| as | Assamese | asmtrain.jsonl | asmval.jsonl |
| bn | Bengali | bentrain.jsonl | benval.jsonl |
| gu | Gujarati | gutrain.jsonl | guval.jsonl |
| hi | Hindi | hintrain.jsonl | hinval.jsonl |
| kn | Kannada | kantrain.jsonl | kanval.jsonl |
| ml | Malayalam | maltrain.jsonl | malval.jsonl |
| mr | Marathi | martrain.jsonl | marval.jsonl |
| ne | Nepali | neptrain.jsonl | nepval.jsonl |
| or | Odia | ortrain.jsonl | orval.jsonl |
| pa | Punjabi | pantrain.jsonl | panval.jsonl |
| sa | Sanskrit | santrain.jsonl | sanval.jsonl |
| ta | Tamil | tamtrain.jsonl | tamval.jsonl |
| te | Telugu | teltrain.jsonl | telval.jsonl |
| ur | Urdu | urdtrain.jsonl | urdval.jsonl |

## Usage

```python
from datasets import load_dataset

# Load Hindi training data
dataset = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")


# Access the data
for example in dataset:
    print(f"Query: {example['query']}")
    print(f"Answers: {example['answers']}")
    print(f"Passages: {len(example['passages'])}")
    break
```

## Dataset Structure

Each example in the dataset contains:

### Translation Metadata
- `source_lang` (string): Source language code (e.g., "eng_Latn")
- `target_lang` (string): Target language code (e.g., "asm_Beng")  
- `meta` (dict): Translation model metadata including:
  - `model_name` (string): Name of the translation model used
  - `temperature` (float): Sampling temperature
  - `max_tokens` (int): Maximum tokens generated
  - `top_p` (float): Top-p sampling parameter
  - `frequency_penalty` (float): Frequency penalty
  - `presence_penalty` (float): Presence penalty

### Main Content
- `query` (string): The translated search query
- `Answer` (string): The translated answer
- `query_id` (int): Unique identifier for the query
- `query_type` (string): Type/category of the query

### Passages
- `passages` (dict): Contains passage information:
  - `is_selected` (list): List indicating which passages are selected (1) or not (0)
  - `English_passages` (list): List of original English passages
  - `Translated_passages` (list): List of translated passages

### Original English Content
- `Eng_Query` (string): Original English query
- `Eng_Answer` (string): Original English answer

## Example

```python
{
    "source_lang": "eng_Latn",
    "target_lang": "asm_Beng", 
    "meta": {
        "model_name": "ckpt-3epochs-sft-then-400k-kd",
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    },
    "query": "মেনহাটন প্ৰকল্পৰ সফলতাৰ তাৎক্ষণিক প্ৰভাৱ কি আছিল?",
    "Answer": "মেনহাটন প্ৰকল্পৰ সফলতাৰ তাৎক্ষণিক প্ৰভাৱ আছিল...",
    "query_id": 1185869,
    "query_type": "DESCRIPTION",
    "passages": {
        "is_selected": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "English_passages": ["The presence of communication amid scientific minds...", ...],
        "Translated_passages": ["বৈজ্ঞানিক মনৰ মাজত যোগাযোগৰ উপস্থিতি...", ...]
    },
    "Eng_Query": ")what was the immediate impact of the success of the manhattan project?",
    "Eng_Answer": "The immediate impact of the success of the manhattan project was..."
}
```
## 📖 Citation

If you use IndicMSMARCO in your research, please cite:

```bibtex
@dataset{indic_msmarco_2024,
  title={IndicRAGSuite: LargeScale Datasets and a Benchmark for Indian Language RAG Systems},
  author={Pasunuti Prasanjith,Prathmesh B More,Anoop Kunchukuttan, Raj Dabre},
  year={2025},
  {journal = {arXiv preprint arXiv:2506.01615},
  url={https://huggingface.co/datasets/ai4bharat/IndicMSMARCO}
}
```
## License

Please refer to the original MS MARCO dataset license terms.