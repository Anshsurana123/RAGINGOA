import json
import datasets
import os

# Language mapping from your file structure
_LANGUAGES = [
    "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", 
    "sa", "ta", "te", "ur"
]

_LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali", 
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu"
}

class MSMarcoTranslationsConfig(datasets.BuilderConfig):
    """BuilderConfig for MS MARCO Translations dataset."""
    
    def __init__(self, language=None, **kwargs):
        super(MSMarcoTranslationsConfig, self).__init__(**kwargs)
        self.language = language

class MSMarcoTranslations(datasets.GeneratorBasedBuilder):
    """MS MARCO dataset translated to Indic languages."""
    
    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        MSMarcoTranslationsConfig(
            name=lang,
            language=lang,
            version=VERSION,
            description=f"MS MARCO dataset translated to {_LANGUAGE_NAMES[lang]} ({lang})"
        ) for lang in _LANGUAGES
    ]
    
    DEFAULT_CONFIG_NAME = "hi"  # Default to Hindi
    
    def _info(self):
        return datasets.DatasetInfo(
            description="MS MARCO dataset translated to various Indic languages",
            features=datasets.Features({
                # Translation metadata
                "source_lang": datasets.Value("string"),
                "target_lang": datasets.Value("string"),
                "meta": datasets.Features({
                    "model_name": datasets.Value("string"),
                    "temperature": datasets.Value("float32"),
                    "max_tokens": datasets.Value("int32"),
                    "top_p": datasets.Value("float32"),
                    "frequency_penalty": datasets.Value("float32"),
                    "presence_penalty": datasets.Value("float32"),
                }),
                
                # Main content
                "query": datasets.Value("string"),  # Translated query
                "Answer": datasets.Value("string"),  # Translated answer
                "query_id": datasets.Value("int32"),
                "query_type": datasets.Value("string"),
                
                # Passages in both languages
                "passages": datasets.Features({
                    "is_selected": datasets.Sequence(datasets.Value("int32")),
                    "English_passages": datasets.Sequence(datasets.Value("string")),
                    "Translated_passages": datasets.Sequence(datasets.Value("string")),
                }),
                
                # Original English content
                "Eng_Query": datasets.Value("string"),
                "Eng_Answer": datasets.Value("string"),
            }),
            supervised_keys=None,
            homepage="https://microsoft.github.io/msmarco/",
            citation="""@misc{msmarco,
title={MS MARCO: A Human Generated MAchine Reading COmprehension Dataset},
author={Tri Nguyen and Mir Rosenberg and Xia Song and Jianfeng Gao and Saurabh Tiwary and Rangan Majumder and Li Deng},
year={2016},
eprint={1611.09268},
archivePrefix={arXiv},
primaryClass={cs.CL}
}""",
        )
    
    def _split_generators(self, dl_manager):
        language = self.config.language
        
        # Define file paths based on your folder structure
        urls = {}
        
        # Check for train file
        train_file = f"train/{language}train.jsonl"
        urls["train"] = train_file
        
        # Check for validation file
        val_file = f"validation/{language}val.jsonl"
        urls["validation"] = val_file
        
        # Download files
        downloaded_files = dl_manager.download(urls)
        
        splits = []
        
        # Add train split
        if "train" in downloaded_files:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": downloaded_files["train"]}
                )
            )
        
        # Add validation split
        if "validation" in downloaded_files:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": downloaded_files["validation"]}
                )
            )
        
        return splits
    
    def _generate_examples(self, filepath):
        """Generate examples from the dataset files."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract and clean the data according to your format
                        yield idx, {
                            # Translation metadata
                            "source_lang": data.get("source_lang", ""),
                            "target_lang": data.get("target_lang", ""),
                            "meta": {
                                "model_name": data.get("meta", {}).get("model_name", ""),
                                "temperature": float(data.get("meta", {}).get("temperature", 0.0)),
                                "max_tokens": int(data.get("meta", {}).get("max_tokens", 0)),
                                "top_p": float(data.get("meta", {}).get("top_p", 1.0)),
                                "frequency_penalty": float(data.get("meta", {}).get("frequency_penalty", 0.0)),
                                "presence_penalty": float(data.get("meta", {}).get("presence_penalty", 0.0)),
                            },
                            
                            # Main content
                            "query": data.get("query", ""),  # Translated query
                            "Answer": data.get("Answer", ""),  # Translated answer
                            "query_id": int(data.get("query_id", 0)),
                            "query_type": data.get("query_type", ""),
                            
                            # Passages
                            "passages": {
                                "is_selected": data.get("passages", {}).get("is_selected", []),
                                "English_passages": data.get("passages", {}).get("English_passages", []),
                                "Translated_passages": data.get("passages", {}).get("Translated_passages", []),
                            },
                            
                            # Original English content
                            "Eng_Query": data.get("Eng_Query", ""),
                            "Eng_Answer": data.get("Eng_Answer", ""),
                        }
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {idx}: {e}")
                        continue
                    except (ValueError, TypeError) as e:
                        print(f"Error processing data in line {idx}: {e}")
                        continue