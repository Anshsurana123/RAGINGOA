"""
Fine-Tuning Qwen2.5-0.5B-Instruct on Multilingual RAG Dataset (Google Colab / CUDA).

Requirements:
    pip install torch transformers peft trl datasets accelerate bitsandbytes

Usage in Google Colab:
    1. Upload `rag_sft_dataset.jsonl`
    2. Run: python train_colab.py
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "rag_sft_dataset.jsonl"
OUTPUT_DIR = "qwen2.5_0.5b_indic_rag_lora"
MERGED_DIR = "qwen2.5_0.5b_indic_rag_merged"

# 2. Load Dataset
print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Loaded {len(dataset)} examples. Shuffling and splitting...")
dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.05)

# 3. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Format Conversations for SFTTrainer
def format_chat_template(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

formatted_train = dataset["train"].map(format_chat_template)
formatted_eval = dataset["test"].map(format_chat_template)

# 5. Load Model with QLoRA (4-bit) if CUDA is available
is_cuda = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda}")

if is_cuda:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )

# 6. Apply LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 7. Training Arguments optimized for full 126k dataset
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16 if is_cuda else 2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    fp16=is_cuda,
    optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
    report_to="none",
    save_total_limit=1,
)

# 8. Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_train,
    eval_dataset=formatted_eval,
    dataset_text_field="text",
    max_seq_length=384,
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting LoRA Fine-Tuning...")
trainer.train()

# 9. Save LoRA Adapter
print(f"Saving LoRA Adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 10. Merge LoRA with Base Model for fast CPU inference
print("Merging LoRA weights with Base Model for CPU inference...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if is_cuda else torch.float32,
    device_map="auto" if is_cuda else "cpu",
    trust_remote_code=True
)
from peft import PeftModel
merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
merged_model = merged_model.merge_and_unload()

print(f"Saving Merged Model to {MERGED_DIR}...")
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("\n Training & Merging Completed Successfully!")
print(f"Model ready at: {MERGED_DIR}")
