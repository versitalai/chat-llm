import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import SFTTrainer
import os
import json

# Configuration
MODEL_ID = "HuggingFaceTB/SmolLM-135M" # Using a tiny model for accessibility
DATA_PATH = "data/processed/casual_chat_train.jsonl"
OUTPUT_DIR = "models/chat-llm-base"

def train():
    print(f"Starting SFT training for {MODEL_ID}...")
    
    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # 2. Load Processed Dataset
    def gen_from_jsonl(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 3. Setup SFT Trainer
    # We use the ChatML format processed in preprocess.py
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="messages", # SFTTrainer handles the ChatML list format
        max_seq_length=512,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=1,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            fp16=torch.cuda.is_available(),
        ),
    )

    print("Beginning training loop...")
    trainer.train()
    
    # 4. Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
