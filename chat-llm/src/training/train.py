import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import os
import json

# Configuration
MODEL_ID = "HuggingFaceTB/SmolLM-135M"
DATA_PATH = "data/processed/casual_chat_train.jsonl"
OUTPUT_DIR = "models/chat-llm-base"

# Explicit ChatML Template String
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)

def train():
    print(f"Starting SFT training for {MODEL_ID}...")
    
    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    # We still set it on the tokenizer just in case
    tokenizer.chat_template = CHAT_TEMPLATE
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # 2. Load Processed Dataset
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 3. Setup SFT Trainer
    # FIX: We pass the chat_template EXPLICITLY to the trainer to override any 
    # internal tokenizer failures and avoid the ValueError.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
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
        processing_class=tokenizer,
    )
    
    # Manually assign the template to the trainer's processing class just to be safe
    trainer.processing_class.chat_template = CHAT_TEMPLATE

    print("Beginning training loop...")
    trainer.train()
    
    # 4. Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
