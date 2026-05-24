import json
import os
from datasets import load_dataset
from tqdm import tqdm

# Configuration
# Switching to a static Parquet-based dataset to avoid "Dataset scripts" errors
DATASET_CONFIG = {
    "ultrachat": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "split": "train",
        "map_fn": "map_ultrachat"
    }
}

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "casual_chat_train.jsonl")

# Assistant-speak patterns to scrub (more aggressive now)
FORBIDDEN_PHRASES = [
    "as an ai",
    "as a language model",
    "how can i help you",
    "i am here to assist",
    "my purpose is to",
    "please let me know if",
    "i'm sorry, but i cannot",
    "i don't have feelings",
    "i am a computer program",
    "certainly!",
    "of course!",
    "i would be happy to",
    "here is a detailed",
    "in summary"
]

def is_clean(text):
    """Check if the text contains robotic assistant patterns."""
    if not text: return False
    text_lower = text.lower()
    return not any(phrase in text_lower for phrase in FORBIDDEN_PHRASES)

def map_ultrachat(example):
    """
    Maps UltraChat to ChatML format.
    UltraChat uses a 'messages' list already.
    """
    messages = example.get('messages', [])
    if not messages or len(messages) < 2:
        return None
    
    # We want to ensure the conversation doesn't sound like a textbook
    # We filter out any conversation where the assistant sounds too robotic
    for msg in messages:
        if msg['role'] == 'assistant':
            if not is_clean(msg['content']):
                return None
    
    # We inject a casual system prompt for the base training
    final_messages = [{"role": "system", "content": "You are a friendly person chatting casually."}]
    final_messages.extend(messages)
    
    return {"messages": final_messages}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_processed_data = []
    
    for ds_name, config in DATASET_CONFIG.items():
        print(f"Processing {ds_name}...")
        try:
            # load_dataset for Parquet files is fast and doesn't require trust_remote_code
            dataset = load_dataset(config['path'], split=config['split'])
        except Exception as e:
            print(f"Critical failure loading {ds_name}: {e}")
            continue
        
        map_func = globals()[config['map_fn']]
        
        # Process a subset (e.g., 50k) to keep training fast for the base model
        subset_size = min(50000, len(dataset))
        for i in tqdm(range(subset_size)):
            example = dataset[i]
            processed = map_func(example)
            if processed:
                all_processed_data.append(processed)
    
    print(f"Total samples collected: {len(all_processed_data)}")
    
    if not all_processed_data:
        print("Error: No data was processed. Even the static dataset failed.")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_processed_data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully saved processed data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
