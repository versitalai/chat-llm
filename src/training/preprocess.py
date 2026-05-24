import json
import os
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_CONFIG = {
    "empathetic_dialogues": {
        "path": "facebook/empathetic_dialogues",
        "split": "train",
        "map_fn": "map_empathetic"
    },
    "personachat": {
        "path": "bavard/personachat_truecased",
        "split": "train",
        "map_fn": "map_personachat"
    }
}

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "casual_chat_train.jsonl")

# Assistant-speak patterns to scrub
FORBIDDEN_PHRASES = [
    "as an ai",
    "as a language model",
    "how can i help you",
    "i am here to assist",
    "my purpose is to",
    "please let me know if",
    "i'm sorry, but i cannot",
    "i don't have feelings",
    "i am a computer program"
]

def is_clean(text):
    """Check if the text contains robotic assistant patterns."""
    if not text: return False
    text_lower = text.lower()
    return not any(phrase in text_lower for phrase in FORBIDDEN_PHRASES)

def map_empathetic(example):
    """
    Maps Empathetic Dialogues to ChatML format.
    """
    # Some versions of this dataset use 'utterances'
    utterances = example.get('utterances', [])
    if not utterances or len(utterances) < 2:
        return None
    
    messages = []
    messages.append({"role": "system", "content": "You are a friendly and empathetic person chatting casually."})
    
    for i in range(0, len(utterances), 2):
        user_text = utterances[i]
        ai_text = utterances[i+1] if i+1 < len(utterances) else ""
        
        if not is_clean(ai_text):
            return None
            
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": ai_text})
        
    return {"messages": messages}

def map_personachat(example):
    """
    Maps PersonaChat to ChatML format.
    """
    persona = example.get('persona', "")
    dialogue = example.get('dialogue', "")
    
    if not dialogue: return None
    
    turns = dialogue.split(' ')
    if len(turns) < 2:
        return None
        
    messages = []
    messages.append({"role": "system", "content": f"You are a person with the following traits: {persona}"})
    
    for i in range(0, len(turns), 2):
        user_text = turns[i]
        ai_text = turns[i+1] if i+1 < len(turns) else ""
        
        if not is_clean(ai_text):
            return None
            
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": ai_text})
        
    return {"messages": messages}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_processed_data = []
    
    for ds_name, config in DATASET_CONFIG.items():
        print(f"Processing {ds_name}...")
        try:
            # Try to load without trust_remote_code first, 
            # but if it's a script-based dataset, we use a different approach:
            # We can try loading via 'parquet' if the dataset has been converted, 
            # but for the standard names, we'll try the most compatible load.
            dataset = load_dataset(config['path'], split=config['split'])
        except Exception as e:
            print(f"Standard load failed for {ds_name}: {e}")
            print("Attempting to load with trust_remote_code=True...")
            try:
                dataset = load_dataset(config['path'], split=config['split'], trust_remote_code=True)
            except Exception as e2:
                print(f"All load attempts failed for {ds_name}: {e2}")
                continue
        
        map_func = globals()[config['map_fn']]
        
        for example in tqdm(dataset):
            processed = map_func(example)
            if processed:
                all_processed_data.append(processed)
    
    print(f"Total samples collected: {len(all_processed_data)}")
    
    if not all_processed_data:
        print("Error: No data was processed. Please check dataset paths.")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_processed_data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully saved processed data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
