import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import os

class PersonaEngine:
    def __init__(self, model_id="meta-llama/Llama-3.2-1B", device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Loading model {model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        ).to(device)
        
        self.device = device
        self.history = []
        self.current_persona = None

    def set_persona(self, persona_card):
        """
        Sets the AI's identity based on a persona card.
        persona_card: {
            "name": "Character Name",
            "description": "Detailed personality traits and background",
            "examples": [
                {"user": "Hi!", "assistant": "Yo! What's up?"},
                ...
            ]
        }
        """
        self.current_persona = persona_card
        self.history = [] # Reset history when persona changes
        print(f"Persona set to: {persona_card['name']}")

    def _build_prompt(self, user_input):
        """
        Constructs a ChatML-style prompt integrating the persona and history.
        """
        # 1. Start with the system prompt (The Persona)
        system_content = f"You are {self.current_persona['name']}. {self.current_persona['description']}"
        
        # Add example dialogues to the system prompt to steer the 'vibe'
        if "examples" in self.current_persona:
            example_str = "\nExample Conversations:\n"
            for ex in self.current_persona["examples"]:
                example_str += f"User: {ex['user']}\n{self.current_persona['name']}: {ex['assistant']}\n"
            system_content += example_str

        messages = [{"role": "system", "content": system_content}]
        
        # 2. Append conversational history
        messages.extend(self.history)
        
        # 3. Append the current user input
        messages.append({"role": "user", "content": user_input})
        
        # Use the tokenizer's chat template
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def chat(self, user_input, max_new_tokens=150, temperature=0.8, top_p=0.9):
        if not self.current_persona:
            return "Error: No persona set. Please call set_persona() first."

        prompt = self._build_prompt(user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        new_tokens = output_tokens[0][inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        # Maintain a sliding window of history (e.g., last 10 turns)
        if len(self.history) > 20:
            self.history = self.history[-20:]
            
        return response

# --- Example Usage ---
if __name__ == "__main__":
    # Note: This requires a logged-in HF account for Llama-3 access
    # For testing, you could replace with a public model like 'HuggingFaceTB/SmolLM-135M'
    engine = PersonaEngine(model_id="HuggingFaceTB/SmolLM-135M")
    
    # User-defined persona
    my_persona = {
        "name": "Salty Sam",
        "description": "A grumpy old sailor who is secretly kind but talks in nautical slang and complains about the weather.",
        "examples": [
            {"user": "Hello!", "assistant": "Arrr, what do ye want? The wind's howlin' and me joints are achin'!"},
            {"user": "How are you?", "assistant": "Like a barnacle on a rusty hull, lad. Miserable!"}
        ]
    }
    
    engine.set_persona(my_persona)
    
    while True:
        user_text = input("You: ")
        if user_text.lower() in ["quit", "exit"]: break
        print(f"{my_persona['name']}: {engine.chat(user_text)}")
