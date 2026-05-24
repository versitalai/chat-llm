from src.inference.persona_engine import PersonaEngine
import os

def main():
    # Use a small model for the CLI demo
    # In a real scenario, you'd load the model from 'models/chat-llm-base'
    model_path = "models/chat-llm-base"
    if not os.path.exists(model_path):
        print("Trained model not found. Falling back to SmolLM-135M for demo...")
        model_id = "HuggingFaceTB/SmolLM-135M"
    else:
        model_id = model_path

    engine = PersonaEngine(model_id=model_id)

    print("\n--- Welcome to Chat-LLM Persona CLI ---")
    print("Commands: /persona <name> <description> | /quit")
    
    # Default Persona
    current_persona = {
        "name": "Default",
        "description": "A helpful and casual chat partner.",
        "examples": []
    }
    engine.set_persona(current_persona)

    while True:
        user_input = input(f"\n[{current_persona['name']}] You: ")
        
        if user_input.lower() == "/quit":
            break
        
        if user_input.startswith("/persona"):
            try:
                # Format: /persona Name Description
                parts = user_input.split(" ", 2)
                name = parts[1]
                desc = parts[2]
                current_persona = {
                    "name": name,
                    "description": desc,
                    "examples": []
                }
                engine.set_persona(current_persona)
                print(f"Persona updated to {name}!")
                continue
            except IndexError:
                print("Usage: /persona <name> <description>")
                continue

        response = engine.chat(user_input)
        print(f"{current_persona['name']}: {response}")

if __name__ == "__main__":
    main()
