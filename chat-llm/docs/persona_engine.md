# Persona Engine Guide

The Persona Engine is the core inference wrapper for Chat-LLM. It transforms a base model into a specific character on-the-fly, ensuring that the "soul" of the AI is determined by the user, not the training.

## How it Works
Instead of baking a personality into the model's weights, the engine uses a **Dynamic System Prompt**. 

1. **Persona Card:** The engine accepts a "Persona Card" (a JSON object) containing a name, description, and a few example dialogues.
2. **Prompt Assembly:** It assembles a `ChatML` prompt:
   - `System`: [Name] + [Description] + [Examples]
   - `History`: [Last N turns of conversation]
   - `User`: [Current input]
3. **Steering:** By providing example dialogues in the system prompt, the engine steers the model's tone and cadence (e.g., making it use slang or specific quirks) without requiring a full fine-tune for every character.

## Using the Engine
To create a new character, simply define a persona card:

```python
persona = {
    "name": "Cyber-Punk Fixer",
    "description": "A street-smart dealer of illegal tech in a neon city. Speaks in short, clipped sentences.",
    "examples": [
        {"user": "Got any decks?", "assistant": "Depends on your credits. Check the alley at midnight."}
    ]
}
engine.set_persona(persona)
```

## Configuration
The `chat()` method allows tuning for different "vibes":
- **High Temperature (0.8 - 1.2):** More creative, unpredictable, and "human" (ideal for casual chat).
- **Low Temperature (0.2 - 0.5):** More consistent and focused (ideal for structured roleplay).
