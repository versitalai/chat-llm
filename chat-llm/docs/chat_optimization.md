# Optimization Strategy: Casual Chatting vs. Info Fetching

To move away from the "robotic assistant" feel and toward a "casual chat" persona, we will focus on the following dimensions:

## 1. Dataset Curation
Instead of focusing on Wikipedia, textbooks, or Q&A pairs (like SQuAD), we will prioritize:
- **Dialogue Corpora:** High-quality movie scripts, novel dialogues, and curated social media threads.
- **Persona-Driven Data:** Datasets that exhibit consistent personality traits, humor, and emotional range.
- **Multi-turn Conversations:** Prioritizing long-form interactions to improve coherence and memory of the conversation flow.

## 2. Training Objectives
- **DPO (Direct Preference Optimization):** Use preference pairs where "casual/natural" responses are ranked higher than "informative/robotic" ones.
- **SFT (Supervised Fine-Tuning):** Fine-tune on "style-rich" datasets that emphasize conversational fillers, colloquialisms, and varied sentence structures.

## 3. Inference Tuning
- **Temperature & Top-P:** Implement dynamic sampling to allow for more creativity and less predictability in casual conversation.
- **System Prompting:** Design system prompts that explicitly discourage "As an AI language model..." phrasing and encourage a specific, relatable persona.

## 4. Evaluation Metrics
Standard benchmarks (MMLU, GSM8K) are less relevant here. We will use:
- **Human Preference:** A/B testing for "naturalness."
- **Engagement Metrics:** Measuring the length and depth of conversations.
- **Persona Consistency:** Evaluating if the model maintains its character over long interactions.
