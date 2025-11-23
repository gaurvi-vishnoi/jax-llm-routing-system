import os
import json
import pandas as pd
from config import CLEAN_CSV


"""
This script converts tickets_small.csv into a fine-tuning dataset for OpenAI.

Each training example becomes a ChatML dialog:

{
  "messages": [
     {"role": "system", "content": "You are a helpful support assistant..."},
     {"role": "user", "content": "<ticket text>"},
     {"role": "assistant", "content": "<high quality LLM generated answer>"}
  ]
}

Later, create_finetune.py will upload this JSONL to OpenAI and start fine-tuning.
"""


OUT_PATH = "llm/finetune_data.jsonl"

SYSTEM_PROMPT = (
    "You are a highly skilled customer support assistant. "
    "Your job is to answer customer issues clearly, politely, and helpfully. "
    "Give short, direct responses in simple language."
)

# Simple heuristic to generate a synthetic answer

def generate_answer(text, category):
    """
    For now, a basic synthetic answer.  
    Later, orchestrator.py can replace this with OpenAI GPT calls.
    """
    if category == "Billing":
        return "I understand you're facing a billing issue. I've reviewed your account and the charge will be corrected."
    if category == "Refund":
        return "I've started your refund request. You will receive the amount in 3-5 business days."
    if category == "Account":
        return "It seems like an account access issue. Please reset your password using the link we sent."
    if category == "Bug":
        return "Thanks for reporting this bug. Our engineering team is investigating the issue."
    if category == "Technical":
        return "Let me help you with this technical issue. Please try restarting and updating the app."

    # fallback
    return "Thank you for contacting support. We are here to help."



# Build JSONL fine-tuning dataset

def build_finetune_data():
    print("Loading cleaned dataset:", CLEAN_CSV)
    df = pd.read_csv(CLEAN_CSV)

    os.makedirs("llm", exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = row["text"]
            category = row["category"]

            answer = generate_answer(text, category)

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer}
                ]
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("Fine-tuning dataset created successfully!")
    print("Saved at:", OUT_PATH)


if __name__ == "__main__":
    build_finetune_data()
