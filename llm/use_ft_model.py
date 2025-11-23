import json
from openai import OpenAI
from config import OPENAI_API_KEY, CATEGORIES

"""
use_ft_model.py
----------------

This module simulates a fine-tuned customer support model using:

• Category-aware prompts
• Few-shot examples per category
• Clean structured output
• Deterministic behavior (temperature=0.2)
• Compatible with orchestrator + guardrails

This is the recommended approach when fine-tuning is not available.
"""

client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------------------------------------
# Category-Specific Few-Shot Examples
# -----------------------------------------------------------
FEW_SHOTS = {
    "Billing": [
        (
            "I was charged twice this month.",
            "I'm sorry for the double charge. I've reviewed your account and initiated a correction. You'll see the refund in 3–5 business days."
        ),
        (
            "Why was my subscription renewed?",
            "Your subscription renewed because auto-renew was enabled. I can help stop renewal or issue a refund if needed."
        ),
    ],

    "Refund": [
        (
            "I want a refund for my order.",
            "I've started your refund request. The amount will return to your payment method within 5–7 business days."
        ),
        (
            "My payment failed — refund?",
            "Failed payments are not charged, but I'm happy to double-check your account and confirm."
        ),
    ],

    "Account": [
        (
            "I cannot log in to my account.",
            "Please try resetting your password using the link we just emailed. That usually resolves access issues."
        ),
        (
            "My account was locked.",
            "Your account was locked for security. Please reset your password or contact support to regain access."
        ),
    ],

    "Bug": [
        (
            "The app keeps freezing.",
            "Thanks for reporting this. I've logged the issue with engineering. A fix will be rolled out soon."
        ),
        (
            "A feature is not working.",
            "Thanks for flagging this. We're aware of the bug and working on a resolution."
        ),
    ],

    "Technical": [
        (
            "The app is not loading.",
            "Please restart the app and check your internet connection. If it persists, reinstalling typically fixes it."
        ),
        (
            "Why can't I update?",
            "Updates require enough storage and a stable connection. Please clear space and try again."
        ),
    ],
}


# -----------------------------------------------------------
# Build Prompt (Category-Aware System Instruction)
# -----------------------------------------------------------
def build_prompt(ticket_text: str, category: str):
    """
    Builds a deterministic prompt with:
    - Base system message
    - Two few-shot examples from that category
    - Final user query
    """

    system_msg = (
        "You are a professional customer support assistant. "
        "Write short, polite, helpful responses. "
        "Keep answers actionable and avoid unnecessary details. "
        "Always address the user's problem directly."
    )

    messages = [{"role": "system", "content": system_msg}]

    # Select few-shot examples for this category
    examples = FEW_SHOTS.get(category, [])

    # Add up to 2 few-shot pairs
    for user_msg, assistant_msg in examples[:2]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # Add final user query
    messages.append({"role": "user", "content": ticket_text})

    return messages


# -----------------------------------------------------------
# Generate the Final Support Answer
# -----------------------------------------------------------
def generate_support_answer(ticket_text: str, category: str, kb_hits=None):

    rag_context = ""
    if kb_hits:
        rag_context = "Relevant Knowledge Base Information:\n"
        for hit in kb_hits:
            rag_context += f"- {hit['title']}: {hit.get('content','')}\n"

    messages = build_prompt(ticket_text, category)

    if rag_context:
        messages.append({"role": "system", "content": rag_context})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content



# -----------------------------------------------------------
# CLI Test
# -----------------------------------------------------------
if __name__ == "__main__":
    print("Testing support answer generator...\n")

    sample_text = "I was charged incorrectly this month."
    sample_cat = "Billing"

    output = generate_support_answer(sample_text, sample_cat)

    print("Category:", sample_cat)
    print("Response:", output)
