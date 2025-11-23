from openai import OpenAI
from config import OPENAI_API_KEY

"""
judge.py
--------

Evaluates the quality of the support answer using an LLM.

The judge model returns:
- A numeric score between 0 and 1
- Brief reasoning

This is optional in the pipeline but gives a quality control layer.
"""

client = OpenAI(api_key=OPENAI_API_KEY)


JUDGE_PROMPT = """
You are a strict evaluation agent. 
Your job is to evaluate a support assistant's answer.

Evaluate based on:
1. Clarity (Is it easy to understand?)
2. Helpfulness (Does it solve the problem?)
3. Correctness (Any hallucinations or wrong info?)
4. Tone (Is it polite & professional?)
5. Category alignment (Is this correct for the predicted category?)

Return ONLY a JSON object with this format:

{
  "score": <float between 0 and 1>,
  "reasoning": "<short explanation>"
}
"""


def judge_answer_quality(user_message: str, assistant_answer: str, category: str):
    """
    Uses GPT to evaluate the quality of the generated support answer.
    """

    prompt = f"""
User message:
\"\"\"{user_message}\"\"\"

Predicted category: {category}

Assistant answer:
\"\"\"{assistant_answer}\"\"\"

Now evaluate using the rules above.
Return ONLY valid JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # fast + cheap
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        content = response.choices[0].message.content

        # parse JSON
        import json
        data = json.loads(content)

        score = float(data.get("score", 0.0))
        reasoning = data.get("reasoning", "")

        # normalize score
        if score < 0: score = 0.0
        if score > 1: score = 1.0

        return score, reasoning

    except Exception as e:
        # fallback if judge fails
        return 0.5, f"Judge failed ({e}). Used fallback score."


# -----------------------------------------------------------
# CLI test
# -----------------------------------------------------------
if __name__ == "__main__":

    user = "I was charged twice this month."
    answer = "Sorry, your charge seems wrong. A refund will be processed."
    category = "Billing"

    score, reasoning = judge_answer_quality(user, answer, category)

    print("\n=== JUDGE OUTPUT ===")
    print("Score:", score)
    print("Reasoning:", reasoning)
