import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    ROUTER_MODEL_PATH,
    ROUTER_CONFIG,
    CATEGORIES,
    NUM_CLASSES,
)

from pipeline.compute_embeddings import compute_single_embedding
from llm.use_ft_model import generate_support_answer
from pipeline.judge import judge_answer_quality
from pipeline.logger import log_interaction

# Import router architecture
from models.jax_router import RouterModel


"""
orchestrator.py
----------------

Main brain of the system:

1. Embed user message
2. Load saved JAX router parameters
3. Rebuild Flax router model
4. Predict category + confidence
5. Apply guardrails (fallback if low confidence)
6. Generate LLM support answer
7. Judge the final answer quality
"""

# ----------------------------------------------
# 🔹 Load router parameters + Rebuild Flax model
# ----------------------------------------------
def load_router_model():
    if not os.path.exists(ROUTER_MODEL_PATH):
        raise FileNotFoundError(f"Router model not found: {ROUTER_MODEL_PATH}")

    print(f"📦 Loading router model: {ROUTER_MODEL_PATH}")

    with open(ROUTER_MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    params = data["params"]  # Saved parameters only

    # Rebuild the architecture
    model = RouterModel(
        hidden_dim=ROUTER_CONFIG["hidden_dim"],
        num_classes=NUM_CLASSES
    )

    return params, model.apply


# ----------------------------------------------
# 🔹 Category Prediction
# ----------------------------------------------
def predict_category(user_message: str):
    """
    Convert text → embedding → JAX router → category prediction.
    """

    # 1. Embed the message
    emb = compute_single_embedding(user_message)
    emb = jnp.array(emb)[None, :]    # Shape: (1, embedding_dim)

    # 2. Load model
    params, apply_fn = load_router_model()

    # 3. Forward pass → logits
    logits = apply_fn({"params": params}, emb)   # (1, NUM_CLASSES)
    probs = np.array(jax.nn.softmax(logits, axis=-1))[0]

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return CATEGORIES[idx], confidence


# ----------------------------------------------
# 🔹 Guardrails (very important)
# ----------------------------------------------
def apply_guardrails(category, confidence, user_message):
    """
    If the router is not confident enough (< 0.60),
    return a safe fallback category + message.
    """

    LOW_CONF_THRESHOLD = 0.60

    if confidence < LOW_CONF_THRESHOLD:
        return {
            "category": "Uncertain",
            "router_confidence": confidence,
            "answer": (
                "I want to help! Could you please give me a bit more detail "
                "so I can understand your issue correctly?"
            ),
            "judge_score": None,
            "judge_reasoning": "Skipped due to low router confidence."
        }

    return None  # Safe to continue


from pipeline.logger import log_interaction

# ----------------------------------------------
# 🔹 Full Pipeline: Router → RAG → LLM → Judge → Guardrails → LOG
# ----------------------------------------------
def run_pipeline(user_message: str):
    # Step 1 – router category
    category, confidence = predict_category(user_message)

    # Step 2 – RAG knowledge search
    from rag.retriever import retrieve_top_k
    kb_hits = retrieve_top_k(user_message, top_k=3, category=category)

    # Step 3 – Few-shot + RAG answer
    answer = generate_support_answer(user_message, category, kb_hits)

    # Step 4 – Judge scoring
    score, reasoning = judge_answer_quality(user_message, answer, category)

    # Step 5 – Guardrails
    if confidence < 0.55:
        guardrail_action = "ROUTER_LOW_CONFIDENCE_ESCALATE"
        guardrail_notes = "Low router confidence → escalate to human."
    elif score < 0.50:
        guardrail_action = "LLM_POOR_ANSWER_RETRY"
        guardrail_notes = "LLM answer judged low quality → retry."
    else:
        guardrail_action = "OK"
        guardrail_notes = "All checks passed."

    # ✅ Log to CSV
    log_interaction(
        user_message=user_message,
        category=category,
        router_conf=confidence,
        judge_score=score,
        judge_reasoning=reasoning,
        guardrail_action=guardrail_action,
        guardrail_notes=guardrail_notes,
    )

    # Object Streamlit expects
    return {
        "category": category,
        "router_confidence": float(confidence),
        "kb_hits": kb_hits,
        "answer": answer,
        "final_answer": answer,
        "judge_score": float(score),
        "judge_reasoning": reasoning,
        "guardrail_action": guardrail_action,
    }



# ----------------------------------------------
# 🔹 CLI Manual Test
# ----------------------------------------------
if __name__ == "__main__":
    test = "I did not receive my refund for last month yet."
    out = run_pipeline(test)

    print("\n=== PIPELINE OUTPUT ===")
    print("Predicted Category:", out["category"])
    print("Confidence:", out["router_confidence"])
    print("\nGenerated Answer:\n", out["answer"])
    print("\nJudge Score:", out["judge_score"])
    print("Judge Reasoning:", out["judge_reasoning"])



