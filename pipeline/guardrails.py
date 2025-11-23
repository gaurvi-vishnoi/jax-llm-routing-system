from dataclasses import dataclass


@dataclass
class GuardrailConfig:
    min_router_conf: float = 0.6
    min_judge_score: float = 0.6


CFG = GuardrailConfig()


def apply_guardrails(
    category: str,
    router_conf: float,
    judge_score: float,
    user_message: str,
    answer: str,
):
    """
    Returns a dict with:
      - final_answer
      - action: "normal" | "ask_clarification" | "escalate"
      - notes: free-form text for logging
    Priority:
      1) Low router confidence → ask clarification
      2) Very low judge score → escalate
    """
    action = "normal"
    final_answer = answer
    notes = []

    if router_conf < CFG.min_router_conf:
        action = "ask_clarification"
        notes.append("low_router_confidence")
        final_answer = (
            "I want to make sure I understand your issue correctly before proceeding.\n\n"
            f"My current guess is that this is related to **{category}**, "
            "but I'm not fully confident.\n\n"
            "Could you please clarify your problem in a bit more detail?"
        )

    if judge_score < CFG.min_judge_score:
        # Escalation overrides previous action
        action = "escalate"
        notes.append("low_judge_score")
        final_answer = (
            final_answer
            + "\n\nI've also forwarded this conversation to a human support specialist "
              "to double-check and help you further."
        )

    return {
        "final_answer": final_answer,
        "action": action,
        "notes": ",".join(notes),
    }
