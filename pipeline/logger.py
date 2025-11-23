import csv
import os
from datetime import datetime

from config import INTERACTIONS_LOG, LOGS_DIR


FIELDNAMES = [
    "timestamp",
    "user_message",
    "predicted_category",
    "router_confidence",
    "judge_score",
    "judge_reasoning",
    "guardrail_action",
    "guardrail_notes",
]


def _ensure_log_file():
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(INTERACTIONS_LOG):
        with open(INTERACTIONS_LOG, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def log_interaction(
    user_message: str,
    category: str,
    router_conf: float,
    judge_score: float,
    judge_reasoning: str,
    guardrail_action: str,
    guardrail_notes: str,
):
    _ensure_log_file()
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_message": user_message,
        "predicted_category": category,
        "router_confidence": router_conf,
        "judge_score": judge_score,
        "judge_reasoning": judge_reasoning,
        "guardrail_action": guardrail_action,
        "guardrail_notes": guardrail_notes,
    }
    with open(INTERACTIONS_LOG, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
