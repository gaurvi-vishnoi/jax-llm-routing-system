import pandas as pd
from pathlib import Path
from config import RAW_CSV, CLEAN_CSV, CATEGORIES, DATA_DIR



# Category Mapping Function

def map_to_target_category(text: str) -> str:
    """Maps ticket text into one of the 5 target categories."""
    t = text.lower()

    # Billing issues
    if any(w in t for w in ["bill", "billing", "payment", "charged", "invoice", "subscription"]):
        return "Billing"

    # Refund issues
    if any(w in t for w in ["refund", "money back", "return", "reverse", "credit back"]):
        return "Refund"

    # Account issues
    if any(w in t for w in ["login", "password", "account", "username", "sign in", "signin", "locked out"]):
        return "Account"

    # Bug/issues/errors
    if any(w in t for w in ["bug", "crash", "error", "not working", "issue", "failed", "broken"]):
        return "Bug"

    # Default fallback
    return "Technical"



# Build cleaned + balanced dataset

def build_dataset(limit_per_category: int = 400):
    """Creates tickets_small.csv using `body` as the text column."""

    Path(DATA_DIR).mkdir(exist_ok=True)

    print("Loading raw dataset:", RAW_CSV)
    df = pd.read_csv(RAW_CSV)

    # CHECK that 'body' exists
    if "body" not in df.columns:
        raise ValueError("ERROR: Column 'body' not found in dataset.")

    # Use body as our ticket text
    df = df[["body"]].rename(columns={"body": "text"})
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    print("Mapping categories")
    df["category"] = df["text"].apply(map_to_target_category)

    # Balance dataset
    print("Balancing dataset to", limit_per_category, "per category")
    frames = []
    for cat in CATEGORIES:
        frames.append(df[df["category"] == cat].head(limit_per_category))

    final_df = pd.concat(frames).sample(frac=1.0, random_state=42).reset_index(drop=True)

    final_df.to_csv(CLEAN_CSV, index=False)

    print("\n Dataset cleaned and saved!")
    print("Output:", CLEAN_CSV)
    print("Shape:", final_df.shape)
    print(final_df.head())

if __name__ == "__main__":
    build_dataset()
