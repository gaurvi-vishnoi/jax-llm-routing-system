import numpy as np
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CLEAN_CSV,
    TRAIN_EMB_PATH,
    TRAIN_LABELS_PATH,
    CATEGORIES,
)

# Initialize client
client = OpenAI(api_key=OPENAI_API_KEY)

def compute_single_embedding(text: str) -> np.ndarray:
    """
    Compute embedding for ONE support ticket text.
    Used by the orchestrator + dashboard at runtime.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    emb = response.data[0].embedding
    return np.array(emb, dtype=np.float32)


def encode_labels(categories):
    """
    Convert category strings → integer labels using config.CATEGORIES
    Example:
        "Billing"   → 0
        "Technical" → 1
    """
    label_to_id = {cat: i for i, cat in enumerate(CATEGORIES)}
    encoded = np.array([label_to_id[c] for c in categories], dtype=np.int32)
    return encoded, label_to_id


def batch_iter(items, batch_size=64):
    """Yield successive batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def compute_and_save_embeddings():
    """
    Generates embeddings + integer labels from the cleaned CSV
    and saves them to TRAIN_EMB_PATH / TRAIN_LABELS_PATH for the JAX router.
    """
    print("Loading cleaned dataset:", CLEAN_CSV)
    df = pd.read_csv(CLEAN_CSV)

    texts = df["text"].astype(str).tolist()
    labels = df["category"].tolist()

    # Encode categories → integers
    print("Encoding labels")
    y, label_map = encode_labels(labels)

    all_embeddings = []

    print("\n Generating embeddings for training data...")
    for batch in tqdm(list(batch_iter(texts, batch_size=64))):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        embs = [item.embedding for item in response.data]
        all_embeddings.extend(embs)

    X = np.array(all_embeddings, dtype=np.float32)

    # Save to disk
    print("\nSaving embeddings...")
    np.save(TRAIN_EMB_PATH, X)
    np.save(TRAIN_LABELS_PATH, y)

    print("\n Embeddings + labels saved!")
    print(TRAIN_EMB_PATH)
    print(TRAIN_LABELS_PATH)
    print("Shape:", X.shape, y.shape)
    print("Label Map:", label_map)


if __name__ == "__main__":
    compute_and_save_embeddings()
