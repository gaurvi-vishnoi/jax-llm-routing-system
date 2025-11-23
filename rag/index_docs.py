import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    KB_CSV,
    KB_EMB_PATH,
    KB_META_PATH,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def build_kb_index():
    if not os.path.exists(KB_CSV):
        raise FileNotFoundError(f"KB CSV not found: {KB_CSV}")

    print(" Loading knowledge base:", KB_CSV)
    df = pd.read_csv(KB_CSV)

    texts = []
    metadata = []

    for _, row in df.iterrows():
        text = f"{row.get('title', '')}. {row.get('content', '')}"
        texts.append(text)
        metadata.append(
            {
                "id": int(row.get("id", 0)),
                "category": str(row.get("category", "")),
                "title": str(row.get("title", "")),
            }
        )

    print("Computing embeddings for KB docs")
    embeddings = []
    for i in tqdm(range(0, len(texts), 64)):
        batch = texts[i : i + 64]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        embs = [item.embedding for item in resp.data]
        embeddings.extend(embs)

    X = np.array(embeddings, dtype=np.float32)

    print("Saving KB embeddings & metadata")
    np.save(KB_EMB_PATH, X)
    with open(KB_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("KB index built!")
    print("Embeddings:", KB_EMB_PATH)
    print("Metadata:", KB_META_PATH)
    print("Shape:", X.shape)


if __name__ == "__main__":
    build_kb_index()
