import os
import json
import numpy as np
from typing import Optional

from config import KB_EMB_PATH, KB_META_PATH
from pipeline.compute_embeddings import compute_single_embedding

# Cached globals (load once)
_KB_EMB = None
_KB_META = None


# -------------------------------------------------------------
# Load Knowledge Base (embeddings + metadata)
# -------------------------------------------------------------
def _load_kb():
    """
    Loads the knowledge base embeddings and metadata into memory.
    Cached after first load.
    """
    global _KB_EMB, _KB_META

    if _KB_EMB is not None and _KB_META is not None:
        return _KB_EMB, _KB_META

    if not os.path.exists(KB_EMB_PATH) or not os.path.exists(KB_META_PATH):
        raise FileNotFoundError(
            "KB index not found. Run: python -m rag.index_docs"
        )

    # Load embeddings
    _KB_EMB = np.load(KB_EMB_PATH)

    # Load metadata
    with open(KB_META_PATH, "r", encoding="utf-8") as f:
        _KB_META = json.load(f)

    return _KB_EMB, _KB_META


# -------------------------------------------------------------
# Semantic search over knowledge base
# -------------------------------------------------------------
def search_knowledge(
    query: str,
    top_k: int = 3,
    category: Optional[str] = None
):
    """
    Returns top_k KB entries sorted by cosine similarity.
    Can optionally filter by category.
    """
    kb_emb, kb_meta = _load_kb()

    # Query embedding
    query_emb = np.array(compute_single_embedding(query), dtype=np.float32)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

    # Normalize KB embeddings
    kb_norm = kb_emb / (np.linalg.norm(kb_emb, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity
    sims = kb_norm @ query_emb

    # Preselect more candidates
    idxs = np.argsort(-sims)[: top_k * 4]

    results = []
    for idx in idxs:
        meta = kb_meta[int(idx)]

        # Category filtering (optional)
        if category and meta.get("category") and meta["category"] != category:
            continue

        results.append(
            {
                "similarity": float(sims[idx]),
                "id": meta.get("id"),
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "content": meta.get("content", ""),  # Required for RAG prompt
            }
        )

        if len(results) >= top_k:
            break

    return results


# -------------------------------------------------------------
# Wrapper for orchestrator
# -------------------------------------------------------------
def retrieve_top_k(
    query: str,
    top_k: int = 3,
    category: Optional[str] = None
):
    """
    Clean wrapper used by orchestrator.py to call RAG search.
    """
    return search_knowledge(query, top_k=top_k, category=category)
