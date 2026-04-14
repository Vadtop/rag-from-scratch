import logging
import re
import time
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_emb_cache: dict[str, list[float]] = {}


def cached_embed_query(text: str) -> list[float]:
    if text in _emb_cache:
        return _emb_cache[text]
    from huggingface_rag import embed_query

    emb = embed_query(text)
    _emb_cache[text] = emb
    return emb


def batch_embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    from huggingface_rag import embed_texts

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = embed_texts(batch)
        all_embs.extend(embs)
    return all_embs


def rerank(query: str, documents: list[str], top_k: int = 3) -> list[dict]:
    if not documents:
        return []

    query_emb = cached_embed_query(query)
    doc_embs = batch_embed(documents)

    scores = []
    for i, doc_emb in enumerate(doc_embs):
        q = np.array(query_emb)
        d = np.array(doc_emb)
        sim = float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-8))
        scores.append({"index": i, "document": documents[i], "score": sim})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


BANNED_PATTERNS = [
    r"as an ai language model",
    r"i cannot (help|assist|provide)",
    r"i'm sorry,? i can'?t",
    r"my instructions prevent me",
    r"i do not have (access|permission)",
]

PII_PATTERN = re.compile(
    r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b"
    r"|\b\d{16}\b"
    r"|\b[A-Z]{2}\d{6}\b",
    re.IGNORECASE,
)


def guardrails_check(answer: str) -> dict[str, Any]:
    lower = answer.lower()
    issues = []

    for pat in BANNED_PATTERNS:
        if re.search(pat, lower):
            issues.append({"type": "refusal", "pattern": pat})
            break

    pii_hits = PII_PATTERN.findall(answer)
    if pii_hits:
        issues.append({"type": "pii_detected", "count": len(pii_hits)})

    if len(answer) > 2000:
        issues.append({"type": "too_long", "length": len(answer)})

    if len(answer) < 10:
        issues.append({"type": "too_short", "length": len(answer)})

    return {"safe": len(issues) == 0, "issues": issues}


def compute_rag_metrics(
    query: str,
    answer: str,
    context_chunks: list[str],
    ground_truth: str | None = None,
) -> dict[str, float]:
    query_emb = cached_embed_query(query)
    answer_emb = cached_embed_query(answer)

    q = np.array(query_emb)
    a = np.array(answer_emb)
    answer_relevancy = float(
        np.dot(q, a) / (np.linalg.norm(q) * np.linalg.norm(a) + 1e-8)
    )

    faithfulness = 0.0
    if context_chunks:
        ctx_emb = batch_embed(context_chunks)
        ctx_matrix = np.array(ctx_emb)
        answer_norm = a / (np.linalg.norm(a) + 1e-8)
        similarities = np.dot(ctx_matrix, answer_norm)
        faithfulness = float(np.max(similarities))

    recall_at_k = 0.0
    if ground_truth:
        gt_emb = cached_embed_query(ground_truth)
        g = np.array(gt_emb)
        best_sim = 0.0
        for chunk in context_chunks:
            c_emb = cached_embed_query(chunk)
            c = np.array(c_emb)
            sim = float(np.dot(g, c) / (np.linalg.norm(g) * np.linalg.norm(c) + 1e-8))
            best_sim = max(best_sim, sim)
        recall_at_k = best_sim

    return {
        "answer_relevancy": round(answer_relevancy, 4),
        "faithfulness": round(faithfulness, 4),
        "recall_at_k": round(recall_at_k, 4),
    }
