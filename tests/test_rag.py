import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_rerank_basic():
    from advanced_rag import rerank

    with (
        patch("advanced_rag.cached_embed_query", return_value=[0.1] * 384),
        patch(
            "advanced_rag.batch_embed",
            return_value=[[0.1] * 384, [0.9] * 384, [0.5] * 384],
        ),
    ):
        results = rerank("test query", ["doc1", "doc2", "doc3"], top_k=2)
    assert len(results) == 2
    assert all("score" in r for r in results)
    assert all("document" in r for r in results)
    assert all("index" in r for r in results)


def test_rerank_empty():
    from advanced_rag import rerank

    results = rerank("test", [], top_k=3)
    assert results == []


def test_rerank_top_k_exceeds_docs():
    from advanced_rag import rerank

    with (
        patch("advanced_rag.cached_embed_query", return_value=[0.1] * 384),
        patch("advanced_rag.batch_embed", return_value=[[0.5] * 384]),
    ):
        results = rerank("test", ["only doc"], top_k=5)
    assert len(results) == 1


def test_guardrails_safe():
    from advanced_rag import guardrails_check

    result = guardrails_check("RAG uses vector databases for semantic search.")
    assert result["safe"] is True
    assert result["issues"] == []


def test_guardrails_refusal():
    from advanced_rag import guardrails_check

    result = guardrails_check("As an AI language model, I cannot help with that.")
    assert result["safe"] is False
    assert any(i["type"] == "refusal" for i in result["issues"])


def test_guardrails_pii():
    from advanced_rag import guardrails_check

    result = guardrails_check("Call me at 123-45-6789 for details.")
    assert result["safe"] is False
    assert any(i["type"] == "pii_detected" for i in result["issues"])


def test_guardrails_too_short():
    from advanced_rag import guardrails_check

    result = guardrails_check("ok")
    assert result["safe"] is False
    assert any(i["type"] == "too_short" for i in result["issues"])


def test_guardrails_too_long():
    from advanced_rag import guardrails_check

    long_text = "word " * 500
    result = guardrails_check(long_text)
    assert result["safe"] is False
    assert any(i["type"] == "too_long" for i in result["issues"])


def test_compute_rag_metrics():
    from advanced_rag import compute_rag_metrics

    with (
        patch("advanced_rag.cached_embed_query") as mock_emb,
        patch("advanced_rag.batch_embed") as mock_batch,
    ):
        mock_emb.side_effect = lambda t: (
            [0.5] * 384
            if "answer" in t.lower() or "retrieval" in t.lower()
            else [0.3] * 384
        )
        mock_batch.return_value = [[0.5] * 384, [0.4] * 384]
        metrics = compute_rag_metrics(
            "What is RAG?",
            "RAG is retrieval augmented generation.",
            ["RAG uses retrieval", "Vector search finds docs"],
        )
    assert "answer_relevancy" in metrics
    assert "faithfulness" in metrics
    assert "recall_at_k" in metrics
    assert 0 <= metrics["answer_relevancy"] <= 1


def test_compute_rag_metrics_with_ground_truth():
    from advanced_rag import compute_rag_metrics

    with (
        patch("advanced_rag.cached_embed_query", return_value=[0.5] * 384),
        patch("advanced_rag.batch_embed", return_value=[[0.5] * 384]),
    ):
        metrics = compute_rag_metrics(
            "What is RAG?",
            "RAG combines retrieval and generation.",
            ["RAG uses retrieval"],
            ground_truth="RAG is retrieval augmented generation.",
        )
    assert "recall_at_k" in metrics


def test_cached_embed_query_caches():
    from advanced_rag import cached_embed_query, _emb_cache

    _emb_cache.clear()
    with patch("huggingface_rag.embed_query", return_value=[0.1] * 384) as mock_emb:
        result1 = cached_embed_query("test text caching unique")
        result2 = cached_embed_query("test text caching unique")
    assert mock_emb.call_count == 1
    assert result1 == result2
    _emb_cache.clear()


def test_batch_embed_batches():
    from advanced_rag import batch_embed

    texts = [f"text batch {i}" for i in range(10)]
    with patch(
        "huggingface_rag.embed_texts", return_value=[[0.1] * 384] * 5
    ) as mock_emb:
        result = batch_embed(texts, batch_size=5)
    assert mock_emb.call_count == 2
    assert len(result) == 10
