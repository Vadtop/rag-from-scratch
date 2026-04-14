import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_SHEETS_ID", "")

from api import chunk_text, app
from httpx import AsyncClient, ASGITransport


def test_chunk_text_basic():
    text = "A" * 600
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.strip()


def test_chunk_text_short():
    chunks = chunk_text("Short text", chunk_size=500, overlap=100)
    assert len(chunks) == 1


def test_chunk_text_empty():
    chunks = chunk_text("", chunk_size=500, overlap=100)
    assert len(chunks) == 0


def test_chunk_text_overlap():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    if len(chunks) > 1:
        overlap_region = chunks[0][-50:]
        assert overlap_region in chunks[1]


@pytest.mark.anyio
async def test_root_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "endpoints" in data


@pytest.mark.anyio
async def test_stats_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/stats")
    assert resp.status_code == 200
    assert "total_chunks" in resp.json()


@pytest.mark.anyio
async def test_reset_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.delete("/reset")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_query_no_documents():
    with patch("api.get_embedding", return_value=[0.1] * 384):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/query", json={"query": "test"})
    assert resp.status_code == 200
    assert "No documents" in resp.json()["answer"]


@pytest.mark.anyio
async def test_guardrails_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/guardrails?answer=Hello%20world%20this%20is%20fine")
    assert resp.status_code == 200
    assert "safe" in resp.json()


@pytest.mark.anyio
async def test_query_hf_no_documents():
    with patch("api.embed_query", return_value=[0.1] * 384):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/query_hf", json={"query": "test"})
    assert resp.status_code == 200
    assert "No documents" in resp.json()["answer"]


@pytest.mark.anyio
async def test_query_advanced_no_documents():
    with patch("api.get_embedding", return_value=[0.1] * 384):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/query_advanced", json={"query": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "guardrails" in data
    assert "metrics" in data


@pytest.mark.anyio
async def test_agent_session_lifecycle():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.delete("/agent/test_session_cleanup")
    assert resp.status_code == 200
