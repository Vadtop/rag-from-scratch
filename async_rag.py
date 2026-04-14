import asyncio
import logging
from typing import Any

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

_session: aiohttp.ClientSession | None = None


async def get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def close_session():
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


async def async_embed_query(text: str) -> list[float]:
    from huggingface_rag import get_embedding_model

    model = get_embedding_model()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: model.encode([text], show_progress_bar=False).tolist()
    )
    return result[0]


async def async_embed_texts(
    texts: list[str], batch_size: int = 32
) -> list[list[float]]:
    from huggingface_rag import get_embedding_model

    model = get_embedding_model()

    async def _embed_batch(batch: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: model.encode(batch, show_progress_bar=False).tolist()
        )

    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tasks.append(_embed_batch(batch))

    results = await asyncio.gather(*tasks)
    all_embs = []
    for r in results:
        all_embs.extend(r)
    return all_embs


async def async_rerank(query: str, documents: list[str], top_k: int = 3) -> list[dict]:
    if not documents:
        return []

    query_emb, doc_embs = await asyncio.gather(
        async_embed_query(query),
        async_embed_texts(documents),
    )

    scores = []
    for i, doc_emb in enumerate(doc_embs):
        q = np.array(query_emb)
        d = np.array(doc_emb)
        sim = float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-8))
        scores.append({"index": i, "document": documents[i], "score": sim})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


async def async_get_completion(
    messages: list[dict],
    api_key: str = "",
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "deepseek/deepseek-chat",
) -> str:
    session = await get_session()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    async with session.post(
        f"{base_url}/chat/completions", headers=headers, json=payload
    ) as resp:
        data = await resp.json()
        return data["choices"][0]["message"]["content"]
