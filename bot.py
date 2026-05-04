import os
import logging
import asyncio

from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

router = Router()

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "deepseek/deepseek-chat"


def _get_vector_store():
    from vector_store import VectorStore

    vs = VectorStore()
    return vs


def _get_embedding(text):
    import requests

    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": "openai/text-embedding-3-small", "input": text},
    )
    return response.json()["data"][0]["embedding"]


def _get_completion(messages):
    import requests

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": LLM_MODEL, "messages": messages, "temperature": 0},
    )
    return response.json()["choices"][0]["message"]["content"]


def _rag_ask(question: str) -> str:
    vs = _get_vector_store()
    if vs.count() == 0:
        return (
            "База знаний пока пуста. Загрузите документы через веб-интерфейс "
            f"({os.environ.get('RAILWAY_PUBLIC_DOMAIN', 'localhost')}) "
            "и повторите вопрос."
        )

    query_emb = _get_embedding(question)
    results = vs.search(query_emb, top_k=3)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = [
        f"[{m['source']}, chunk {m['chunk_id']+1}]\n{d}"
        for d, m in zip(documents, metadatas)
    ]
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "Ты AI-агент с доступом к базе знаний. "
        "Отвечай на вопросы используя предоставленный контекст. "
        "Если контекста недостаточно — отвечай из общих знаний, "
        "но предупреди об этом. Отвечай на русском языке."
    )
    if context:
        system_prompt += f"\n\nБаза знаний:\n{context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    answer = _get_completion(messages)
    return answer


@router.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "👋 Привет! Я RAG AI-агент.\n\n"
        "Умею:\n"
        "• /ask ВОПРОС — задать вопрос, ответ из базы знаний (RAG + DeepSeek)\n"
        "• /start — это приветствие\n\n"
        "Загрузите документы через веб-интерфейс, а потом спрашивайте!"
    )


@router.message(Command("ask"))
async def cmd_ask(message: types.Message):
    question = message.text.removeprefix("/ask").strip()
    if not question:
        await message.answer("Использование: /ask <ваш вопрос>")
        return

    await message.chat.do("typing")
    try:
        answer = await asyncio.to_thread(_rag_ask, question)
        await message.answer(answer[:4096])
    except Exception as e:
        logger.exception("RAG error in /ask")
        await message.answer(f"Ошибка: {e}")


@router.message(F.text)
async def fallback(message: types.Message):
    question = message.text.strip()
    if not question:
        return
    await message.chat.do("typing")
    try:
        answer = await asyncio.to_thread(_rag_ask, question)
        await message.answer(answer[:4096])
    except Exception as e:
        logger.exception("RAG error in fallback")
        await message.answer(f"Ошибка: {e}")


_bot: Bot | None = None
_dp: Dispatcher | None = None
_polling_task: asyncio.Task | None = None


async def start_bot():
    global _bot, _dp, _polling_task

    token = os.environ.get("BOT_TOKEN", "")
    if not token:
        logger.warning("BOT_TOKEN not set — Telegram bot disabled")
        return

    _bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    _dp = Dispatcher()
    _dp.include_router(router)

    async def _poll():
        try:
            await _dp.start_polling(_bot)
        except Exception:
            logger.exception("Bot polling error")

    _polling_task = asyncio.create_task(_poll())
    logger.info("Telegram bot started (polling)")


async def stop_bot():
    global _polling_task
    if _polling_task and not _polling_task.done():
        _polling_task.cancel()
        try:
            await _polling_task
        except asyncio.CancelledError:
            pass
    if _dp:
        await _dp.shutdown()
    if _bot:
        await _bot.session.close()
    logger.info("Telegram bot stopped")
