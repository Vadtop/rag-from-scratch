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

_agent_sessions: dict[int, list] = {}


def _get_vector_store():
    from vector_store import VectorStore

    return VectorStore()


def _chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


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


def _upload_text(filename: str, text: str) -> int:
    vs = _get_vector_store()
    chunks = _chunk_text(text, chunk_size=500, overlap=100)
    for i, chunk in enumerate(chunks):
        embedding = _get_embedding(chunk)
        vs.add_chunk(
            chunk_id=f"{filename}_chunk_{i}",
            content=chunk,
            embedding=embedding,
            metadata={
                "source": filename,
                "chunk_id": i,
                "total_chunks": len(chunks),
            },
        )
    return len(chunks)


def _rag_ask(question: str, chat_id: int) -> str:
    vs = _get_vector_store()
    if vs.count() == 0:
        return (
            "База знаний пуста. Отправь мне документ (файл .txt) "
            "или используй /upload чтобы загрузить текст, потом спрашивай!"
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
    sources = list(set(m["source"] for m in metadatas))

    system_prompt = (
        "Ты AI-агент с доступом к базе знаний. "
        "Отвечай на вопросы используя предоставленный контекст. "
        "Если контекста недостаточно — отвечай из общих знаний, "
        "но предупреди об этом. Отвечай на русском языке."
    )
    if context:
        system_prompt += f"\n\nБаза знаний:\n{context}"

    history = _agent_sessions.get(chat_id, [])

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": question})

    answer = _get_completion(messages)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    _agent_sessions[chat_id] = history

    if sources:
        answer += f"\n\n📎 Источники: {', '.join(sources)}"

    return answer


@router.message(CommandStart())
async def cmd_start(message: types.Message):
    vs = _get_vector_store()
    docs_count = vs.count()
    await message.answer(
        "👋 Привет! Я RAG AI-агент.\n\n"
        "Умею:\n"
        "• Отправить файл .txt — загружу в базу знаний\n"
        "• /ask ВОПРОС — отвечу из базы знаний (RAG + DeepSeek)\n"
        "• /upload ТЕКСТ — загружу текст вручную\n"
        "• /stats — статистика базы знаний\n"
        "• /reset — очистить базу и историю диалога\n\n"
        f"📄 Документов в базе: {docs_count} чанков\n\n"
        "Просто отправь .txt файл или напиши вопрос!"
    )


@router.message(Command("ask"))
async def cmd_ask(message: types.Message):
    question = message.text.removeprefix("/ask").strip()
    if not question:
        await message.answer("Использование: /ask <ваш вопрос>")
        return

    await message.chat.do("typing")
    try:
        answer = await asyncio.to_thread(_rag_ask, question, message.chat.id)
        await message.answer(answer[:4096])
    except Exception as e:
        logger.exception("RAG error in /ask")
        await message.answer(f"Ошибка: {e}")


@router.message(Command("upload"))
async def cmd_upload(message: types.Message):
    text = message.text.removeprefix("/upload").strip()
    if not text:
        await message.answer("Использование: /upload <текст для загрузки в базу знаний>")
        return

    await message.chat.do("typing")
    try:
        n = await asyncio.to_thread(_upload_text, "manual_upload.txt", text)
        vs = _get_vector_store()
        await message.answer(f"✅ Загружено {n} чанков. Всего в базе: {vs.count()}")
    except Exception as e:
        logger.exception("Upload error")
        await message.answer(f"Ошибка загрузки: {e}")


@router.message(Command("stats"))
async def cmd_stats(message: types.Message):
    vs = _get_vector_store()
    sources = vs.get_all_sources()
    await message.answer(
        f"📊 База знаний:\n"
        f"Всего чанков: {vs.count()}\n"
        f"Документов: {len(sources)}\n"
        f"Источники: {', '.join(sources) if sources else 'пусто'}"
    )


@router.message(Command("reset"))
async def cmd_reset(message: types.Message):
    vs = _get_vector_store()
    vs.clear()
    _agent_sessions.pop(message.chat.id, None)
    await message.answer("🗑 База знаний и история диалога очищены.")


@router.message(F.document)
async def handle_document(message: types.Message):
    doc = message.document
    if not doc.file_name or not doc.file_name.endswith(".txt"):
        await message.answer("Пока принимаю только .txt файлы. Отправь текстовый файл!")
        return

    await message.answer(f"📥 Загружаю {doc.file_name}...")
    await message.chat.do("typing")

    try:
        file = await message.bot.get_file(doc.file_id)
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.telegram.org/file/bot{os.environ.get('BOT_TOKEN', '')}/{file.file_path}"
            ) as resp:
                content = await resp.text()

        n = await asyncio.to_thread(_upload_text, doc.file_name, content)
        vs = _get_vector_store()
        await message.answer(
            f"✅ {doc.file_name} загружен!\n"
            f"Чанков: {n}\n"
            f"Всего в базе: {vs.count()}\n\n"
            "Теперь можешь спрашивать — /ask ВОПРОС или просто напиши вопрос!"
        )
    except Exception as e:
        logger.exception("Document upload error")
        await message.answer(f"Ошибка загрузки файла: {e}")


@router.message(F.text)
async def fallback(message: types.Message):
    question = message.text.strip()
    if not question or question.startswith("/"):
        return
    await message.chat.do("typing")
    try:
        answer = await asyncio.to_thread(_rag_ask, question, message.chat.id)
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
