"""
Google Sheets интеграция — запись вопросов и ответов агента.

Требует:
- GOOGLE_SHEETS_CREDENTIALS_JSON — путь к файлу credentials.json (Service Account)
- GOOGLE_SHEETS_ID — ID таблицы из URL (между /d/ и /edit)

Установка: pip install gspread google-auth
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_sheet = None


def _get_sheet():
    """Ленивая инициализация подключения к Google Sheets."""
    global _sheet
    if _sheet is not None:
        return _sheet

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        creds_path = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON", "credentials.json")
        sheet_id = os.environ.get("GOOGLE_SHEETS_ID", "")

        if not sheet_id:
            logger.warning("GOOGLE_SHEETS_ID not set — sheets logging disabled")
            return None

        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(sheet_id)

        # Используем первый лист или создаём "RAG Log"
        try:
            _sheet = spreadsheet.worksheet("RAG Log")
        except Exception:
            _sheet = spreadsheet.add_worksheet(title="RAG Log", rows=1000, cols=6)
            # Заголовки
            _sheet.append_row(["Timestamp", "Question", "Answer", "Sources", "Model", "Chunks Used"])

        logger.info("Google Sheets connected: %s", sheet_id)
        return _sheet

    except ImportError:
        logger.warning("gspread not installed — run: pip install gspread google-auth")
        return None
    except Exception as e:
        logger.warning("Google Sheets init error: %s", e)
        return None


def log_query(
    question: str,
    answer: str,
    sources: list,
    model: str = "deepseek/deepseek-chat",
    chunks_used: int = 0,
) -> bool:
    """
    Записать вопрос и ответ в Google Sheets.
    Возвращает True если успешно, False если ошибка (не критично).
    """
    sheet = _get_sheet()
    if sheet is None:
        return False

    try:
        row = [
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            question[:500],
            answer[:1000],
            ", ".join(sources) if sources else "",
            model,
            str(chunks_used),
        ]
        sheet.append_row(row)
        logger.debug("Logged to Google Sheets: %s", question[:50])
        return True
    except Exception as e:
        logger.warning("Google Sheets log error: %s", e)
        return False
