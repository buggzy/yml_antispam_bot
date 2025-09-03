import os
import sys
import glob
from typing import List, Dict

import pytest
from dotenv import load_dotenv

# Добавляем корневую директорию проекта в sys.path для импорта main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Загружаем переменные окружения из .env
load_dotenv()

from main import ModerationClient, load_settings


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MESSAGES_DIR = os.path.join(PROJECT_ROOT, "tests", "messages")
ALLOWED_DIR = os.path.join(MESSAGES_DIR, "allowed")
DISALLOWED_DIR = os.path.join(MESSAGES_DIR, "disallowed")
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.json")


def _collect_message_files(directory: str) -> List[str]:
    files: List[str] = []
    files.extend(glob.glob(os.path.join(directory, "*.md")))
    files.extend(glob.glob(os.path.join(directory, "*.txt")))
    return sorted(files)


ALLOWED_FILES = _collect_message_files(ALLOWED_DIR)
DISALLOWED_FILES = _collect_message_files(DISALLOWED_DIR)


requires_openai_key = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Integration test requires OPENAI_API_KEY environment variable",
)


def _make_client() -> ModerationClient:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Инструкции совместимы с пользовательским промптом, который требует decisions[]
    system_prompt = (
        "Ты модератор чата. Возвращай только JSON и строго следуй формату, \n"
        "который описывает пользователь в своём сообщении."
    )
    return ModerationClient(api_key=os.environ["OPENAI_API_KEY"], model=model, system_prompt=system_prompt)


def _load_rules() -> List[Dict[str, str]]:
    settings = load_settings(SETTINGS_PATH)
    rules = settings.get("forbidden_topics", [])
    assert isinstance(rules, list) and len(rules) > 0, "В settings.json должны быть правила forbidden_topics"
    return rules


@pytest.mark.asyncio
@requires_openai_key
@pytest.mark.parametrize("path", ALLOWED_FILES)
async def test_allowed_messages_not_blocked(path: str) -> None:
    rules = _load_rules()
    client = _make_client()

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    is_forbidden, reason, rationale, decisions = await client.classify(text=text, rules=rules)
    if is_forbidden:
        print(f"\n--- СОДЕРЖИМОЕ ФАЙЛА {path} ---")
        print(text)
        print(f"--- МОТИВАЦИЯ ИИ ---")
        print(f"Reason: {reason}")
        print(f"Rationale: {rationale}")
        print(f"Decisions: {decisions}")
        print("--- КОНЕЦ ---\n")
    assert is_forbidden is False, (
        f"Сообщение {path} ошибочно классифицировано как запрещённое.\n"
        f"Содержимое: {text[:200]}{'...' if len(text) > 200 else ''}\n"
        f"Reason: {reason}\n"
        f"Rationale: {rationale}\n"
        f"Decisions: {decisions}"
    )


@pytest.mark.asyncio
@requires_openai_key
@pytest.mark.parametrize("path", DISALLOWED_FILES)
async def test_disallowed_messages_blocked(path: str) -> None:
    rules = _load_rules()
    client = _make_client()

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    is_forbidden, reason, rationale, decisions = await client.classify(text=text, rules=rules)
    if not is_forbidden:
        print(f"\n--- СОДЕРЖИМОЕ ФАЙЛА {path} ---")
        print(text)
        print(f"--- МОТИВАЦИЯ ИИ ---")
        print(f"Reason: {reason}")
        print(f"Rationale: {rationale}")
        print(f"Decisions: {decisions}")
        print("--- КОНЕЦ ---\n")
    assert is_forbidden is True, (
        f"Сообщение {path} должно быть заблокировано.\n"
        f"Содержимое: {text[:200]}{'...' if len(text) > 200 else ''}\n"
        f"Reason: {reason}\n"
        f"Rationale: {rationale}\n"
        f"Decisions: {decisions}"
    )

    # Если модель вернула reason, он должен быть из settings.json
    if reason is not None:
        allowed_reasons = {item.get("reason") for item in rules}
        assert reason in allowed_reasons