import asyncio
import logging
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.markdown import hbold
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class ModerationClient:
    """Обёртка для вызова OpenAI с возвратом структурированного результата."""

    def __init__(self, api_key: str, model: str, system_prompt: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def classify(self, text: str, rules: List[Dict[str, str]]) -> Tuple[bool, Optional[str]]:
        """
        Возвращает (is_forbidden, reason). reason должен быть одним из значений reason из правил.
        """
        # Формируем инструкцию с явным списком правил
        rules_json = json.dumps(rules, ensure_ascii=False)
        user_prompt = (
            "Тебе дан список правил в формате JSON со структурами {\"prompt\": str, \"reason\": str}.\n" \
            "Текст сообщения ниже.\n" \
            "Определи, нарушает ли сообщение какое-либо правило по смыслу (даже если нет точных совпадений слов).\n" \
            "Если нарушает, верни строго JSON вида {\"is_forbidden\": true, \"reason\": \"<одно из reason из списка>\"}.\n" \
            "Если нет, верни строго {\"is_forbidden\": false}. Никакого другого текста не пиши.\n\n" \
            f"Правила: {rules_json}\n" \
            f"Сообщение: {text}"
        )

        # OpenAI Python SDK 1.x не поддерживает async напрямую, используем asyncio.to_thread
        def _sync_call() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or "{}"

        content = await asyncio.to_thread(_sync_call)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Попытка вычленить JSON из ответа
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(content[start : end + 1])
                except Exception:
                    data = {"is_forbidden": False}
            else:
                data = {"is_forbidden": False}

        is_forbidden = bool(data.get("is_forbidden", False))
        reason = data.get("reason") if is_forbidden else None

        # Если reason не из словаря, обнулим
        allowed_reasons = {item["reason"] for item in rules}
        if is_forbidden and (not isinstance(reason, str) or reason not in allowed_reasons):
            # Попробуем маппинг по самому близкому prompt через простую эвристику (падение назад)
            reason = next(iter(allowed_reasons), None)

        return is_forbidden, reason


def load_settings(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def main() -> None:
    load_dotenv()

    def get_bool_env(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "Ты модератор чата. Верни ТОЛЬКО JSON с ключами is_forbidden и reason.",
    )
    debug_mode = get_bool_env("DEBUG", False)

    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.info("Starting YML antispam bot: model=%s, debug=%s", openai_model, debug_mode)

    if not telegram_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY не задан в .env")

    settings = load_settings(os.path.join(os.path.dirname(__file__), "settings.json"))
    rules = settings.get("forbidden_topics", [])
    if not isinstance(rules, list) or not rules:
        raise RuntimeError("В settings.json отсутствуют правила forbidden_topics")

    moderation = ModerationClient(api_key=openai_key, model=openai_model, system_prompt=system_prompt)

    bot = Bot(token=telegram_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        await message.reply("Бот антиспама активен. Добавьте меня администратором группы с правами удаления сообщений.")

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        if debug_mode:
            logging.debug(
                "Incoming message: chat_id=%s user_id=%s text=%r",
                getattr(message.chat, "id", None),
                getattr(getattr(message, "from_user", None), "id", None),
                (message.text or "")[:500],
            )
        # Игнорируем собственные сообщения бота и команды
        if message.from_user and (message.from_user.is_bot or (message.text and message.text.startswith("/"))):
            return

        text = message.text or ""
        if not text.strip():
            return

        try:
            is_forbidden, reason = await moderation.classify(text=text, rules=rules)
        except Exception:
            if debug_mode:
                logging.exception("OpenAI classify failed")
            # В сомнительных случаях не удаляем
            return

        if debug_mode:
            logging.debug("Classification result: forbidden=%s reason=%r", is_forbidden, reason)

        if not is_forbidden:
            return

        chat_id = message.chat.id
        user_mention = (
            f"{hbold(message.from_user.full_name)}" if message.from_user else "пользователь"
        )
        reason_text = reason or "нарушение правил"

        # Удаляем исходное сообщение
        try:
            await message.delete()
        except Exception:
            if debug_mode:
                logging.exception("Failed to delete violating message")
            # Нет прав — выходим
            return

        # Публикуем уведомление и удаляем через 60 секунд
        notice_text = f"сообщение от {user_mention} удалено по причине {hbold(reason_text)}"
        try:
            notice_msg = await bot.send_message(chat_id=chat_id, text=notice_text)
        except Exception:
            if debug_mode:
                logging.exception("Failed to send notice message")
            return

        async def delete_notice_later(chat: int, message_id: int) -> None:
            await asyncio.sleep(60)
            try:
                await bot.delete_message(chat_id=chat, message_id=message_id)
            except Exception:
                if debug_mode:
                    logging.exception("Failed to delete notice message")
                pass

        asyncio.create_task(delete_notice_later(chat_id, notice_msg.message_id))

    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass

