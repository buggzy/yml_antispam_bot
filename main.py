import asyncio
import logging
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Set

from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatMemberStatus
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.markdown import hbold
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# Совместимость: asyncio.to_thread недоступен в Python < 3.9
try:  # Python 3.9+
    _to_thread = asyncio.to_thread  # type: ignore[attr-defined]
except AttributeError:  # Python 3.8
    async def _to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

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
    async def classify(
        self,
        text: str,
        rules: List[Dict[str, str]],
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[List[bool]]]:
        """
        Возвращает (is_forbidden, reason). reason должен быть одним из значений reason из правил.
        """
        # Формируем инструкцию только с prompt'ами, убираем reason из промпта
        rules_prompts_only = [{"prompt": rule["prompt"]} for rule in rules]
        rules_json = json.dumps(rules_prompts_only, ensure_ascii=False)
        user_prompt = (
            "Тебе дан список правил в формате JSON со структурами {\"prompt\": str}.\n" \
            "Ниже дан текст сообщения.\n\n" \
            "Верни СТРОГО JSON:\n" \
            "{\n  \"decisions\": [true|false, ...]  // по одному boolean на каждое правило из списка, в том же порядке,\n" \
            "  \"rationale\": \"краткое обоснование решения в свободной форме\"\n}\n\n" \
            "Если массив decisions отсутствует, верни хотя бы {\"decisions\": []}.\n" \
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

        content = await _to_thread(_sync_call)
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

        # Попытка новой схемы: decisions[] и rationale
        decisions_raw = data.get("decisions")
        rationale: Optional[str] = None
        decisions_out: Optional[List[bool]] = None
        is_forbidden: bool
        reason: Optional[str]

        if isinstance(decisions_raw, list):
            # нормализуем длину и типы
            decisions_bool: List[bool] = []
            for i, v in enumerate(decisions_raw[: len(rules)]):
                decisions_bool.append(bool(v))
            if len(decisions_bool) < len(rules):
                decisions_bool.extend([False] * (len(rules) - len(decisions_bool)))

            is_forbidden = any(decisions_bool)
            decisions_out = decisions_bool
            reason = None
            if is_forbidden:
                try:
                    first_idx = next(idx for idx, val in enumerate(decisions_bool) if val)
                    reason = rules[first_idx].get("reason")  # type: ignore[union-attr]
                except StopIteration:
                    reason = None
            rationale = data.get("rationale") if isinstance(data.get("rationale"), str) else None
        else:
            # fallback к старой схеме {is_forbidden, reason}
            is_forbidden = bool(data.get("is_forbidden", False))
            reason = data.get("reason") if is_forbidden else None
            rationale = None
            decisions_out = None

        # Если reason не из словаря, обнулим
        allowed_reasons = {item["reason"] for item in rules}
        if is_forbidden and (not isinstance(reason, str) or reason not in allowed_reasons):
            # Попробуем маппинг по самому близкому prompt через простую эвристику (падение назад)
            reason = next(iter(allowed_reasons), None)

        return is_forbidden, reason, rationale, decisions_out


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
    system_prompt = "Ты модератор чата. Верни ТОЛЬКО JSON с ключами is_forbidden и reason."
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
    group_command = str(settings.get("group_command", "yml_antispam")).strip() or "yml_antispam"
    if not isinstance(rules, list) or not rules:
        raise RuntimeError("В settings.json отсутствуют правила forbidden_topics")

    moderation = ModerationClient(api_key=openai_key, model=openai_model, system_prompt=system_prompt)

    bot = Bot(token=telegram_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    # Кэшируем идентификатор бота и храним флаги уведомлений о потере прав на чат
    me = await bot.get_me()
    bot_id: int = me.id
    admin_loss_notified_chats: Set[int] = set()

    async def get_bot_rights(chat_id: int) -> Tuple[bool, bool]:
        """Возвращает (is_admin, can_delete_messages)."""
        try:
            member = await bot.get_chat_member(chat_id=chat_id, user_id=bot_id)
            status = getattr(member, "status", None)
            # Нормализуем статус к строке
            if isinstance(status, ChatMemberStatus):
                status_str = status.value
            else:
                status_str = str(status)
            if status_str == "creator":
                return True, True
            if status_str == "administrator":
                return True, bool(getattr(member, "can_delete_messages", False))
            return False, False
        except Exception:
            return False, False

    # В личке реагируем на /start
    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        chat_type = getattr(message.chat, "type", "") if message.chat else ""
        if chat_type == "private":
            await message.reply("Бот антиспама активен. Добавьте меня администратором группы с правами удаления сообщений.")
        # В группах игнорируем /start

    # В группах реагируем на команду из settings
    @dp.message(Command(group_command))
    async def cmd_group_info(message: Message) -> None:
        chat_id = getattr(message.chat, "id", None)
        chat_type = getattr(message.chat, "type", "") if message.chat else ""
        if chat_id and chat_type in {"group", "supergroup"}:
            is_admin, can_delete = await get_bot_rights(chat_id)
            if is_admin and can_delete:
                await message.reply("Бот антиспама активен.")
            else:
                await message.reply("Боту нужны права администратора с удалением сообщений.")

    async def process_text_content(message: Message, text_value: str) -> None:
        if debug_mode:
            logging.debug(
                "Incoming message: chat_id=%s user_id=%s text=%r",
                getattr(message.chat, "id", None),
                getattr(getattr(message, "from_user", None), "id", None),
                (text_value or "")[:500],
            )
        # Игнорируем собственные сообщения бота и команды
        if message.from_user and (message.from_user.is_bot or (text_value and text_value.startswith("/"))):
            return

        text = text_value or ""
        if not text.strip():
            return

        try:
            is_forbidden, reason, rationale, decisions = await moderation.classify(
                text=text,
                rules=rules,
            )
        except Exception:
            if debug_mode:
                logging.exception("OpenAI classify failed")
            # В сомнительных случаях не удаляем
            return

        if debug_mode:
            logging.debug(
                "Classification result: forbidden=%s reason=%r rationale=%r decisions=%r",
                is_forbidden,
                reason,
                rationale,
                decisions,
            )

        if not is_forbidden:
            return

        chat_id = message.chat.id
        user_mention = (
            f"{hbold(message.from_user.full_name)}" if message.from_user else "пользователь"
        )
        reason_text = reason or "нарушение правил"

        # Удаляем сообщение (исходное или отредактированное)
        try:
            await message.delete()
        except Exception:
            if debug_mode:
                logging.exception("Failed to delete violating message")
            # Проверим права администратора и, если прав нет, оповестим один раз на чат
            chat_id_for_perm = getattr(message.chat, "id", None)
            if chat_id_for_perm is not None:
                try:
                    is_admin, can_delete = await get_bot_rights(chat_id_for_perm)
                    if (not can_delete) and chat_id_for_perm not in admin_loss_notified_chats:
                        admin_loss_notified_chats.add(chat_id_for_perm)
                        try:
                            await bot.send_message(
                                chat_id=chat_id_for_perm,
                                text=(
                                    "Не могу удалять сообщения: у меня нет права администратора на удаление. "
                                    "Верните права, чтобы модерация работала."
                                ),
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
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

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        await process_text_content(message, message.text or "")

    @dp.edited_message(F.text)
    async def on_edited_text(message: Message) -> None:
        await process_text_content(message, message.text or "")

    # Подписи к фото (новые сообщения)
    @dp.message(F.photo & F.caption)
    async def on_photo_caption(message: Message) -> None:
        await process_text_content(message, message.caption or "")

    # Редактирование подписи к медиа (включая фото)
    @dp.edited_message(F.caption)
    async def on_edited_caption(message: Message) -> None:
        await process_text_content(message, message.caption or "")

    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass

