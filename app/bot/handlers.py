from __future__ import annotations

from telegram import Update
from telegram.error import TimedOut
from telegram.ext import ContextTypes

from app.bot.keyboards import build_main_keyboard
from app.channels.telegram import build_incoming_message


MAX_TELEGRAM_MESSAGE_LEN = 3200

WELCOME_MESSAGE = (
    "Hola. Soy un asistente virtual especializado en comercio ambulatorio.\n\n"
    "Puedo orientarte con consultas sobre requisitos, autorizaciones, modulos, "
    "zonas rigidas, ferias, SISA y articulos de las ordenanzas 108-2012-MDP/C "
    "y 227-2019-MDP/C.\n\n"
    "Trabajo con memoria conversacional por chat. Si luego haces una pregunta "
    "de seguimiento, intentare entenderla usando el contexto previo.\n\n"
    "Si la base documental no respalda una respuesta con claridad, te lo dire "
    "con honestidad."
)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start."""

    if update.message is None:
        return
    await update.message.reply_text(WELCOME_MESSAGE, reply_markup=build_main_keyboard())


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""

    if update.message is None:
        return
    await update.message.reply_text(
        "Puedes preguntarme, por ejemplo:\n"
        "- Que requisitos necesito\n"
        "- Cuanto mide un modulo\n"
        "- Cuanto se paga de SISA\n"
        "- Que zonas son rigidas\n"
        "- Que dice el articulo 7\n"
        "- Explicame la autorizacion municipal\n\n"
        "Comandos utiles:\n"
        "/reset para borrar el contexto de este chat\n"
        "/estado para ver el modo actual del asistente"
    )


async def reset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset and clear the current chat memory."""

    if update.effective_chat is None or update.message is None:
        return

    assistant = context.application.bot_data["assistant_service"]
    assistant.reset_conversation("telegram", str(update.effective_chat.id))
    await update.message.reply_text(
        "Listo. Borre el contexto conversacional de este chat. "
        "La siguiente consulta empezara como una conversacion nueva."
    )


async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /estado and show a short runtime status."""

    if update.message is None:
        return

    assistant = context.application.bot_data["assistant_service"]
    await _send_text_safely(update, assistant.get_runtime_status())


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resolve user free-text messages."""

    incoming = build_incoming_message(update)
    if incoming is None:
        return

    assistant = context.application.bot_data["assistant_service"]
    payload = assistant.answer_chat_message(incoming)

    response = payload.answer.strip()
    if payload.sources:
        response += "\n\nFuente(s): " + "; ".join(payload.sources[:3])

    await _send_text_safely(update, response)


async def _send_text_safely(update: Update, text: str) -> None:
    """Send long responses in smaller chunks and retry once on timeout."""

    if update.message is None:
        return

    for part in _split_message(text):
        try:
            await update.message.reply_text(part)
        except TimedOut:
            await update.message.reply_text(part)


def _split_message(text: str, limit: int = MAX_TELEGRAM_MESSAGE_LEN) -> list[str]:
    """Split a long Telegram response by paragraph boundaries."""

    compact = text.strip()
    if len(compact) <= limit:
        return [compact]

    parts: list[str] = []
    current = ""
    for paragraph in compact.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            parts.append(current)

        while len(paragraph) > limit:
            parts.append(paragraph[:limit].rstrip())
            paragraph = paragraph[limit:].lstrip()
        current = paragraph

    if current:
        parts.append(current)
    return parts
