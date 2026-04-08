from __future__ import annotations

import logging

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from app.bot.handlers import (
    help_handler,
    message_handler,
    reset_handler,
    start_handler,
    status_handler,
)
from app.services.assistant_service import AssistantService


def build_telegram_application(
    token: str,
    assistant_service: AssistantService,
    logger: logging.Logger,
) -> Application:
    """Create and configure the Telegram bot application."""

    app = (
        Application.builder()
        .token(token)
        .connect_timeout(20)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(20)
        .build()
    )
    app.bot_data["assistant_service"] = assistant_service

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("reset", reset_handler))
    app.add_handler(CommandHandler("estado", status_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.getChild("telegram_bot").info("Aplicacion de Telegram configurada")
    return app
