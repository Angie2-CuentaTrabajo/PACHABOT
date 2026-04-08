from __future__ import annotations

from app.bot.telegram_bot import build_telegram_application
from app.main import container


def main() -> None:
    """Start the Telegram bot in polling mode."""

    token = container.settings.telegram_bot_token
    if not token:
        raise RuntimeError(
            "Falta TELEGRAM_BOT_TOKEN. Crea tu archivo .env a partir de .env.example."
        )

    app = build_telegram_application(
        token=token,
        assistant_service=container.assistant_service,
        logger=container.retrieval_service.logger.parent,
    )
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
