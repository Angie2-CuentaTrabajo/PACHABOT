from __future__ import annotations

from telegram import Update

from app.channels.schemas import IncomingChatMessage


def build_incoming_message(update: Update) -> IncomingChatMessage | None:
    """Convert a Telegram update into a channel-agnostic chat message."""

    if update.effective_message is None or not update.effective_message.text:
        return None

    chat = update.effective_chat
    user = update.effective_user
    return IncomingChatMessage(
        channel="telegram",
        session_id=str(chat.id) if chat else "telegram-unknown",
        user_id=str(user.id) if user else "telegram-user-unknown",
        text=update.effective_message.text.strip(),
        user_display_name=user.full_name if user else "",
        metadata={
            "chat_type": chat.type if chat else "",
            "message_id": update.effective_message.message_id,
            "username": user.username if user else "",
        },
    )
