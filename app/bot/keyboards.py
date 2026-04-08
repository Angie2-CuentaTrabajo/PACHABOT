from __future__ import annotations

from telegram import ReplyKeyboardMarkup


def build_main_keyboard() -> ReplyKeyboardMarkup:
    """Provide example citizen queries."""

    return ReplyKeyboardMarkup(
        [
            ["Que requisitos necesito", "Cuanto mide un modulo"],
            ["Cuanto se paga de SISA", "Que zonas son rigidas"],
            ["Que dice el articulo 7", "Explicame la autorizacion"],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )
