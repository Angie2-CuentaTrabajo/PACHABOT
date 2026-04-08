from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IncomingChatMessage:
    channel: str
    session_id: str
    user_id: str
    text: str
    user_display_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
