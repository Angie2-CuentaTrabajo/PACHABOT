from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Persist JSON using UTF-8 and indentation."""

    if isinstance(payload, list):
        serializable = [asdict(item) if is_dataclass(item) else item for item in payload]
    else:
        serializable = payload
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read JSON from disk."""

    return json.loads(path.read_text(encoding="utf-8"))
