from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.core.logger import setup_logging
from app.utils.helpers import ensure_directory


def main() -> None:
    """Reset processed data and local vectorstore artifacts."""

    settings = get_settings()
    logger = setup_logging(settings.log_level)

    if settings.vectorstore_dir.exists():
        shutil.rmtree(settings.vectorstore_dir)
        logger.info("Vectorstore eliminado: %s", settings.vectorstore_dir)
    ensure_directory(settings.vectorstore_dir)

    if settings.processed_chunks_file.exists():
        settings.processed_chunks_file.unlink()
        logger.info("Archivo eliminado: %s", settings.processed_chunks_file)


if __name__ == "__main__":
    main()
