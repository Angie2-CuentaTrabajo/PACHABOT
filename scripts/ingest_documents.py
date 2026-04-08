from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.core.logger import setup_logging
from app.services.document_service import DocumentService
from app.services.retrieval_service import RetrievalService


def main() -> None:
    """Process raw documents and rebuild the local vectorstore."""

    settings = get_settings()
    logger = setup_logging(settings.log_level)
    document_service = DocumentService(settings, logger)
    retrieval_service = RetrievalService(settings, logger)

    chunks = document_service.build_chunks()
    if not chunks:
        logger.warning("No se generaron chunks. Revisa data/raw/")
        return

    retrieval_service.build_index(chunks)
    logger.info("Ingesta completada con %s chunks", len(chunks))


if __name__ == "__main__":
    main()
