from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.models.schemas import DocumentChunk
from app.utils.chunking import split_text_into_chunks
from app.utils.helpers import ensure_directory, write_json
from app.utils.text_cleaner import clean_text


DOCUMENT_TITLES = {
    "ordenanza_108_2012": "Ordenanza 108-2012-MDP/C",
    "ordenanza_227_2019": "Ordenanza 227-2019-MDP/C",
}


class DocumentService:
    """Read, clean and chunk source documents."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger.getChild("document_service")

    def discover_raw_documents(self) -> list[Path]:
        """Return supported text documents found in the raw data directory."""

        ensure_directory(self.settings.raw_data_dir)
        return sorted(self.settings.raw_data_dir.glob("*.txt"))

    def build_chunks(self) -> list[DocumentChunk]:
        """Generate chunks from all raw documents and persist the processed output."""

        documents = self.discover_raw_documents()
        all_chunks: list[DocumentChunk] = []

        if not documents:
            self.logger.warning("No se encontraron documentos en %s", self.settings.raw_data_dir)
            return all_chunks

        for file_path in documents:
            document_id = file_path.stem
            source_title = DOCUMENT_TITLES.get(document_id, file_path.stem.replace("_", " ").title())
            raw_text = file_path.read_text(encoding="utf-8")
            cleaned_text = clean_text(raw_text)
            chunks = split_text_into_chunks(
                cleaned_text,
                document_id=document_id,
                source_title=source_title,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            self.logger.info("Documento %s dividido en %s chunks", file_path.name, len(chunks))
            all_chunks.extend(chunks)

        ensure_directory(self.settings.processed_data_dir)
        write_json(self.settings.processed_chunks_file, all_chunks)
        self.logger.info("Chunks procesados guardados en %s", self.settings.processed_chunks_file)
        return all_chunks
