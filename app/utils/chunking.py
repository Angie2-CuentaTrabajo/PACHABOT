from __future__ import annotations

import re

from app.models.schemas import DocumentChunk
from app.utils.text_cleaner import clean_text


TITLE_PATTERN = re.compile(r"^(T[IÍ]TULO\s+[A-Z0-9IVXLC]+.*)$", re.IGNORECASE)
ARTICLE_HEADER_PATTERN = re.compile(
    r"^Art[íi]culo\s+([0-9]+[A-Z]?)\s*[°º]?\s*(?:[.\-:]|-\s)",
    re.IGNORECASE,
)
SUBSECTION_PATTERN = re.compile(r"^[A-ZÁÉÍÓÚÑ0-9 ,;:/().\-]{3,}$")


def split_text_into_chunks(
    text: str,
    *,
    document_id: str,
    source_title: str,
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[DocumentChunk]:
    """Split legal text into meaningful chunks using titles and article headers."""

    cleaned = clean_text(text)
    structured_chunks = _extract_structured_legal_chunks(
        cleaned,
        document_id=document_id,
        source_title=source_title,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if structured_chunks:
        return structured_chunks

    return _fallback_paragraph_chunks(
        cleaned,
        document_id=document_id,
        source_title=source_title,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def _extract_structured_legal_chunks(
    text: str,
    *,
    document_id: str,
    source_title: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Chunk legal documents, preserving titles and ignoring preamble article references."""

    lines = [line.strip() for line in text.splitlines()]
    chunks: list[DocumentChunk] = []
    chunk_number = 1
    base_title = ""
    current_title = ""
    title_buffer: list[str] = []
    seen_regulation_body = False
    preamble_buffer: list[str] = []

    index = 0
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue

        if TITLE_PATTERN.match(line):
            seen_regulation_body = True
            base_title = line
            current_title = line
            title_buffer = [line]
            if index + 1 < len(lines):
                next_line = lines[index + 1]
                if next_line and not TITLE_PATTERN.match(next_line) and not ARTICLE_HEADER_PATTERN.match(next_line):
                    title_buffer.append(next_line)
                    base_title = f"{line} | {next_line}"
                    current_title = base_title
                    index += 1
            index += 1
            continue

        if not seen_regulation_body:
            preamble_buffer.append(line)
            index += 1
            continue

        article_match = ARTICLE_HEADER_PATTERN.match(line)
        if article_match:
            article_lines = [line]
            article_label = article_match.group(1)
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if TITLE_PATTERN.match(next_line) or ARTICLE_HEADER_PATTERN.match(next_line):
                    break
                if next_line:
                    article_lines.append(next_line)
                index += 1

            article_text = "\n".join(article_lines)
            prefixed_text = article_text if not title_buffer else "\n".join(title_buffer + [article_text])
            article_chunks = _split_large_legal_block(
                prefixed_text,
                document_id=document_id,
                source_title=source_title,
                section_title=current_title,
                article_label=article_label,
                chunk_number_start=chunk_number,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            chunks.extend(article_chunks)
            chunk_number += len(article_chunks)
            continue

        if _looks_like_subsection_heading(line):
            current_title = f"{base_title} | {line}" if base_title else line
            title_buffer = [current_title]
            index += 1
            continue

        # Non-empty lines inside the regulation body that are not article headers become contextual chunks.
        contextual_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if not next_line:
                index += 1
                if contextual_lines:
                    break
                continue
            if TITLE_PATTERN.match(next_line) or ARTICLE_HEADER_PATTERN.match(next_line):
                break
            contextual_lines.append(next_line)
            index += 1

        contextual_text = "\n".join(contextual_lines)
        contextual_chunks = _split_large_legal_block(
            contextual_text if not title_buffer else "\n".join(title_buffer + [contextual_text]),
            document_id=document_id,
            source_title=source_title,
            section_title=current_title or "PREAMBULO",
            article_label="",
            chunk_number_start=chunk_number,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(contextual_chunks)
        chunk_number += len(contextual_chunks)

    if preamble_buffer:
        preamble_text = "\n".join(preamble_buffer).strip()
        preamble_chunks = _split_large_legal_block(
            preamble_text,
            document_id=document_id,
            source_title=source_title,
            section_title="PREAMBULO",
            article_label="",
            chunk_number_start=chunk_number,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(preamble_chunks)

    return chunks


def _looks_like_subsection_heading(line: str) -> bool:
    """Identify uppercase legal subsection headings like DEFINICIONES or CRITERIOS."""

    if not line or TITLE_PATTERN.match(line) or ARTICLE_HEADER_PATTERN.match(line):
        return False
    if len(line) > 120:
        return False
    return SUBSECTION_PATTERN.match(line) is not None


def _split_large_legal_block(
    text: str,
    *,
    document_id: str,
    source_title: str,
    section_title: str,
    article_label: str,
    chunk_number_start: int,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Split a legal block while preserving article identity."""

    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[DocumentChunk] = []
    buffer = ""
    chunk_number = chunk_number_start

    for paragraph in paragraphs:
        candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    source_title=source_title,
                    text=buffer,
                    section_title=section_title,
                    article_label=article_label,
                    chunk_number=chunk_number,
                )
            )
            chunk_number += 1
            carry = buffer[-overlap:] if overlap > 0 else ""
            buffer = f"{carry}\n{paragraph}".strip()
        else:
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                slice_text = paragraph[start:end].strip()
                chunks.append(
                    _build_chunk(
                        document_id=document_id,
                        source_title=source_title,
                        text=slice_text,
                        section_title=section_title,
                        article_label=article_label,
                        chunk_number=chunk_number,
                    )
                )
                chunk_number += 1
                start = max(end - overlap, start + 1)
            buffer = ""

    if buffer:
        chunks.append(
            _build_chunk(
                document_id=document_id,
                source_title=source_title,
                text=buffer,
                section_title=section_title,
                article_label=article_label,
                chunk_number=chunk_number,
            )
        )

    return chunks


def _build_chunk(
    *,
    document_id: str,
    source_title: str,
    text: str,
    section_title: str,
    article_label: str,
    chunk_number: int,
) -> DocumentChunk:
    """Create a chunk with consistent metadata."""

    return DocumentChunk(
        chunk_id=f"{document_id}-{chunk_number:03d}",
        document_id=document_id,
        source_title=source_title,
        text=text,
        section_title=section_title,
        article_label=article_label,
        metadata={
            "chunk_number": chunk_number,
            "section_title": section_title,
            "article_label": article_label,
        },
    )


def _fallback_paragraph_chunks(
    text: str,
    *,
    document_id: str,
    source_title: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Fallback splitter for non-legal or poorly structured text."""

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[DocumentChunk] = []
    buffer = ""
    chunk_number = 1

    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    source_title=source_title,
                    text=buffer,
                    section_title="",
                    article_label="",
                    chunk_number=chunk_number,
                )
            )
            chunk_number += 1
            carry = buffer[-overlap:] if overlap > 0 else ""
            buffer = f"{carry}\n\n{paragraph}".strip()
        else:
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                slice_text = paragraph[start:end].strip()
                chunks.append(
                    _build_chunk(
                        document_id=document_id,
                        source_title=source_title,
                        text=slice_text,
                        section_title="",
                        article_label="",
                        chunk_number=chunk_number,
                    )
                )
                chunk_number += 1
                start = max(end - overlap, start + 1)
            buffer = ""

    if buffer:
        chunks.append(
            _build_chunk(
                document_id=document_id,
                source_title=source_title,
                text=buffer,
                section_title="",
                article_label="",
                chunk_number=chunk_number,
            )
        )

    return chunks
