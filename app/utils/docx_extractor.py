from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from app.utils.text_cleaner import fix_mojibake

WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_docx_text(path: Path) -> str:
    """Extract visible text from a DOCX file using only the standard library."""

    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []

    for paragraph in root.findall(".//w:body/w:p", WORD_NAMESPACE):
        paragraph_parts: list[str] = []
        for node in paragraph.iter():
            tag = _strip_namespace(node.tag)
            if tag == "t" and node.text:
                paragraph_parts.append(node.text)
            elif tag == "tab":
                paragraph_parts.append("\t")
            elif tag in {"br", "cr"}:
                paragraph_parts.append("\n")

        paragraph_text = "".join(paragraph_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return _postprocess_extracted_text("\n".join(paragraphs))


def _postprocess_extracted_text(text: str) -> str:
    """Normalize extracted DOCX text for downstream legal chunking."""

    text = html.unescape(text)
    text = fix_mojibake(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Improve legal document readability before indexing.
    text = re.sub(r"(?i)\b(T[IÍ]TULO\s+[A-Z0-9IVXLC]+)\b", r"\n\1", text)
    text = re.sub(r"(?i)\b(Art[íi]culo\s+[0-9]+[A-Z]?[°º]?)", r"\n\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_namespace(tag: str) -> str:
    """Return the local XML tag name."""

    if "}" not in tag:
        return tag
    return tag.split("}", 1)[1]
