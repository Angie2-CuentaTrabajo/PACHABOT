from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.logger import setup_logging
from app.utils.docx_extractor import extract_docx_text
from app.utils.helpers import ensure_directory


DEFAULT_MAPPINGS = {
    ROOT_DIR / "data" / "raw" / "ordenanza_108_2012.txt": Path(
        r"C:\Users\PC\Downloads\ORDENANZA Nº 108-2012-MDP_C.docx"
    ),
    ROOT_DIR / "data" / "raw" / "ordenanza_227_2019.txt": Path(
        r"C:\Users\PC\Downloads\ORDENANZA N° 227-2019-MDP_C.docx"
    ),
}


def main() -> None:
    """Extract raw text files from the municipal DOCX sources."""

    logger = setup_logging("INFO")
    target_dir = ROOT_DIR / "data" / "raw"
    ensure_directory(target_dir)

    for output_path, source_path in DEFAULT_MAPPINGS.items():
        if not source_path.exists():
            logger.warning("No se encontró el documento fuente: %s", source_path)
            continue

        text = extract_docx_text(source_path)
        output_path.write_text(text, encoding="utf-8-sig")
        logger.info("Documento extraído: %s -> %s", source_path.name, output_path.name)


if __name__ == "__main__":
    main()
