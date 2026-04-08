from app.utils.chunking import split_text_into_chunks


def test_split_text_into_multiple_chunks() -> None:
    text = ("Parrafo de prueba. " * 60) + "\n\n" + ("Otro parrafo. " * 60)
    chunks = split_text_into_chunks(
        text,
        document_id="demo",
        source_title="Demo",
        chunk_size=300,
        overlap=50,
    )

    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "demo-001"
    assert chunks[0].source_title == "Demo"


def test_split_legal_text_preserves_article_metadata() -> None:
    text = (
        "TÍTULO III\n"
        "DE LA AUTORIZACION MUNICIPAL\n"
        "Artículo 6°.- Para ejercer el comercio informal se requiere autorización municipal.\n"
        "Artículo 7°.- La autorización municipal es personal e intransferible.\n"
    )
    chunks = split_text_into_chunks(
        text,
        document_id="ordenanza_demo",
        source_title="Ordenanza Demo",
        chunk_size=400,
        overlap=50,
    )

    assert len(chunks) >= 2
    assert chunks[0].section_title.startswith("TÍTULO III")
    assert chunks[0].article_label == "6"
    assert chunks[1].article_label == "7"


def test_split_legal_text_ignores_preamble_article_references() -> None:
    text = (
        "CONSIDERANDO:\n"
        "Que, el artículo 194° de la Constitución Política del Perú establece principios generales.\n"
        "TÍTULO I\n"
        "DEFINICIONES\n"
        "Artículo 2°.- Se entiende por comercio ambulatorio la actividad autorizada.\n"
    )
    chunks = split_text_into_chunks(
        text,
        document_id="ordenanza_demo",
        source_title="Ordenanza Demo",
        chunk_size=400,
        overlap=50,
    )

    assert any(chunk.section_title == "PREAMBULO" and chunk.article_label == "" for chunk in chunks)
    assert any(chunk.article_label == "2" for chunk in chunks)
