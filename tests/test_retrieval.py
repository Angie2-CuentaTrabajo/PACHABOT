from app.config import get_settings
from app.core.logger import setup_logging
from app.models.schemas import DocumentChunk
from app.services.retrieval_service import RetrievalService


def test_retrieval_returns_relevant_chunk() -> None:
    settings = get_settings()
    logger = setup_logging("INFO")
    service = RetrievalService(settings, logger)
    chunks = [
        DocumentChunk(
            chunk_id="doc-001",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="La autorización municipal exige requisitos y documentación básica.",
            section_title="TÍTULO III | DE LA AUTORIZACION MUNICIPAL",
            article_label="6",
            metadata={},
        ),
        DocumentChunk(
            chunk_id="doc-002",
            document_id="ordenanza_227_2019",
            source_title="Ordenanza 227-2019-MDP/C",
            text="Las zonas rígidas prohíben el comercio ambulatorio en vías determinadas.",
            section_title="TÍTULO IV | REGULACION DEL COMERCIO INFORMAL EN LA VIA PUBLICA",
            article_label="13",
            metadata={},
        ),
    ]
    service.build_index(chunks)
    results = service.search("que requisitos necesito para la autorizacion", top_k=2)

    assert results
    assert results[0].document_id == "ordenanza_108_2012"


def test_retrieval_prioritizes_exact_article_match() -> None:
    settings = get_settings()
    logger = setup_logging("INFO")
    service = RetrievalService(settings, logger)
    chunks = [
        DocumentChunk(
            chunk_id="doc-001",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Artículo 6°. Para ejercer el comercio informal se requiere autorización.",
            section_title="TÍTULO III | DE LA AUTORIZACION MUNICIPAL",
            article_label="6",
            metadata={},
        ),
        DocumentChunk(
            chunk_id="doc-002",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Artículo 7°. La autorización municipal es personal e intransferible.",
            section_title="TÍTULO III | DE LA AUTORIZACION MUNICIPAL",
            article_label="7",
            metadata={},
        ),
    ]
    service.build_index(chunks)
    results = service.search("que dice el articulo 7", top_k=2)

    assert results
    assert results[0].article_label == "7"


def test_retrieval_prefers_definition_over_preamble() -> None:
    settings = get_settings()
    logger = setup_logging("INFO")
    service = RetrievalService(settings, logger)
    chunks = [
        DocumentChunk(
            chunk_id="doc-001",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Es política de la gestión municipal regular el comercio ambulatorio en el distrito.",
            section_title="PREAMBULO",
            article_label="",
            metadata={},
        ),
        DocumentChunk(
            chunk_id="doc-002",
            document_id="ordenanza_108_2012",
            source_title="Ordenanza 108-2012-MDP/C",
            text="Artículo 2°.- Se entiende por comercio en la vía pública la actividad económica autorizada y temporal.",
            section_title="TÍTULO I | DEFINICIONES",
            article_label="2",
            metadata={},
        ),
    ]
    service.build_index(chunks)
    results = service.search("que es el comercio ambulatorio", top_k=2)

    assert results
    assert results[0].section_title.startswith("TÍTULO I")
