from __future__ import annotations

import re

from app.config import Settings
from app.models.domain import QueryIntent
from app.models.schemas import ConversationTurn, KnowledgeBundle, RetrievedChunk, RoutedQuery
from app.services.query_rewriter import QueryRewriter
from app.services.retrieval_service import RetrievalService
from app.utils.text_cleaner import normalize_for_search


ARTICLE_PATTERN = re.compile(r"art[ií]culo\s+([0-9]+[A-Z]?)", re.IGNORECASE)
COMMON_TYPO_HINTS = {
    "ke": "que",
    "qe": "que",
    "q": "que",
    "kuanto": "cuanto",
    "kanto": "cuanto",
    "komo": "como",
    "dnd": "donde",
    "sona": "zona",
    "sissa": "sisa",
    "modlo": "modulo",
}

INTENT_EXPANSIONS = {
    QueryIntent.REQUISITOS: "requisitos autorizacion tramite declaracion jurada dni",
    QueryIntent.MODULOS: "modulo medidas dimensiones especificaciones tecnicas",
    QueryIntent.PAGOS_SISA: "sisa pago tributo monto diario",
    QueryIntent.ZONAS_RIGIDAS: "zonas rigidas zonas prohibidas via publica",
    QueryIntent.AUTORIZACIONES: "autorizacion permiso vigencia renovacion",
    QueryIntent.FERIAS: "ferias feria temporal autorizacion especial",
    QueryIntent.PROHIBICIONES: "prohibiciones sanciones restricciones",
}


class DocumentToolkit:
    """Prepare queries, recover document evidence and package it for answer generation."""

    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
        query_rewriter: QueryRewriter,
        logger,
    ) -> None:
        self.settings = settings
        self.retrieval_service = retrieval_service
        self.query_rewriter = query_rewriter
        self.logger = logger.getChild("document_toolkit")

    def prepare_query(self, question: str) -> tuple[str, list[str]]:
        """Normalize only the most obvious typos before retrieval."""

        corrected_question = self._correct_spelling(question)
        notes: list[str] = []
        if normalize_for_search(corrected_question) != normalize_for_search(question):
            notes.append(
                f"Se interpreto la consulta corrigiendo posibles errores de escritura: {corrected_question}"
            )
        return corrected_question, notes

    def gather_knowledge(
        self,
        question: str,
        routed_query: RoutedQuery,
        history: list[ConversationTurn],
        *,
        original_question: str | None = None,
        preparation_notes: list[str] | None = None,
    ) -> KnowledgeBundle:
        """Collect document evidence using a rewritten query plus lightweight multi-search."""

        original_question = original_question or question
        preparation_notes = preparation_notes or []
        effective_query = self.query_rewriter.rewrite(question, history)
        search_queries = self._build_search_queries(question, effective_query, routed_query)

        merged: dict[str, RetrievedChunk] = {}
        for search_query in search_queries:
            results = self.retrieval_service.search(
                search_query,
                top_k=max(self.settings.retrieval_top_k + 1, 5),
                history=history,
            )
            for item in results:
                previous = merged.get(item.chunk_id)
                if previous is None or item.score > previous.score:
                    merged[item.chunk_id] = item

        final_chunks = self._finalize_chunks(question, effective_query, list(merged.values()))
        confidence = final_chunks[0].score if final_chunks else 0.0
        sources = list(dict.fromkeys(self._format_source(chunk) for chunk in final_chunks))

        notes = list(preparation_notes)
        if normalize_for_search(effective_query) != normalize_for_search(question):
            notes.append("La consulta se reformulo usando el historial reciente.")

        return KnowledgeBundle(
            original_query=original_question,
            effective_query=effective_query,
            chunks=final_chunks,
            confidence=confidence,
            sources=sources,
            notes=notes,
            search_queries=search_queries,
            tools_used=["query_rewrite", "document_search"],
        )

    def _build_search_queries(
        self,
        question: str,
        effective_query: str,
        routed_query: RoutedQuery,
    ) -> list[str]:
        """Create a small set of search queries without overfitting to hand-written rules."""

        queries = [effective_query]
        normalized_question = normalize_for_search(question)

        article_match = ARTICLE_PATTERN.search(question)
        if article_match:
            queries.append(f"Articulo {article_match.group(1)} comercio ambulatorio")

        if any(pattern in normalized_question for pattern in ("que es", "defin", "se entiende", "concepto")):
            queries.append(f"{effective_query} definiciones")

        expansion = INTENT_EXPANSIONS.get(routed_query.intent)
        if expansion:
            queries.append(f"{effective_query} {expansion}")

        return list(dict.fromkeys(query.strip() for query in queries if query.strip()))

    def _finalize_chunks(
        self,
        question: str,
        effective_query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Stabilize retrieved results with a light final rerank."""

        normalized_question = normalize_for_search(question)
        article_match = ARTICLE_PATTERN.search(effective_query)
        article_label = article_match.group(1).upper() if article_match else ""
        asks_definition = any(
            pattern in normalized_question
            for pattern in ("que es", "defin", "se entiende", "concepto")
        )

        reranked: list[RetrievedChunk] = []
        for chunk in chunks:
            bonus = 0.0
            normalized_text = normalize_for_search(chunk.text)
            normalized_section = normalize_for_search(chunk.section_title)

            if article_label and chunk.article_label and chunk.article_label.upper() == article_label:
                bonus += 0.45
            if asks_definition and (
                "definiciones" in normalized_section or "se entiende por" in normalized_text
            ):
                bonus += 0.20

            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_title=chunk.source_title,
                    text=chunk.text,
                    score=chunk.score + bonus,
                    section_title=chunk.section_title,
                    article_label=chunk.article_label,
                    metadata=chunk.metadata,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[: self.settings.retrieval_top_k]

    def _format_source(self, chunk: RetrievedChunk) -> str:
        """Create a concise source string."""

        parts = [chunk.source_title]
        if chunk.article_label:
            parts.append(f"Art. {chunk.article_label}")
        elif chunk.section_title:
            parts.append(chunk.section_title)
        return " | ".join(parts)

    def _correct_spelling(self, question: str) -> str:
        """Correct only a small whitelist of obvious typo forms."""

        normalized_question = normalize_for_search(question)
        tokens = re.findall(r"[a-záéíóúñ0-9]+", normalized_question)
        corrected_tokens = [COMMON_TYPO_HINTS.get(token, token) for token in tokens]
        return " ".join(corrected_tokens)
