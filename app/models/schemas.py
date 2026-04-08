from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.models.domain import QueryIntent


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    source_title: str
    text: str
    section_title: str = ""
    article_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_title: str
    text: str
    score: float
    section_title: str = ""
    article_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoutedQuery:
    original_query: str
    normalized_query: str
    intent: QueryIntent
    in_domain: bool
    matched_keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnswerPayload:
    answer: str
    sources: list[str]
    intent: QueryIntent
    in_domain: bool
    confidence: float
    used_llm: bool


@dataclass(slots=True)
class ConversationTurn:
    role: str
    text: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeBundle:
    original_query: str
    effective_query: str
    chunks: list[RetrievedChunk]
    confidence: float
    sources: list[str]
    notes: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
