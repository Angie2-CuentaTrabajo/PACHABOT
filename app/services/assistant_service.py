from __future__ import annotations

import logging

from app.channels.schemas import IncomingChatMessage
from app.config import Settings
from app.memory.conversation_store import ConversationMemoryStore
from app.models.domain import QueryIntent
from app.models.schemas import AnswerPayload, ConversationTurn
from app.services.llm_service import LLMService
from app.services.query_router import QueryRouter
from app.tools.document_toolkit import DocumentToolkit
from app.utils.text_cleaner import normalize_for_search


GREETING_MARKERS = (
    "hola",
    "buenas",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "que tal",
)

THANKS_MARKERS = (
    "gracias",
    "muchas gracias",
    "ok gracias",
    "vale gracias",
)

ACKNOWLEDGEMENT_MARKERS = (
    "si",
    "sí",
    "si ayudame",
    "sí ayudame",
    "si, ayudame",
    "sí, ayudame",
    "ayudame",
    "ayúdame",
    "dale",
    "ok",
    "okay",
    "claro",
    "continua",
    "continúa",
    "sigue",
)


class AssistantService:
    """Main conversational orchestrator for all supported chat channels."""

    def __init__(
        self,
        settings: Settings,
        router: QueryRouter,
        document_toolkit: DocumentToolkit,
        llm_service: LLMService,
        memory_store: ConversationMemoryStore,
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self.router = router
        self.document_toolkit = document_toolkit
        self.llm_service = llm_service
        self.memory_store = memory_store
        self.logger = logger.getChild("assistant_service")

    def answer_chat_message(self, message: IncomingChatMessage) -> AnswerPayload:
        """Resolve a channel-agnostic incoming message into a safe answer."""

        question = message.text.strip()
        normalized_question = normalize_for_search(question)
        history = self.memory_store.get_recent_history(
            message.channel,
            message.session_id,
            limit=self.settings.memory_history_limit,
        )

        if self._is_greeting(normalized_question):
            answer = (
                "Hola. Puedo ayudarte con comercio ambulatorio"
                + (
                    " y, en este modo, tambien conversar de temas generales."
                    if self.settings.allow_general_chat
                    else "."
                )
            )
            return self._remember_social_reply(message, question, answer)

        if self._is_thanks(normalized_question):
            return self._remember_social_reply(
                message,
                question,
                "Con gusto. Si quieres, seguimos con otra consulta.",
            )

        if self._is_acknowledgement(normalized_question):
            return self._remember_social_reply(
                message,
                question,
                (
                    "Claro. Dime una consulta concreta y yo intento interpretarla. "
                    "Por ejemplo: que necesito para vender, que dice el articulo 7 "
                    "o cuanto se paga de SISA."
                ),
            )

        memory_payload = self._answer_from_memory_if_applicable(normalized_question, history)
        if memory_payload is not None:
            self._remember_exchange(message, question, memory_payload)
            return memory_payload

        prepared_question, preparation_notes = self.document_toolkit.prepare_query(question)
        routed = self.router.route(prepared_question)
        knowledge = self.document_toolkit.gather_knowledge(
            prepared_question,
            routed,
            history,
            original_question=question,
            preparation_notes=preparation_notes,
        )

        has_document_context = bool(knowledge.chunks) and (
            knowledge.confidence >= self.settings.confidence_threshold
        )
        if has_document_context:
            answer, used_llm = self.llm_service.generate_answer(
                knowledge.effective_query,
                knowledge.chunks,
                history=history,
            )
            payload = self._build_payload(
                answer=answer,
                sources=knowledge.sources[: self.settings.assistant_max_sources],
                confidence=knowledge.confidence,
                used_llm=used_llm,
                in_domain=True,
                intent=routed.intent if routed.intent != QueryIntent.OUT_OF_SCOPE else QueryIntent.GENERAL,
            )
            self._remember_exchange(message, question, payload)
            return payload

        if self.settings.allow_general_chat and not routed.in_domain:
            answer, used_llm = self.llm_service.generate_general_answer(
                question,
                history=history,
            )
            payload = self._build_payload(
                answer=answer,
                sources=[],
                confidence=0.25 if used_llm else 0.15,
                used_llm=used_llm,
                in_domain=False,
                intent=QueryIntent.OUT_OF_SCOPE,
            )
            self._remember_exchange(message, question, payload)
            return payload

        if not routed.in_domain:
            payload = self._build_payload(
                answer=(
                    "Por ahora solo atiendo consultas sobre comercio ambulatorio "
                    "basadas en las ordenanzas municipales cargadas."
                ),
                sources=[],
                confidence=0.0,
                used_llm=False,
                in_domain=False,
                intent=QueryIntent.OUT_OF_SCOPE,
            )
            self._remember_exchange(message, question, payload)
            return payload

        answer, used_llm = self.llm_service.generate_answer(
            knowledge.effective_query,
            [],
            history=history,
        )
        payload = self._build_payload(
            answer=answer,
            sources=[],
            confidence=knowledge.confidence,
            used_llm=used_llm,
            in_domain=True,
            intent=routed.intent,
        )
        self._remember_exchange(message, question, payload)
        return payload

    def answer_user_query(self, question: str) -> AnswerPayload:
        """Convenience wrapper for local/API usage without a Telegram update."""

        return self.answer_chat_message(
            IncomingChatMessage(
                channel="local",
                session_id="local-default",
                user_id="local-user",
                text=question,
            )
        )

    def reset_conversation(self, channel: str, session_id: str) -> None:
        """Clear stored memory for a specific conversation session."""

        self.memory_store.reset_session(channel, session_id)

    def get_runtime_status(self) -> str:
        """Expose a short runtime summary for operational commands."""

        llm_mode = "externo" if self.llm_service.client is not None else "fallback local"
        chunk_count = len(self.document_toolkit.retrieval_service.chunks)
        return (
            f"Canal activo: Telegram\n"
            f"Modo de respuesta: {llm_mode}\n"
            f"Proveedor configurado: {self.settings.llm_provider}\n"
            f"Modelo configurado: {self.settings.chat_model}\n"
            f"Modo libre fuera de dominio: {'si' if self.settings.allow_general_chat else 'no'}\n"
            f"Fragmentos indexados: {chunk_count}\n"
            f"Memoria por sesion: hasta {self.settings.memory_max_turns} turnos"
        )

    def _build_payload(
        self,
        *,
        answer: str,
        sources: list[str],
        confidence: float,
        used_llm: bool,
        in_domain: bool,
        intent: QueryIntent,
    ) -> AnswerPayload:
        """Create a normalized answer payload."""

        return AnswerPayload(
            answer=answer.strip(),
            sources=sources,
            intent=intent,
            in_domain=in_domain,
            confidence=confidence,
            used_llm=used_llm,
        )

    def _remember_social_reply(
        self,
        message: IncomingChatMessage,
        question: str,
        answer: str,
    ) -> AnswerPayload:
        """Store a social or operational reply in memory."""

        payload = self._build_payload(
            answer=answer,
            sources=[],
            confidence=1.0,
            used_llm=False,
            in_domain=True,
            intent=QueryIntent.GENERAL,
        )
        self._remember_exchange(message, question, payload)
        return payload

    def _remember_exchange(
        self,
        message: IncomingChatMessage,
        user_text: str,
        payload: AnswerPayload,
    ) -> None:
        """Persist the user and assistant turns for follow-up questions."""

        session_metadata = {
            "user_id": message.user_id,
            "user_display_name": message.user_display_name,
            "channel_metadata": message.metadata,
        }
        self.memory_store.append_turn(
            message.channel,
            message.session_id,
            ConversationTurn(
                role="user",
                text=user_text,
                metadata=session_metadata,
            ),
        )
        self.memory_store.append_turn(
            message.channel,
            message.session_id,
            ConversationTurn(
                role="assistant",
                text=payload.answer,
                metadata={
                    "sources": payload.sources,
                    "confidence": payload.confidence,
                    "used_llm": payload.used_llm,
                },
            ),
        )

    def _is_greeting(self, normalized_question: str) -> bool:
        """Detect short greeting-only messages."""

        return any(
            normalized_question == marker or normalized_question.startswith(f"{marker} ")
            for marker in GREETING_MARKERS
        )

    def _is_thanks(self, normalized_question: str) -> bool:
        """Detect simple acknowledgement messages."""

        return any(
            normalized_question == marker or normalized_question.startswith(f"{marker} ")
            for marker in THANKS_MARKERS
        )

    def _is_acknowledgement(self, normalized_question: str) -> bool:
        """Detect generic confirmations that are not actual document questions."""

        return any(
            normalized_question == marker or normalized_question.startswith(f"{marker} ")
            for marker in ACKNOWLEDGEMENT_MARKERS
        )

    def _answer_from_memory_if_applicable(
        self,
        normalized_question: str,
        history: list[ConversationTurn],
    ) -> AnswerPayload | None:
        """Answer simple chat-history questions directly from memory."""

        if not history:
            return None

        asks_first_question = any(
            marker in normalized_question
            for marker in (
                "que te pregunte primero",
                "que te dije primero",
                "cual fue mi primera pregunta",
                "que te pregunte al comienzo",
            )
        )
        if asks_first_question:
            first_user_turn = next(
                (
                    turn
                    for turn in history
                    if turn.role == "user"
                    and not self._is_greeting(normalize_for_search(turn.text))
                    and not self._is_acknowledgement(normalize_for_search(turn.text))
                ),
                None,
            )
            answer = (
                "Todavia no tengo una pregunta previa clara en este chat."
                if first_user_turn is None
                else f'La primera consulta clara que me hiciste en este tramo del chat fue: "{first_user_turn.text}".'
            )
            return self._build_payload(
                answer=answer,
                sources=[],
                confidence=1.0,
                used_llm=False,
                in_domain=True,
                intent=QueryIntent.GENERAL,
            )

        asks_last_answer = any(
            marker in normalized_question
            for marker in (
                "que me respondiste",
                "que dijiste",
                "que me acabas de decir",
                "que respondiste",
            )
        )
        if asks_last_answer:
            last_assistant_turn = next((turn for turn in reversed(history) if turn.role == "assistant"), None)
            answer = (
                "Todavia no tengo una respuesta previa registrada en este chat."
                if last_assistant_turn is None
                else f'Lo ultimo que te respondi fue: "{last_assistant_turn.text}".'
            )
            return self._build_payload(
                answer=answer,
                sources=[],
                confidence=1.0,
                used_llm=False,
                in_domain=True,
                intent=QueryIntent.GENERAL,
            )

        return None
