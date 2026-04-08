from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.config import Settings
from app.core.prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_answer_messages,
    build_general_chat_messages,
    build_query_rewrite_messages,
)
from app.models.schemas import ConversationTurn, RetrievedChunk
from app.utils.text_cleaner import normalize_for_search

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


CURRENCY_PATTERN = re.compile(r"S/\s*\.?\s*\d+(?:\.\d+)?", re.IGNORECASE)
MEASURE_PATTERN = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(m|metros?|dias|meses|anos)\b", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"art[ií]culo\s+([0-9]+[A-Z]?)", re.IGNORECASE)
SPANISH_STOPWORDS = {
    "que",
    "cual",
    "como",
    "cuando",
    "donde",
    "cuanto",
    "una",
    "uno",
    "unos",
    "unas",
    "para",
    "sobre",
    "del",
    "de",
    "la",
    "el",
    "los",
    "las",
    "un",
    "por",
    "con",
    "se",
    "me",
    "puedes",
    "puede",
    "dice",
    "dime",
    "explica",
    "explicame",
    "explicacion",
    "norma",
    "ordenanza",
    "articulo",
    "mide",
    "pago",
    "dura",
    "vigencia",
    "plazo",
    "caso",
    "pasa",
    "resolver",
    "situacion",
    "seguimiento",
    "osea",
    "entonces",
    "primero",
}


@dataclass(slots=True)
class QuestionFrame:
    kind: str
    normalized_question: str
    article_label: str = ""
    subject: str = ""
    focus_terms: list[str] = field(default_factory=list)
    current_question: str = ""
    wants_confirmation: bool = False
    wants_plain_explanation: bool = False


class LLMService:
    """Generate answers using an external LLM when available, or a safer fallback."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger.getChild("llm_service")
        self.client = None
        self.provider = settings.llm_provider.lower().strip()

        if OpenAI is None or settings.llm_mode == "mock":
            return

        if self.provider == "grok" and settings.grok_api_key:
            self.client = OpenAI(
                api_key=settings.grok_api_key,
                base_url=settings.grok_base_url,
            )
            return

        if self.provider == "openai" and settings.openai_api_key:
            client_kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                client_kwargs["base_url"] = settings.openai_base_url
            self.client = OpenAI(**client_kwargs)

    def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        history: list[ConversationTurn] | None = None,
        orchestration_notes: list[str] | None = None,
    ) -> tuple[str, bool]:
        """Return an answer and whether an external LLM was used."""

        history = history or []
        _ = orchestration_notes or []

        if self.client is None:
            return self._fallback_answer(question, chunks), False

        try:
            messages = build_answer_messages(question, chunks, history)
            answer = self._call_provider(
                system_prompt=SYSTEM_PROMPT,
                messages=messages,
                temperature=0.15,
                warning_label="respuesta municipal",
            )
            return answer, True
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Fallo el uso del LLM externo: %s", exc)
            if self._should_disable_external_client(exc):
                self.logger.warning(
                    "Se desactivara temporalmente el proveedor externo y se continuara en fallback local."
                )
                self.client = None
            return self._fallback_answer(question, chunks), False

    def generate_general_answer(
        self,
        question: str,
        *,
        history: list[ConversationTurn] | None = None,
    ) -> tuple[str, bool]:
        """Answer a free-form question outside the municipal domain."""

        history = history or []

        if self.client is None:
            return self._fallback_general_answer(question), False

        try:
            messages = build_general_chat_messages(question, history)
            answer = self._call_provider(
                system_prompt=GENERAL_CHAT_SYSTEM_PROMPT,
                messages=messages,
                temperature=0.35,
                warning_label="chat general",
            )
            return answer, True
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Fallo el chat general con el proveedor externo: %s", exc)
            if self._should_disable_external_client(exc):
                self.logger.warning(
                    "Se desactivara temporalmente el proveedor externo y el chat general seguira en fallback local."
                )
                self.client = None
            return self._fallback_general_answer(question), False

    def rewrite_query(
        self,
        question: str,
        *,
        history: list[ConversationTurn] | None = None,
    ) -> str:
        """Rewrite a follow-up question into a clearer retrieval query."""

        history = history or []

        if self.client is None:
            return question

        try:
            messages = build_query_rewrite_messages(question, history)
            rewritten = self._call_provider(
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                messages=messages,
                temperature=0.0,
                warning_label="query rewriting",
            )
            cleaned = rewritten.strip().strip("\"'")
            first_line = cleaned.splitlines()[0].strip() if cleaned else question
            return first_line or question
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Fallo la reescritura de consulta con el proveedor externo: %s", exc)
            if self._should_disable_external_client(exc):
                self.logger.warning(
                    "Se desactivara temporalmente el proveedor externo y la reescritura seguira en modo heuristico."
                )
                self.client = None
            return question

    def _call_provider(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float,
        warning_label: str,
    ) -> str:
        """Call the external provider using the most compatible API available."""

        full_messages = [{"role": "system", "content": system_prompt}, *messages]

        try:
            response = self.client.responses.create(
                model=self.settings.chat_model,
                input=full_messages,
            )
            output_text = getattr(response, "output_text", "").strip()
            if output_text:
                return output_text
        except Exception as exc:
            self.logger.warning(
                "Fallo el endpoint responses para %s; se intentara chat.completions: %s",
                warning_label,
                exc,
            )

        completion = self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=full_messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()

    def _fallback_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Build a local answer by inferring the user's intent from the question and evidence."""

        frame = self._analyze_question(question)
        evidence = self._collect_evidence(frame, chunks)

        if not evidence:
            return (
                "Encontre informacion relacionada, pero no con la claridad suficiente "
                "como para darte una respuesta segura en este momento."
            )

        if frame.kind == "article":
            return self._compose_article_answer(frame, evidence)
        if frame.kind == "definition":
            return self._compose_definition_answer(frame, evidence)
        if frame.kind == "case":
            return self._compose_case_answer(frame, evidence)
        if frame.kind == "quantity":
            return self._compose_quantity_answer(frame, evidence)
        if frame.kind == "procedure":
            return self._compose_procedure_answer(evidence)
        if frame.kind == "location":
            return self._compose_location_answer(evidence)
        return self._compose_general_answer(evidence)

    def _fallback_general_answer(self, question: str) -> str:
        """Offer a basic general-chat fallback when no unrestricted LLM is active."""

        normalized_question = normalize_for_search(question)

        if any(greeting in normalized_question for greeting in ("hola", "buenas", "que tal")):
            return "Hola. Aqui estoy. Puedes preguntarme lo que quieras."

        if "como estas" in normalized_question or "como te va" in normalized_question:
            return "Estoy bien y listo para conversar contigo. Si quieres, probemos con una pregunta general."

        if any(
            marker in normalized_question
            for marker in ("quien eres", "que puedes hacer", "como funcionas")
        ):
            return (
                "Ahora mismo puedo conversar tambien fuera del tema municipal. "
                "Eso si: como en este entorno no hay un LLM general activo, mi modo libre "
                "todavia es basico comparado con ChatGPT completo."
            )

        if "chiste" in normalized_question:
            return "Claro: por que el expediente nunca se perdio? Porque siempre estaba bien foliado."

        if any(marker in normalized_question for marker in ("consejo", "recomiendame", "que opinas")):
            return (
                "Puedo intentarlo, pero te hablo con honestidad: para responder con libertad real "
                "como un asistente general necesito un LLM externo activo. En este modo local puedo "
                "mantener una conversacion basica, pero no reemplazo a ChatGPT completo."
            )

        return (
            "Puedo abrir la conversacion a temas generales, pero para responder con total libertad "
            "como ChatGPT normal necesito activar un proveedor LLM. En este modo local todavia estoy "
            "limitado a respuestas generales bastante basicas."
        )

    def _analyze_question(self, question: str) -> QuestionFrame:
        """Infer the general question type and its main subject."""

        normalized_question = normalize_for_search(question)
        current_question = question.split("Seguimiento:")[-1].strip()
        normalized_current = normalize_for_search(current_question)
        article_match = ARTICLE_PATTERN.search(current_question) or ARTICLE_PATTERN.search(question)
        focus_terms = self._extract_focus_terms(normalized_question)
        subject = " ".join(focus_terms[:3]).strip()
        wants_confirmation = self._looks_like_confirmation(normalized_current)
        wants_plain_explanation = any(
            pattern in normalized_current
            for pattern in (
                "hablame",
                "hablame de",
                "explicame",
                "cuentame",
                "resumeme",
                "en simple",
                "con tus palabras",
            )
        )

        if article_match:
            return QuestionFrame(
                kind="article",
                normalized_question=normalized_question,
                article_label=article_match.group(1).upper(),
                subject=subject,
                focus_terms=focus_terms,
                current_question=normalized_current,
                wants_confirmation=wants_confirmation,
                wants_plain_explanation=wants_plain_explanation,
            )

        if self._looks_like_case(normalized_current) or self._looks_like_case(normalized_question):
            kind = "case"
        elif "sisa" in normalized_question or ("pago" in normalized_question and "tribut" in normalized_question):
            kind = "quantity"
        elif any(
            pattern in normalized_current or pattern in normalized_question
            for pattern in ("que es", "defin", "significa", "se entiende")
        ) or wants_plain_explanation:
            kind = "definition"
        elif any(
            pattern in normalized_current or pattern in normalized_question
            for pattern in ("cuanto", "monto", "valor", "mide", "medida", "plazo", "vigencia", "dura")
        ) or (wants_confirmation and any(token in normalized_question for token in ("sisa", "modulo", "monto", "sol"))):
            kind = "quantity"
        elif any(
            pattern in normalized_current or pattern in normalized_question
            for pattern in ("como", "requis", "tramite", "solicitud", "permiso", "autoriz")
        ):
            kind = "procedure"
        elif any(
            pattern in normalized_current or pattern in normalized_question
            for pattern in ("donde", "zona", "ubicacion", "lugar", "via publica")
        ):
            kind = "location"
        else:
            kind = "summary"

        return QuestionFrame(
            kind=kind,
            normalized_question=normalized_question,
            subject=subject,
            focus_terms=focus_terms,
            current_question=normalized_current,
            wants_confirmation=wants_confirmation,
            wants_plain_explanation=wants_plain_explanation,
        )

    def _collect_evidence(
        self,
        frame: QuestionFrame,
        chunks: list[RetrievedChunk],
    ) -> list[tuple[float, str, RetrievedChunk]]:
        """Score candidate evidence sentences using the question type and retrieved context."""

        candidates: list[tuple[float, str, RetrievedChunk]] = []
        for chunk in chunks:
            cleaned_text = self._strip_legal_headers(chunk.text)
            for sentence in self._split_candidates(cleaned_text):
                score = self._score_candidate(frame, chunk, sentence)
                if score <= 0:
                    continue
                candidates.append((score, sentence, chunk))

        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, str, RetrievedChunk]] = []
        seen_sentences: set[str] = set()
        for item in ranked:
            fingerprint = normalize_for_search(item[1])
            if fingerprint in seen_sentences:
                continue
            seen_sentences.add(fingerprint)
            selected.append(item)
            if len(selected) >= 3:
                break
        return selected

    def _score_candidate(
        self,
        frame: QuestionFrame,
        chunk: RetrievedChunk,
        sentence: str,
    ) -> float:
        """Assign a relevance score to an evidence sentence."""

        normalized_sentence = normalize_for_search(sentence)
        if len(normalized_sentence) < 18:
            return -1.0
        if normalized_sentence.startswith(("titulo ", "capitulo ", "base legal")):
            return -1.0

        score = 0.1
        if len(normalized_sentence) > 280:
            score -= 0.20
        overlap = sum(1 for token in frame.focus_terms if token in normalized_sentence)
        score += overlap * 0.55
        if frame.focus_terms and overlap == 0:
            score -= 0.20

        if frame.kind == "article" and chunk.article_label == frame.article_label:
            score += 1.2

        if frame.kind == "definition":
            if any(marker in normalized_sentence for marker in ("se entiende por", " es la ", " es el ", "areas de la via publica")):
                score += 1.0

        if frame.kind == "case":
            if any(
                marker in normalized_sentence
                for marker in (
                    "se requiere",
                    "autorizacion",
                    "prohib",
                    "no autoriza",
                    "revoc",
                    "retiro",
                    "sancion",
                    "requisitos",
                )
            ):
                score += 0.9

        if frame.kind == "quantity":
            if CURRENCY_PATTERN.search(sentence) or MEASURE_PATTERN.search(sentence):
                score += 0.9
            if any(marker in normalized_sentence for marker in ("diario", "mensual", "anual", "meses", "metros", "m")):
                score += 0.3
            if "sisa" in frame.normalized_question and "sisa" in normalized_sentence:
                score += 0.7
            if any(marker in normalized_sentence for marker in ("numero de comprobante", "tupa", "diario oficial")):
                score -= 0.65

        if frame.kind == "procedure":
            if any(marker in normalized_sentence for marker in ("debera", "solicitud", "presentar", "requisitos", "tramite")):
                score += 0.8

        if frame.kind == "location":
            if any(marker in normalized_sentence for marker in ("zona", "areas", "via publica", "ubicacion", "no autoriza", "prohib")):
                score += 0.8

        if "considerando" in normalized_sentence or "por cuanto" in normalized_sentence:
            score -= 0.5
        if any(
            marker in normalized_sentence
            for marker in (
                "facultades conferidas",
                "senores regidores",
                "ley n",
                "diario oficial",
                "tupa",
            )
        ):
            score -= 0.55

        return score

    def _compose_article_answer(
        self,
        frame: QuestionFrame,
        evidence: list[tuple[float, str, RetrievedChunk]],
    ) -> str:
        """Compose a summary for article-specific questions."""

        primary_sentence = self._humanize_sentence(evidence[0][1])
        return (
            f"Si lo pongo en palabras simples, el articulo {frame.article_label} dice que "
            f"{self._ensure_sentence_starts_lower(primary_sentence)}."
        )

    def _compose_definition_answer(
        self,
        frame: QuestionFrame,
        evidence: list[tuple[float, str, RetrievedChunk]],
    ) -> str:
        """Compose a definition-like answer in plain language."""

        subject = self._resolve_subject_label(frame)
        primary_sentence = self._humanize_sentence(evidence[0][1])
        primary_lower = self._ensure_sentence_starts_lower(primary_sentence)
        normalized_primary = normalize_for_search(primary_sentence)

        if any(
            normalized_primary.startswith(marker)
            for marker in ("es el ", "es la ", "es un ", "es una ")
        ):
            response = f"Claro. En simple, {subject} {primary_lower}"
        elif subject in normalized_primary:
            response = f"Claro. En simple, {primary_lower}"
        else:
            response = f"Claro. En simple, {subject} es {primary_lower}"

        extra_fact = self._extract_currency_fact([item[1] for item in evidence]) or self._extract_measure_fact([item[1] for item in evidence])
        if extra_fact:
            response += f" Ademas, aparece este dato concreto: {extra_fact}."
        return response

    def _compose_case_answer(
        self,
        frame: QuestionFrame,
        evidence: list[tuple[float, str, RetrievedChunk]],
    ) -> str:
        """Compose a reasoned orientation for a practical case."""

        humanized = [self._humanize_sentence(item[1]) for item in evidence]
        first = self._ensure_sentence_starts_lower(humanized[0])
        normalized_joined = normalize_for_search(" ".join(humanized))

        if "sin autorizacion" in frame.normalized_question and "autorizacion" in normalized_joined:
            response = f"No. {self._sentence_case(first)}."
            if len(humanized) > 1:
                second = self._ensure_sentence_starts_lower(humanized[1])
                response += f" Ademas, {second}."
            return response

        response = f"En un caso asi, lo mas importante es que {first}."

        if len(humanized) > 1:
            second = self._ensure_sentence_starts_lower(humanized[1])
            response += f" Ademas, {second}."

        if len(humanized) > 2:
            third = self._ensure_sentence_starts_lower(humanized[2])
            response += f" Como referencia adicional, {third}."

        response += (
            " Si quieres, te lo bajo a un caso concreto con conclusion y articulo aplicable."
        )
        return response

    def _compose_quantity_answer(
        self,
        frame: QuestionFrame,
        evidence: list[tuple[float, str, RetrievedChunk]],
    ) -> str:
        """Compose an answer for amounts, measures, terms or durations."""

        subject = self._resolve_subject_label(frame)
        humanized = [self._humanize_sentence(item[1]) for item in evidence]
        joined = " ".join(humanized)
        currency_fact = self._extract_currency_fact([item[1] for item in evidence])
        measure_fact = self._extract_measure_fact([item[1] for item in evidence])
        period_fact = self._extract_period_fact([item[1] for item in evidence])
        subject_opening = self._sentence_case(subject)
        subject_lower = subject.lower()

        if "modulo" in subject_lower and measure_fact:
            if "entre modulos" in normalize_for_search(joined) or "uno del otro" in normalize_for_search(joined):
                return (
                    f"No veo una medida unica del tamano del modulo. Lo mas claro que encuentro "
                    f"es {measure_fact}, pero ese dato parece referirse a la distancia o separacion "
                    "entre modulos, no al tamano exacto del modulo."
                )
            return f"{subject_opening} tiene como medida mas clara recuperada {measure_fact}."

        if "sisa" in subject_lower and currency_fact:
            period_suffix = f" {period_fact}" if period_fact else ""
            if frame.wants_confirmation:
                return f"Si. En lo que aparece cargado, el SISA figura como un pago de {currency_fact}{period_suffix}."
            return (
                f"Claro. El SISA aparece como un pago municipal y el monto mas claro que encuentro "
                f"es {currency_fact}{period_suffix}."
            )

        if currency_fact and "modulo" not in subject_lower:
            if frame.wants_confirmation:
                return f"Si. Lo mas claro que encuentro sobre {subject} es {currency_fact}."
            return f"Lo mas claro que encuentro sobre {subject} es {currency_fact}."

        if measure_fact:
            return f"Lo mas claro que encuentro sobre {subject} es {measure_fact}."

        if "modulo" in subject_lower:
            joined_normalized = normalize_for_search(joined)
            if any(
                marker in joined_normalized
                for marker in ("parametros tecnicos", "especificaciones tecnicas", "medidas establecidas")
            ):
                return (
                    "No veo una medida exacta del modulo en los articulos recuperados. "
                    "Lo que si aparece es que el modulo debe ajustarse a parametros tecnicos "
                    "o especificaciones aprobadas por la municipalidad."
                )
            return (
                "No veo una medida exacta del modulo en los articulos recuperados. "
                f"Lo mas cercano que encuentro es que {self._ensure_sentence_starts_lower(humanized[0])}."
            )

        return (
            f"No veo un valor o medida exacta sobre {subject}. "
            f"Lo mas cercano que encuentro es que {self._ensure_sentence_starts_lower(humanized[0])}."
        )

    def _compose_procedure_answer(
        self,
        evidence: list[tuple[float, str, RetrievedChunk]],
    ) -> str:
        """Compose an answer for procedure or requirement questions."""

        humanized = [self._humanize_sentence(item[1]) for item in evidence]
        first = self._ensure_sentence_starts_lower(humanized[0])
        if len(evidence) == 1:
            return f"En pocas palabras, {first}."
        second = self._ensure_sentence_starts_lower(humanized[1])
        return f"En pocas palabras, {first}. Ademas, {second}."

    def _compose_location_answer(
        self,
        evidence: list[tuple[float, str, RetrievedChunk]],
    ) -> str:
        """Compose an answer for zones, locations or restrictions."""

        first = self._ensure_sentence_starts_lower(self._humanize_sentence(evidence[0][1]))
        return f"En pocas palabras, {first}."

    def _compose_general_answer(
        self,
        evidence: list[tuple[float, str, RetrievedChunk]],
    ) -> str:
        """Compose a general explanatory answer."""

        humanized = [self._humanize_sentence(item[1]) for item in evidence]
        first = self._ensure_sentence_starts_lower(humanized[0])
        if len(evidence) == 1:
            return f"En pocas palabras, {first}."
        second = self._ensure_sentence_starts_lower(humanized[1])
        return f"En pocas palabras, {first}. Ademas, {second}."

    def _extract_focus_terms(self, normalized_question: str) -> list[str]:
        """Extract the main semantic tokens from the user's question."""

        tokens = re.findall(r"[a-záéíóúñ0-9]+", normalized_question)
        return [
            token
            for token in tokens
            if len(token) > 2 and token not in SPANISH_STOPWORDS
        ]

    def _looks_like_case(self, normalized_question: str) -> bool:
        """Detect when the user is presenting a practical scenario."""

        return any(
            pattern in normalized_question
            for pattern in (
                "que pasa si",
                "si un",
                "si una",
                "si el",
                "si la",
                "en un caso",
                "en este caso",
                "tengo un caso",
                "te planteo",
                "supongamos",
                "por ejemplo",
                "una persona",
                "un comerciante",
                "un vendedor",
                "puedo ",
                "podria ",
                "se puede",
                "esta permitido",
                "sin autorizacion",
                "seria valido",
                "seria legal",
                "corresponde",
            )
        )

    def _looks_like_confirmation(self, normalized_question: str) -> bool:
        """Detect short follow-ups that are asking for confirmation."""

        stripped = normalized_question.strip()
        return any(
            marker in stripped
            for marker in (
                "entonces",
                "osea",
                "o sea",
                "si es",
                "osea que",
                "eso quiere decir",
            )
        ) or stripped.startswith(("1 ", "un sol", "s/", "s/."))

    def _split_candidates(self, text: str) -> list[str]:
        """Split a legal fragment into candidate evidence units."""

        raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])|(?<=;)\s+|\n+", text)
        candidates: list[str] = []
        for part in raw_parts:
            compact = " ".join(part.split()).strip(" -")
            if not compact:
                continue
            candidates.append(compact)
        return candidates

    def _extract_currency_fact(self, sentences: list[str]) -> str | None:
        """Extract the clearest monetary amount from evidence."""

        for sentence in sentences:
            currency_match = CURRENCY_PATTERN.search(sentence)
            if currency_match:
                return self._normalize_currency(currency_match.group(0))
        return None

    def _extract_measure_fact(self, sentences: list[str]) -> str | None:
        """Extract the clearest non-monetary measure or duration from evidence."""

        for sentence in sentences:
            measure_match = MEASURE_PATTERN.search(sentence)
            if measure_match:
                value = measure_match.group(1).replace(",", ".")
                unit = measure_match.group(2)
                return f"{value} {unit}"
        return None

    def _extract_period_fact(self, sentences: list[str]) -> str:
        """Extract a simple period qualifier such as daily or annual."""

        joined = normalize_for_search(" ".join(sentences))
        if "diario" in joined:
            return "al dia"
        if "mensual" in joined or "mes" in joined:
            return "al mes"
        if "anual" in joined or "ano" in joined:
            return "al ano"
        return ""

    def _normalize_currency(self, raw_amount: str) -> str:
        """Normalize currencies to a readable local format."""

        normalized = raw_amount.upper().replace(" ", "")
        normalized = normalized.replace("S/.", "S/")
        match = re.search(r"S/(\d+(?:\.\d+)?)", normalized)
        if not match:
            return "S/ 1.00"
        value = float(match.group(1))
        return f"S/ {value:.2f}"

    def _resolve_subject_label(self, frame: QuestionFrame) -> str:
        """Return a cleaner human-readable label for the main topic."""

        normalized = frame.normalized_question
        if "sisa" in normalized:
            return "el SISA"
        if "zona rigida" in normalized:
            return "la zona rigida"
        if "autoriz" in normalized:
            return "la autorizacion municipal"
        if "modulo" in normalized:
            return "el modulo"
        if "feria" in normalized:
            return "la feria"
        if frame.subject:
            return frame.subject
        return "el tema consultado"

    def _humanize_sentence(self, sentence: str) -> str:
        """Rewrite the strongest legal sentence into plainer citizen language."""

        compact = self._trim_text(self._strip_legal_headers(sentence), limit=260)
        normalized = normalize_for_search(compact)
        currency_fact = self._extract_currency_fact([compact])

        if "pago por concepto de sisa" in normalized:
            amount = currency_fact or "S/ 1.00"
            if "diario" in normalized:
                return f"el comerciante debe pagar SISA y el monto que aparece es {amount} al dia"
            return f"el comerciante debe pagar SISA y el monto que aparece es {amount}"

        if "se requiere" in normalized and "autoriz" in normalized and "via publica" in normalized:
            return "para vender en la via publica se necesita autorizacion municipal previa"

        if "autorizacion municipal es personal e intransferible" in normalized:
            return "la autorizacion es personal y no se puede transferir ni prestar a otra persona"

        if "vigencia anual" in normalized and "renov" in normalized:
            return "la autorizacion dura un ano y puede renovarse si la municipalidad la evalua favorablemente"

        if "zonas rigidas" in normalized and "no se autoriza" in normalized:
            return "una zona rigida es un espacio de la via publica donde no esta permitido vender"

        if "mobiliario desmontable" in normalized and "actividad comercial" in normalized:
            if "especificaciones tecnicas" in normalized:
                return (
                    "el modulo es una estructura desmontable para vender y debe respetar "
                    "las especificaciones tecnicas aprobadas por la municipalidad"
                )
            return "el modulo es una estructura desmontable usada para desarrollar la actividad comercial"

        if "no se otorgara o se anulara" in normalized and "autorizacion" in normalized:
            return "la municipalidad puede negar o dejar sin efecto la autorizacion si se incumplen las condiciones previstas"

        simplified = compact
        replacements = (
            (r"\bse encuentra obligado al pago por concepto de\b", "debe pagar"),
            (r"\bse encuentra obligado a\b", "debe"),
            (r"\bse requiere de\b", "se necesita"),
            (r"\bse requiere\b", "se necesita"),
            (r"\bno se autoriza el ejercicio del comercio en la via publica\b", "no se permite vender en la via publica"),
            (r"\bsera registrado en el padron correspondiente\b", "debe quedar registrada en el padron correspondiente"),
            (r"\bdebera presentar\b", "debe presentar"),
            (r"\badjuntar\b", "presentar"),
            (r"\bcuyo monto sera de\b", "y el monto es de"),
            (r"\bcuyo monto es de\b", "y el monto es de"),
            (r"\bun nuevo sol\b", "S/ 1.00"),
        )
        for pattern, replacement in replacements:
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)

        simplified = simplified.replace("(S/. 1.00)", "S/ 1.00")
        simplified = simplified.replace("(S/.1.00)", "S/ 1.00")
        simplified = simplified.replace("(S/ 1.00)", "S/ 1.00")
        simplified = re.sub(r"\s+", " ", simplified).strip(" .")
        return simplified

    def _sentence_case(self, text: str) -> str:
        """Uppercase the first letter of a sentence-sized fragment."""

        compact = text.strip()
        if not compact:
            return compact
        return compact[:1].upper() + compact[1:]

    def _strip_legal_headers(self, text: str) -> str:
        """Remove title/article heading noise before local summarization."""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_lines: list[str] = []

        for line in lines:
            normalized_line = normalize_for_search(line)
            if normalized_line.startswith("titulo "):
                continue
            if normalized_line.startswith("capitulo "):
                continue
            if normalized_line.startswith("base legal"):
                continue
            if normalized_line.startswith("de la ") and len(normalized_line.split()) <= 8:
                continue
            cleaned_lines.append(line)

        cleaned_text = " ".join(cleaned_lines) if cleaned_lines else text
        cleaned_text = re.sub(
            r"art[ií]culo\s+[0-9]+[a-z]?\s*[°º]?\s*[.\-:]*\s*",
            "",
            cleaned_text,
            flags=re.IGNORECASE,
        )
        cleaned_text = re.sub(
            r"^(?:\d+(?:\.\d+)?\s+)?[A-Za-zÁÉÍÓÚÑáéíóúñ\s]{3,40}\.-\s*",
            "",
            cleaned_text,
            flags=re.IGNORECASE,
        )
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text

    def _ensure_sentence_starts_lower(self, text: str) -> str:
        """Normalize leading casing so the answer reads naturally."""

        compact = self._trim_text(text)
        if compact.endswith("."):
            compact = compact[:-1]
        if not compact:
            return compact
        return compact[:1].lower() + compact[1:]

    def _trim_text(self, text: str, limit: int = 220) -> str:
        """Keep snippets compact."""

        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _should_disable_external_client(self, exc: Exception) -> bool:
        """Disable the provider after persistent permission or model errors."""

        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "403",
                "permission",
                "credits or licenses",
                "model not found",
            )
        )
