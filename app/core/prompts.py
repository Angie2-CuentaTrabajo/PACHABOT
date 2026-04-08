from __future__ import annotations

from typing import Iterable

from app.models.schemas import ConversationTurn, RetrievedChunk


SYSTEM_PROMPT = """Eres un asistente municipal especializado en comercio ambulatorio.

Tu trabajo es responder como una persona que realmente entendio las ordenanzas, no como un motor que pega parrafos.

Reglas obligatorias:
1. Responde solo en espanol.
2. Explica con tus propias palabras y con tono claro, cercano y profesional.
3. No uses frases como "segun la evidencia recuperada" o "la norma senala que" salvo que sean estrictamente necesarias.
4. No copies bloques largos del documento. Si citas algo, que sea breve y util.
5. Si el historial muestra una repregunta, interpreta el contexto antes de responder.
6. Si el usuario plantea un caso practico, analiza el caso y responde con criterio: que parece permitido, que parece prohibido y que tendria que confirmarse.
7. Si la informacion documental no alcanza, dilo con honestidad y sugiere verificarlo con la municipalidad.
8. Si tienes fuentes claras, puedes mencionarlas de forma breve al final, pero no conviertas la respuesta en una lista tecnica.
"""


GENERAL_CHAT_SYSTEM_PROMPT = """Eres un asistente general en espanol.

Responde con naturalidad, claridad y tono humano.
No suenes robotico.
Si no sabes algo con certeza, dilo con honestidad.
"""


QUERY_REWRITE_SYSTEM_PROMPT = """Reescribe preguntas de ciudadanos para mejorar la busqueda en ordenanzas municipales sobre comercio ambulatorio.

Reglas:
1. Devuelve una sola pregunta reformulada.
2. Resuelve pronombres o referencias vagas usando el historial.
3. Si la pregunta ya es clara, devuelvela casi igual.
4. No expliques nada. No uses comillas. No agregues etiquetas.
"""


def build_context_block(chunks: Iterable[RetrievedChunk]) -> str:
    """Render a compact reference block for the answer model."""

    parts: list[str] = []
    for chunk in chunks:
        header = chunk.source_title
        if chunk.article_label:
            header += f", Art. {chunk.article_label}"
        elif chunk.section_title:
            header += f", {chunk.section_title}"
        parts.append(f"[{header}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _history_messages(history: list[ConversationTurn], *, limit: int = 8) -> list[dict[str, str]]:
    """Convert recent history into provider-compatible chat messages."""

    messages: list[dict[str, str]] = []
    for turn in history[-limit:]:
        role = "assistant" if turn.role == "assistant" else "user"
        messages.append({"role": role, "content": turn.text})
    return messages


def build_answer_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[ConversationTurn],
) -> list[dict[str, str]]:
    """Build messages for municipal answer generation."""

    messages = _history_messages(history)

    if not chunks:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Pregunta actual: {question}\n\n"
                    "No se recuperaron fragmentos documentales suficientemente claros. "
                    "Si puedes orientar sin inventar, hazlo. Si no, di con honestidad que "
                    "la base disponible no basta para responder con seguridad."
                ),
            }
        )
        return messages

    context_block = build_context_block(chunks[:4])
    messages.append(
        {
            "role": "user",
            "content": (
                "Documentos de referencia:\n\n"
                f"{context_block}\n\n"
                "---\n\n"
                f"Pregunta actual: {question}\n\n"
                "Responde de forma natural, con sintesis propia y sin pegar el documento."
            ),
        }
    )
    return messages


def build_general_chat_messages(
    question: str,
    history: list[ConversationTurn],
) -> list[dict[str, str]]:
    """Build messages for unrestricted general chat."""

    messages = _history_messages(history)
    messages.append({"role": "user", "content": question})
    return messages


def build_query_rewrite_messages(
    question: str,
    history: list[ConversationTurn],
) -> list[dict[str, str]]:
    """Build messages for query rewriting."""

    messages = _history_messages(history, limit=6)
    messages.append(
        {
            "role": "user",
            "content": (
                "Reformula esta pregunta para buscar mejor en las ordenanzas, "
                "quitando ambiguedades y usando el contexto si hace falta:\n"
                f"{question}"
            ),
        }
    )
    return messages
