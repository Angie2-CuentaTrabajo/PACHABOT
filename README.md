# Asistente Virtual de Comercio Ambulatorio

Prototipo local en Python de un asistente conversacional para consultas ciudadanas sobre comercio ambulatorio en una municipalidad peruana. Hoy funciona en Telegram, pero la arquitectura ya queda separada por canal para crecer luego a WhatsApp o web.

## Objetivo

El proyecto responde consultas normativas e informativas usando como base principal:

- Ordenanza 108-2012-MDP/C
- Ordenanza 227-2019-MDP/C

Prioridades del prototipo:

- responder en espanol claro y ciudadano
- limitarse al dominio de comercio ambulatorio
- basarse en la documentacion cargada
- citar ordenanza y articulo cuando sea posible
- permitir preguntas libres y preguntas de seguimiento
- seguir funcionando incluso si el LLM externo no esta disponible

## Arquitectura actual

El proyecto ya no es solo un bot con busqueda simple. Ahora tiene estas capas:

- `channels/`: adapta Telegram al formato interno del asistente
- `memory/`: guarda el historial reciente por chat
- `tools/`: concentra herramientas documentales para reescribir consultas y recuperar evidencia
- `services/`: contiene el orquestador, el router, la recuperacion y la capa LLM
- `api/`: expone un endpoint HTTP para probar el asistente fuera de Telegram

## Decision tecnica del RAG

Para este prototipo se usa un motor local con `scikit-learn`, `TF-IDF` y similitud coseno. Se eligio asi porque:

- es facil de correr en local
- no depende de embeddings externos
- es mas simple para un prototipo universitario
- deja el proyecto listo para migrar despues a FAISS, ChromaDB o embeddings externos

Encima de esa base ya existe una capa mas inteligente:

- segmentacion juridica por titulo y articulo
- busqueda hibrida con prioridad legal
- reescritura de consultas de seguimiento
- memoria por sesion
- respuesta sintetizada por LLM o fallback local

## Estructura

```text
project_root/
|-- app/
|   |-- api/
|   |   |-- __init__.py
|   |   `-- schemas.py
|   |-- bot/
|   |   |-- __init__.py
|   |   |-- handlers.py
|   |   |-- keyboards.py
|   |   `-- telegram_bot.py
|   |-- channels/
|   |   |-- __init__.py
|   |   |-- schemas.py
|   |   `-- telegram.py
|   |-- core/
|   |   |-- __init__.py
|   |   |-- logger.py
|   |   `-- prompts.py
|   |-- memory/
|   |   |-- __init__.py
|   |   `-- conversation_store.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- domain.py
|   |   `-- schemas.py
|   |-- services/
|   |   |-- __init__.py
|   |   |-- assistant_service.py
|   |   |-- document_service.py
|   |   |-- llm_service.py
|   |   |-- query_router.py
|   |   `-- retrieval_service.py
|   |-- tools/
|   |   |-- __init__.py
|   |   `-- document_toolkit.py
|   |-- utils/
|   |   |-- __init__.py
|   |   |-- chunking.py
|   |   |-- docx_extractor.py
|   |   |-- helpers.py
|   |   `-- text_cleaner.py
|   |-- __init__.py
|   |-- config.py
|   `-- main.py
|-- data/
|   |-- processed/
|   |   |-- chunks.json
|   |   `-- conversations/
|   |-- raw/
|   |   |-- ordenanza_108_2012.txt
|   |   `-- ordenanza_227_2019.txt
|   `-- vectorstore/
|-- scripts/
|   |-- import_docx_documents.py
|   |-- ingest_documents.py
|   `-- reset_vectorstore.py
|-- tests/
|   |-- test_assistant_conversation.py
|   |-- test_chunking.py
|   |-- test_retrieval.py
|   `-- test_router.py
|-- .env.example
|-- README.md
|-- requirements.txt
`-- run.py
```

## Instalacion paso a paso

### 1. Crear entorno virtual

En PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activacion:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```powershell
Copy-Item .env.example .env
```

Completa al menos:

- `TELEGRAM_BOT_TOKEN`
- `LLM_PROVIDER`
- `CHAT_MODEL`

Si vas a usar xAI/Grok:

- `GROK_API_KEY`
- `GROK_BASE_URL=https://api.x.ai/v1`

Si todavia no tienes un LLM operativo, puedes dejar:

```env
LLM_MODE=mock
```

## Variables principales

Ejemplo base:

```env
TELEGRAM_BOT_TOKEN=
LLM_PROVIDER=grok
GROK_API_KEY=
GROK_BASE_URL=https://api.x.ai/v1
OPENAI_API_KEY=
OPENAI_BASE_URL=
APP_ENV=development
LOG_LEVEL=INFO
EMBEDDING_MODEL=local-tfidf
CHAT_MODEL=grok-4
LLM_MODE=mock
ALLOW_GENERAL_CHAT=false
RETRIEVAL_TOP_K=4
CHUNK_SIZE=700
CHUNK_OVERLAP=120
CONFIDENCE_THRESHOLD=0.12
MEMORY_HISTORY_LIMIT=12
MEMORY_MAX_TURNS=40
ASSISTANT_MAX_SOURCES=3
```

## Cargar documentos desde Word

Si tienes las ordenanzas en `.docx`, puedes importarlas a `data/raw/` asi:

```powershell
python scripts/import_docx_documents.py
```

Ese script extrae el texto y lo deja listo para la siguiente etapa.

## Construir el indice documental

Despues de tener los `.txt` en `data/raw/`:

```powershell
python scripts/ingest_documents.py
```

Este proceso:

- limpia texto
- segmenta por secciones y articulos
- genera `chunks.json`
- construye el indice local en `data/vectorstore/`

Si quieres reiniciar el indice:

```powershell
python scripts/reset_vectorstore.py
python scripts/ingest_documents.py
```

## Ejecutar el bot de Telegram

```powershell
python run.py
```

Comandos disponibles en Telegram:

- `/start`
- `/help`
- `/reset`
- `/estado`

## Probar por HTTP

Tambien puedes levantar FastAPI:

```powershell
uvicorn app.main:app --reload
```

Endpoints disponibles:

- `GET /health`
- `GET /info`
- `POST /chat`
- `POST /chat/reset`

Ejemplo de prueba:

```json
POST /chat
{
  "channel": "api",
  "session_id": "demo-1",
  "user_id": "usuario-demo",
  "text": "Que dice el articulo 7"
}
```

## Ejemplos de preguntas

- `Que requisitos necesito para una autorizacion`
- `Cuanto mide un modulo`
- `Cuanto se paga de SISA`
- `Que zonas son rigidas`
- `Que dice el articulo 7`
- `Explicame la autorizacion municipal`
- `Y cuanto dura?`

## Comportamiento esperado

- si la consulta esta dentro del dominio, el sistema busca evidencia y responde con lenguaje natural
- si la consulta es de seguimiento, intenta aprovechar el contexto del chat
- si la evidencia es debil, responde con honestidad y evita inventar
- si la consulta esta fuera del dominio, limita la conversacion a comercio ambulatorio
- si el LLM externo falla, el bot sigue respondiendo con un fallback local

## Pruebas

```powershell
pytest tests/test_chunking.py tests/test_retrieval.py tests/test_router.py tests/test_assistant_conversation.py
```

## Limitaciones actuales

- la calidad sigue dependiendo de la calidad del texto cargado
- el RAG es local y sencillo, aunque ya tiene memoria y reescritura conversacional
- no hay OCR
- no hay panel admin
- WhatsApp aun no esta conectado, aunque la arquitectura ya esta separada por canal
- si xAI devuelve `403` por falta de creditos, el proyecto trabajara en fallback local

## Proximos pasos sugeridos

- agregar parser de PDF con limpieza juridica mas fuerte
- incorporar herramientas externas o busqueda web controlada
- sumar un canal WhatsApp reutilizando `channels/`
- reemplazar TF-IDF por embeddings semanticos cuando crezca la base documental
- agregar observabilidad por chat y metricas de recuperacion
