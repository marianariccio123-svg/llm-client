# Unified Async LLM Client

Cliente asíncrono unificado para modelos de lenguaje (LLM), con una interfaz
común para OpenAI, Anthropic y Google Gemini. Proyecto de pre-entrega para
el curso de AI Engineering.

## Entregas del curso

Este repositorio acumula las pre-entregas. Cada una vive en su propia carpeta,
con su README, sus dependencias y sus instrucciones de ejecución:

| Entrega | Carpeta | Tema |
|---|---|---|
| 1 | raíz del repo | Cliente async unificado para OpenAI, Anthropic y Gemini |
| 2 | [`pre-entrega-2/`](pre-entrega-2/) | Pipeline de extracción estructurada con LangChain |
| 3 | [`pre-entrega-3/`](pre-entrega-3/) | RAG local con LangChain + ChromaDB |
| 4 | [`pre-entrega-4/`](pre-entrega-4/) | RAG escalable en Pinecone + recuperador híbrido + evaluación |

Lo que sigue documenta la **Pre-entrega 1**.

## ¿Qué hace?

- Permite instanciar un proveedor (OpenAI, Anthropic o Gemini) bajo la misma interfaz.
- Todas las llamadas son asíncronas (`async/await`), no bloquean el programa.
- Soporta modo streaming: recibe la respuesta token por token a medida que se genera.
- Valida mensajes y configuración con Pydantic antes de llamar a cualquier API.
- Maneja errores de red y de límite de tasa sin romper el programa: reintenta
  automáticamente los errores temporales (con backoff exponencial) y distingue
  un límite de cuota diario (no reintentable) de una saturación momentánea del
  servidor (sí reintentable).

## Estructura del proyecto
llm-client/
├── schemas.py # Modelos de datos con Pydantic (ChatMessage, ModelConfig, ModelResponse)
├── base_client.py # Clase abstracta BaseLLMClient (el "contrato" común)
├── openai_client.py # Implementación para OpenAI (AsyncOpenAI)
├── anthropic_client.py # Implementación para Anthropic (AsyncAnthropic)
├── gemini_client.py # Implementación para Gemini (google-genai), probada en vivo
├── manager.py # AsyncLLMManager: elige el proveedor por configuración
├── main.py # Script de prueba (modo normal + streaming)
├── requirements.txt
├── .env.example
└── .gitignore


## Requisitos

- Python 3.12+ (probado con 3.14, totalmente compatible: el proyecto no usa
  ninguna sintaxis exclusiva de 3.12)
- Una API key de al menos un proveedor. Se recomienda **Gemini**, porque
  Google AI Studio entrega una API key gratuita: https://aistudio.google.com/apikey

## Instalación

```bash
# Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows (PowerShell)
# source venv/bin/activate       # Mac/Linux

# Instalar dependencias
pip install -r requirements.txt
```

## Variables de entorno

GEMINI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=


El `.env` nunca se sube al repositorio (está en `.gitignore`).

## Cómo correr la prueba

```bash
python main.py
```

Esto va a:
1. Instanciar `AsyncLLMManager` con el proveedor `"gemini"`.
2. Preguntar "¿Qué es la entropía?" en **modo normal** (respuesta completa de una vez).
3. Repetir la misma pregunta en **modo streaming** (imprimiendo fragmento por fragmento a medida que llega).

Para probar con otro proveedor, cambiá el parámetro `provider` en `main.py`
(`"openai"`, `"anthropic"` o `"gemini"`) y completá la API key correspondiente en `.env`.

## Limitaciones conocidas

- El tier gratuito de Gemini limita a 20 peticiones por día por modelo. Si
  aparece un error de "cuota diaria agotada", es esperable y está documentado
  en el manejo de errores de `gemini_client.py` (no reintenta, falla rápido
  con un mensaje claro).
- `OpenAIClient` y `AnthropicClient` están implementados y cumplen la interfaz
  común (`BaseLLMClient`), pero no fueron probados en vivo en este entorno por
  no contar con API keys de pago; la validación funcional completa se hizo
  con `GeminiClient`.