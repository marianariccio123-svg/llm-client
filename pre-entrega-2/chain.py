"""
chain.py

Cadena LCEL que extrae entidades técnicas de un texto usando LangChain:
Prompt -> Modelo -> Output Parser (structured output), con reintentos.
"""

from dotenv import load_dotenv

load_dotenv()

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import TechExtraction


SYSTEM_INSTRUCTIONS = """\
Sos un asistente técnico especializado en analizar arquitecturas de software \
y logs de error. A partir del texto que te pasa el usuario, extraé:

- Las tecnologías mencionadas (frameworks, bases de datos, lenguajes, servicios).
- El nivel de criticidad de la situación descrita (baja, media o alta).
- Un resumen técnico breve y preciso de la situación.

Si el texto es ambiguo o menciona pocas tecnologías, hacé tu mejor esfuerzo \
igual: nunca dejes la lista de tecnologías vacía ni el resumen incompleto.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        ("human", "{texto}"),
    ]
)


def get_chat_model():
    """
    Instancia el modelo de chat según la variable de entorno PROVIDER
    (openai, anthropic o gemini). Reutiliza las mismas API keys del
    Módulo 1 (.env). Incluye timeout para evitar quedar colgado
    ante una API que no responde.
    """
    provider = os.getenv("PROVIDER", "gemini").lower()

    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            timeout=30,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model="claude-3-5-haiku-latest",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
            timeout=30,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
            timeout=30,
        )
    else:
        raise ValueError(f"Proveedor '{provider}' no soportado.")


# --- Ensamblado de la cadena LCEL con reintentos ---
model = get_chat_model()

chain = (prompt | model.with_structured_output(TechExtraction)).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def process_text(text: str) -> TechExtraction | None:
    """
    Ejecuta la cadena LCEL de forma asíncrona sobre un texto de entrada.
    Devuelve el objeto TechExtraction ya validado, o None si falla
    incluso después de los reintentos automáticos.
    """
    logger.info("Procesando texto (%d caracteres)...", len(text))
    try:
        result = await chain.ainvoke({"texto": text})
        logger.info("Extracción validada OK: %s", result.model_dump())
        return result
    except Exception as e:
        logger.error("Falló la extracción tras los reintentos: %s", e)
        return None