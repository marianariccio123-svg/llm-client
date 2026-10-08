"""
tests/stub_llm.py

Un modelo de chat simulado, con guion.

Por qué existe
--------------
Las trazas de ejecución demuestran que el orquestador funciona, pero cuestan
llamadas a la API, tardan, y dependen de que el modelo decida hoy lo mismo que
decidió ayer. Eso sirve como evidencia y no sirve como test: un test que puede
fallar porque el modelo tuvo otra idea no dice nada sobre el código.

`LLMGuionado` reemplaza al modelo por una cola de respuestas prefabricadas. Con
eso se puede ejercitar el grafo entero —el ruteo del supervisor, los ciclos
ReAct de los especialistas, la validación, la síntesis— de forma determinista,
en milisegundos y sin gastar un solo token. Lo que queda bajo test es lo que
escribimos nosotros: la topología, los reducers, el paso de contexto y las
herramientas.

Qué tiene que soportar
----------------------
Para poder enchufarse donde va el modelo real tiene que responder a tres cosas:

- `bind_tools()`, porque `create_react_agent` la llama.
- `with_structured_output()`, porque el supervisor la usa para que su decisión
  venga tipada.
- `_generate()`, que es el método que BaseChatModel usa por debajo (y del que
  deriva gratis la variante asíncrona).
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict, Field


class GuionAgotado(AssertionError):
    """
    Se pidió una respuesta más de las que tiene el guion.

    Es `AssertionError` y no `RuntimeError` a propósito: si el grafo llamó al
    modelo más veces de las previstas, el test tiene que fallar como un test
    (con un mensaje que diga cuántas llamadas hubo), no reventar como un bug de
    infraestructura.
    """


class LLMGuionado(BaseChatModel):
    """
    Modelo de chat que devuelve respuestas de una cola, en orden.

    Args:
        respuestas: los `AIMessage` que va a devolver, uno por llamada. Pueden
            llevar `tool_calls` para simular que el modelo pidió herramientas.
        estructuradas: los objetos que devuelve la rama de
            `with_structured_output`, en orden. El supervisor consume de acá.

    Atributos de inspección, para que los tests puedan afirmar sobre el
    contexto que recibió cada agente y no solo sobre el resultado:
        prompts_recibidos: la lista de mensajes de cada llamada.
        prompts_estructurados: ídem, para la rama estructurada.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    respuestas: list[BaseMessage] = Field(default_factory=list)
    estructuradas: list[Any] = Field(default_factory=list)
    prompts_recibidos: list[list[BaseMessage]] = Field(default_factory=list)
    prompts_estructurados: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "llm-guionado"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts_recibidos.append(list(messages))

        if not self.respuestas:
            raise GuionAgotado(
                f"El grafo pidió una respuesta número {len(self.prompts_recibidos)} "
                f"pero el guion se agotó. Agregá una respuesta más, o revisá por "
                f"qué el ciclo dio una vuelta extra."
            )

        return ChatResult(generations=[ChatGeneration(message=self.respuestas.pop(0))])

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> BaseChatModel:
        """
        Acepta las herramientas y se devuelve a sí mismo.

        El stub no necesita el esquema de las herramientas: las `tool_calls`
        vienen escritas en el guion. Devolver `self` (y no un wrapper) mantiene
        una sola cola de respuestas, así el orden del guion es el orden real de
        las llamadas.
        """
        return self

    def with_structured_output(
        self, schema: type[BaseModel] | Any, **kwargs: Any
    ) -> Runnable:
        """
        Devuelve un runnable que entrega los objetos de `estructuradas`.

        Valida contra el esquema pedido antes de entregar, para que un guion
        mal escrito falle en el test en vez de propagar un objeto incompatible
        hasta el nodo que lo consume.
        """

        def _responder(entrada: Any) -> Any:
            mensajes = entrada if isinstance(entrada, list) else [entrada]
            self.prompts_estructurados.append(list(mensajes))

            if not self.estructuradas:
                raise GuionAgotado(
                    f"El supervisor pidió la decisión número "
                    f"{len(self.prompts_estructurados)} pero el guion de salidas "
                    f"estructuradas se agotó."
                )

            siguiente = self.estructuradas.pop(0)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                if not isinstance(siguiente, schema):
                    raise AssertionError(
                        f"El guion entregó un {type(siguiente).__name__} donde el "
                        f"código esperaba un {schema.__name__}."
                    )
            return siguiente

        return RunnableLambda(_responder)


def mensaje_con_herramientas(*llamadas: tuple[str, dict[str, Any]]) -> AIMessage:
    """
    Arma un `AIMessage` que pide herramientas, como lo haría el modelo real.

    Args:
        llamadas: pares `(nombre_herramienta, argumentos)`.

    Uso:
        mensaje_con_herramientas(("listar_incidentes", {"meses": 12}))
    """
    return AIMessage(
        content="",
        tool_calls=[
            {"name": nombre, "args": args, "id": f"call_{i}"}
            for i, (nombre, args) in enumerate(llamadas)
        ],
    )
