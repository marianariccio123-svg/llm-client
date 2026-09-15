# Pipeline de Extracción de Entidades Técnicas

Pipeline que recibe un texto técnico sin procesar (descripción de arquitectura,
log de error, etc.) y devuelve un objeto validado con las tecnologías
mencionadas, el nivel de criticidad y un resumen técnico. Construido con
LangChain (LCEL) sobre la lógica de clientes async del Módulo 1.

## ¿Qué hace?

- Define un esquema Pydantic (`TechExtraction`) que valida la estructura de salida.
- Arma un `ChatPromptTemplate` modular (sin f-strings hardcodeadas).
- Ensambla una cadena LCEL: `prompt | model.with_structured_output(TechExtraction)`.
- Envuelve la cadena completa con `.with_retry()`: reintenta automáticamente
  ante errores temporales del servidor (503, timeouts) o ante un JSON
  que no valide contra el esquema.
- Expone `process_text()`, una función asíncrona (`ainvoke`) con logs que
  muestran el proceso de validación y los reintentos en tiempo real.

## Estructura