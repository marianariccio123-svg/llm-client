# Dockerfile
#
# El contexto de build es la RAÍZ del repo, no esta carpeta. Tiene que serlo:
# la API reutiliza el orquestador de `pre-entrega-6/` y el corpus de
# documentación de `pre-entrega-4/`, así que los tres directorios tienen que
# entrar en la imagen.
#
# Se construye desde docker-compose.yml, que ya pasa el contexto correcto:
#   docker compose up --build

FROM python:3.12-slim

# No escribir .pyc y no bufferear stdout: en un contenedor los logs tienen que
# salir cuando ocurren, no cuando se llena el buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Las dependencias primero y en su propia capa: mientras requirements.txt no
# cambie, Docker reusa la capa y un rebuild por un cambio de código no vuelve
# a bajar todo.
COPY pre-entrega-7/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# El código de las tres entregas que participan.
COPY pre-entrega-4/data ./pre-entrega-4/data
COPY pre-entrega-6 ./pre-entrega-6
COPY pre-entrega-7 ./pre-entrega-7

WORKDIR /app/pre-entrega-7

EXPOSE 8000

# Un solo worker de uvicorn a propósito. El paralelismo de esta app lo da su
# propio pool de corrutinas (CANTIDAD_WORKERS), y cada proceso de uvicorn
# levantaría un pool entero más: cuatro procesos serían cuatro veces la
# concurrencia configurada contra el proveedor del modelo, que es justo lo que
# el limitador de tasa trata de evitar.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
