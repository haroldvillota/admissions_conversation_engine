# --- Etapa 1: Build (Instalación y Compilación) ---
FROM python:3.12-slim AS builder

# Instalamos uv desde su imagen oficial (más rápido y seguro que pip)
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /bin/

WORKDIR /app

# Copiamos solo los metadatos del proyecto para aprovechar la caché de capas
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

# Temporalmente para exponerlo a langsmith studio
COPY langgraph.json ./

# Instalamos dependencias y el proyecto. 
# --no-dev: Evita instalar librerías de test/linting en producción.
RUN uv sync --frozen --no-dev

# --- Etapa 2: Runtime (Imagen ligera para ejecución) ---
FROM python:3.12-slim

WORKDIR /app

# Seguridad: Creamos un usuario sin privilegios para no correr como root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copiamos los artefactos necesarios de la etapa 'builder'
# .venv: contiene las librerías externas ya instaladas
# src: contiene el código fuente que el venv necesita para funcionar
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/alembic.ini /app/alembic.ini
COPY --from=builder /app/alembic /app/alembic

# Temporalmente para exponerlo a langsmith studio
COPY --from=builder /app/langgraph.json /app/langgraph.json
RUN mkdir -p /app/.langgraph_api && chown -R appuser:appuser /app

# Configuración de Python para contenedores
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cambiamos al usuario de ejecución
USER appuser

# Ejecuta la aplicación. Para correr migraciones manualmente (ej. desde CI/CD):
#   docker run --rm --env-file .env <image> alembic upgrade head
CMD ["start"]
