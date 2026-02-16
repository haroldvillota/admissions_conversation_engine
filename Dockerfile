# --- Etapa 1: Build (Instalación y Compilación) ---
FROM python:3.11-slim AS builder

# Instalamos uv desde su imagen oficial (más rápido y seguro que pip)
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /bin/

WORKDIR /app

# Copiamos solo los metadatos del proyecto para aprovechar la caché de capas
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Instalamos dependencias y el proyecto. 
# --no-dev: Evita instalar librerías de test/linting en producción.
RUN uv sync --frozen --no-dev

# --- Etapa 2: Runtime (Imagen ligera para ejecución) ---
FROM python:3.11-slim

WORKDIR /app

# Seguridad: Creamos un usuario sin privilegios para no correr como root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copiamos los artefactos necesarios de la etapa 'builder'
# .venv: contiene las librerías externas ya instaladas
# src: contiene el código fuente que el venv necesita para funcionar
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Configuración de Python para contenedores
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cambiamos al usuario de ejecución
USER appuser

# Ejecutamos el script definido en [project.scripts] en el pyproject.toml
CMD ["start"]