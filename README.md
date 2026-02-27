# Agente Conversacional con LangGraph (Python)

Este repositorio implementa un **agente conversacional** usando el ecosistema **Langchain 1.x LangGraph 1.x**, organizado bajo un **marco de arquitectura hexagonal simplificado**.

---

## 🧱 Arquitectura (Hexagonal Simplificada)

El código se organiza en **tres capas**, con dependencias siempre hacia el interior:

```
infraestructura  →  aplicación  →  dominio
```

**Nota de diseño**

Aunque se usa una arquitectura hexagonal simplificada, en este proyecto se permite de forma intencional el uso directo de componentes de **LangChain/LangGraph** en las capas de **dominio** y **aplicación**.  
Esta decisión prioriza simplicidad y velocidad de desarrollo, asumiendo explícitamente el acoplamiento con el framework como parte del diseño.

## 🧠 Estado Conversacional

El graph trabaja sobre un **estado explícito y tipado** (`AgentState`) y un **contexto de runtime** (`ContextSchema`).

### `AgentState` (estado del graph)

* `messages`: historial conversacional (acumulado con `add_messages`)
* `language`: idioma detectado (`str | None`)
* `language_confidence`: confianza de detección (`float | None`)
* `guardrail_allowed`: bandera de seguridad (`bool`)
* `guardrail_reason`: motivo (`OK | PROHIBITED_TOPIC | OUT_OF_SCOPE | INJECTION`)
* `case_node`: nodo de enrutamiento según el caso de uso(`str | None`)

### `ContextSchema` (runtime.context)

* `chat_id`: identificador de conversación
* `channel_id`: canal de origen
* `case`: caso de negocio (`off_hours | low_scoring | overflow | max_retries`)
* `user_name`: nombre del usuario

* El estado es **inmutable por contrato** (se retorna uno nuevo o actualizado).
* Cada nodo **lee estado → decide → devuelve cambios**.

Esto permite:

* Razonamiento claro del LLM
* Trazabilidad
* Testeo aislado

---

## 🔁 LangGraph (v1.x)

* El flujo conversacional se modela como un **graph dirigido**.
* Cada **nodo** representa una acción clara (ej. interpretar input, responder, enrutar).
* Las **transiciones** dependen del estado, no de efectos colaterales.

👉 Principio clave: *El graph describe el flujo, no la lógica de negocio*.

---

## 📝 Estilo de Código (Reglas No Negociables)

Este proyecto prioriza **legibilidad y mantenimiento a largo plazo**.

### Código Autodocumentado

* **Nombres largos y descriptivos** > comentarios explicativos.
* La intención debe ser evidente leyendo la firma de una función.

```python
def determine_next_conversation_step(state: ConversationState) -> ConversationStep:
    ...
```

---

### Encapsulación de Intención

* Si un bloque necesita explicación → **extraer función**.
* Las funciones cuentan la historia del sistema.

---

### Tipado Estático

* Todas las funciones públicas usan **type hints**.
* El tipado define contratos y expectativas para humanos y LLMs.

---


## 🧩 Paradigma: Programación Orientada a Objetos (POO)

El sistema adopta **Programación Orientada a Objetos** como modelo principal para estructurar responsabilidades y expresar intención.

### Principios Aplicados

* **Clases como unidades de intención**: cada clase representa un concepto del dominio, un caso de uso o un adaptador técnico claro.
* **Métodos describen comportamientos**, no procedimientos genéricos.
* **Estado encapsulado**: los objetos protegen sus invariantes internas.

---

### Uso de POO por Capa

**Dominio**

* Entidades modelados como clases.
* El estado conversacional se expresa como una clase tipada e inmutable por contrato.
* Sin herencia innecesaria; se prioriza composición.

**Aplicación**

* Casos de uso y nodos del graph se representan como clases o servicios.
* Cada clase orquesta un único propósito claro.
* Dependencias inyectadas vía constructor.

**Infraestructura**

* Adaptadores implementados como clases concretas.
* Implementan interfaces definidas en capas internas.

---

## 🐳 Contenedores

Este proyecto separa claramente los flujos de **producción** y **desarrollo local**:

- `Dockerfile`: imagen de **producción** (`uv sync --no-dev`).
- `Dockerfile.dev`: imagen de **desarrollo local** (incluye dependencias `dev`).
- `docker-compose-dev.yml`: orquestación local de `app` + `db` (PostgreSQL con pgvector).

### Producción (`Dockerfile`)

- Construye una imagen multi-stage optimizada para runtime.
- Inicia con `CMD ["start"]`.
- Incluye `alembic.ini` y `alembic/` para permitir ejecutar migraciones cuando se necesite.
- **No** corre migraciones automáticamente al iniciar.

Build:

```bash
docker build -t admissions-conversation-engine:prod .
```

Run:

```bash
docker run --rm -p 2024:2024 --env-file .env admissions-conversation-engine:prod
```

### Desarrollo local (`Dockerfile.dev` + `docker-compose-dev.yml`)

- El contenedor `app` se construye con `Dockerfile.dev`.
- Ejecuta migraciones al arrancar (`alembic upgrade head`) y luego inicia el script CLI del proyecto (`uv run cli`), definido en `pyproject.toml` (`[project.scripts] cli = ...`).
- Monta `./src:/app/src` para reflejar cambios de código en local.
- Levanta `db` con `pgvector/pgvector:pg17`.

**1. Crear el archivo `.env`**

```bash
cp env-example .env
# Editar .env y completar las API keys (LLM__DEFAULT__API_KEY, RAG__VECTOR_STORE__DSN, CHECKPOINTER__DSN, etc.)
```

**2. Levantar servicios de desarrollo**

```bash
docker compose -f docker-compose-dev.yml build
docker compose -f docker-compose-dev.yml run --rm app-cli
```

**3. Detener y limpiar**

```bash
docker compose -f docker-compose-dev.yml down
docker compose -f docker-compose-dev.yml down -v
```

### Build manual (desarrollo)

```bash
docker build -t admissions-conversation-engine:dev -f Dockerfile.dev .
```

### Run manual (desarrollo)

```bash
docker run --rm -it -p 2024:2024 --env-file .env admissions-conversation-engine:dev cli
```

## 🗄️ Migraciones con Alembic

El proyecto incluye Alembic y la migración inicial:

- `alembic/versions/e88610257584_create_vectorstore_view.py`

### Ejecutar migraciones en local

```bash
make migrate
```

### Migraciones en Docker (desarrollo)

`Dockerfile.dev` ejecuta `alembic upgrade head` automáticamente al iniciar el contenedor.

### Migraciones en producción (CI/CD)

La imagen de producción no ejecuta migraciones automáticamente. Se deben correr como paso manual en el pipeline de CI/CD:

```bash
docker run --rm --env-file .env admissions-conversation-engine:prod alembic upgrade head
```

---

## ⚙️ Variables de entorno

El sistema usa variables con `__` para configuración anidada. Puedes usar `env-example` como plantilla.

### Requeridas (según esquema actual)

- `RAG__VECTOR_STORE__KIND`
- `RAG__VECTOR_STORE__DSN`
- `RAG__VECTOR_STORE__COLLECTION`
- `RAG__EMBEDDINGS__MODEL`
- `LLM__DEFAULT__MODEL`
- `LLM__DEFAULT__API_KEY`
- `LLM__GUARDRAIL__MODEL`
- `LLM__REACT__MODEL`
- `LLM__TRANSLATOR__MODEL`
- `CHECKPOINTER__KIND`
- `CHECKPOINTER__DSN`
- `TENANT__INSTITUTION`
- `TENANT__TERMS_OF_SERVICE`
- `TENANT__ALLOWED_TOPICS`
- `TENANT__TONE`
- `TENANT__LANGUAGE_FALLBACK`
- `TENANT__ALLOWED_LANGUAGES`

### Opcionales comunes

- `RAG__VECTOR_STORE__TOP_K`
- `RAG__EMBEDDINGS__PROVIDER`
- `RAG__EMBEDDINGS__API_KEY`
- `RAG__EMBEDDINGS__BATCH_SIZE`
- `LLM__DEFAULT__PROVIDER`
- `LLM__DEFAULT__TEMPERATURE`
- `LLM__GUARDRAIL__TEMPERATURE`
- `LLM__REACT__TEMPERATURE`
- `LLM__TRANSLATOR__TEMPERATURE`
- `OBSERVABILITY__PROVIDER`
- `OBSERVABILITY__PUBLIC_KEY`
- `OBSERVABILITY__SECRET_KEY`
- `OBSERVABILITY__BASE_URL`

### Nota sobre fallback de LLM

`config_bootstrap` completa `guardrail`, `react` y `translator` con valores de `llm.default` solo si esos campos vienen vacíos (`None` o `""`) una vez cargada la configuración.

Para embeddings, `rag.embeddings.api_key` es opcional. Si no se define, el `AgentBuilder` usa `llm.default.api_key` como fallback.

---
