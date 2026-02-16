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

El graph trabaja sobre un **estado explícito y tipado**, que al inicio incluye:

* Historial de mensajes
* Nombre del usuario

Ejemplo conceptual:

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

## 🐳 Docker

Se incluyen dos Dockerfiles:

- `Dockerfile`: imagen de producción (instala dependencias sin `dev`).
- `Dockerfile.dev`: imagen de desarrollo (incluye dependencias `dev`).

### Build

```bash
docker build -t admissions-conversation-engine:prod .
docker build -t admissions-conversation-engine:dev -f Dockerfile.dev .
```

### Run

```bash
# usando .env local
docker run --rm -it -p 2024:2024 --env-file .env admissions-conversation-engine:dev

```

---

## ⚙️ Variables de entorno

El sistema usa variables con `__` para configuración anidada. Puedes usar `env-example` como plantilla.

### Requeridas

- `LLM__DEFAULT__API_KEY`
- `LLM__DEFAULT__MODEL`
- `LLM__TRANSLATOR__MODEL`
- `CHECKPOINTER__DSN`
- `RAG__VECTOR_STORE__DSN`
- `TENANT__INSTITUTION`
- `TENANT__TERMS_OF_SERVICE`
- `TENANT__ALLOWED_TOPICS`
- `TENANT__TONE`
- `TENANT__LANGUAGE_FALLBACK`
- `TENANT__ALLOWED_LANGUAGES`

### Opcionales comunes

- `LLM__DEFAULT__TEMPERATURE`
- `LLM__GUARDRAIL__MODEL`
- `LLM__REACT__MODEL`
- `LLM__TRANSLATOR__TEMPERATURE`
- `OBSERVABILITY__PUBLIC_KEY`
- `OBSERVABILITY__SECRET_KEY`
- `OBSERVABILITY__BASE_URL`

---
