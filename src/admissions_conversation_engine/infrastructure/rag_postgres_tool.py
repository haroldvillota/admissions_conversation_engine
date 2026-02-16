from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from admissions_conversation_engine.infrastructure.config.app_config import RagConfig
from langchain_core.tools import tool

@dataclass
class PostgresVectorStoreTool:
    rag_config: RagConfig
    embeddings_api_key: str | None

    def __post_init__(self) -> None:
        self._vector_store: PGVector | None = None

    @tool
    def search(self, query: str) -> str:
        """
        Busca información oficial y actualizada sobre procesos de admisiones
        en la base de conocimiento interna de la institución.

        CUÁNDO USARLA:
        - Cuando el usuario haga preguntas específicas sobre:
        - Requisitos de ingreso
        - Documentación necesaria
        - Proceso de admisión
        - Fechas y plazos (si están disponibles)
        - Modalidades de acceso
        - Pasos para inscribirse
        - Información académica relacionada con admisiones

        NO LA USES:
        - Para saludos, confirmaciones o respuestas cortas.
        - Para temas prohibidos (precios, becas, convalidaciones, etc.).
        - Si ya tienes suficiente información para responder sin consultar.
        - Para preguntas fuera del ámbito de admisiones.

        IMPORTANTE:
        - Si no encuentras información relevante, responde indicando que no
        puedes ayudar con esa pregunta.
        - Nunca menciones que consultaste una base de conocimiento.
        - Nunca inventes información que no esté en los resultados.

        Devuelve:
        - Un texto con la información relevante encontrada.
        - Puede devolver un texto vacío si no hay resultados.
        """
            
        if not query.strip():
            return ""

        vector_store = self._get_vector_store()
        documents = vector_store.similarity_search(
            query=query,
            k=self.rag_config.vector_store.top_k,
        )

        if not documents:
            return ""

        return "\n\n".join(document.page_content for document in documents if document.page_content)

    def _get_vector_store(self) -> PGVector:
        if self._vector_store is not None:
            return self._vector_store

        if self.rag_config.vector_store.kind != "postgresql":
            raise ValueError(f"Unsupported vector store kind: {self.rag_config.vector_store.kind}")

        if self.rag_config.embeddings.provider != "openai":
            raise ValueError(f"Unsupported embeddings provider: {self.rag_config.embeddings.provider}")

        embeddings = OpenAIEmbeddings(
            model=self.rag_config.embeddings.model,
            api_key=self.embeddings_api_key,
            chunk_size=self.rag_config.embeddings.batch_size,
        )

        self._vector_store = PGVector(
            embeddings=embeddings,
            collection_name=self.rag_config.vector_store.collection,
            connection=self.rag_config.vector_store.dsn,
            use_jsonb=True,
        )
        return self._vector_store
