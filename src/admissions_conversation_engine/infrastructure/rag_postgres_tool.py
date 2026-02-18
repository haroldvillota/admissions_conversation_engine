from __future__ import annotations

from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from admissions_conversation_engine.infrastructure.config.app_config import RagConfig
from langchain_core.tools import tool

from langchain_core.tools import BaseTool

class PostgresVectorStoreTool(BaseTool):
    name: str = "search_admissions"
    description: str = """
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

    def __init__(
        self,
        rag_config: RagConfig,
        embeddings_api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rag_config = rag_config
        self.embeddings_api_key = embeddings_api_key
        self._vector_store: PGVector | None = None
        
    rag_config: RagConfig = None
    embeddings_api_key: str = None

    def __post_init__(self) -> None:
        super().__init__()
        self._vector_store: PGVector | None = None


    def _run(self, query: str) -> str:
        if not query.strip():
            return ""

        vector_store = self._get_vector_store()
        documents = vector_store.similarity_search(
            query=query,
            k=self.rag_config.vector_store.top_k,
        )
        if not documents:
            return ""

        return "\n\n".join(d.page_content for d in documents if d.page_content)

    async def _arun(self, query: str) -> str:
        # TODO
        return self._run(query)
    
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

    def search(self, query: str) -> str:
        return self._run(query)

    def make_search_tool(self):
        @tool(name_or_callable=self.name, description=self.description)
        def search_admissions(query: str) -> str:
            return self.search(query)

        return search_admissions
