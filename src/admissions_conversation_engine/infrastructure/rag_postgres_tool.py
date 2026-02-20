from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

from admissions_conversation_engine.infrastructure.config.app_config import RagConfig

from typing import Optional
from pydantic import Field

class PostgresVectorStoreTool(BaseTool):
    name: str = "search_admissions"
    description: str = (
        "Busca información oficial y actualizada sobre procesos de admisiones "
        "en la base de conocimiento interna de la institución."
    )
    rag_config: RagConfig = Field(...)
    _engine: Optional[PGEngine] = None
    _vector_store: Optional[PGVectorStore] = None

    def _run(self, query: str) -> str:
        if not query.strip():
            return ""

        store = self._get_vector_store()
        docs = store.similarity_search(query, k=self.rag_config.vector_store.top_k)
        
        if not docs:
            return ""

        return "\n\n".join(d.page_content for d in docs if d.page_content)

    async def _arun(self, query: str) -> str:
        # TODO
        return self._run(query)

    def _get_vector_store(self) -> PGVectorStore:
        if self._vector_store is not None:
            return self._vector_store

        if self.rag_config.vector_store.kind != "postgresql":
            raise ValueError(f"Unsupported vector store kind: {self.rag_config.vector_store.kind}")

        if self.rag_config.embeddings.provider != "openai":
            raise ValueError(f"Unsupported embeddings provider: {self.rag_config.embeddings.provider}")

        if not self.rag_config.embeddings.api_key:
            raise ValueError("Missing embeddings api_key in rag_config.embeddings.api_key")

        embeddings = OpenAIEmbeddings(
            model=self.rag_config.embeddings.model,
            api_key=self.rag_config.embeddings.api_key,
            chunk_size=self.rag_config.embeddings.batch_size,
        )

        engine = PGEngine.from_connection_string(url=self.rag_config.vector_store.dsn)
        self._engine = engine

        self._vector_store = PGVectorStore.create_sync(
            engine=engine,
            table_name=self.rag_config.vector_store.collection,
            embedding_service=embeddings
        )

        return self._vector_store
