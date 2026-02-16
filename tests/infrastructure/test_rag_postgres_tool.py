from admissions_conversation_engine.infrastructure.config.app_config import RagConfig
from admissions_conversation_engine.infrastructure.rag_postgres_tool import (
    PostgresVectorStoreTool,
)


def _rag_config() -> RagConfig:
    return RagConfig.model_validate(
        {
            "vector_store": {
                "kind": "postgresql",
                "dsn": "postgresql://u:p@h:5432/db",
                "collection": "rag_docs",
                "top_k": 2,
            },
            "embeddings": {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "batch_size": 128,
            },
        }
    )


def test_rag_postgres_tool_returns_joined_page_content(monkeypatch) -> None:
    class FakeDoc:
        def __init__(self, page_content: str):
            self.page_content = page_content

    class FakePGVector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def similarity_search(self, query: str, k: int):
            assert query == "admisiones"
            assert k == 2
            return [FakeDoc("doc 1"), FakeDoc("doc 2")]

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.rag_postgres_tool.OpenAIEmbeddings",
        lambda **kwargs: {"embeddings": kwargs},
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.rag_postgres_tool.PGVector",
        lambda **kwargs: FakePGVector(**kwargs),
    )

    tool = PostgresVectorStoreTool(rag_config=_rag_config(), embeddings_api_key="secret")

    result = tool.search("admisiones")

    assert result == "doc 1\n\ndoc 2"


def test_rag_postgres_tool_returns_empty_string_for_blank_query(monkeypatch) -> None:
    tool = PostgresVectorStoreTool(rag_config=_rag_config(), embeddings_api_key="secret")

    result = tool.search("   ")

    assert result == ""


def test_rag_postgres_tool_raises_for_unsupported_vector_store_kind() -> None:
    config = _rag_config().model_copy(
        update={
            "vector_store": _rag_config().vector_store.model_copy(update={"kind": "memory"}),
        }
    )
    tool = PostgresVectorStoreTool(rag_config=config, embeddings_api_key="secret")

    try:
        tool.search("hola")
        assert False, "Expected ValueError for unsupported vector store kind"
    except ValueError as error:
        assert "Unsupported vector store kind" in str(error)
