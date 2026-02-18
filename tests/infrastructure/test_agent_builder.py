from admissions_conversation_engine.infrastructure.agent_builder import AgentBuilder
from admissions_conversation_engine.infrastructure.config.app_config import AppConfig


class FakeStateGraph:
    last_instance = None

    def __init__(self, state_type, context_schema=None):
        self.state_type = state_type
        self.context_schema = context_schema
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []
        self.compile_checkpointer = None
        FakeStateGraph.last_instance = self

    def add_node(self, name, node):
        self.nodes[name] = node

    def add_edge(self, source, target):
        self.edges.append((source, target))

    def add_conditional_edges(self, source, router, mapping):
        self.conditional_edges.append((source, mapping))

    def compile(self, checkpointer=None):
        self.compile_checkpointer = checkpointer
        return {"graph": "compiled", "checkpointer": checkpointer}


def _app_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "rag": {
                "vector_store": {
                    "kind": "postgresql",
                    "dsn": "postgresql://u:p@h:5432/db",
                    "collection": "rag_docs",
                    "top_k": 4,
                },
                "embeddings": {
                    "provider": "openai",
                    "model": "text-embedding-3-large",
                    "batch_size": 128,
                },
            },
            "llm": {
                "default": {
                    "provider": "openai",
                    "api_key": "default-key",
                    "model": "gpt-4.1-mini",
                    "temperature": 0.0,
                },
                "guardrail": {
                    "provider": "openai",
                    "api_key": "guardrail-key",
                    "model": "gpt-4.1-mini",
                    "temperature": 0.0,
                },
                "react": {
                    "provider": "openai",
                    "api_key": "react-key",
                    "model": "gpt-4.1",
                    "temperature": 0.0,
                },
                "translator": {
                    "provider": "openai",
                    "api_key": "translator-key",
                    "model": "gpt-4.1-nano",
                    "temperature": 0.0,
                },
            },
            "checkpointer": {
                "kind": "postgresql",
                "dsn": "postgresql://u:p@h:5432/db",
            },
            "observability": {
                "provider": "langfuse",
                "public_key": "pk",
                "secret_key": "sk",
                "base_url": "https://cloud.langfuse.com",
            },
            "tenant": {
                "institution": "Universidad Europea",
                "terms_of_service": "https://example.com/tos",
                "allowed_topics": "Admisiones",
                "tone": "Empatico",
                "language_fallback": "en-US",
                "allowed_languages": "es-ES,en-US",
            },
        }
    )


def test_agent_builder_builds_graph_with_expected_nodes_and_compiles(monkeypatch) -> None:
    class FakeLLM:
        def __init__(self, model: str):
            self.model = model

        def bind_tools(self, _tools):
            return self

    class FakeLLMFactory:
        def __init__(self, profile):
            self._profile = profile

        def build_llm(self):
            return FakeLLM(self._profile.model)

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.StateGraph",
        FakeStateGraph,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.LLMFactory",
        FakeLLMFactory,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.render_guardrail_prompt",
        lambda _: "guardrail-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.render_language_detector_prompt",
        lambda _: "language-detector-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.render_case_off_hours_prompt",
        lambda _: "off-hours-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.render_case_low_scoring_prompt",
        lambda _: "low-scoring-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.render_case_overflow_prompt",
        lambda _: "overflow-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.render_case_max_retries_prompt",
        lambda _: "max-retries-prompt",
    )

    checkpointer = object()
    builder = AgentBuilder(app_config=_app_config(), checkpointer=checkpointer)

    compiled = builder.build()
    graph = FakeStateGraph.last_instance

    assert compiled == {"graph": "compiled", "checkpointer": checkpointer}
    assert graph is not None
    assert graph.compile_checkpointer is checkpointer
    assert set(graph.nodes) == {
        "setup",
        "language_detector",
        "guardrail",
        "case_router",
        "off_hours_node",
        "tools",
        "low_scoring_node",
        "overflow_node",
        "max_retries_node",
    }
    assert ("setup", "language_detector") in graph.edges
    assert ("language_detector", "guardrail") in graph.edges
