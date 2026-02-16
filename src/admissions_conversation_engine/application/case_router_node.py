from langgraph.runtime import Runtime
from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from langgraph.graph import END

class CaseRouterNode:
    def __call__(self, state: AgentState, runtime: Runtime[ContextSchema]) -> str:
        """
        Devuelve el nombre del siguiente nodo basado en runtime.context.case
        """
        case = runtime.context.case

        mapping = {
            "off_hours": "off_hours_node",
            "low_scoring": "low_scoring_node",
            "overflow": "overflow_node",
            "max_retries": "max_retries_node",
        }

        next_node = mapping.get(runtime.context.case, "END")
        return {"next_node": next_node}
