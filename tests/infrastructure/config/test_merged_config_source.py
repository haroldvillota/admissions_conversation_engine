from admissions_conversation_engine.infrastructure.config.config_source import ConfigSource
from admissions_conversation_engine.infrastructure.config.merged_config_source import (
    MergedConfigSource,
)


class StaticConfigSource(ConfigSource):
    def __init__(self, values: dict):
        self._values = values

    def load_configuration_values(self) -> dict:
        return self._values


def test_merged_config_source_applies_override_for_selected_keys() -> None:
    base_source = StaticConfigSource({"A": "1", "B": "2"})
    override_source = StaticConfigSource({"B": "20", "C": "30"})
    source = MergedConfigSource(
        base_source=base_source,
        override_source=override_source,
        override_keys=["B", "C"],
    )

    result = source.load_configuration_values()

    assert result == {"A": "1", "B": "20", "C": "30"}


def test_merged_config_source_ignores_empty_override_values() -> None:
    base_source = StaticConfigSource({"A": "1", "B": "2"})
    override_source = StaticConfigSource({"B": "", "A": None})
    source = MergedConfigSource(
        base_source=base_source,
        override_source=override_source,
        override_keys=["A", "B"],
    )

    result = source.load_configuration_values()

    assert result == {"A": "1", "B": "2"}
