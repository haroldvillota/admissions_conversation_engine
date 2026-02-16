from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config_source import ConfigSource


@dataclass(frozen=True)
class MergedConfigSource(ConfigSource):
    base_source: ConfigSource
    override_source: ConfigSource
    override_keys: Sequence[str]

    def load_configuration_values(self) -> Mapping[str, Any]:
        base_values = dict(self.base_source.load_configuration_values())
        override_values = dict(self.override_source.load_configuration_values())

        for key in self.override_keys:
            if key in override_values and override_values[key] not in (None, ""):
                base_values[key] = override_values[key]

        return base_values
