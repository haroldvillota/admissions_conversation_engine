from __future__ import annotations

from typing import Any, Mapping, Protocol


class ConfigSource(Protocol):
    def load_configuration_values(self) -> Mapping[str, Any]:
        ...
