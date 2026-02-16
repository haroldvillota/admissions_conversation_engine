from __future__ import annotations

import os
from typing import Any, Mapping

from .config_source import ConfigSource


class EnvironmentVariableConfigSource(ConfigSource):
    def load_configuration_values(self) -> Mapping[str, Any]:
        return dict(os.environ)
