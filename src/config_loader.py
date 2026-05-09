"""Central configuration loader."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if overrides:
        config = _deep_merge(config, overrides)

    config["_project_root"] = str(PROJECT_ROOT)
    return config


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from config['paths'] relative to project root."""
    return PROJECT_ROOT / config["paths"][key]
