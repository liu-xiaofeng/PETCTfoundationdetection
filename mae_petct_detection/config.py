from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_config_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    base_name = config.pop("_base_", None)
    if base_name:
        base_path = (path.parent / base_name).resolve()
        base = _load_config_tree(base_path, (*stack, path))
        config = _deep_update(base, config)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = _load_config_tree(path)
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parents[1])
    return config


def resolve_path(value: str | Path | None, config: dict[str, Any]) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (Path(config["_project_root"]) / path).resolve()
