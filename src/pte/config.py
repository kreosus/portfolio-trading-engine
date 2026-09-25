"""Config loading. One YAML file holds every tunable; nothing is hard-coded elsewhere."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def with_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of cfg with dotted-key overrides, e.g. {"smc.swing_n": 5}."""
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[parts[-1]] = value
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
