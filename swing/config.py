"""Load config/strategy.yaml into a nested attribute-access object."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "strategy.yaml"


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def load_raw(path: str | Path = DEFAULT_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load(path: str | Path = DEFAULT_PATH, overrides: dict | None = None) -> SimpleNamespace:
    """Load the strategy config. `overrides` is a nested dict merged on top (used by tests and sweeps)."""
    raw = load_raw(path)
    if overrides:
        raw = merge(raw, overrides)
    return _to_ns(raw)


def merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out
