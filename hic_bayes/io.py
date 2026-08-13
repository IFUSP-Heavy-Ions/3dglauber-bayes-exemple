"""Path helpers for separated experimental data and model calculations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

MODEL_KEYS = {
    "design",
    "design_low",
    "design_high",
    "design_ranges",
    "simulation",
    "simulation_low",
    "simulation_high",
    "validation_design",
    "validation_simulation",
}

DATA_KEYS = {"experiment", "exp_error"}


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(path))))


def _path_from_section(section: dict[str, Any], key: str) -> Path | None:
    value = section.get(key)
    if value is None:
        return None
    path = _expand_path(value)
    if path.is_absolute():
        return path
    base_dir = _expand_path(section.get("base_dir", "."))
    return base_dir / path


def table_path(cfg: dict[str, Any], key: str, *, kind: str | None = None) -> Path:
    """Resolve a CSV path from ``data:``/``model:`` config sections.

    The new convention is ``data`` for experimental inputs and ``model`` for
    design/simulation tables.  For older YAML files, model keys fall back to the
    ``data`` section so existing analyses keep working until their configs are
    migrated.
    """
    if kind is None:
        kind = "data" if key in DATA_KEYS else "model"

    sections: list[dict[str, Any]] = []
    if kind == "data":
        sections = [cfg.get("data", {})]
    elif kind == "model":
        sections = [cfg.get("model", {}), cfg.get("data", {})]
    else:
        raise ValueError(f"Unknown table kind {kind!r}; use 'data' or 'model'.")

    for section in sections:
        path = _path_from_section(section, key)
        if path is not None:
            return path

    section_names = "model/data" if kind == "model" else "data"
    raise KeyError(f"Could not resolve {key!r} from {section_names} config section.")


def read_table(cfg: dict[str, Any], key: str, *, kind: str | None = None) -> pd.DataFrame:
    return pd.read_csv(table_path(cfg, key, kind=kind))


def read_array(cfg: dict[str, Any], key: str, *, kind: str | None = None):
    return read_table(cfg, key, kind=kind).values
