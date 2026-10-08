"""Configuration helpers: YAML loading and repository paths."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """Return an absolute path inside the repository."""
    return REPO_ROOT.joinpath(*parts)


def data_root() -> Path:
    """Directory holding the raw downloads (see data/README.md).

    Defaults to ``<repo>/data/raw``; override with the VIGIL_DATA environment variable.
    """
    return Path(os.environ.get("VIGIL_DATA", repo_path("data", "raw")))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` (``name`` may include the extension)."""
    p = Path(name)
    if not p.suffix:
        p = repo_path("configs", f"{name}.yaml")
    elif not p.is_absolute():
        p = repo_path("configs", name)
    return load_yaml(p)


def merge(base: dict, override: dict | None) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
