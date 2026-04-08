"""Common filesystem path helpers shared by CLIs."""

from __future__ import annotations

from pathlib import Path


def default_project_root() -> Path:
    """Prefer cwd when it looks like repo root; else walk up from this module."""
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file():
        return cwd
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        if (d / "pyproject.toml").is_file():
            return d
    return cwd

