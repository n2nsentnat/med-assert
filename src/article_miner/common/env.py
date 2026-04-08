"""Environment/bootstrap helpers."""

from __future__ import annotations

from dotenv import load_dotenv


def load_project_env() -> None:
    """Load .env into process env if present (no override)."""
    load_dotenv(override=False)

