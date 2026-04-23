"""Backward-compatible module path for Uvicorn: ``med_assert.interfaces.api.app:app``.

Implementation lives in :mod:`med_assert.interfaces.api.http_app`.
"""

from med_assert.interfaces.api.http_app import app, run

__all__ = ["app", "run"]
