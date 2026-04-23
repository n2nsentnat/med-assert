"""Protocol for HTTP text GET (testability / DIP)."""

from __future__ import annotations

from typing import Any, Protocol


class HttpTextClient(Protocol):
    """Minimal surface used by ``EntrezPubMedGateway``.

    Callers should pass an implementation that applies NCBI-friendly rate limiting
    and retries (e.g. :class:`~med_assert.infrastructure.collect.resilient_http.ResilientHttpClient`).
    The protocol itself does not enforce that—only the concrete client does.
    """

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        """Perform GET with optional query params and return decoded body text."""
