"""In-memory view of recent activity for the live monitoring page.

Placeholder with the interface the request path relies on; the monitoring
implementation replaces the bodies.
"""

from __future__ import annotations

from typing import Any

from .observe import RequestRecord


class Live:
    def __init__(self, ctx: Any = None):
        self.ctx = ctx

    def on_request_start(self, rec: RequestRecord) -> None:
        pass

    def on_request_done(self, rec: RequestRecord) -> None:
        pass
