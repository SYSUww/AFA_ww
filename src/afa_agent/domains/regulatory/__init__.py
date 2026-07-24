from __future__ import annotations

from typing import Any

__all__ = ["RegulatoryPlugin"]


def __getattr__(name: str) -> Any:
    """Keep retrieval imports independent from the legacy rule solver."""

    if name == "RegulatoryPlugin":
        from .plugin import RegulatoryPlugin

        return RegulatoryPlugin
    raise AttributeError(name)
