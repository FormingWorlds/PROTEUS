from __future__ import annotations


def none_if_none(val: str) -> str | None:
    """Convert 'none' string into None literal."""
    return None if val == 'none' else val
