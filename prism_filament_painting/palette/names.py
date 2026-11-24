"""Comparison keys for spool names and inventory fields."""

import re


def normalize_key(key: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (key or "").lower())
