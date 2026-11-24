"""Expected Quilt error classes."""

from __future__ import annotations


class QuiltInputError(ValueError):
    """Raised for invalid caller input."""


class QuiltResourceError(RuntimeError):
    """Raised when live host resources cannot safely admit a request."""


class QuiltInternalError(Exception):
    """Raised for internal invariant violations (server bugs, not caller input).

    Deliberately neither QuiltInputError nor QuiltResourceError so the /generate
    handler lets it surface as HTTP 500 rather than mislabeling an internal bug as a
    client 400 or a resource 503.
    """
