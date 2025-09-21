"""Compatibility utilities for legacy imports.

The historical test-suite referenced ``GPMAdapter``; expose the new
``Gpm`` continual model under the same name to avoid cascading changes.
"""

from models.gpm import Gpm as GPMAdapter

__all__ = ["GPMAdapter"]
