"""Backward compatibility wrapper for legacy imports.

Historically the project exposed ``GPMModel`` from ``models.gpm_model``.  The
new integration lives in :mod:`models.gpm` under :class:`models.gpm.Gpm`.
This module simply subclasses the new implementation so external references
continue to work without modification.
"""

from models.gpm import Gpm


class GPMModel(Gpm):
    """Alias of :class:`models.gpm.Gpm` for backward compatibility."""

    NAME = "gpm"


__all__ = ["GPMModel"]

