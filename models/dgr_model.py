"""Backward compatibility wrapper for the original DGR integration."""

from models.dgr import Dgr


class DGRModel(Dgr):
    """Alias of :class:`models.dgr.Dgr` for legacy import paths."""

    NAME = "dgr"


__all__ = ["DGRModel"]

