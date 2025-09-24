"""DGR model entrypoint for Mammoth integrations.

Historically this module re-exported the legacy wrapper that proxied calls to the
original reference implementation. That approach forced the use of the bespoke
classifier shipped with the paper, which diverges substantially from the
ResNet-based backbones used throughout this codebase and led to severe accuracy
drops in the Einstellung benchmark.

The project now exposes the Mammoth-native adapter that keeps the ContinualModel
interface while generating replay samples with the VAE presented in the paper.
This lets DGR train the same backbone as the other strategies and removes the
domain gap that previously crippled performance.
"""

from models.dgr_mammoth_adapter import DGRMammothAdapter


class DGRModel(DGRMammothAdapter):
    """Deep Generative Replay using the Mammoth-native adapter."""

    NAME = "dgr"
    _COND_VAE_CLS = None


__all__ = ["DGRModel"]
