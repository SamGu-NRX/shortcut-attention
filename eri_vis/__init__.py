"""
ERI Visualization System

A comprehensive toolkit for visualizing Einstellung Rigidity Index (ERI) metrics
in continual learning experiments.
"""

__version__ = "1.0.0"
__author__ = "ERI Research Team"

from .data_loader import ERIDataLoader
from .dataset import ERITimelineDataset

__all__ = [
    "ERIDataLoader",
    "ERITimelineDataset",
]
