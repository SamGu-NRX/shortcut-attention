"""
ERITimelineDataset - Core data structure for ERI timeline data.

This module provides the ERITimelineDataset class for storing and manipulating
timeline data from Einstellung experiments.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class ERITimelineDataset:
    """
    Core data structure for ERI timeline data.

    This class holds timeline data from Einstellung experiments with methods
    for filtering, alignment, and export.
    """
    data: pd.DataFrame
    metadata: Dict[str, Any]
    methods: List[str]
    splits: List[str]
    seeds: List[int]
    epoch_range: Tuple[float, float]

    def get_method_data(self, method: str) -> pd.DataFrame:
        """Get data for a specific method."""
        return self.data[self.data['method'] == method].copy()

    def get_split_data(self, split: str) -> pd.DataFrame:
        """Get data for a specific split."""
        return self.data[self.data['split'] == split].copy()

    def get_seed_data(self, seed: int) -> pd.DataFrame:
        """Get data for a specific seed."""
        return self.data[self.data['seed'] == seed].copy()

    def align_epochs(self, epochs: np.ndarray) -> "ERITimelineDataset":
        """Align dataset to common epoch grid (placeholder for now)."""
        # This will be implemented in task 2
        raise NotImplementedError("align_epochs will be implemented in task 2")
