"""Utility helpers for checkpoint discovery and metadata."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List


def find_existing_checkpoints(
    strategy: str,
    backbone: str,
    seed: int,
    *,
    dataset: str,
    checkpoint_root: Path = Path("checkpoints"),
) -> List[Path]:
    """Return matching checkpoints sorted by modification time (newest first)."""
    if not checkpoint_root.exists():
        return []

    buffer_size = "500" if strategy == "derpp" else "0"
    n_epochs = "20" if "vit" in backbone.lower() else "50"

    patterns = [
        f"{strategy}_{dataset}_*_{buffer_size}_{n_epochs}_*_last.pt",
        f"{strategy}_{dataset}_*_{buffer_size}_{n_epochs}_*_1.pt",
    ]

    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(str(checkpoint_root / pattern)))

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches


_CHECKPOINT_REGEX = re.compile(
    r"([^_]+)_([^_]+)_([^_]+)_(\d+)_(\d+)_(\d{8}-\d{6})_([^_]+)_(.+)\.pt"
)


def describe_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Parse a Mammoth checkpoint filename for logging/reporting."""
    info: Dict[str, Any] = {
        "path": str(checkpoint_path),
        "size_mb": f"{checkpoint_path.stat().st_size / (1024 * 1024):.1f}",
        "modified": checkpoint_path.stat().st_mtime,
        "filename": checkpoint_path.name,
    }

    match = _CHECKPOINT_REGEX.match(checkpoint_path.name)
    if match:
        info.update(
            {
                "model": match.group(1),
                "dataset": match.group(2),
                "config": match.group(3),
                "buffer_size": match.group(4),
                "n_epochs": match.group(5),
                "timestamp": match.group(6),
                "uid": match.group(7),
                "suffix": match.group(8),
            }
        )

    return info
