"""
Utility functions for dataset management.
"""

from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import DataLoader
import numpy as np
from argparse import Namespace

def fix_class_names_order(class_names: List[str], class_order: List[int]) -> List[str]:
    """
    Reorder class names according to the class order.
    
    Args:
        class_names: Original list of class names
        class_order: List of class indices in desired order
        
    Returns:
        Reordered list of class names
    """
    if class_order is None:
        return class_names
    return [class_names[i] for i in class_order]

def store_masked_loaders(
    train_dataset: Any,
    test_dataset: Any,
    class_order: List[int],
    args: Namespace
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Store dataloaders for each task in sequence.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        class_order: List of class indices in order
        args: Arguments with batch_size, workers, etc.
        
    Returns:
        Tuple of (train_loaders, test_loaders)
    """
    train_loaders, test_loaders = [], []
    N_CLASSES_PER_TASK = 2  # For CIFAR-10 custom setup

    for i in range(0, len(class_order), N_CLASSES_PER_TASK):
        train_mask = np.isin(train_dataset.targets, class_order[i:i + N_CLASSES_PER_TASK])
        test_mask = np.isin(test_dataset.targets, class_order[i:i + N_CLASSES_PER_TASK])

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=args.drop_last
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders
