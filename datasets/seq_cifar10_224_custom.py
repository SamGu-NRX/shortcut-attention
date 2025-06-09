"""Custom CIFAR-10 dataset with specific 2-task structure for shortcut investigation."""

import logging
from typing import Dict, List, Tuple, Optional
from argparse import Namespace

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from datasets.seq_cifar10_224 import SequentialCIFAR10224
from datasets.utils.dataset_utils import fix_class_names_order

class SequentialCIFAR10Custom224(SequentialCIFAR10224):
    # Class attributes required by parent
    class_order: Optional[List[int]] = None
    class_names: Optional[List[str]] = None
    args: Namespace
    """Custom version of Sequential CIFAR10 Dataset with specific task organization.
    Version with ViT backbone, images resized to 224x224.

    Tasks:
    - Task 1: airplane, automobile (potential shortcuts: sky, road)
    - Task 2: bird, truck (potential shortcuts: sky, road/wheels)
    """

    NAME = 'seq-cifar10-224-custom'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 2  # Custom number of tasks
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    
    # Custom class order defining the subset of CIFAR-10 classes we use
    # Indices correspond to original CIFAR-10 class indices:
    # 0: airplane, 1: automobile, 2: bird, 9: truck
    CUSTOM_CLASS_ORDER = [0, 1, 2, 9]

    def __init__(self, args) -> None:
        """Initialize the dataset with given arguments.

        Args:
            args: Configuration object that must contain:
                - base_path: Path to dataset storage
                - custom_class_order: Optional comma-separated indices for class order
        """
        try:
            # Ensure required args are present
            args.base_path = getattr(args, 'base_path', './data/')

            # Handle class order setup
            provided_order = None
            if hasattr(args, 'custom_class_order') and args.custom_class_order:
                try:
                    provided_order = [int(i) for i in args.custom_class_order.split(',')]
                    logging.info(f"{self.NAME}: Using provided class order: {args.custom_class_order}")
                except (ValueError, AttributeError) as e:
                    logging.warning(f"{self.NAME}: Invalid custom_class_order format: {e}")
                    provided_order = None

            if provided_order is None:
                args.custom_class_order = ','.join(map(str, self.CUSTOM_CLASS_ORDER))
                provided_order = self.CUSTOM_CLASS_ORDER
                logging.info(f"{self.NAME}: Using default class order: {args.custom_class_order}")

            # Set and validate class order
            self.class_order = provided_order[:self.N_CLASSES]
            if len(self.class_order) != self.N_CLASSES:
                raise ValueError(
                    f"Number of classes in class order ({len(self.class_order)}) "
                    f"doesn't match N_CLASSES ({self.N_CLASSES})"
                )

            # Initialize parent with validated args
            super().__init__(args)
            self._validate_task_setup()
            
            # Log successful initialization
            logging.info(
                f"{self.NAME} initialized:\n"
                f"- Tasks: {self.N_TASKS}\n"
                f"- Classes per task: {self.N_CLASSES_PER_TASK}\n"
                f"- Total classes: {self.N_CLASSES}"
            )
            
        except Exception as e:
            logging.error(f"{self.NAME} initialization failed: {str(e)}")
            raise

    def _validate_task_setup(self) -> None:
        """Validate the task setup.
        
        Raises:
            ValueError: If any validation check fails
        """
        # Validate custom class order indices
        if not all(0 <= idx <= 9 for idx in self.CUSTOM_CLASS_ORDER):
            raise ValueError(
                f"Invalid CIFAR-10 indices in CUSTOM_CLASS_ORDER: {self.CUSTOM_CLASS_ORDER}. "
                f"All indices must be between 0 and 9."
            )
        
        # Class order has been initialized in __init__, just validate indices
        assert self.class_order is not None, "class_order was not initialized"
        assert len(self.class_order) == self.N_CLASSES, (
            f"Class order length ({len(self.class_order)}) does not match "
            f"N_CLASSES ({self.N_CLASSES})"
        )
            
        logging.debug(f"{self.NAME}: Task setup validated successfully")

    def get_class_names(self) -> List[str]:
        """Get ordered list of class names based on the initialized class order."""
        if self.class_names is not None:
            return self.class_names

        try:
            if self.class_order is None:
                raise ValueError("class_order must be initialized before getting class names")
                
            all_classes = CIFAR10(self.args.base_path + 'CIFAR10', 
                                train=True, download=True).classes
            self.class_names = [all_classes[i] for i in self.class_order]
            
            if len(self.class_names) != self.N_CLASSES:
                raise ValueError(
                    f"Got {len(self.class_names)} class names but expected {self.N_CLASSES}"
                )
                
            logging.debug(f"{self.NAME}: Initialized class names: {self.class_names}")
            return self.class_names
                
        except Exception as e:
            logging.error(f"{self.NAME}: Failed to get class names: {str(e)}")
            raise

    def get_task_labels(self) -> Dict[int, List[str]]:
        """Get mapping from task ID to list of class names for that task.
        
        Returns:
            Dict[int, List[str]]: Mapping from task ID to list of class names.
            Example: {0: ['airplane', 'automobile'], 1: ['bird', 'truck']}
            
        Raises:
            ValueError: If class names list is invalid or task structure is incorrect
        """
        try:
            class_names = self.get_class_names()
            if len(class_names) != self.N_CLASSES:
                raise ValueError(
                    f"Number of class names ({len(class_names)}) "
                    f"doesn't match N_CLASSES ({self.N_CLASSES})"
                )
            
            task_labels = {}
            for task_id in range(self.N_TASKS):
                start_idx = task_id * self.N_CLASSES_PER_TASK
                end_idx = (task_id + 1) * self.N_CLASSES_PER_TASK
                
                if end_idx > len(class_names):
                    raise ValueError(
                        f"Task {task_id} requires classes up to index {end_idx-1}, "
                        f"but only have {len(class_names)} classes"
                    )
                
                task_labels[task_id] = class_names[start_idx:end_idx]
                logging.debug(
                    f"{self.NAME}: Task {task_id} classes: {task_labels[task_id]} "
                    f"(indices {start_idx}-{end_idx-1})"
                )
            
            logging.info(f"{self.NAME}: Task labels initialized successfully")
            return task_labels
            
        except Exception as e:
            logging.error(f"{self.NAME}: Failed to get task labels: {str(e)}")
            raise
