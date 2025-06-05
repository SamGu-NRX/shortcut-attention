# Custom dataset for CIFAR-10, 224x224 for ViTs, with a specific 2-task structure.

import logging
from argparse import Namespace
from typing import Tuple, List
import torch

import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from datasets.seq_cifar10 import TCIFAR10, MyCIFAR10, base_path # Reuses these from original seq_cifar10
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders,
                                              fix_class_names_order)
from datasets.utils import set_default_from_args
# from utils.prompt_templates import templates # Remove if not using prompts for this custom version

class SequentialCIFAR10Custom224(ContinualDataset):
    """
    Sequential CIFAR10 Dataset, resized to 224x224, with a custom
    2-task structure for shortcut investigation.
    Task 1: airplane (0), automobile (1)
    Task 2: bird (2), truck (9)
    """

    NAME = 'seq-cifar10-224-custom'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 2 # Custom number of tasks
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS # Should be 4
    
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    SIZE = (224, 224) # Images are resized to this

    # Transforms are inherited from seq-cifar10-224 essentially
    TRANSFORM = transforms.Compose(
        [transforms.Resize(SIZE), # Resize to 224x224
         transforms.RandomCrop(SIZE, padding=28), # Standard padding for 224 crop from larger
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    TEST_TRANSFORM = transforms.Compose(
        [transforms.Resize(SIZE), 
         transforms.ToTensor(), 
         transforms.Normalize(MEAN, STD)])

    # Custom class order for the 4 selected classes (original CIFAR-10 indices)
    CUSTOM_CLASS_ORDER_INDICES = [0, 1, 2, 9] 

    def __init__(self, args: Namespace, transform_type: str = 'weak'): # Added transform_type
        # Set defaults for args expected by ContinualDataset and its utilities
        if not hasattr(args, 'permute_classes'):
            args.permute_classes = False 
        
        # If custom_class_order is not provided by the user, use this dataset's definition
        if not hasattr(args, 'custom_class_order') or args.custom_class_order is None:
            args.custom_class_order = ','.join(map(str, self.CUSTOM_CLASS_ORDER_INDICES))
            logging.info(
                f"Using default custom_class_order for {self.NAME}: {args.custom_class_order}"
            )
        
        if not hasattr(args, 'custom_task_order'):
            args.custom_task_order = None
        
        if not hasattr(args, 'validation_mode'):
            args.validation_mode = 'current' 
            logging.info(f"Setting default args.validation_mode to '{args.validation_mode}' for {self.NAME}")
        elif args.validation is None and args.validation_mode in ['class-il', 'task-il']:
            logging.warning(f"args.validation is None, but validation_mode is '{args.validation_mode}'. "
                            f"Changing to 'current'.")
            args.validation_mode = 'current'

        # N_CLASSES for this dataset is fixed at 4. Ensure args reflects this for fix_class_names_order.
        if not hasattr(args, 'N_CLASSES') or args.N_CLASSES != self.N_CLASSES:
            args.N_CLASSES = self.N_CLASSES
            logging.info(f"Setting/Overriding args.N_CLASSES to {self.N_CLASSES} for {self.NAME}")
        
        if not hasattr(args, 'batch_size'):
            try:
                args.batch_size = self.get_batch_size() 
            except AttributeError: 
                args.batch_size = 32 
            logging.info(f"Setting args.batch_size to {args.batch_size} for {self.NAME} in __init__")

        if not hasattr(args, 'drop_last'):
            args.drop_last = False 
            logging.info(f"Setting args.drop_last to {args.drop_last} for {self.NAME} in __init__")
        
        if not hasattr(args, 'num_workers'):
            args.num_workers = 0 
            logging.info(f"Setting args.num_workers to {args.num_workers} for {self.NAME} in __init__")

        super().__init__(args) 

        # Manually set self.class_order based on our custom logic
        current_custom_order_str = getattr(self.args, 'custom_class_order', None)
        if current_custom_order_str and not self.args.permute_classes:
            try:
                # Ensure we only take N_CLASSES worth of indices
                self.class_order = [int(x) for x in current_custom_order_str.split(',')][:self.N_CLASSES]
                logging.info(f"{self.NAME} __init__: Manually set self.class_order to {self.class_order}")
            except ValueError:
                logging.error(f"Could not parse args.custom_class_order: {current_custom_order_str}. Falling back.")
                self.class_order = list(range(self.N_CLASSES)) 
        elif self.args.permute_classes: # Should not be the case for this custom dataset
            if not hasattr(self, 'class_order') or self.class_order is None:
                 logging.error(f"{self.NAME} __init__: Permutation requested, but self.class_order still not set by superclass!")
                 self.class_order = list(range(self.N_CLASSES)) 
            else:
                logging.info(f"{self.NAME} __init__: Using self.class_order {self.class_order} (from permutation)")
        else: 
            self.class_order = list(range(self.N_CLASSES)) # Fallback if no custom_class_order
            logging.info(f"{self.NAME} __init__: Defaulting self.class_order to {self.class_order}")

        if len(self.class_order) != self.N_CLASSES:
            logging.error(f"CRITICAL: self.class_order length ({len(self.class_order)}) "
                          f"mismatches self.N_CLASSES ({self.N_CLASSES}).")
        
        # Handle strong augmentation if specified (similar to seq_cifar10_custom)
        if transform_type == 'strong':
            logging.info(f"Using strong augmentation for {self.NAME}")
            self.TRANSFORM = transforms.Compose(
                [transforms.Resize(self.SIZE),
                 transforms.RandomCrop(self.SIZE, padding=4), # Adjusted padding for 224
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 transforms.ToTensor(),
                 transforms.Normalize(self.MEAN, self.STD)])
        # else: self.TRANSFORM is already set at class level

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        # Uses MyCIFAR10 and TCIFAR10 which don't do resizing themselves;
        # resizing is handled by self.TRANSFORM and self.TEST_TRANSFORM.
        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=self.TRANSFORM)
        test_dataset = TCIFAR10(base_path() + 'CIFAR10', train=False,
                                download=True, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform(): # This refers to the class-level TRANSFORM
        return transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10Custom224.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone(): # Static method
        return "vit" # Default backbone for this 224x224 dataset

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCIFAR10Custom224.MEAN, SequentialCIFAR10Custom224.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialCIFAR10Custom224.MEAN, SequentialCIFAR10Custom224.STD)

    @set_default_from_args('n_epochs')
    def get_epochs(self): # Instance method
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self): # Instance method
        return 32

    def get_class_names(self) -> List[str]:
        if self.class_names is not None:
            return self.class_names
        
        all_cifar_classes = CIFAR10(base_path() + 'CIFAR10', train=True, download=True).classes
        
        # Ensure self.args has N_CLASSES for fix_class_names_order
        if not hasattr(self.args, 'N_CLASSES') or self.args.N_CLASSES != self.N_CLASSES:
            logging.warning(f"get_class_names: self.args.N_CLASSES is missing or incorrect. "
                            f"Setting to {self.N_CLASSES} for fix_class_names_order.")
            self.args.N_CLASSES = self.N_CLASSES 
            
        # fix_class_names_order reorders all_cifar_classes based on args.custom_class_order
        # or args.permute_classes. It does NOT truncate to args.N_CLASSES if custom_class_order is used.
        reordered_all_names = fix_class_names_order(all_cifar_classes, self.args)
        
        # We must select our N_CLASSES based on self.class_order (the indices)
        if hasattr(self, 'class_order') and self.class_order is not None and len(self.class_order) == self.N_CLASSES:
            try:
                # Use the indices from self.class_order to pick from the original CIFAR names
                selected_names = [all_cifar_classes[i] for i in self.class_order]
            except IndexError as e:
                logging.error(f"IndexError in get_class_names with self.class_order {self.class_order}: {e}. Falling back.")
                # Fallback: take the first N_CLASSES from the reordered_all_names if class_order is bad
                selected_names = reordered_all_names[:self.N_CLASSES]
        else:
            logging.error(f"self.class_order is not set correctly in get_class_names. Using first {self.N_CLASSES} from reordered names.")
            selected_names = reordered_all_names[:self.N_CLASSES]
            
        self.class_names = selected_names
        logging.info(f"Resolved class names for {self.NAME}: {self.class_names}")
        
        if len(self.class_names) != self.N_CLASSES:
            logging.error(f"CRITICAL MISMATCH in get_class_names: N_CLASSES ({self.N_CLASSES}) "
                          f"and resolved class names len ({len(self.class_names)}). ")
            # Fallback to generic names if still a mismatch
            self.class_names = [f"fallback_class_{i}" for i in range(self.N_CLASSES)]

        return self.class_names

    def get_task_labels(self): # Copied from previous seq_cifar10_custom
        class_names = self.get_class_names() 
        if len(class_names) != self.N_CLASSES:
             logging.error(f"get_task_labels: Critical - class_names length ({len(class_names)}) "
                           f"still mismatches N_CLASSES ({self.N_CLASSES}). Task labels will be wrong.")
             return {
                i: [f'task{i}_class{j}_err' for j in range(self.N_CLASSES_PER_TASK)]
                for i in range(self.N_TASKS)
            }
        task_labels = {}
        for task_id in range(self.N_TASKS):
            start_idx = task_id * self.N_CLASSES_PER_TASK
            end_idx = (task_id + 1) * self.N_CLASSES_PER_TASK
            if end_idx > len(class_names):
                logging.error(f"Not enough class names to form task {task_id}. "
                              f"Need up to index {end_idx-1}, have {len(class_names)} names.")
                task_labels[task_id] = [f"err_task{task_id}_cls{j}" for j in range(self.N_CLASSES_PER_TASK)]
            else:
                task_labels[task_id] = class_names[start_idx:end_idx]
        logging.debug(f"Generated task labels for {self.NAME}: {task_labels}")
        return task_labels

    # @staticmethod # Removed if not used for this custom version
    # def get_prompt_templates():
    #     return templates['cifar100'] # Or 'cifar10' or remove