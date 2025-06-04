# datasets/seq_cifar10_custom.py

from argparse import Namespace
import logging
from typing import Tuple, List 

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10 

from utils.conf import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders,
                                              fix_class_names_order)
from datasets.utils import set_default_from_args


class TCIFAR10Custom(CIFAR10):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR10Custom, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class MyCIFAR10Custom(CIFAR10):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR10Custom, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        return img, target, not_aug_img


class SequentialCIFAR10Custom(ContinualDataset):
    NAME = 'seq-cifar10-custom'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 2
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS 
    SIZE = (32, 32)
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    CUSTOM_CLASS_ORDER_INDICES = [0, 1, 2, 9] 
    
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __init__(self, args: Namespace, transform_type: str = 'weak'):
        if not hasattr(args, 'permute_classes'):
            args.permute_classes = False
        
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
                            f"Changing to 'current' to avoid ValueError in store_masked_loaders.")
            args.validation_mode = 'current'

        if not hasattr(args, 'N_CLASSES'):
            args.N_CLASSES = self.N_CLASSES
            logging.info(f"Setting args.N_CLASSES to {self.N_CLASSES} for {self.NAME}")
        elif args.N_CLASSES != self.N_CLASSES:
            logging.warning(f"args.N_CLASSES ({args.N_CLASSES}) differs from dataset's N_CLASSES ({self.N_CLASSES}). Using dataset's.")
            args.N_CLASSES = self.N_CLASSES
        
        if not hasattr(args, 'batch_size'):
            try:
                args.batch_size = self.get_batch_size() 
            except AttributeError: 
                args.batch_size = 32 
            logging.info(f"Setting args.batch_size to {args.batch_size} for {self.NAME} in __init__")

        if not hasattr(args, 'drop_last'):
            args.drop_last = False 
            logging.info(f"Setting args.drop_last to {args.drop_last} for {self.NAME} in __init__")
        
        # Ensure num_workers is on args for create_seeded_dataloader
        if not hasattr(args, 'num_workers'):
            args.num_workers = 0 # Default to 0 for simplicity/testing
            logging.info(f"Setting args.num_workers to {args.num_workers} for {self.NAME} in __init__")


        super().__init__(args) 

        current_custom_order_str = getattr(self.args, 'custom_class_order', None)
        if current_custom_order_str and not self.args.permute_classes:
            try:
                self.class_order = [int(x) for x in current_custom_order_str.split(',')][:self.N_CLASSES]
                logging.info(f"{self.NAME} __init__: Manually set self.class_order to {self.class_order} from args.custom_class_order")
            except ValueError:
                logging.error(f"Could not parse args.custom_class_order: {current_custom_order_str}. Falling back.")
                self.class_order = list(range(self.N_CLASSES)) 
        elif self.args.permute_classes:
            if not hasattr(self, 'class_order') or self.class_order is None:
                 logging.error(f"{self.NAME} __init__: Permutation requested, but self.class_order still not set by superclass!")
                 self.class_order = list(range(self.N_CLASSES)) 
            else:
                logging.info(f"{self.NAME} __init__: Using self.class_order {self.class_order} (likely from permutation by superclass)")
        else: 
            self.class_order = list(range(self.N_CLASSES))
            logging.info(f"{self.NAME} __init__: Defaulting self.class_order to {self.class_order}")

        if len(self.class_order) != self.N_CLASSES:
            logging.error(f"CRITICAL: self.class_order length ({len(self.class_order)}) "
                          f"mismatches self.N_CLASSES ({self.N_CLASSES}) after manual setting.")

        assert transform_type in ['weak', 'strong'], "Transform type must be either 'weak' or 'strong'."
        if transform_type == 'strong':
            logging.info("Using strong augmentation for CIFAR10 Custom")
            self.TRANSFORM = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 transforms.ToTensor(),
                 transforms.Normalize(SequentialCIFAR10Custom.MEAN, SequentialCIFAR10Custom.STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM
        train_dataset = MyCIFAR10Custom(base_path() + 'CIFAR10', train=True,
                                      download=True, transform=transform)
        test_dataset = TCIFAR10Custom(base_path() + 'CIFAR10', train=False,
                                    download=True, transform=self.TEST_TRANSFORM)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        return transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10Custom.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone(): 
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCIFAR10Custom.MEAN, SequentialCIFAR10Custom.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialCIFAR10Custom.MEAN, SequentialCIFAR10Custom.STD)

    @set_default_from_args('n_epochs')
    def get_epochs(self): 
        return 50

    @set_default_from_args('batch_size') 
    def get_batch_size(self): 
        return 32 

    def get_class_names(self) -> List[str]:
        if self.class_names is not None:
            return self.class_names
        
        all_cifar_classes = CIFAR10(base_path() + 'CIFAR10', train=True, download=True).classes
        
        if not hasattr(self.args, 'N_CLASSES') or self.args.N_CLASSES != self.N_CLASSES:
            logging.warning(f"get_class_names: self.args.N_CLASSES is missing or incorrect. "
                            f"Temporarily setting to {self.N_CLASSES} for fix_class_names_order.")
            self.args.N_CLASSES = self.N_CLASSES 
            
        ordered_names_from_util = fix_class_names_order(all_cifar_classes, self.args)
        
        selected_names: List[str]
        if hasattr(self, 'class_order') and self.class_order is not None and len(self.class_order) == self.N_CLASSES:
            try:
                selected_names = [all_cifar_classes[i] for i in self.class_order]
                logging.info(f"Selected class names using self.class_order: {selected_names}")
            except IndexError:
                logging.error(f"IndexError selecting names with self.class_order {self.class_order}. Falling back.")
                selected_names = ordered_names_from_util[:self.N_CLASSES] # Fallback to first N from util
        else: 
            logging.warning(f"self.class_order not suitable or not set correctly ({getattr(self, 'class_order', 'Not Set')}). "
                            f"Using first {self.N_CLASSES} from fix_class_names_order result.")
            selected_names = ordered_names_from_util[:self.N_CLASSES]
            
        self.class_names = selected_names
        logging.info(f"Resolved class names for {self.NAME}: {self.class_names}")
        
        if len(self.class_names) != self.N_CLASSES:
            logging.error(f"CRITICAL MISMATCH in get_class_names: N_CLASSES ({self.N_CLASSES}) "
                          f"and resolved class names len ({len(self.class_names)}). ")
            self.class_names = [f"fallback_class_{i}" for i in range(self.N_CLASSES)]
            logging.error(f"Using fallback class names: {self.class_names}")

        return self.class_names
    
    def get_task_labels(self):
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