# experiments/default_args.py

"""
Single Source of Truth for Experiment Arguments.

This file defines a comprehensive set of default arguments required by the
Mammoth framework. Both the training and analysis scripts import this
configuration to ensure consistency and prevent AttributeError issues.
"""

import torch

def get_base_args() -> dict:
    """
    Returns a dictionary containing a complete set of default arguments
    for the shortcut investigation experiment.
    """
    args = {
        # ==========================================
        # Core Experiment Settings
        # ==========================================
        "dataset": "seq-cifar10-224-custom",
        "model": "derpp",  # Default, will be overridden
        "backbone": "vit",
        "n_epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",

        # ==========================================
        # Required Mammoth Arguments
        # ==========================================
        "base_path": "./data/",
        "results_path": None,
        "savecheck": "task",
        "ckpt_name": None,
        "debug_mode": 0,
        "non_verbose": 1,
        "code_optimization": 0,

        # --- Logging and Reporting ---
        "tensorboard": False,
        "csv_log": False,
        "notes": None,

        # --- Dataset & Tasking ---
        "seed": 42,
        "start_from": None,
        "stop_after": None,
        "joint": False,
        "label_perc": 1.0,
        "label_perc_by_class": 1.0,
        "validation": None,
        "validation_mode": "current",
        "noise_rate": 0.0,
        "transform_type": "weak",
        "custom_class_order": None,
        "permute_classes": False,
        "custom_task_order": None, # <-- The missing argument
        "drop_last": False,
        "num_workers": 0,
        "num_classes": 4,

        # --- Model & Training ---
        "fitting_mode": "epochs",
        "n_iters": None,
        "minibatch_size": 32,
        "optimizer": "sgd",
        "optim_wd": 0.0,
        "optim_mom": 0.0,
        "optim_nesterov": False,
        "lr_scheduler": None,
        "distributed": "no",
        "load_best_args": False,
        "inference_only": 0,

        # --- Metrics ---
        "enable_other_metrics": False,
        "eval_future": False,
        "ignore_other_metrics": 0,

        # --- ViT Specific (for backbone instantiation) ---
        "pretrained": False,
        "pretrain_type": "in21k-ft-in1k",
    }
    return args