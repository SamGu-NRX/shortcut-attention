"""
Integrated Methods Registry for ERI Visualization System

This module provides a registration system for adapted continual learning methods
(GPM, DGR, and hybrid approaches) that have been integrated with the Mammoth framework
for ERI (Einstellung Rigidity Index) evaluation.

The registry automatically discovers and registers adapted methods, provides
configuration validation, and integrates seamlessly with existing Mammoth
model loading infrastructure.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import yaml
import logging
import importlib
from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass, field
from pathlib import Path

from models.utils.continual_model import ContinualModel
from utils.conf import warn_once


@dataclass
class MethodMetadata:
    """Metadata for an integrated method."""
    name: str
    class_name: str
    module_name: str
    config_file: str
    description: str
    computational_requirements: Dict[str, str] = field(default_factory=dict)
    compatibility: Dict[str, Union[bool, List[str]]] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    usage_notes: str = ""
    tuning_guidelines: str = ""


class IntegratedMethodRegistry:
    """
    Registry for adapted continual learning methods integrated with Mammoth.

    This registry provides automatic discovery and registration of adapted methods,
    configuration validation, and seamless integration with existing Mammoth
    model loading infrastructure.
    """

    # Registry of integrated methods
    _methods: Dict[str, MethodMetadata] = {}
    _initialized = False

    # Base configuration directory
    CONFIG_DIR = Path(__file__).parent / "config"

    # Integrated method definitions
    INTEGRATED_METHODS = {
        'gpm': {
            'class_name': 'GPMModel',
            'module_name': 'gpm_model',
            'config_file': 'gpm.yaml',
            'description': 'Gradient Projection Memory with SVD-based subspace extraction adapted from original GPM implementation'
        },
        'dgr': {
            'class_name': 'DGRModel',
            'module_name': 'dgr_model',
            'config_file': 'dgr.yaml',
            'description': 'Deep Generative Replay using VAE-based sample generation adapted from original DGR implementation'
        },
        'gpm_dgr_hybrid': {
            'class_name': 'GPMDGRHybridModel',
            'module_name': 'gpm_dgr_hybrid_model',
            'config_file': 'gpm_dgr_hybrid.yaml',
            'description': 'Hybrid method combining adapted GPM gradient projection with DGR generative replay'
        }
    }

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry by discovering and registering all integrated methods."""
        if cls._initialized:
            return

        logging.info("Initializing Integrated Methods Registry...")

        for method_name, method_info in cls.INTEGRATED_METHODS.items():
            try:
                cls._register_method(method_name, method_info)
                logging.info(f"Registered integrated method: {method_name}")
            except Exception as e:
                warn_once(f"Failed to register integrated method {method_name}: {e}")

        cls._initialized = True
        logging.info(f"Integrated Methods Registry initialized with {len(cls._methods)} methods")

    @classmethod
    def _register_method(cls, method_name: str, method_info: Dict[str, str]) -> None:
        """Register a single integrated method."""
        config_path = cls.CONFIG_DIR / method_info['config_file']

        # Load configuration if it exists
        config_data = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as e:
                warn_once(f"Failed to load config for {method_name}: {e}")

        # Create method metadata
        metadata = MethodMetadata(
            name=method_name,
            class_name=method_info['class_name'],
            module_name=method_info['module_name'],
            config_file=method_info['config_file'],
            description=method_info['description'],
            computational_requirements=config_data.get('computational_notes', {}),
            compatibility=config_data.get('compatibility', {}),
            hyperparameters=cls._extract_hyperparameters(config_data),
            usage_notes=config_data.get('usage_notes', ''),
            tuning_guidelines=config_data.get('tuning_guidelines', '')
        )

        cls._methods[method_name] = metadata

    @classmethod
    def _extract_hyperparameters(cls, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hyperparameters from configuration data."""
        # Skip metadata fields and extract actual hyperparameters
        skip_keys = {
            'model', 'description', 'computational_notes', 'usage_notes',
            'tuning_guidelines', 'compatibility', 'references', 'reference',
            'implementation_notes'
        }

        hyperparams = {}
        for k, v in config_data.items():
            if k not in skip_keys:
                if isinstance(v, dict):
                    # Handle nested configurations (like gpm: and dgr: in hybrid methods)
                    for nested_k, nested_v in v.items():
                        hyperparams[f"{k}_{nested_k}"] = nested_v
                else:
                    hyperparams[k] = v

        return hyperparams

    @classmethod
    def get_available_methods(cls) -> List[str]:
        """Get list of available integrated method names."""
        cls.initialize()
        return list(cls._methods.keys())

    @classmethod
    def get_method_metadata(cls, method_name: str) -> Optional[MethodMetadata]:
        """Get metadata for a specific method."""
        cls.initialize()
        return cls._methods.get(method_name)

    @classmethod
    def is_integrated_method(cls, method_name: str) -> bool:
        """Check if a method is an integrated method."""
        cls.initialize()
        return method_name in cls._methods

    @classmethod
    def create_method(cls, method_name: str, backbone, loss, args, transform=None, dataset=None) -> ContinualModel:
        """
        Create an instance of an integrated method.

        Args:
            method_name: Name of the method to create
            backbone: Neural network backbone
            loss: Loss function
            args: Arguments namespace
            transform: Data transform (optional)
            dataset: Dataset instance (optional)

        Returns:
            Instance of the requested ContinualModel

        Raises:
            ValueError: If method is not found or cannot be created
            ImportError: If method module cannot be imported
        """
        cls.initialize()

        if method_name not in cls._methods:
            raise ValueError(f"Unknown integrated method: {method_name}. "
                           f"Available methods: {list(cls._methods.keys())}")

        metadata = cls._methods[method_name]

        try:
            # Import the method module
            module = importlib.import_module(f'models.{metadata.module_name}')
            method_class = getattr(module, metadata.class_name)

            # Create and return the method instance
            return method_class(backbone, loss, args, transform, dataset)

        except ImportError as e:
            raise ImportError(f"Failed to import method {method_name} from models.{metadata.module_name}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Method class {metadata.class_name} not found in models.{metadata.module_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create method {method_name}: {e}")

    @classmethod
    def validate_configuration(cls, method_name: str, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration for a method.

        Args:
            method_name: Name of the method
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        cls.initialize()

        if method_name not in cls._methods:
            return [f"Unknown method: {method_name}"]

        metadata = cls._methods[method_name]
        errors = []

        # Load expected configuration schema
        config_path = cls.CONFIG_DIR / metadata.config_file
        if not config_path.exists():
            return [f"Configuration file not found: {metadata.config_file}"]

        try:
            with open(config_path, 'r') as f:
                expected_config = yaml.safe_load(f) or {}
        except Exception as e:
            return [f"Failed to load configuration schema: {e}"]

        # Validate required parameters (basic validation)
        expected_params = cls._extract_hyperparameters(expected_config)

        # Check for unknown parameters
        for param in config:
            if param not in expected_params and param != 'model':
                errors.append(f"Unknown parameter: {param}")

        # Type validation for known parameters
        for param, expected_value in expected_params.items():
            if param in config:
                if type(config[param]) != type(expected_value):
                    errors.append(f"Parameter {param} should be {type(expected_value).__name__}, "
                                f"got {type(config[param]).__name__}")

        return errors

    @classmethod
    def generate_documentation(cls, method_name: Optional[str] = None) -> str:
        """
        Generate documentation for integrated methods.

        Args:
            method_name: Specific method to document (None for all methods)

        Returns:
            Formatted documentation string
        """
        cls.initialize()

        if method_name:
            if method_name not in cls._methods:
                return f"Method {method_name} not found."
            methods_to_document = [method_name]
        else:
            methods_to_document = list(cls._methods.keys())

        doc_parts = []
        doc_parts.append("# Integrated Methods Documentation\n")
        doc_parts.append("This document describes the adapted continual learning methods "
                        "integrated with the Mammoth framework for ERI evaluation.\n")

        for method in methods_to_document:
            metadata = cls._methods[method]

            doc_parts.append(f"## {method.upper()}")
            doc_parts.append(f"**Class:** {metadata.class_name}")
            doc_parts.append(f"**Module:** models.{metadata.module_name}")
            doc_parts.append(f"**Config:** {metadata.config_file}\n")

            doc_parts.append(f"**Description:**")
            doc_parts.append(f"{metadata.description}\n")

            if metadata.hyperparameters:
                doc_parts.append("**Hyperparameters:**")
                for param, value in metadata.hyperparameters.items():
                    doc_parts.append(f"- `{param}`: {value} ({type(value).__name__})")
                doc_parts.append("")

            if metadata.compatibility:
                doc_parts.append("**Compatibility:**")
                if isinstance(metadata.compatibility, dict):
                    for key, value in metadata.compatibility.items():
                        if isinstance(value, list):
                            doc_parts.append(f"- {key}: {', '.join(value)}")
                        else:
                            doc_parts.append(f"- {key}: {value}")
                else:
                    doc_parts.append(f"- {metadata.compatibility}")
                doc_parts.append("")

            if metadata.usage_notes:
                doc_parts.append("**Usage Notes:**")
                doc_parts.append(metadata.usage_notes)
                doc_parts.append("")

            if metadata.tuning_guidelines:
                doc_parts.append("**Tuning Guidelines:**")
                doc_parts.append(metadata.tuning_guidelines)
                doc_parts.append("")

            doc_parts.append("---\n")

        return "\n".join(doc_parts)

    @classmethod
    def get_method_names_for_mammoth(cls) -> Dict[str, Type[ContinualModel]]:
        """
        Get method names and classes in format expected by Mammoth's get_model_names().

        Returns:
            Dictionary mapping method names to ContinualModel classes
        """
        cls.initialize()

        method_classes = {}

        for method_name, metadata in cls._methods.items():
            try:
                module = importlib.import_module(f'models.{metadata.module_name}')
                method_class = getattr(module, metadata.class_name)

                # Use the class NAME attribute if available, otherwise use method_name
                class_name = getattr(method_class, 'NAME', method_name)
                method_classes[class_name.replace('_', '-')] = method_class

            except Exception as e:
                warn_once(f"Failed to load integrated method {method_name}: {e}")
                # Skip failed methods instead of storing exceptions
                continue

        return method_classes


# Integration with existing Mammoth model loading
def extend_mammoth_model_names(existing_names: Dict[str, Type[ContinualModel]]) -> Dict[str, Type[ContinualModel]]:
    """
    Extend existing Mammoth model names with integrated methods.

    Args:
        existing_names: Existing model names from Mammoth

    Returns:
        Extended dictionary including integrated methods
    """
    integrated_names = IntegratedMethodRegistry.get_method_names_for_mammoth()

    # Merge with existing names, integrated methods take precedence for conflicts
    extended_names = existing_names.copy()
    extended_names.update(integrated_names)

    return extended_names


# Convenience functions for external use
def get_integrated_method_names() -> List[str]:
    """Get list of available integrated method names."""
    return IntegratedMethodRegistry.get_available_methods()


def create_integrated_method(method_name: str, backbone, loss, args, transform=None, dataset=None) -> ContinualModel:
    """Create an instance of an integrated method."""
    return IntegratedMethodRegistry.create_method(method_name, backbone, loss, args, transform, dataset)


def is_integrated_method(method_name: str) -> bool:
    """Check if a method is an integrated method."""
    return IntegratedMethodRegistry.is_integrated_method(method_name)


def generate_integrated_methods_documentation(method_name: Optional[str] = None) -> str:
    """Generate documentation for integrated methods."""
    return IntegratedMethodRegistry.generate_documentation(method_name)
