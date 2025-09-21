"""
Tests for Integrated Methods Registry

This module contains comprehensive tests for the IntegratedMethodRegistry class,
including registration, configuration validation, method creation, and documentation
generation functionality.
"""

import pytest
import tempfile
import yaml
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from argparse import Namespace

from models.integrated_methods_registry import (
    IntegratedMethodRegistry,
    MethodMetadata,
    extend_mammoth_model_names,
    get_integrated_method_names,
    create_integrated_method,
    is_integrated_method,
    generate_integrated_methods_documentation
)


class TestMethodMetadata:
    """Test MethodMetadata dataclass."""

    def test_method_metadata_creation(self):
        """Test creating MethodMetadata instance."""
        metadata = MethodMetadata(
            name="test_method",
            class_name="TestModel",
            module_name="test_module",
            config_file="test.yaml",
            description="Test method description"
        )

        assert metadata.name == "test_method"
        assert metadata.class_name == "TestModel"
        assert metadata.module_name == "test_module"
        assert metadata.config_file == "test.yaml"
        assert metadata.description == "Test method description"
        assert metadata.computational_requirements == {}
        assert metadata.compatibility == {}
        assert metadata.hyperparameters == {}
        assert metadata.usage_notes == ""
        assert metadata.tuning_guidelines == ""


class TestIntegratedMethodRegistry:
    """Test IntegratedMethodRegistry class."""

    def setup_method(self):
        """Reset registry state before each test."""
        IntegratedMethodRegistry._methods = {}
        IntegratedMethodRegistry._initialized = False

    def test_registry_initialization(self):
        """Test registry initialization."""
        with patch.object(IntegratedMethodRegistry, '_register_method') as mock_register:
            IntegratedMethodRegistry.initialize()

            # Should register all integrated methods
            assert mock_register.call_count == len(IntegratedMethodRegistry.INTEGRATED_METHODS)
            assert IntegratedMethodRegistry._initialized is True

    def test_registry_initialization_idempotent(self):
        """Test that initialization is idempotent."""
        with patch.object(IntegratedMethodRegistry, '_register_method') as mock_register:
            IntegratedMethodRegistry.initialize()
            IntegratedMethodRegistry.initialize()  # Second call

            # Should only register once
            assert mock_register.call_count == len(IntegratedMethodRegistry.INTEGRATED_METHODS)

    def test_register_method_with_config(self):
        """Test registering a method with configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary config file
            config_path = Path(temp_dir) / "test.yaml"
            config_data = {
                'model': 'test_method',
                'param1': 'value1',
                'param2': 42,
                'description': 'Test description',
                'computational_notes': {'memory': 'low'},
                'compatibility': {'class-il': True},
                'usage_notes': 'Test usage',
                'tuning_guidelines': 'Test tuning'
            }

            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            # Mock CONFIG_DIR to point to temp directory
            with patch.object(IntegratedMethodRegistry, 'CONFIG_DIR', Path(temp_dir)):
                method_info = {
                    'class_name': 'TestModel',
                    'module_name': 'test_module',
                    'config_file': 'test.yaml',
                    'description': 'Test method'
                }

                IntegratedMethodRegistry._register_method('test_method', method_info)

                # Check that method was registered correctly
                assert 'test_method' in IntegratedMethodRegistry._methods
                metadata = IntegratedMethodRegistry._methods['test_method']

                assert metadata.name == 'test_method'
                assert metadata.class_name == 'TestModel'
                assert metadata.module_name == 'test_module'
                assert metadata.config_file == 'test.yaml'
                assert metadata.description == 'Test method'
                assert metadata.hyperparameters == {'param1': 'value1', 'param2': 42}
                assert metadata.computational_requirements == {'memory': 'low'}
                assert metadata.compatibility == {'class-il': True}
                assert metadata.usage_notes == 'Test usage'
                assert metadata.tuning_guidelines == 'Test tuning'

    def test_register_method_without_config(self):
        """Test registering a method without configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock CONFIG_DIR to point to temp directory (no config file)
            with patch.object(IntegratedMethodRegistry, 'CONFIG_DIR', Path(temp_dir)):
                method_info = {
                    'class_name': 'TestModel',
                    'module_name': 'test_module',
                    'config_file': 'nonexistent.yaml',
                    'description': 'Test method'
                }

                IntegratedMethodRegistry._register_method('test_method', method_info)

                # Check that method was registered with empty config
                assert 'test_method' in IntegratedMethodRegistry._methods
                metadata = IntegratedMethodRegistry._methods['test_method']

                assert metadata.name == 'test_method'
                assert metadata.hyperparameters == {}
                assert metadata.computational_requirements == {}

    def test_extract_hyperparameters(self):
        """Test hyperparameter extraction from config data."""
        config_data = {
            'model': 'test',
            'param1': 'value1',
            'param2': 42,
            'description': 'Test description',
            'computational_notes': 'Notes',
            'usage_notes': 'Usage',
            'tuning_guidelines': 'Guidelines',
            'compatibility': {'class-il': True}
        }

        hyperparams = IntegratedMethodRegistry._extract_hyperparameters(config_data)

        expected = {'param1': 'value1', 'param2': 42}
        assert hyperparams == expected

    def test_get_available_methods(self):
        """Test getting available method names."""
        IntegratedMethodRegistry._methods = {
            'method1': MagicMock(),
            'method2': MagicMock()
        }
        IntegratedMethodRegistry._initialized = True

        methods = IntegratedMethodRegistry.get_available_methods()
        assert set(methods) == {'method1', 'method2'}

    def test_get_method_metadata(self):
        """Test getting method metadata."""
        mock_metadata = MagicMock()
        IntegratedMethodRegistry._methods = {'test_method': mock_metadata}
        IntegratedMethodRegistry._initialized = True

        result = IntegratedMethodRegistry.get_method_metadata('test_method')
        assert result is mock_metadata

        result = IntegratedMethodRegistry.get_method_metadata('nonexistent')
        assert result is None

    def test_is_integrated_method(self):
        """Test checking if method is integrated."""
        IntegratedMethodRegistry._methods = {'test_method': MagicMock()}
        IntegratedMethodRegistry._initialized = True

        assert IntegratedMethodRegistry.is_integrated_method('test_method') is True
        assert IntegratedMethodRegistry.is_integrated_method('nonexistent') is False

    def test_create_method_success(self):
        """Test successful method creation."""
        # Mock method metadata
        metadata = MethodMetadata(
            name='test_method',
            class_name='TestModel',
            module_name='test_module',
            config_file='test.yaml',
            description='Test'
        )
        IntegratedMethodRegistry._methods = {'test_method': metadata}
        IntegratedMethodRegistry._initialized = True

        # Mock module and class
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.TestModel = mock_class
            mock_import.return_value = mock_module

            # Test method creation
            backbone = MagicMock()
            loss = MagicMock()
            args = MagicMock()

            result = IntegratedMethodRegistry.create_method(
                'test_method', backbone, loss, args
            )

            assert result is mock_instance
            mock_import.assert_called_once_with('models.test_module')
            mock_class.assert_called_once_with(backbone, loss, args, None, None)

    def test_create_method_unknown_method(self):
        """Test creating unknown method raises ValueError."""
        IntegratedMethodRegistry._methods = {}
        IntegratedMethodRegistry._initialized = True

        with pytest.raises(ValueError, match="Unknown integrated method"):
            IntegratedMethodRegistry.create_method('unknown', None, None, None)

    def test_create_method_import_error(self):
        """Test method creation with import error."""
        metadata = MethodMetadata(
            name='test_method',
            class_name='TestModel',
            module_name='nonexistent_module',
            config_file='test.yaml',
            description='Test'
        )
        IntegratedMethodRegistry._methods = {'test_method': metadata}
        IntegratedMethodRegistry._initialized = True

        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError, match="Failed to import method"):
                IntegratedMethodRegistry.create_method('test_method', None, None, None)

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_path = Path(temp_dir) / "test.yaml"
            config_data = {
                'model': 'test',
                'param1': 'value1',
                'param2': 42
            }

            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            # Mock registry
            metadata = MethodMetadata(
                name='test_method',
                class_name='TestModel',
                module_name='test_module',
                config_file='test.yaml',
                description='Test'
            )
            IntegratedMethodRegistry._methods = {'test_method': metadata}
            IntegratedMethodRegistry._initialized = True

            with patch.object(IntegratedMethodRegistry, 'CONFIG_DIR', Path(temp_dir)):
                # Test valid configuration
                config_to_validate = {'param1': 'new_value', 'param2': 100}
                errors = IntegratedMethodRegistry.validate_configuration('test_method', config_to_validate)

                assert errors == []

    def test_validate_configuration_unknown_method(self):
        """Test configuration validation for unknown method."""
        IntegratedMethodRegistry._methods = {}
        IntegratedMethodRegistry._initialized = True

        errors = IntegratedMethodRegistry.validate_configuration('unknown', {})
        assert len(errors) == 1
        assert "Unknown method" in errors[0]

    def test_validate_configuration_type_error(self):
        """Test configuration validation with type errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_path = Path(temp_dir) / "test.yaml"
            config_data = {
                'model': 'test',
                'param1': 'string_value',
                'param2': 42
            }

            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            # Mock registry
            metadata = MethodMetadata(
                name='test_method',
                class_name='TestModel',
                module_name='test_module',
                config_file='test.yaml',
                description='Test'
            )
            IntegratedMethodRegistry._methods = {'test_method': metadata}
            IntegratedMethodRegistry._initialized = True

            with patch.object(IntegratedMethodRegistry, 'CONFIG_DIR', Path(temp_dir)):
                # Test configuration with wrong types
                config_to_validate = {'param1': 123, 'param2': 'wrong_type'}
                errors = IntegratedMethodRegistry.validate_configuration('test_method', config_to_validate)

                assert len(errors) == 2
                assert any("param1 should be str" in error for error in errors)
                assert any("param2 should be int" in error for error in errors)

    def test_generate_documentation_single_method(self):
        """Test documentation generation for single method."""
        metadata = MethodMetadata(
            name='test_method',
            class_name='TestModel',
            module_name='test_module',
            config_file='test.yaml',
            description='Test method description',
            hyperparameters={'param1': 'value1', 'param2': 42},
            compatibility={'class-il': True, 'datasets': ['cifar10', 'cifar100']},
            usage_notes='Test usage notes',
            tuning_guidelines='Test tuning guidelines'
        )
        IntegratedMethodRegistry._methods = {'test_method': metadata}
        IntegratedMethodRegistry._initialized = True

        doc = IntegratedMethodRegistry.generate_documentation('test_method')

        assert 'TEST_METHOD' in doc
        assert 'TestModel' in doc
        assert 'test_module' in doc
        assert 'Test method description' in doc
        assert 'param1' in doc
        assert 'param2' in doc
        assert 'Test usage notes' in doc
        assert 'Test tuning guidelines' in doc

    def test_generate_documentation_all_methods(self):
        """Test documentation generation for all methods."""
        metadata1 = MethodMetadata(
            name='method1',
            class_name='Model1',
            module_name='module1',
            config_file='config1.yaml',
            description='Method 1'
        )
        metadata2 = MethodMetadata(
            name='method2',
            class_name='Model2',
            module_name='module2',
            config_file='config2.yaml',
            description='Method 2'
        )
        IntegratedMethodRegistry._methods = {'method1': metadata1, 'method2': metadata2}
        IntegratedMethodRegistry._initialized = True

        doc = IntegratedMethodRegistry.generate_documentation()

        assert 'METHOD1' in doc
        assert 'METHOD2' in doc
        assert 'Method 1' in doc
        assert 'Method 2' in doc

    def test_get_method_names_for_mammoth(self):
        """Test getting method names in Mammoth format."""
        # Mock method class
        mock_class = MagicMock()
        mock_class.NAME = 'test_method_name'

        metadata = MethodMetadata(
            name='test_method',
            class_name='TestModel',
            module_name='test_module',
            config_file='test.yaml',
            description='Test'
        )
        IntegratedMethodRegistry._methods = {'test_method': metadata}
        IntegratedMethodRegistry._initialized = True

        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.TestModel = mock_class
            mock_import.return_value = mock_module

            result = IntegratedMethodRegistry.get_method_names_for_mammoth()

            assert 'test-method-name' in result
            assert result['test-method-name'] is mock_class


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Reset registry state before each test."""
        IntegratedMethodRegistry._methods = {}
        IntegratedMethodRegistry._initialized = False

    def test_get_integrated_method_names(self):
        """Test get_integrated_method_names function."""
        with patch.object(IntegratedMethodRegistry, 'get_available_methods', return_value=['method1', 'method2']):
            result = get_integrated_method_names()
            assert result == ['method1', 'method2']

    def test_create_integrated_method(self):
        """Test create_integrated_method function."""
        with patch.object(IntegratedMethodRegistry, 'create_method', return_value='mock_instance') as mock_create:
            result = create_integrated_method('test', 'backbone', 'loss', 'args')
            assert result == 'mock_instance'
            mock_create.assert_called_once_with('test', 'backbone', 'loss', 'args', None, None)

    def test_is_integrated_method_function(self):
        """Test is_integrated_method function."""
        with patch.object(IntegratedMethodRegistry, 'is_integrated_method', return_value=True) as mock_check:
            result = is_integrated_method('test')
            assert result is True
            mock_check.assert_called_once_with('test')

    def test_generate_integrated_methods_documentation_function(self):
        """Test generate_integrated_methods_documentation function."""
        with patch.object(IntegratedMethodRegistry, 'generate_documentation', return_value='mock_doc') as mock_gen:
            result = generate_integrated_methods_documentation('test')
            assert result == 'mock_doc'
            mock_gen.assert_called_once_with('test')


class TestMammothIntegration:
    """Test integration with Mammoth model loading."""

    def test_extend_mammoth_model_names(self):
        """Test extending Mammoth model names with integrated methods."""
        existing_names = {'existing_method': 'ExistingClass'}
        integrated_names = {'integrated_method': 'IntegratedClass'}

        with patch.object(IntegratedMethodRegistry, 'get_method_names_for_mammoth', return_value=integrated_names):
            result = extend_mammoth_model_names(existing_names)

            assert 'existing_method' in result
            assert 'integrated_method' in result
            assert result['existing_method'] == 'ExistingClass'
            assert result['integrated_method'] == 'IntegratedClass'

    def test_extend_mammoth_model_names_conflict_resolution(self):
        """Test that integrated methods take precedence in conflicts."""
        existing_names = {'conflicting_method': 'ExistingClass'}
        integrated_names = {'conflicting_method': 'IntegratedClass'}

        with patch.object(IntegratedMethodRegistry, 'get_method_names_for_mammoth', return_value=integrated_names):
            result = extend_mammoth_model_names(existing_names)

            # Integrated method should take precedence
            assert result['conflicting_method'] == 'IntegratedClass'


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def setup_method(self):
        """Reset registry state before each test."""
        IntegratedMethodRegistry._methods = {}
        IntegratedMethodRegistry._initialized = False

    def test_full_integration_workflow(self):
        """Test complete workflow from registration to method creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create realistic config file
            config_path = Path(temp_dir) / "gpm.yaml"
            config_data = {
                'model': 'gpm',
                'gpm_threshold_base': 0.97,
                'gpm_threshold_increment': 0.003,
                'gpm_activation_samples': 512,
                'description': 'GPM method',
                'compatibility': {'class-il': True}
            }

            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            # Mock the registry with realistic data
            with patch.object(IntegratedMethodRegistry, 'CONFIG_DIR', Path(temp_dir)):
                with patch.object(IntegratedMethodRegistry, 'INTEGRATED_METHODS', {
                    'gpm': {
                        'class_name': 'GPMModel',
                        'module_name': 'gpm_model',
                        'config_file': 'gpm.yaml',
                        'description': 'GPM method'
                    }
                }):
                    # Initialize registry
                    IntegratedMethodRegistry.initialize()

                    # Check registration
                    assert 'gpm' in IntegratedMethodRegistry.get_available_methods()

                    # Check metadata
                    metadata = IntegratedMethodRegistry.get_method_metadata('gpm')
                    assert metadata is not None
                    assert metadata.name == 'gpm'
                    assert metadata.class_name == 'GPMModel'
                    assert 'gpm_threshold_base' in metadata.hyperparameters

                    # Test configuration validation
                    valid_config = {'gpm_threshold_base': 0.90}
                    errors = IntegratedMethodRegistry.validate_configuration('gpm', valid_config)
                    assert errors == []

                    # Test documentation generation
                    doc = IntegratedMethodRegistry.generate_documentation('gpm')
                    assert 'gpm' in doc
                    assert 'GPM method' in doc

    def test_error_handling_during_registration(self):
        """Test error handling during method registration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid config file
            config_path = Path(temp_dir) / "invalid.yaml"
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [")

            with patch.object(IntegratedMethodRegistry, 'CONFIG_DIR', Path(temp_dir)):
                with patch.object(IntegratedMethodRegistry, 'INTEGRATED_METHODS', {
                    'invalid_method': {
                        'class_name': 'InvalidModel',
                        'module_name': 'invalid_module',
                        'config_file': 'invalid.yaml',
                        'description': 'Invalid method'
                    }
                }):
                    # Should not raise exception, but should warn
                    with patch('models.integrated_methods_registry.warn_once') as mock_warn:
                        IntegratedMethodRegistry.initialize()

                        # Should still be initialized even with errors
                        assert IntegratedMethodRegistry._initialized is True

                        # Should have registered the method with empty config
                        assert 'invalid_method' in IntegratedMethodRegistry._methods
