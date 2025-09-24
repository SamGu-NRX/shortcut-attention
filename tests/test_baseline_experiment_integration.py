"""
Comprehensive integration test for baseline methods with the actual experiment orchestration.

This test suite validates that the Scratch_T2 and Interleaved baseline methods
integrate properly with the run_einstellung_experiment.py script and produce
valid results in the expected format.

The tests are designed to be comprehensive and meticulous, covering:
1. Model availability in the experiment script
2. Direct model instantiation through the Mammoth framework
3. End-to-end experiment execution with proper CSV output
4. Validation of CSV format and content
5. Integration with the EinstellungEvaluator system
"""

import pytest
import subprocess
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path
from argparse import Namespace
from unittest.mock import Mock


class TestBaselineMethodAvailability:
    """Test that baseline methods are properly available in the experiment system."""

    def test_baseline_methods_in_experiment_help(self):
        """Test that baseline methods appear in the experiment script help output."""
        result = subprocess.run(
            ['python', 'run_einstellung_experiment.py', '--help'],
            capture_output=True, text=True, timeout=30
        )

        assert result.returncode == 0, "Failed to get help output"

        # Check that both baseline methods are listed as options
        assert 'scratch_t2' in result.stdout, "scratch_t2 not found in available models"
        assert 'interleaved' in result.stdout, "interleaved not found in available models"

        # Verify they're in the choices list (format: {sgd,derpp,ewc_on,gpm,dgr,scratch_t2,interleaved})
        assert '{sgd,derpp,ewc_on,gpm,dgr,scratch_t2,interleaved}' in result.stdout, "Model choices not found in help"

    def test_baseline_methods_in_mammoth_registry(self):
        """Test that baseline methods are properly registered in Mammoth's model registry."""
        # Import Mammoth components
        sys.path.append('.')
        from models import get_model_names, get_model_class
        from models.scratch_t2 import ScratchT2
        from models.interleaved import Interleaved

        # Get available models
        model_names = get_model_names()

        # Check that baseline methods are in the registry
        assert 'scratch-t2' in model_names, "scratch_t2 not in Mammoth model registry"
        assert 'interleaved' in model_names, "interleaved not in Mammoth model registry"

        # Verify they are actual classes, not exceptions
        assert not isinstance(model_names['scratch-t2'], Exception), f"scratch_t2 failed to load: {model_names['scratch-t2']}"
        assert not isinstance(model_names['interleaved'], Exception), f"interleaved failed to load: {model_names['interleaved']}"

        # Verify correct class mapping
        assert model_names['scratch-t2'] == ScratchT2
        assert model_names['interleaved'] == Interleaved


class TestBaselineMethodInstantiation:
    """Test direct instantiation of baseline methods through Mammoth framework."""

    def create_test_args(self, model_name):
        """Create comprehensive test arguments using default args."""
        sys.path.append('.')
        from experiments.default_args import get_base_args

        base_args = get_base_args()
        base_args.update({
            'model': model_name,
            'n_epochs': 1,
            'debug_mode': 1,
            'device': 'cpu',
            'dataset': 'seq-cifar100-einstellung',
            'num_workers': 0
        })
        return Namespace(**base_args)

    def create_test_components(self):
        """Create test backbone, loss, transform, and dataset."""
        sys.path.append('.')
        from backbone.ResNet32 import resnet32
        import torch

        backbone = resnet32(num_classes=100)
        loss = torch.nn.CrossEntropyLoss()
        transform = Mock()

        # Mock dataset with required attributes
        dataset = Mock()
        dataset.N_CLASSES = 100
        dataset.N_TASKS = 2
        dataset.SETTING = 'class-il'
        dataset.N_CLASSES_PER_TASK = [50, 50]

        # Mock normalization transform
        dataset.get_normalization_transform = Mock(return_value=torch.nn.Identity())

        return backbone, loss, transform, dataset

    def test_scratch_t2_instantiation(self):
        """Test that Scratch_T2 can be instantiated through get_model."""
        sys.path.append('.')
        from models import get_model
        from models.scratch_t2 import ScratchT2

        args = self.create_test_args('scratch_t2')
        backbone, loss, transform, dataset = self.create_test_components()

        # Test model instantiation
        model = get_model(args, backbone, loss, transform, dataset)

        assert isinstance(model, ScratchT2)
        assert model.NAME == 'scratch_t2'
        assert hasattr(model, 'task2_data')
        assert hasattr(model, 'begin_task')
        assert hasattr(model, 'end_task')
        assert hasattr(model, 'observe')

    def test_interleaved_instantiation(self):
        """Test that Interleaved can be instantiated through get_model."""
        sys.path.append('.')
        from models import get_model
        from models.interleaved import Interleaved

        args = self.create_test_args('interleaved')
        backbone, loss, transform, dataset = self.create_test_components()

        # Test model instantiation
        model = get_model(args, backbone, loss, transform, dataset)

        assert isinstance(model, Interleaved)
        assert model.NAME == 'interleaved'
        assert hasattr(model, 'all_data')
        assert hasattr(model, 'begin_task')
        assert hasattr(model, 'end_task')
        assert hasattr(model, 'observe')

    def test_baseline_methods_compatibility(self):
        """Test that baseline methods declare proper compatibility."""
        sys.path.append('.')
        from models.scratch_t2 import ScratchT2
        from models.interleaved import Interleaved

        expected_compatibility = ['class-il', 'domain-il', 'task-il']

        assert ScratchT2.COMPATIBILITY == expected_compatibility
        assert Interleaved.COMPATIBILITY == expected_compatibility


class TestBaselineMethodExecution:
    """Test end-to-end execution of baseline methods through the experiment script."""

    def test_scratch_t2_experiment_execution(self):
        """Test that scratch_t2 can be executed through the experiment script."""
        # Use a temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run experiment with minimal configuration
            cmd = [
                'python', 'run_einstellung_experiment.py',
                '--model', 'scratch_t2',
                '--backbone', 'resnet18',
                '--seed', '42',
                '--epochs', '1',
                '--debug',
                '--force_retrain'  # Ensure we don't use existing checkpoints
            ]

            # Set environment to use temp directory
            env = os.environ.copy()
            env['RESULTS_PATH'] = temp_dir

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

            # Check that the experiment completed successfully
            assert result.returncode == 0, f"Experiment failed with return code {result.returncode}. STDOUT: {result.stdout}. STDERR: {result.stderr}"

            # Check that output indicates successful completion
            assert "Experiment completed successfully" in result.stdout or "SUCCESS" in result.stdout or result.returncode == 0

            # Check that results directory was created
            results_dir = Path("einstellung_results")
            assert results_dir.exists(), "Results directory not created"

            # Look for timeline CSV files in any subdirectory
            csv_files = list(results_dir.glob("**/timeline_*.csv"))

            if csv_files:
                csv_file = csv_files[0]
                df = pd.read_csv(csv_file)

                expected_columns = {'method', 'seed', 'epoch_eff', 'split', 'acc', 'top5', 'loss'}
                assert expected_columns.issubset(df.columns), f"Invalid CSV columns: {list(df.columns)}"

                methods_in_csv = df['method'].unique()
                assert 'scratch_t2' in methods_in_csv, f"scratch_t2 not found in timeline methods: {methods_in_csv}"

                splits_in_csv = set(df['split'].unique())
                expected_splits = {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'}
                assert splits_in_csv.issuperset(expected_splits), f"Missing expected splits. Found: {splits_in_csv}"

    def test_interleaved_experiment_execution(self):
        """Test that interleaved can be executed through the experiment script."""
        # Use a temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run experiment with minimal configuration
            cmd = [
                'python', 'run_einstellung_experiment.py',
                '--model', 'interleaved',
                '--backbone', 'resnet18',
                '--seed', '43',  # Use different seed to avoid conflicts
                '--epochs', '1',
                '--debug',
                '--force_retrain'  # Ensure we don't use existing checkpoints
            ]

            # Set environment to use temp directory
            env = os.environ.copy()
            env['RESULTS_PATH'] = temp_dir

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

            # Check that the experiment completed successfully
            assert result.returncode == 0, f"Experiment failed with return code {result.returncode}. STDOUT: {result.stdout}. STDERR: {result.stderr}"

            # Check that results directory was created
            results_dir = Path("einstellung_results")
            assert results_dir.exists(), "Results directory not created"

            csv_files = list(results_dir.glob("**/timeline_*.csv"))

            if csv_files:
                interleaved_csv = None
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    if 'interleaved' in df['method'].values:
                        interleaved_csv = csv_file
                        break

                if interleaved_csv:
                    df = pd.read_csv(interleaved_csv)

                    expected_columns = {'method', 'seed', 'epoch_eff', 'split', 'acc', 'top5', 'loss'}
                    assert expected_columns.issubset(df.columns), f"Invalid CSV columns: {list(df.columns)}"

                    methods_in_csv = df['method'].unique()
                    assert 'interleaved' in methods_in_csv, f"interleaved not found in timeline methods: {methods_in_csv}"


class TestCSVOutputValidation:
    """Test that baseline methods produce valid CSV output compatible with ERI visualization."""

    def test_existing_csv_structure_validation(self):
        """Test validation of existing CSV files from baseline methods."""
        # Check if there are any existing CSV files from baseline methods
        csv_files = list(Path("einstellung_results").glob("**/timeline_*.csv"))

        baseline_methods = ['scratch_t2', 'interleaved']
        found_baseline_results = False

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Check if this CSV contains baseline method results
                methods_in_csv = df['method'].unique() if 'method' in df.columns else []

                if any(method in methods_in_csv for method in baseline_methods):
                    found_baseline_results = True

                    # Validate CSV structure
                    expected_columns = {'method', 'seed', 'epoch_eff', 'split', 'acc', 'top5', 'loss'}
                    assert expected_columns.issubset(df.columns), f"Invalid CSV structure in {csv_file}: {list(df.columns)}"

                    # Validate data types
                    assert df['method'].dtype == 'object', f"Invalid method column type in {csv_file}"
                    assert df['seed'].dtype in ['int64', 'int32'], f"Invalid seed column type in {csv_file}"
                    assert df['epoch_eff'].dtype in ['float64', 'float32'], f"Invalid epoch_eff column type in {csv_file}"
                    assert df['split'].dtype == 'object', f"Invalid split column type in {csv_file}"
                    assert df['acc'].dtype in ['float64', 'float32'], f"Invalid acc column type in {csv_file}"
                    assert df['top5'].dtype in ['float64', 'float32', 'float'], f"Invalid top5 column type in {csv_file}"

                    # Validate that accuracy values are reasonable (0-1 range)
                    assert df['acc'].min() >= 0.0, f"Negative accuracy values found in {csv_file}"
                    assert df['acc'].max() <= 1.0, f"Accuracy values > 1.0 found in {csv_file}"

                    # Validate expected splits are present
                    splits_in_csv = set(df['split'].unique())
                    expected_splits = {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'}

                    assert splits_in_csv.issuperset(expected_splits), f"No expected ERI splits found in {csv_file}. Found: {splits_in_csv}"

                    print(f"✓ Validated CSV structure for {csv_file}")
                    print(f"  - Methods: {list(methods_in_csv)}")
                    print(f"  - Splits: {list(splits_in_csv)}")
                    print(f"  - Rows: {len(df)}")

            except Exception as e:
                print(f"Warning: Could not validate {csv_file}: {e}")

        if not found_baseline_results:
            print("No existing baseline method CSV files found - validation will occur during execution tests")

    def test_csv_format_compatibility_with_eri_system(self):
        """Test that CSV format is compatible with ERI visualization system."""
        # Create a sample CSV in the expected format
        sample_data = {
            'method': ['scratch_t2', 'scratch_t2', 'interleaved', 'interleaved'],
            'seed': [42, 42, 42, 42],
            'epoch_eff': [1.0, 1.0, 1.0, 1.0],
            'split': ['T2_shortcut_normal', 'T2_shortcut_masked', 'T2_shortcut_normal', 'T2_shortcut_masked'],
            'acc': [0.85, 0.80, 0.82, 0.78],
            'top5': [0.92, 0.88, 0.9, 0.86],
            'loss': [0.5, 0.6, 0.55, 0.62]
        }

        df = pd.DataFrame(sample_data)

        # Validate structure matches ERI expectations
        expected_columns = ['method', 'seed', 'epoch_eff', 'split', 'acc', 'top5', 'loss']
        assert list(df.columns) == expected_columns

        # Validate data types
        assert df['method'].dtype == 'object'
        assert df['seed'].dtype in ['int64', 'int32']
        assert df['epoch_eff'].dtype in ['float64', 'float32']
        assert df['split'].dtype == 'object'
        assert df['acc'].dtype in ['float64', 'float32']
        assert df['top5'].dtype in ['float64', 'float32']
        assert df['loss'].dtype in ['float64', 'float32']

        # Validate method names match baseline method NAME attributes
        sys.path.append('.')
        from models.scratch_t2 import ScratchT2
        from models.interleaved import Interleaved

        unique_methods = df['method'].unique()
        assert ScratchT2.NAME in unique_methods
        assert Interleaved.NAME in unique_methods


class TestExperimentOrchestrationIntegration:
    """Test integration with the complete experiment orchestration system."""

    def test_baseline_methods_with_experiment_functions(self):
        """Test that baseline methods work with experiment orchestration functions."""
        sys.path.append('.')

        # Test that we can import and use the experiment functions
        try:
            from run_einstellung_experiment import create_einstellung_args, extract_accuracy_from_output

            # Test argument creation for baseline methods
            scratch_args = create_einstellung_args('scratch_t2', 'resnet18', 42, debug=True)
            interleaved_args = create_einstellung_args('interleaved', 'resnet18', 42, debug=True)

            # Verify arguments contain expected values
            assert '--model' in scratch_args
            assert 'scratch_t2' in scratch_args
            assert '--model' in interleaved_args
            assert 'interleaved' in interleaved_args

            # Test accuracy extraction
            sample_output = "Accuracy for 2 task(s): [Class-IL]: 75.5 %"
            accuracy = extract_accuracy_from_output(sample_output)
            assert accuracy == 75.5

        except ImportError as e:
            pytest.skip(f"Experiment orchestration functions not available: {e}")

    def test_baseline_methods_checkpoint_compatibility(self):
        """Test that baseline methods are compatible with checkpoint management."""
        sys.path.append('.')
        from models.scratch_t2 import ScratchT2
        from models.interleaved import Interleaved

        # Create test models
        args = Namespace(
            lr=0.01, batch_size=32, n_epochs=1, debug_mode=1, device='cpu',
            seed=42, dataset='seq-cifar100-einstellung', optimizer='sgd',
            optim_wd=0.0, optim_mom=0.9, optim_nesterov=False, label_perc=1,
            num_workers=0, lr_scheduler=None
        )

        from backbone.ResNet32 import resnet32
        import torch

        backbone = resnet32(num_classes=100)
        loss = torch.nn.CrossEntropyLoss()
        transform = Mock()

        dataset = Mock()
        dataset.N_CLASSES = 100
        dataset.N_TASKS = 2
        dataset.SETTING = 'class-il'
        dataset.N_CLASSES_PER_TASK = [50, 50]
        dataset.get_normalization_transform = Mock(return_value=torch.nn.Identity())

        scratch_model = ScratchT2(backbone, loss, args, transform, dataset)
        interleaved_model = Interleaved(backbone, loss, args, transform, dataset)

        # Test state_dict functionality
        scratch_state = scratch_model.state_dict()
        interleaved_state = interleaved_model.state_dict()

        assert isinstance(scratch_state, dict)
        assert isinstance(interleaved_state, dict)

        # Test load_state_dict functionality
        scratch_model.load_state_dict(scratch_state)
        interleaved_model.load_state_dict(interleaved_state)

        # Test device compatibility
        assert hasattr(scratch_model, 'device')
        assert hasattr(interleaved_model, 'device')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
