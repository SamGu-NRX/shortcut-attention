#!/usr/bin/env python3
"""
Einstellung Integration Setup Script

This script sets up the Einstellung Effect testing framework within the Mammoth
continual learning system. It handles:

1. Environment verification
2. Dataset registration
3. Argument integration
4. Configuration setup
5. End-to-end testing

Usage:
    python setup_einstellung_integration.py [--test] [--cleanup]
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EinstellungSetup:
    """Handles the complete setup of Einstellung integration."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_complete = False
        self.test_results = {}

    def check_environment(self) -> bool:
        """Verify the environment is ready for Einstellung integration."""
        logger.info("Checking environment...")

        checks = {
            "Python version": self._check_python_version(),
            "Required packages": self._check_required_packages(),
            "Mammoth structure": self._check_mammoth_structure(),
            "CUDA availability": self._check_cuda_availability(),
        }

        all_passed = all(checks.values())

        logger.info("Environment check results:")
        for check, passed in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {check}: {status}")

        return all_passed

    def _check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            logger.info(f"  Python {version.major}.{version.minor}.{version.micro} ✓")
            return True
        else:
            logger.error(f"  Python {version.major}.{version.minor}.{version.micro} ✗ (requires 3.8+)")
            return False

    def _check_required_packages(self) -> bool:
        """Check if required packages are installed."""
        required_packages = [
            'torch', 'torchvision', 'numpy', 'PIL', 'argparse', 'logging'
        ]

        optional_packages = [
            'timm'  # For ViT backbones
        ]

        missing_required = []
        missing_optional = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_required.append(package)

        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(package)

        if missing_required:
            logger.error(f"  Missing required packages: {missing_required}")
            return False

        if missing_optional:
            logger.warning(f"  Missing optional packages: {missing_optional}")
            logger.warning("  ViT backbones may not work without timm")

        logger.info("  All required packages available ✓")
        return True

    def _check_mammoth_structure(self) -> bool:
        """Verify Mammoth project structure."""
        required_files = [
            'main.py',
            'utils/args.py',
            'datasets/__init__.py',
            'models/__init__.py',
            'backbone/__init__.py'
        ]

        missing = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing.append(file_path)

        if missing:
            logger.error(f"  Missing Mammoth files: {missing}")
            return False
        else:
            logger.info("  Mammoth structure verified ✓")
            return True

    def _check_cuda_availability(self) -> bool:
        """Check CUDA availability."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                logger.info(f"  CUDA available: {device_count} device(s) ✓")
            else:
                logger.warning("  CUDA not available - will use CPU")
            return True  # Not critical for setup
        except ImportError:
            logger.warning("  PyTorch not available - CUDA check skipped")
            return True

    def verify_integration_files(self) -> bool:
        """Verify all Einstellung integration files are present."""
        logger.info("Verifying integration files...")

        required_files = [
            'datasets/seq_cifar100_einstellung.py',
            'datasets/seq_cifar100_einstellung_224.py',
            'datasets/configs/seq-cifar100/einstellung-224.yaml',
            'utils/args.py',  # Should contain Einstellung arguments
            'run_einstellung_experiment.py',
            'test_einstellung_diagnostics.py'
        ]

        missing = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing.append(file_path)

        if missing:
            logger.error(f"  Missing integration files: {missing}")
            return False
        else:
            logger.info("  All integration files present ✓")
            return True

    def test_dataset_registration(self) -> bool:
        """Test that Einstellung datasets are properly registered."""
        logger.info("Testing dataset registration...")

        try:
            # Test argument registration
            try:
                from utils.args import add_experiment_args

                parser = argparse.ArgumentParser()
                add_experiment_args(parser)
                args = parser.parse_args([])

                # Check for Einstellung arguments
                einstellung_args = [
                    'einstellung_patch_size',
                    'einstellung_patch_color',
                    'einstellung_apply_shortcut',
                    'einstellung_mask_shortcut',
                    'einstellung_evaluation_subsets',
                    'einstellung_extract_attention',
                    'einstellung_adaptation_threshold'
                ]

                missing_args = []
                for arg in einstellung_args:
                    if not hasattr(args, arg):
                        missing_args.append(arg)

                if missing_args:
                    logger.error(f"  Missing Einstellung arguments: {missing_args}")
                    return False

                logger.info("  Einstellung arguments registered ✓")

            except ImportError as e:
                logger.error(f"  Failed to import utils.args: {e}")
                return False

            # Test dataset discovery
            try:
                from datasets import CONTINUAL_DATASET
                einstellung_datasets = [
                    'seq-cifar100-einstellung',
                    'seq-cifar100-einstellung-224'
                ]

                missing_datasets = []
                for dataset_name in einstellung_datasets:
                    if dataset_name not in CONTINUAL_DATASET:
                        missing_datasets.append(dataset_name)

                if missing_datasets:
                    logger.error(f"  Missing datasets: {missing_datasets}")
                    return False

                logger.info("  Einstellung datasets registered ✓")

            except ImportError as e:
                logger.error(f"  Failed to import datasets: {e}")
                return False

            return True

        except Exception as e:
            logger.error(f"  Dataset registration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_dataset_loading(self) -> bool:
        """Test loading Einstellung datasets."""
        logger.info("Testing dataset loading...")

        try:
            from datasets import get_dataset
            from argparse import Namespace

            # Test basic dataset
            args = Namespace(
                dataset='seq-cifar100-einstellung',
                base_path='./data/',
                batch_size=32,
                seed=42,
                custom_class_order=None,
                permute_classes=False,
                custom_task_order=None,
                label_perc=1,
                label_perc_by_class=1,
                validation=None,
                validation_mode='current',
                joint=False,
                start_from=None,
                stop_after=None,
                enable_other_metrics=False,
                eval_future=False,
                noise_rate=0.0,
                drop_last=False,
                num_workers=0,
                # Einstellung parameters
                einstellung_patch_size=4,
                einstellung_patch_color=[255, 0, 255],
                einstellung_apply_shortcut=True,
                einstellung_mask_shortcut=False,
                einstellung_evaluation_subsets=True,
                einstellung_extract_attention=True,
                einstellung_adaptation_threshold=0.8
            )

            dataset = get_dataset(args)
            logger.info(f"  Basic dataset loaded: {dataset.NAME} ✓")

            # Test 224x224 dataset
            args.dataset = 'seq-cifar100-einstellung-224'
            args.einstellung_patch_size = 16  # Larger for 224x224

            dataset_224 = get_dataset(args)
            logger.info(f"  224x224 dataset loaded: {dataset_224.NAME} ✓")

            return True

        except Exception as e:
            logger.error(f"  Dataset loading test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_minimal_experiment(self) -> bool:
        """Run a minimal Einstellung experiment to verify end-to-end functionality."""
        logger.info("Running minimal experiment...")

        try:
            # Use the experiment runner with minimal settings
            cmd = [
                sys.executable, 'run_einstellung_experiment.py',
                '--model', 'derpp',
                '--backbone', 'resnet18',
                '--seed', '42'
            ]

            logger.info(f"  Running: {' '.join(cmd)}")

            # Run with timeout to prevent hanging
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if process.returncode == 0:
                logger.info("  Minimal experiment completed successfully ✓")
                return True
            else:
                logger.error(f"  Experiment failed with return code {process.returncode}")
                logger.error(f"  STDOUT: {process.stdout}")
                logger.error(f"  STDERR: {process.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("  Experiment timed out")
            return False
        except Exception as e:
            logger.error(f"  Experiment failed: {e}")
            return False

    def create_verification_script(self):
        """Create a verification script for future testing."""
        verification_script = """#!/usr/bin/env python3
\"\"\"
Einstellung Integration Verification Script

Run this script to verify the Einstellung integration is working correctly.
\"\"\"

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def verify_integration():
    \"\"\"Verify all aspects of the Einstellung integration.\"\"\"
    print("Verifying Einstellung Integration...")

    # Test imports
    try:
        from datasets import get_dataset
        from utils.args import add_experiment_args
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    # Test dataset registration
    try:
        from datasets import CONTINUAL_DATASET
        required_datasets = ['seq-cifar100-einstellung', 'seq-cifar100-einstellung-224']
        for dataset in required_datasets:
            if dataset not in CONTINUAL_DATASET:
                print(f"✗ Dataset {dataset} not registered")
                return False
        print("✓ Datasets registered")
    except Exception as e:
        print(f"✗ Dataset registration check failed: {e}")
        return False

    # Test argument registration
    try:
        import argparse
        parser = argparse.ArgumentParser()
        add_experiment_args(parser)
        args = parser.parse_args([])

        einstellung_args = [
            'einstellung_patch_size', 'einstellung_patch_color',
            'einstellung_apply_shortcut', 'einstellung_mask_shortcut',
            'einstellung_evaluation_subsets', 'einstellung_extract_attention',
            'einstellung_adaptation_threshold'
        ]

        for arg in einstellung_args:
            if not hasattr(args, arg):
                print(f"✗ Argument {arg} not registered")
                return False
        print("✓ Arguments registered")
    except Exception as e:
        print(f"✗ Argument registration check failed: {e}")
        return False

    print("\\n🎉 All verification checks passed!")
    return True

if __name__ == '__main__':
    success = verify_integration()
    sys.exit(0 if success else 1)
"""

        verification_path = self.project_root / 'verify_einstellung_integration.py'
        with open(verification_path, 'w') as f:
            f.write(verification_script)

        # Make executable
        verification_path.chmod(0o755)
        logger.info(f"  Created verification script: {verification_path}")

    def create_documentation(self):
        """Create documentation for the Einstellung integration."""
        docs = """# Einstellung Integration Documentation

## Overview

The Einstellung Effect integration adds cognitive rigidity testing capabilities to the Mammoth continual learning framework. It implements artificial shortcuts through colored patches to test how well models adapt when shortcuts are removed.

## Quick Start

### Basic Usage

```bash
# Run a simple Einstellung experiment
python run_einstellung_experiment.py --model derpp --backbone resnet18

# Run with ViT backbone (attention analysis)
python run_einstellung_experiment.py --model derpp --backbone vit_base_patch16_224

# Run comparative analysis
python run_einstellung_experiment.py --comparative
```

### Direct Mammoth Usage

```bash
# Use Einstellung datasets directly with Mammoth
python main.py --dataset seq-cifar100-einstellung --model derpp --backbone resnet18 \\
    --einstellung_patch_size 4 --einstellung_apply_shortcut 1

# 224x224 version for ViT
python main.py --dataset seq-cifar100-einstellung-224 --model derpp --backbone vit \\
    --einstellung_patch_size 16 --einstellung_apply_shortcut 1
```

## Einstellung Parameters

- `--einstellung_patch_size`: Size of shortcut patches (4 for 32x32, 16 for 224x224)
- `--einstellung_patch_color`: RGB color of patches (default: 255 0 255 magenta)
- `--einstellung_apply_shortcut`: Enable shortcut injection (0/1)
- `--einstellung_mask_shortcut`: Mask shortcuts instead of applying them (0/1)
- `--einstellung_evaluation_subsets`: Enable multi-subset evaluation (0/1)
- `--einstellung_extract_attention`: Extract attention maps for ViT (0/1)
- `--einstellung_adaptation_threshold`: Accuracy threshold for adaptation delay (default: 0.8)

## Dataset Structure

### Task Organization
- **Task 1**: 8 superclasses (40 classes) - learned normally
- **Task 2**: 4 superclasses (20 classes) - with artificial shortcuts

### Evaluation Subsets
- `T1_all`: All Task 1 data (normal evaluation)
- `T2_shortcut_normal`: Task 2 shortcut classes with shortcuts
- `T2_shortcut_masked`: Task 2 shortcut classes with shortcuts masked
- `T2_nonshortcut_normal`: Task 2 non-shortcut classes

## Metrics

The integration provides several cognitive rigidity metrics:

1. **ERI (Einstellung Rigidity Index)**: Measures how much performance drops when shortcuts are removed
2. **Performance Deficit**: Accuracy difference between shortcut and non-shortcut conditions
3. **Adaptation Delay**: Time required to reach threshold accuracy after shortcut removal

## Verification

Run the verification script to ensure everything is working:

```bash
python verify_einstellung_integration.py
```

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure datasets are properly registered in `datasets/__init__.py`
2. **Arguments not recognized**: Check that Einstellung arguments are added to `utils/args.py`
3. **CUDA out of memory**: Reduce batch size or use CPU
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

Enable verbose logging for debugging:

```bash
python run_einstellung_experiment.py --model derpp --backbone resnet18 --verbose
```

## Research Applications

This integration enables research into:

- Cognitive rigidity in neural networks
- Shortcut learning and generalization
- Attention mechanisms and interpretability
- Continual learning robustness
- Model adaptation capabilities

## Citation

If you use this integration in your research, please cite the original Einstellung Effect paper and the Mammoth framework.
"""

        docs_path = self.project_root / 'EINSTELLUNG_INTEGRATION_GUIDE.md'
        with open(docs_path, 'w') as f:
            f.write(docs)

        logger.info(f"  Created documentation: {docs_path}")

    def cleanup_diagnostic_files(self):
        """Remove diagnostic files after successful setup."""
        diagnostic_files = [
            'test_einstellung_diagnostics.py',
            'test_einstellung_basic.py'
        ]

        for file_name in diagnostic_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                file_path.unlink()
                logger.info(f"  Removed diagnostic file: {file_name}")

    def run_setup(self, test_mode: bool = False, cleanup: bool = False) -> bool:
        """Run the complete setup process."""
        logger.info("=" * 60)
        logger.info("EINSTELLUNG INTEGRATION SETUP")
        logger.info("=" * 60)

        # Step 1: Environment check
        if not self.check_environment():
            logger.error("Environment check failed. Please fix issues before continuing.")
            return False

        # Step 2: Verify integration files
        if not self.verify_integration_files():
            logger.error("Integration files missing. Please ensure all files are present.")
            return False

        # Step 3: Test dataset registration
        if not self.test_dataset_registration():
            logger.error("Dataset registration failed.")
            return False

        # Step 4: Test dataset loading
        if not self.test_dataset_loading():
            logger.error("Dataset loading failed.")
            return False

        # Step 5: Run minimal experiment (if not in test mode)
        if not test_mode:
            if not self.run_minimal_experiment():
                logger.error("Minimal experiment failed.")
                return False

        # Step 6: Create verification and documentation
        self.create_verification_script()
        self.create_documentation()

        # Step 7: Cleanup (if requested)
        if cleanup:
            self.cleanup_diagnostic_files()

        self.setup_complete = True

        logger.info("=" * 60)
        logger.info("SETUP COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Run verification: python verify_einstellung_integration.py")
        logger.info("2. Read documentation: EINSTELLUNG_INTEGRATION_GUIDE.md")
        logger.info("3. Start experimenting: python run_einstellung_experiment.py --help")

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Einstellung Integration Setup')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (skip experiment execution)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove diagnostic files after setup')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup = EinstellungSetup()
    success = setup.run_setup(test_mode=args.test, cleanup=args.cleanup)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
