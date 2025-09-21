#!/usr/bin/env python3
"""
Integration Test Runner for ERI Visualization System

This script runs comprehensive integration tests for adapted continual learning methods
integrated with the Mammoth framework and ERI evaluation system.

Usage:
    python tests/integration/run_integration_tests.py [--quick] [--method METHOD] [--verbose]
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def check_dependencies():
    """Check that required dependencies are available."""
    required_modules = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'yaml',
        'pytest'
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        logging.error(f"Missing required modules: {missing_modules}")
        logging.error("Please install missing dependencies before running tests")
        return False

    return True

def run_pytest(test_file, args):
    """Run pytest with specified arguments."""
    cmd = ['python', '-m', 'pytest', test_file, '-v']

    if args.quick:
        cmd.extend(['-k', 'not test_end_to_end_pipeline_minimal'])

    if args.method:
        cmd.extend(['-k', f'test_{args.method}'])

    if args.verbose:
        cmd.append('--tb=long')
    else:
        cmd.append('--tb=short')

    # Add coverage if available
    try:
        import coverage
        cmd.extend(['--cov=models', '--cov=eri_vis', '--cov-report=term-missing'])
    except ImportError:
        pass

    logging.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logging.error("Tests timed out after 30 minutes")
        return False
    except Exception as e:
        logging.error(f"Error running tests: {e}")
        return False

def validate_test_environment():
    """Validate that the test environment is properly set up."""
    project_root = Path(__file__).parent.parent.parent

    # Check that required directories exist
    required_dirs = [
        'models',
        'models/config',
        'utils',
        'datasets',
        'backbone'
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            logging.error(f"Required directory not found: {dir_path}")
            return False

    # Check that required config files exist
    config_dir = project_root / 'models' / 'config'
    required_configs = ['gpm.yaml', 'dgr.yaml']

    for config_file in required_configs:
        config_path = config_dir / config_file
        if not config_path.exists():
            logging.error(f"Required config file not found: {config_path}")
            return False

    # Check that integrated methods registry exists
    registry_path = project_root / 'models' / 'integrated_methods_registry.py'
    if not registry_path.exists():
        logging.error(f"Integrated methods registry not found: {registry_path}")
        return False

    logging.info("Test environment validation passed")
    return True

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run ERI integration tests')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only (skip long-running tests)')
    parser.add_argument('--method', type=str,
                       help='Run tests for specific method only')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate environment, do not run tests')

    args = parser.parse_args()

    setup_logging(args.verbose)

    logging.info("Starting ERI Integration Tests")
    logging.info(f"Quick mode: {args.quick}")
    logging.info(f"Method filter: {args.method}")
    logging.info(f"Verbose: {args.verbose}")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Validate test environment
    if not validate_test_environment():
        sys.exit(1)

    if args.validate_only:
        logging.info("Environment validation completed successfully")
        sys.exit(0)

    # Run tests
    test_file = Path(__file__).parent / 'test_integrated_methods.py'

    if not test_file.exists():
        logging.error(f"Test file not found: {test_file}")
        sys.exit(1)

    success = run_pytest(str(test_file), args)

    if success:
        logging.info("All integration tests passed!")
        sys.exit(0)
    else:
        logging.error("Some integration tests failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
