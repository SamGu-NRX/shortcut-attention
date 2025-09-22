#!/usr/bin/env python3
"""
Demo script for running the comprehensive integration test.

This script demonstrates Task 16: Comprehensive Integration Testing
by running the complete test suite and providing a summary of results.
"""

import subprocess
import sys
from pathlib import Path

def run_comprehensive_integration_test():
    """Run the comprehensive integration test and report results."""

    print("🧪 Running Comprehensive Integration Test for Comparative Einstellung Analysis")
    print("=" * 80)
    print()

    print("This test validates the complete comparative analysis pipeline:")
    print("✓ Baseline method integration (Scratch_T2, Interleaved)")
    print("✓ Experiment orchestration and argument handling")
    print("✓ Data aggregation and CSV processing")
    print("✓ Statistical analysis integration")
    print("✓ Visualization pipeline integration")
    print("✓ Error handling and robustness")
    print("✓ Checkpoint management")
    print("✓ Backward compatibility")
    print("✓ End-to-end pipeline simulation")
    print("✓ Performance and memory efficiency")
    print("✓ Comparative experiment runner")
    print()

    # Run the comprehensive integration test
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_comprehensive_integration.py",
        "-v", "--tb=short"
    ]

    print("Running command:", " ".join(cmd))
    print("-" * 80)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("🎉 SUCCESS: All comprehensive integration tests passed!")
            print()
            print("The comparative Einstellung analysis system is fully integrated and working correctly.")
            print("Key validated components:")
            print("  • Baseline methods (Scratch_T2, Interleaved) are properly registered")
            print("  • Experiment orchestration handles all method types")
            print("  • Data aggregation merges CSV files correctly")
            print("  • Statistical analysis processes comparative data")
            print("  • Visualization pipeline generates comparative plots")
            print("  • Error handling is robust across all components")
            print("  • Backward compatibility is maintained")
            print("  • End-to-end pipeline works seamlessly")

            return True
        else:
            print("❌ FAILURE: Some integration tests failed!")
            print(f"Exit code: {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ ERROR: Failed to run integration tests: {e}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    sys.exit(0 if success else 1)
