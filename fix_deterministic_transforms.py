#!/usr/bin/env python3
"""
Fix for deterministic transform application in Einstellung datasets.

This script fixes the inconsistent transform application that causes
cross-method inconsistencies in comparative experiments.
"""

import os
import sys
import re

def fix_transform_consistency():
    """Fix all inconsistent transform applications in Einstellung datasets."""

    files_to_fix = [
        'datasets/seq_cifar100_einstellung.py',
        'datasets/seq_cifar100_einstellung_224.py'
    ]

    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            continue

        print(f"Fixing transforms in {file_path}...")

        with open(file_path, 'r') as f:
            content = f.read()

        # Pattern to find inconsistent transform usage
        # Look for: img = self.transform(img)
        # Replace with: img = apply_deterministic_transform(self.transform, img, index)

        # This pattern matches the problematic lines but ensures we're in a __getitem__ method
        pattern = r'(\s+)if self\.transform is not None:\s*\n\s+img = self\.transform\(img\)'

        def replacement(match):
            indent = match.group(1)
            return f'{indent}if self.transform is not None:\n{indent}    img = apply_deterministic_transform(self.transform, img, index)'

        # Apply the fix
        new_content = re.sub(pattern, replacement, content)

        # Count changes
        changes = len(re.findall(pattern, content))

        if changes > 0:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"  Fixed {changes} inconsistent transform applications")
        else:
            print(f"  No inconsistent transforms found")

def verify_deterministic_imports():
    """Verify that necessary imports are present."""

    files_to_check = [
        'datasets/seq_cifar100_einstellung.py',
        'datasets/seq_cifar100_einstellung_224.py'
    ]

    required_imports = ['torch', 'random', 'numpy as np']

    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r') as f:
            content = f.read()

        missing_imports = []
        for imp in required_imports:
            if f'import {imp}' not in content:
                missing_imports.append(imp)

        if missing_imports:
            print(f"Warning: {file_path} missing imports: {missing_imports}")
        else:
            print(f"✓ {file_path} has all required imports")

def main():
    """Main function."""
    print("🔧 Fixing deterministic transform consistency...")
    print("=" * 60)

    # Verify imports first
    print("Checking imports...")
    verify_deterministic_imports()

    print("\nFixing transform applications...")
    fix_transform_consistency()

    print("\n✅ Transform consistency fixes applied!")
    print("Now run the cross-method validation test to verify the fixes.")

if __name__ == '__main__':
    main()
