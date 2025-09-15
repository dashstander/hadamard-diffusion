#!/usr/bin/env python3
"""
Quick test script to verify the conversion approach works on a small subset.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path to import our utilities
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hadamard_diffusion.data import load_all_hex_hadamard_matrices

def test_single_file():
    """Test loading a single file to verify the approach."""

    # Test with the first file in res0
    test_file = Path("data/raw/res0/dgn_0_12_30_0.txt")

    print(f"Testing with file: {test_file}")

    if not test_file.exists():
        print(f"File {test_file} does not exist!")
        return

    try:
        matrices = load_all_hex_hadamard_matrices(test_file)
        print(f"Successfully loaded {len(matrices)} matrices")

        if matrices:
            # Check the first matrix
            matrix_idx, matrix = matrices[0]
            print(f"First matrix index: {matrix_idx}")
            print(f"First matrix shape: {matrix.shape}")
            print(f"First matrix dtype: {matrix.dtype}")

            # Quick verification
            n = matrix.shape[0]
            product = matrix @ matrix.T
            expected = n * np.eye(n)
            is_valid = np.allclose(product, expected)
            print(f"First matrix is valid Hadamard: {is_valid}")

            # Stack a few matrices to test the array creation
            if len(matrices) >= 5:
                test_matrices = [m for _, m in matrices[:5]]
                stacked = np.stack(test_matrices, axis=0)
                print(f"Successfully stacked 5 matrices: {stacked.shape}")

                # Save as test
                output_dir = Path("data/numpy")
                output_dir.mkdir(exist_ok=True)
                test_file_path = output_dir / "test_5_matrices.npy"
                np.save(test_file_path, stacked)
                print(f"Saved test file: {test_file_path}")

                # Verify saved file
                loaded = np.load(test_file_path)
                print(f"Verified saved file shape: {loaded.shape}")

        else:
            print("No matrices loaded!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_file()