#!/usr/bin/env python3
"""
Script to convert all Hadamard matrices from hex format to numpy arrays.

This script processes all subdirectories in data/raw/, loads the hex-encoded
Hadamard matrices, and saves them as numpy arrays with shape (250000, 32, 32)
in data/numpy/.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path to import our utilities
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hadamard_diffusion.data import load_all_hex_hadamard_matrices

def process_matrices_in_batches(raw_data_dir, batch_size=250000, output_dir=None):
    """
    Process matrices in batches, saving each batch to disk and clearing memory.

    Args:
        raw_data_dir: Path to data/raw directory
        batch_size: Number of matrices per batch
        output_dir: Directory to save batch files

    Returns:
        int: Total number of batches created
    """
    if output_dir is None:
        output_dir = Path(raw_data_dir).parent / "numpy"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_batch = []
    batch_number = 1
    total_processed = 0

    raw_path = Path(raw_data_dir)
    subdirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])

    print(f"Found {len(subdirs)} subdirectories to process")
    print(f"Batch size: {batch_size} matrices per file")
    print(f"Output directory: {output_dir}")

    def save_current_batch():
        nonlocal current_batch, batch_number, total_processed
        if current_batch:
            batch_file = output_dir / f"hadamard_matrices_batch_{batch_number:03d}.npy"
            stacked = np.stack(current_batch, axis=0)

            print(f"\n  💾 Saving batch {batch_number}: {stacked.shape} -> {batch_file}")
            np.save(batch_file, stacked)

            # Verify saved file
            loaded = np.load(batch_file)
            print(f"     Verified: {loaded.shape}, {loaded.nbytes / 1024**2:.1f} MB")

            batch_number += 1
            total_processed += len(current_batch)
            current_batch = []  # Clear memory

    for subdir in subdirs:
        print(f"\nProcessing {subdir.name}...")

        # Get all .txt files in this subdirectory
        txt_files = sorted(subdir.glob("*.txt"))
        print(f"  Found {len(txt_files)} text files")

        for txt_file in txt_files:
            print(f"  Loading {txt_file.name}...")
            try:
                matrices = load_all_hex_hadamard_matrices(txt_file)

                # Extract just the matrix arrays (ignore indices)
                matrix_arrays = [matrix for _, matrix in matrices]

                if matrix_arrays:
                    print(f"    Loaded {len(matrix_arrays)} matrices")

                    # Add matrices to current batch, saving when full
                    for matrix in matrix_arrays:
                        current_batch.append(matrix)

                        if len(current_batch) >= batch_size:
                            save_current_batch()
                            print(f"    Total processed so far: {total_processed}")

                else:
                    print(f"    No valid matrices found in {txt_file.name}")

            except Exception as e:
                print(f"    Error processing {txt_file.name}: {e}")

    # Save any remaining matrices in the final batch
    if current_batch:
        print(f"\nSaving final partial batch with {len(current_batch)} matrices...")
        save_current_batch()

    total_batches = batch_number - 1
    print(f"\n🎉 Processing complete!")
    print(f"   Total matrices processed: {total_processed}")
    print(f"   Total batches created: {total_batches}")
    print(f"   Output files: hadamard_matrices_batch_001.npy to hadamard_matrices_batch_{total_batches:03d}.npy")

    return total_batches

def pad_or_truncate_to_target(matrices, target_count=250000):
    """
    Pad with duplicates or truncate to reach exactly target_count matrices.

    Args:
        matrices: numpy array of shape (n_matrices, 32, 32)
        target_count: desired number of matrices

    Returns:
        numpy array: shape (target_count, 32, 32)
    """
    current_count = matrices.shape[0]

    if current_count == target_count:
        print(f"Already have exactly {target_count} matrices")
        return matrices

    elif current_count > target_count:
        print(f"Truncating from {current_count} to {target_count} matrices")
        return matrices[:target_count]

    else:
        print(f"Padding from {current_count} to {target_count} matrices by duplicating")

        # Calculate how many times to repeat the dataset
        repeats_needed = target_count // current_count
        remainder = target_count % current_count

        # Repeat the full dataset
        repeated = np.tile(matrices, (repeats_needed, 1, 1))

        # Add partial repeat for remainder
        if remainder > 0:
            partial = matrices[:remainder]
            result = np.concatenate([repeated, partial], axis=0)
        else:
            result = repeated

        print(f"Used {repeats_needed} full repeats + {remainder} partial matrices")
        return result

def main():
    """Main conversion function."""

    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    raw_data_dir = project_root / "data" / "raw"
    numpy_output_dir = project_root / "data" / "numpy"

    print(f"Processing matrices from: {raw_data_dir}")
    print(f"Output will be saved to: {numpy_output_dir}")

    # Process matrices in batches
    print("\n" + "="*70)
    print("PROCESSING MATRICES IN BATCHES OF 250,000")
    print("="*70)

    total_batches = process_matrices_in_batches(
        raw_data_dir=raw_data_dir,
        batch_size=250000,
        output_dir=numpy_output_dir
    )

    # Quick verification on the first batch
    print("\n" + "="*50)
    print("VERIFICATION")
    print("="*50)

    first_batch_file = numpy_output_dir / "hadamard_matrices_batch_001.npy"
    if first_batch_file.exists():
        print(f"Verifying first batch: {first_batch_file}")
        loaded = np.load(first_batch_file)
        print(f"  Shape: {loaded.shape}")
        print(f"  Data type: {loaded.dtype}")
        print(f"  Memory: {loaded.nbytes / 1024**2:.1f} MB")

        # Quick verification on a few matrices
        print(f"  Quick validation (checking first 3 matrices):")
        for i in range(min(3, loaded.shape[0])):
            matrix = loaded[i]
            # Check if it's a valid Hadamard matrix: H @ H.T = n * I
            n = matrix.shape[0]
            product = matrix @ matrix.T
            expected = n * np.eye(n)
            is_valid = np.allclose(product, expected)
            print(f"    Matrix {i}: Valid Hadamard = {is_valid}")

    print(f"\n🎉 All done!")
    print(f"   Created {total_batches} batch files")
    print(f"   Each file contains up to 250,000 matrices")
    print(f"   Files: hadamard_matrices_batch_001.npy to hadamard_matrices_batch_{total_batches:03d}.npy")
    print(f"   Ready for training!")

if __name__ == "__main__":
    main()