import numpy as np
import os
from pathlib import Path

def hex_to_hadamard_entries(hex_char):
    """
    Convert a single hex character to 4 Hadamard matrix entries.
    
    Each hex digit represents 4 consecutive entries:
    - 0 (0000 binary) -> [-1, -1, -1, -1]  
    - F (1111 binary) -> [1, 1, 1, 1]
    - etc.
    
    Args:
        hex_char: Single hexadecimal character (0-9, A-F)
        
    Returns:
        list: 4 matrix entries as +1 or -1
    """
    # Convert hex to 4-bit binary
    value = int(hex_char, 16)
    binary = format(value, '04b')
    
    # Convert binary to +1/-1 entries
    entries = []
    for bit in binary:
        if bit == '1':
            entries.append(1)
        else:
            entries.append(-1)
    
    return entries

def parse_hex_matrix_row(hex_string):
    """
    Parse a single row from hex format to Hadamard matrix entries.
    
    Args:
        hex_string: String of hex characters representing one row
        
    Returns:
        numpy array: Row of +1/-1 entries
    """
    row_entries = []
    for hex_char in hex_string.strip():
        entries = hex_to_hadamard_entries(hex_char)
        row_entries.extend(entries)
    
    return np.array(row_entries, dtype=np.int8)

def load_hex_hadamard_matrix(matrix_lines):
    """
    Load a complete Hadamard matrix from hex-encoded lines.
    
    Args:
        matrix_lines: List of hex-encoded strings, one per row
        
    Returns:
        numpy array: Complete Hadamard matrix
    """
    matrix_rows = []
    for line in matrix_lines:
        if line.strip():  # Skip empty lines
            row = parse_hex_matrix_row(line)
            matrix_rows.append(row)
    
    return np.array(matrix_rows)

def load_all_hex_hadamard_matrices(filepath):
    """
    Load all Hadamard matrices from a hex-encoded file.
    
    The file format has:
    - Matrix index number
    - Blank line
    - 32 lines of hex-encoded matrix data
    - Blank line
    - Next matrix...
    
    Args:
        filepath: Path to the hex-encoded matrix file
        
    Returns:
        list: List of (matrix_index, matrix_array) tuples
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    matrices = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for matrix index (should be a number)
        if line.isdigit():
            matrix_index = int(line)
            i += 1
            
            # Skip blank line
            if i < len(lines) and lines[i].strip() == '':
                i += 1
            
            # Read 32 hex lines for the matrix
            matrix_lines = []
            for _ in range(32):
                if i < len(lines):
                    matrix_lines.append(lines[i].strip())
                    i += 1
                else:
                    break
            
            # Parse the matrix
            if len(matrix_lines) == 32:
                try:
                    matrix = load_hex_hadamard_matrix(matrix_lines)
                    matrices.append((matrix_index, matrix))
                    # Only print every 1000th matrix to reduce verbosity
                    if matrix_index % 1000 == 0:
                        print(f"Loaded matrix {matrix_index}: {matrix.shape}")
                except Exception as e:
                    print(f"Error loading matrix {matrix_index}: {e}")
            
            # Skip any trailing blank lines
            while i < len(lines) and lines[i].strip() == '':
                i += 1
        else:
            i += 1
    
    return matrices

def verify_hadamard_property(matrix):
    """
    Verify that a matrix satisfies the Hadamard property: H @ H.T = n * I
    
    Args:
        matrix: numpy array
        
    Returns:
        bool: True if matrix is Hadamard, False otherwise
    """
    n = matrix.shape[0]
    product = matrix @ matrix.T
    expected = n * np.eye(n)
    return np.allclose(product, expected)

def sample_from_hex_matrices(matrices, class_idx=None):
    """
    Sample a matrix from a specific class or random class with hyperoctahedral transformation.
    
    Args:
        matrices: List of (matrix_index, matrix_array) tuples
        class_idx: Index of matrix to sample from (None for random)
        
    Returns:
        tuple: (transformed_matrix, original_index)
    """
    if class_idx is None:
        class_idx = np.random.randint(len(matrices))
    
    original_index, base_matrix = matrices[class_idx]
    
    # Apply random hyperoctahedral transformation
    n = base_matrix.shape[0]
    
    # Random row and column permutations
    row_perm = np.random.permutation(n)
    col_perm = np.random.permutation(n)
    
    # Random sign changes for rows and columns
    row_signs = np.random.choice([-1, 1], n)
    col_signs = np.random.choice([-1, 1], n)
    
    # Apply transformations
    transformed = base_matrix[row_perm][:, col_perm]
    transformed = np.outer(row_signs, col_signs) * transformed
    
    return transformed, original_index

# Example usage and testing
if __name__ == "__main__":
    # Test the hex conversion
    print("Testing hex conversion:")
    print("0 ->", hex_to_hadamard_entries('0'))  # Should be [-1, -1, -1, -1]
    print("F ->", hex_to_hadamard_entries('F'))  # Should be [1, 1, 1, 1]
    print("C ->", hex_to_hadamard_entries('C'))  # Should be [1, 1, -1, -1] (1100 binary)
    
    # Test loading from file
    # Uncomment and modify path as needed:
    matrices = load_all_hex_hadamard_matrices("data/raw/res0/dgn_0_12_30_0.txt")
    print(f"\nLoaded {len(matrices)} matrices")
    
    # # Verify a few matrices
    for i in range(min(5, len(matrices))):
        idx, matrix = matrices[i]
        is_valid = verify_hadamard_property(matrix)
        print(f"Matrix {idx}: Valid Hadamard = {is_valid}")
    
    # # Sample from random matrix
    if matrices:
        sampled_matrix, orig_idx = sample_from_hex_matrices(matrices)
        print(f"\nSampled matrix from original index {orig_idx}")
        print(f"Sampled matrix shape: {sampled_matrix.shape}")
        print(f"Is valid Hadamard: {verify_hadamard_property(sampled_matrix)}")

