#!/usr/bin/env python3

import torch
import sys
sys.path.append('src')

from hadamard_diffusion.layers import HadamardTransformerLayer, MatrixEmbedding

def test_time_conditioning():
    """Test the time-conditioned HadamardTransformerLayer"""

    # Model parameters
    batch_size = 2
    matrix_size = 32
    element_dim = 128
    pool_dim = 64
    num_heads = 8
    head_dim = 16
    ffn_hidden_dim = 256

    # Create sample data
    # Binary matrices with {-1, 1} values
    matrices = torch.randint(0, 2, (batch_size, matrix_size, matrix_size))
    matrices = torch.where(matrices == 0, -1, 1).float()

    # Time steps (diffusion process from 0 to 1)
    timesteps = torch.rand(batch_size)  # Random times in [0, 1]

    # Initialize embedding and transformer layers
    embedding = MatrixEmbedding(embed_dim=element_dim, size=matrix_size)
    transformer = HadamardTransformerLayer(
        element_dim=element_dim,
        pool_dim=pool_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        ffn_hidden_dim=ffn_hidden_dim
    )

    print(f"Input matrices shape: {matrices.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Timesteps values: {timesteps}")

    # Embed matrices
    embedded_matrices = embedding(matrices)
    print(f"Embedded matrices shape: {embedded_matrices.shape}")

    # Forward pass through transformer with time conditioning
    element_reps, row_reps, column_reps = transformer(embedded_matrices, timesteps)

    print(f"Output element reps shape: {element_reps.shape}")
    print(f"Output row reps shape: {row_reps.shape}")
    print(f"Output column reps shape: {column_reps.shape}")

    # Test that different timesteps produce different outputs
    timesteps_early = torch.zeros(batch_size)  # t=0 (pure noise)
    timesteps_late = torch.ones(batch_size)    # t=1 (clean data)

    out_early = transformer(embedded_matrices, timesteps_early)
    out_late = transformer(embedded_matrices, timesteps_late)

    # Check that outputs are different
    diff_early_late = torch.norm(out_early[0] - out_late[0])
    print(f"Difference between t=0 and t=1 outputs: {diff_early_late:.4f}")

    if diff_early_late > 1e-3:
        print("✓ Time conditioning is working - different timesteps produce different outputs")
    else:
        print("⚠ Warning: Time conditioning may not be working properly")

    return True

if __name__ == "__main__":
    test_time_conditioning()