#!/usr/bin/env python3
"""
Simple training script for RADD (Reparameterized Absorbing Discrete Diffusion)
on Hadamard matrices using pre-shuffled data.

This script sets up all hyperparameters and calls the main training function.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hadamard_diffusion.training import train_hadamard_diffusion_preshuffled


def main():
    parser = argparse.ArgumentParser(description="Train RADD model on Hadamard matrices")
    parser.add_argument("--data-path", required=True,
                       help="Path to pre-shuffled numpy file")
    parser.add_argument("--matrix-size", type=int, default=32,
                       help="Size of Hadamard matrices (default: 32)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--save-dir", default="checkpoints",
                       help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--run-name", default=None,
                       help="W&B run name (auto-generated if not provided)")

    args = parser.parse_args()

    # Hyperparameters for RADD training
    model_kwargs = {
        'element_dim': 256,
        'ffn_hidden_dim': 2048,
        'pool_dim': 512,
        'num_heads': 8,
        'head_dim': 64,
        'num_layers': 6
    }

    print("=" * 60)
    print("Training RADD on Hadamard Matrices")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Matrix size: {args.matrix_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Model: RADD (Time-independent)")
    print(f"Graph: Absorbing diffusion")
    print(f"Element hidden dim: {model_kwargs['element_dim_dim']}")
    print(f"Row/Column hidden dim: {model_kwargs['pool_dim']}")
    print(f"Layers: {model_kwargs['num_layers']}")
    print("=" * 60)

    # Train the model
    model, final_metrics = train_hadamard_diffusion_preshuffled(
        file_path=args.data_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        matrix_size=args.matrix_size,
        model_kwargs=model_kwargs,
        save_dir=args.save_dir,
        log_interval=50,
        eval_interval=5,
        eval_batch_size=16,
        model_type='radd',           # Time-independent RADD model
        loss_type='t_dce',           # t-DCE loss for RADD
        graph_type='absorbing',      # Absorbing diffusion
        use_wandb=True,
        wandb_project="hadamard-diffusion",
        wandb_run_name=args.run_name,
        eval_fraction=0.1,
        train_seed=42,
        eval_seed=43
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print("Final metrics:")
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()