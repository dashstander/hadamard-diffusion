import numpy as np
import torch
from torch.utils.data import IterableDataset



class PreShuffledHadamardDataset(IterableDataset):

    def __init__(self, file_path: str, batch_size: int, seed: int, start_idx: int = 0, end_idx: int = None):
        """
        Args:
            file_path: Path to the pre-shuffled numpy memmap file
            batch_size: Batch size for iteration
            seed: Random seed for transformations
            start_idx: Starting index for contiguous subset (inclusive)
            end_idx: Ending index for contiguous subset (exclusive, None for end of file)
        """
        self.matrices = np.lib.format.open_memmap(file_path, mode='r')
        self.batch_size = batch_size
        self.order = self.matrices.shape[-1]
        self.rng = np.random.default_rng(seed)

        # Set up contiguous subset bounds
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else self.matrices.shape[0]

        # Validate bounds
        if self.start_idx < 0 or self.start_idx >= self.matrices.shape[0]:
            raise ValueError(f"start_idx {self.start_idx} out of bounds [0, {self.matrices.shape[0]})")
        if self.end_idx <= self.start_idx or self.end_idx > self.matrices.shape[0]:
            raise ValueError(f"end_idx {self.end_idx} out of bounds ({self.start_idx}, {self.matrices.shape[0]}]")

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def transformations(self, batch_size):
        indices = np.tile(np.arange(self.order), (batch_size, 1))
        row_perm = self.rng.permuted(indices, axis=1)
        col_perm = self.rng.permuted(indices, axis=1)
        signs = self.rng.choice([-1, 1], (batch_size, self.order))
        return row_perm, col_perm, signs

    def __iter__(self):
        i = self.start_idx
        n = self.end_idx
        batch_idx = np.arange(self.batch_size)[:, None]

        while i < n:
            j = min(n, i + self.batch_size)
            actual_batch_size = j - i

            # Get matrices from the contiguous subset
            mats = self.matrices[i:j].astype(np.float32)

            # Apply transformations using actual batch size
            row_perm, col_perm, signs = self.transformations(actual_batch_size)
            batch_idx = np.arange(actual_batch_size)[:, None]

            mats = mats[batch_idx, row_perm, :]
            mats = mats[batch_idx, :, col_perm]
            mats *= signs[:, None, :]
            mats = torch.from_numpy(mats)
            yield mats.pin_memory('cuda')
            i += self.batch_size


def create_train_eval_datasets(file_path: str, batch_size: int, train_seed: int = 42, eval_seed: int = 43, eval_fraction: float = 0.1):
    """
    Create train and eval datasets from a pre-shuffled numpy file.

    Args:
        file_path: Path to the pre-shuffled numpy memmap file
        batch_size: Batch size for both datasets
        train_seed: Random seed for training dataset transformations
        eval_seed: Random seed for evaluation dataset transformations
        eval_fraction: Fraction of data to use for evaluation (0.1 = 10%)

    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    # Get total size by loading metadata
    matrices = np.lib.format.open_memmap(file_path, mode='r')
    total_size = matrices.shape[0]

    # Calculate split point (eval from the end since data is pre-shuffled)
    eval_size = int(total_size * eval_fraction)
    train_size = total_size - eval_size

    # Create train dataset (front portion)
    train_dataset = PreShuffledHadamardDataset(
        file_path=file_path,
        batch_size=batch_size,
        seed=train_seed,
        start_idx=0,
        end_idx=train_size
    )

    # Create eval dataset (back portion)
    eval_dataset = PreShuffledHadamardDataset(
        file_path=file_path,
        batch_size=batch_size,
        seed=eval_seed,
        start_idx=train_size,
        end_idx=total_size
    )

    print(f"Created train dataset: {len(train_dataset)} matrices ({len(train_dataset)//batch_size} batches)")
    print(f"Created eval dataset: {len(eval_dataset)} matrices ({len(eval_dataset)//batch_size} batches)")

    return train_dataset, eval_dataset





