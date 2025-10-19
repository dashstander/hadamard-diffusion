import numpy as np
import torch
from torch.utils.data import IterableDataset



class PreShuffledHadamardDataset(IterableDataset):

    def __init__(self, file_path: str, batch_size: int, seed: int):
        
        self.matrices = np.lib.format.open_memmap(file_path, mode='r')
        self.batch_size = batch_size
        self.order = self.matrices.shape[-1]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.matrices.shape[0]
    
    def transformations(self):
        indices = np.tile(np.arange(self.order), (self.batch_size, 1))
        row_perm = self.rng.permuted(indices, axis=1)
        col_perm = self.rng.permuted(indices, axis=1)
        signs = self.rng.choice([-1, 1], (self.batch_size, self.order))
        return row_perm, col_perm, signs

    def __iter__(self):
        i = 0
        n = len(self)
        batch_idx = np.arange(self.batch_size)[:, None]

        while i < n:
            j = min(n, i + self.batch_size)
            mats = self.matrices[i:j].astype(np.float32)
            row_perm, col_perm, signs = self.transformations()
            mats = mats[batch_idx, row_perm, :]
            mats = mats[batch_idx, :, col_perm]
            mats *= signs[:, None, :]
            mats = torch.from_numpy(mats)
            yield mats.pin_memory('cuda')
            i += self.batch_size





