# File: dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class BiasCorrectionDataset(Dataset):
    def __init__(
        self,
        input_data: dict,
        target_data: dict,
        input_vars: list[str],
        output_vars: list[str],
        input_stats: dict[str, dict] | None = None,
        target_stats: dict[str, dict] | None = None,
        transform: callable | None = None,
    ):
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.input_data = input_data
        self.target_data = target_data
        self.input_stats = input_stats or {}
        self.target_stats = target_stats or {}
        self.transform = transform
        # assume first var defines length
        self.num_samples = next(iter(input_data.values())).shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = []
        for var in self.input_vars:
            arr = self.input_data[var][idx]
            stats = self.input_stats.get(var)
            if stats:
                arr = (arr - stats['mean']) / (stats['std'] + 1e-8)
            x.append(arr)
        x = np.stack(x, axis=0)

        y = []
        for var in self.output_vars:
            arr = self.target_data[var][idx]
            stats = self.target_stats.get(var)
            if stats:
                arr = (arr - stats['mean']) / (stats['std'] + 1e-8)
            y.append(arr)
        y = np.stack(y, axis=0)

        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        if self.transform:
            x_t, y_t = self.transform(x_t, y_t)
        return x_t, y_t
