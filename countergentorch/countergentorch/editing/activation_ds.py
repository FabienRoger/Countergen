from typing import Dict, Iterable, List, Mapping

import torch
from attrs import define
from torch import nn
from countergentorch.editing.activation_utils import get_corresponding_activations

from countergen.types import AugmentedSample, Category


@define
class ActivationsDataset(torch.utils.data.Dataset):
    x_data: torch.Tensor  # dtype: float32
    y_data: torch.Tensor  # dtype: long

    @classmethod
    def from_augmented_samples(cls, samples: Iterable[AugmentedSample], model: nn.Module, modules: Iterable[nn.Module]):
        activations_dict = get_corresponding_activations(samples, model, modules)
        return ActivationsDataset.from_activations_dict(activations_dict)

    @classmethod
    def from_activations_dict(
        cls, activations: Mapping[Category, List[Dict[nn.Module, torch.Tensor]]], device: str = "cpu"
    ):
        """Group all activations regardless of the module and the sequence position.

        All activations must be of the same shape, (seq_len, hid_size)."""

        x_data_l = []
        y_data_l = []
        for i, li in enumerate(activations.values()):
            # Flatten the activations
            all_activations = []
            for d in li:
                for activation in d.values():
                    all_activations.append(activation)
            x = torch.cat(all_activations)
            x_data_l.append(x)
            y = torch.zeros(x.shape[0], dtype=torch.long)
            y[:] = i
            y_data_l.append(y)
        return ActivationsDataset(torch.cat(x_data_l).to(device), torch.cat(y_data_l).to(device))

    def project(self, dir: torch.Tensor):
        dir_norm = (dir / torch.linalg.norm(dir)).to(self.x_data.device)
        new_x_data = self.x_data - torch.outer((self.x_data @ dir_norm), dir_norm)
        return ActivationsDataset(new_x_data, self.y_data)

    def project_(self, dir: torch.Tensor):
        dir_norm = (dir / torch.linalg.norm(dir)).to(self.x_data.device)
        self.x_data -= torch.outer((self.x_data @ dir_norm), dir_norm)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx, :]
        y = self.y_data[idx]
        sample = (x, y)
        return sample
