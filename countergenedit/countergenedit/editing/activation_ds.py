from typing import Dict, Iterable, List, Mapping, Union

import torch
from attrs import define
from torch import nn
from countergenedit.editing.activation_utils import get_corresponding_activations

from countergen.types import AugmentedSample, Category

from countergenedit.tools.math_utils import project


@define
class ActivationsDataset(torch.utils.data.Dataset):
    """Dataset of activations with utilities to compute activations and project them."""

    x_data: torch.Tensor  #: 2D float32 tensor of shape (samples, hidden_dimension)
    y_data: torch.Tensor  #: 1D long tensor of shape (samples,) where one number is one category

    @classmethod
    def from_augmented_samples(
        cls,
        samples: Iterable[AugmentedSample],
        model: nn.Module,
        modules: Union[Iterable[nn.Module], Dict[str, nn.Module]],
    ):
        """Compute the activations of the model on the variations of the samples at the output of the given modules.

        The modules are assumed to have outputs of the same shape.

        If modules is a dictionary, take the values of the dictionary."""

        modules_ = modules.values() if isinstance(modules, dict) else modules

        activations_dict = get_corresponding_activations(samples, model, modules_)
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
        """Return a new dataset where activations have been projected along the dir vector."""
        dir_norm = (dir / torch.linalg.norm(dir)).to(self.x_data.device)
        new_x_data = project(self.x_data, dir_norm[None, :])
        return ActivationsDataset(new_x_data, self.y_data)

    def project_(self, dir: torch.Tensor):
        """Modify activations by projecteding them along the dir vector."""
        dir_norm = (dir / torch.linalg.norm(dir)).to(self.x_data.device)
        self.x_data = project(self.x_data, dir_norm[None, :])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx, :]
        y = self.y_data[idx]
        sample = (x, y)
        return sample
