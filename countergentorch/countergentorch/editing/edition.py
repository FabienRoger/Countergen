import copy
from typing import Dict, Iterable, List
from transformers import GPT2Model
from torch import nn
import torch
from attrs import define


def get_edit_configs(named_modules: Dict[str, nn.Module], dirs: torch.Tensor, has_leftover: bool = False):
    return [ReplacementConfig(name, module, dirs, has_leftover) for name, module in named_modules.items()]


@define
class ReplacementConfig:
    module_name: str
    old_module: nn.Module
    dirs: torch.Tensor
    has_leftover: bool = False


def edit_model(model: nn.Module, configs: Iterable[ReplacementConfig]):
    model = copy.deepcopy(model)
    for config in configs:
        new_module = ProjectionWrapper(config.old_module, config.dirs, config.has_leftover)

        *parent_path, name = config.module_name.split(".")
        parent_name = ".".join(parent_path)
        parent = model.get_submodule(parent_name)
        if hasattr(parent, name):  # Regular case, if it's a regular attribute
            setattr(parent, name, new_module)
        else:  # ModuleList case, if it's the member of a list
            parent[int(name)] = new_module  # type: ignore
    return model


class ProjectionWrapper(nn.Module):
    def __init__(self, wrapped_module: nn.Module, dirs: torch.Tensor, has_leftover: bool = False):
        super().__init__()
        self.wrapped_module = wrapped_module.to(dirs.device)
        self.dirs = dirs
        self.has_leftover = has_leftover

        if not torch.allclose(
            torch.eye(self.dirs.shape[0]).to(dirs.device), torch.einsum("m k, n k -> m n", dirs, dirs), atol=1e-6
        ):
            raise ValueError("Directions should be orthonrmal")

    def forward(self, x: torch.Tensor):
        y = self.wrapped_module(x)

        if self.has_leftover:
            hidden_states, leftover = y
        else:
            hidden_states = y

        hidden_states -= torch.einsum("b n h, m h, m k -> b n k", hidden_states, self.dirs, self.dirs)

        return (hidden_states, leftover) if self.has_leftover else hidden_states
