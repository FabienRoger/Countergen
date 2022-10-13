import copy
from typing import Dict, Iterable, List
from countergenedit.tools.math_utils import project
from transformers import GPT2Model
from torch import nn
import torch
from attrs import define


def get_edit_configs(named_modules: Dict[str, nn.Module], dirs: torch.Tensor, has_leftover: bool = False):
    """Return the configs where the same projections along dirs is done at the output of each module.

    The keys of the named_modules should be the name of the corresponding module in the original model."""
    return [ReplacementConfig(name, module, dirs, has_leftover) for name, module in named_modules.items()]


@define
class ReplacementConfig:
    """Configuration for an edition by projection."""

    module_name: str  #: The name of the module in the original network you wish to replace
    old_module: nn.Module  #: The module object to replace
    dirs: torch.Tensor  #: A 2D Float Tensor of shape (n, hidden_dim) listing vectors along which to project
    #: If True, the output of the module is expected to be (to_proj, sth...) rather than to_proj.
    has_leftover: bool = False


def edit_model_inplace(model: nn.Module, configs: Iterable[ReplacementConfig]):
    """Apply the replacements described in the config."""
    for config in configs:
        new_module = ProjectionWrapper(config.old_module, config.dirs, config.has_leftover)

        *parent_path, name = config.module_name.split(".")
        parent_name = ".".join(parent_path)
        parent = model.get_submodule(parent_name)
        if hasattr(parent, name):  # Regular case, if it's a regular attribute
            setattr(parent, name, new_module)
        else:  # ModuleList case, if it's the member of a list
            parent[int(name)] = new_module  # type: ignore

def edit_model(model: nn.Module, configs: Iterable[ReplacementConfig]) -> nn.Module:
    """Return a new module where the replacements described in the config have been done."""
    model = copy.deepcopy(model)
    edit_model_inplace(model, configs)
    return model


class ProjectionWrapper(nn.Module):
    def __init__(self, wrapped_module: nn.Module, dirs: torch.Tensor, has_leftover: bool = False):
        super().__init__()
        self.wrapped_module = wrapped_module.to(dirs.device)
        self.dirs = dirs
        self.has_leftover = has_leftover

        if not torch.allclose(
            torch.eye(self.dirs.shape[0]).to(dirs.device), torch.einsum("m k, n k -> m n", dirs, dirs), atol=1e-4
        ):
            raise ValueError("Directions should be orthonrmal")

    def forward(self, *args, **kwargs):
        y = self.wrapped_module(*args, **kwargs)

        if self.has_leftover:
            hidden_states, *leftover = y
        else:
            hidden_states = y

        # hidden_states -= torch.einsum("b n h, m h, m k -> b n k", hidden_states, self.dirs, self.dirs)
        hidden_states = project(hidden_states, self.dirs)

        return (hidden_states, *leftover) if self.has_leftover else hidden_states

def recover_model_inplace(model: nn.Module, configs):
    """Cancel the replacements described by the config."""
    for config in configs:
        *parent_path, name = config.module_name.split(".")
        parent_name = ".".join(parent_path)
        parent = model.get_submodule(parent_name)
        if hasattr(parent, name):  # Regular case, if it's a regular attribute
            setattr(parent, name, config.old_module)
        else:  # ModuleList case, if it's the member of a list
            parent[int(name)] = config.old_module  # type: ignore