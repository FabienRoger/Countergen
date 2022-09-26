from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional

import torch
from countergen.tools.utils import unwrap_or
from countergen.types import AugmentedSample, Category
from countergentorch.tools.utils import get_gpt_tokenizer
from torch import nn
from transformers import BatchEncoding, GPT2LMHeadModel


def get_mlp_modules(model: GPT2LMHeadModel, layer_numbers: Optional[List[int]]) -> Dict[str, nn.Module]:
    model_transformer: nn.ModuleList = model.transformer.h  # type: ignore
    layer_numbers_ = unwrap_or(layer_numbers, list(range(len(model_transformer))))
    names = [f"transformer.h.{n}.mlp" for n in layer_numbers_]
    return {name: model.get_submodule(name) for name in names}  # type: ignore


def get_res_modules(model: GPT2LMHeadModel, layer_numbers: Optional[List[int]]) -> Dict[str, nn.Module]:
    model_transformer: nn.ModuleList = model.transformer.h  # type: ignore
    layer_numbers_ = unwrap_or(layer_numbers, list(range(len(model_transformer))))
    names = [f"transformer.h.{n}" for n in layer_numbers_]
    return {name: model.get_submodule(name) for name in names}  # type: ignore


def get_corresponding_activations(
    samples: Iterable[AugmentedSample], model: nn.Module, modules: Iterable[nn.Module]
) -> Mapping[Category, List[Dict[nn.Module, torch.Tensor]]]:
    """For each category, returns a list of activations obtained by running the variations corresponding to this category."""

    tokenizer = get_gpt_tokenizer()

    activations_by_cat = defaultdict(lambda: [])
    for sample in samples:
        for inp, categories in sample.get_variations():
            acts = get_activations(tokenizer(inp, return_tensors="pt"), model, modules)
            for cat in categories:
                activations_by_cat[cat].append(acts)
    return activations_by_cat


Operation = Callable[[torch.Tensor], torch.Tensor]


def get_activations(
    tokens: BatchEncoding, model: nn.Module, modules: Iterable[nn.Module], operation: Operation = lambda x: x
) -> Dict[nn.Module, torch.Tensor]:
    handles = []
    activations: Dict[nn.Module, torch.Tensor] = {}

    def hook_fn(module, inp, out):
        activations[module] = operation(out[0].detach())

    for module in modules:
        handles.append(module.register_forward_hook(hook_fn))
    try:
        model(**tokens.to(model.device))
    except Exception as e:
        raise e
    finally:
        for handle in handles:
            handle.remove()
    return activations


# (module, input, output) -> output
ModificationFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


def run_and_modify(
    tokens: BatchEncoding, model: nn.Module, modification_fns: Dict[nn.Module, ModificationFn] = {}
) -> BatchEncoding:
    handles = []
    for module, f in modification_fns.items():
        handles.append(module.register_forward_hook(f))  # type: ignore
    try:
        out = model(**tokens.to(model.device))
        return out
    except Exception as e:
        raise e
    finally:
        for handle in handles:
            handle.remove()
