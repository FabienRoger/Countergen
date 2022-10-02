from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional

import torch
from countergen.types import AugmentedSample, Category
from countergentorch.tools.utils import get_gpt_tokenizer
from torch import nn
from transformers import BatchEncoding, GPT2LMHeadModel


def get_mlp_modules(model: GPT2LMHeadModel, layer_numbers: Optional[List[int]]) -> Dict[str, nn.Module]:
    model_transformer: nn.ModuleList = model.transformer.h  # type: ignore
    layer_numbers_ = layer_numbers or list(range(len(model_transformer)))
    names = [f"transformer.h.{n}.mlp" for n in layer_numbers_]
    return {name: model.get_submodule(name) for name in names}  # type: ignore


def get_res_modules(model: GPT2LMHeadModel, layer_numbers: Optional[List[int]]) -> Dict[str, nn.Module]:
    model_transformer: nn.ModuleList = model.transformer.h  # type: ignore
    layer_numbers_ = layer_numbers or list(range(len(model_transformer)))
    names = [f"transformer.h.{n}" for n in layer_numbers_]
    return {name: model.get_submodule(name) for name in names}  # type: ignore


def get_corresponding_activations(
    samples: Iterable[AugmentedSample], model: nn.Module, modules: Iterable[nn.Module]
) -> Mapping[Category, List[Dict[nn.Module, torch.Tensor]]]:
    """For each category, returns a list of activations obtained by running the variations corresponding to this category."""

    tokenizer = get_gpt_tokenizer()

    operation = lambda t: t.reshape((-1, t.shape[-1]))

    activations_by_cat = defaultdict(lambda: [])
    for sample in samples:
        for variations in sample.get_variations():
            acts = get_activations(tokenizer(variations.text, return_tensors="pt"), model, modules, operation)
            for cat in variations.categories:
                activations_by_cat[cat].append(acts)
    return activations_by_cat


Operation = Callable[[torch.Tensor], torch.Tensor]


def get_activations(
    tokens: BatchEncoding, model: nn.Module, modules: Iterable[nn.Module], operation: Operation = lambda x: x
) -> Dict[nn.Module, torch.Tensor]:
    handles = []
    activations: Dict[nn.Module, torch.Tensor] = {}

    def hook_fn(module, inp, out):
        out_ = out[0] if isinstance(out, tuple) else out

        activations[module] = operation(out_.detach())

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
