from functools import lru_cache
from typing import (Any, Callable, Dict, Iterable, Mapping, Optional, Sequence,
                    Tuple, TypeVar)

import torch
from transformers import GPT2Tokenizer

T = TypeVar("T")


def concat_dicts(dicts: Sequence[Mapping[Any, torch.Tensor]]) -> Dict[Any, torch.Tensor]:
    if not dicts:
        raise ValueError("dicts is empty")
    keys = dicts[0].keys()
    for d in dicts:
        if d.keys() != keys:
            raise ValueError("dicts must have the same keys")
    return {k: torch.cat([d[k] for d in dicts], dim=-1) for k in keys}


def remove_last_tok(d: Mapping[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
    return {k: t[:, :-1] for k, t in d.items()}


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_gpt_tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained("gpt2")
