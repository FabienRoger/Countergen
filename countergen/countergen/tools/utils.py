from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar

import cattrs
import countergen.config
import openai
from tqdm import tqdm  # type: ignore

T = TypeVar("T")


def other(t: Tuple[T, T], x: T) -> T:
    if x == t[0]:
        if x == t[1]:
            raise ValueError(f"{t} contains two copies of {x}")
        return t[1]
    if x != t[1]:
        raise ValueError(f"{t} does not contain {x}")
    return t[0]


def all_same(l: Sequence[Any]) -> bool:
    return all(x == l[0] for x in l[1:])


def maybe_tqdm(it: Iterable[T], do_tqdm: bool = False, **kwargs) -> Iterable[T]:
    if do_tqdm:
        return tqdm(it, **kwargs)
    else:
        return it


def estimate_paraphrase_length(text: str):
    average_token_length = 3
    safety_margin = 50
    return len(text) // average_token_length + safety_margin


class FromAndToJson:
    """Add a from_json_dict & to_json_dict functions.

    The class is expected to be created using @attrs.define."""

    @classmethod
    def from_json_dict(cls, json_dict: Mapping[str, Any]):
        return cattrs.structure(json_dict, cls)

    def to_json_dict(self) -> Mapping[str, Any]:
        return cattrs.unstructure(self, self.__class__)
