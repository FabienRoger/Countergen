from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

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
    if not l:
        return True
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


def unwrap_float(x: Union[float, Any], err_msg="Value should be a float!") -> float:
    if not isinstance(x, float):
        raise ValueError(err_msg)
    return x


def unwrap_list_of_floats(x: Union[List[float], Any], err_msg="Value should be a list of floats!") -> List[float]:
    if not isinstance(x, list):
        raise ValueError(err_msg)
    if not all(isinstance(y, float) for y in x):
        raise ValueError(err_msg)
    return x
