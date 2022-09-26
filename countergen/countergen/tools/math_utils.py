from math import exp, log2
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, TypeVar, Tuple


def mean(l: Sequence[float]) -> float:
    return sum(l) / len(l)


def geometric_mean(l: Sequence[float]) -> float:
    return 2 ** (mean(list(map(log2, l))))


def perplexity(log_probs: Sequence[float]):
    """Take in natural log probs, returns (average) perplexity"""
    return exp(mean(log_probs))
