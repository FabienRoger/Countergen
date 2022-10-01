from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, TypeVar, Tuple
import torch


def project(dir: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Return dir, but projected in the orthogonal of the subspace spanned by dirs.

    Assume that dirs are already orthonomal, and that the number of dimensions is > 0."""
    inner_products = torch.einsum("n h, h -> n", dirs, dir)
    new_dir = dir - torch.einsum("n, n h -> h", inner_products, dirs)

    return new_dir
