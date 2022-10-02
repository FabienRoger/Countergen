from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, TypeVar, Tuple
import torch


def project(dir: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Return dir, but projected in the orthogonal of the subspace spanned by dirs.

    Assume that dirs are already orthonomal, and that the number of dimensions is > 0."""
    inner_products = torch.einsum("n h, h -> n", dirs, dir)
    new_dir = dir - torch.einsum("n, n h -> h", inner_products, dirs)

    return new_dir

def orthonormalize(dirs: torch.Tensor) -> torch.Tensor:
    """Apply the Gram-Schmidt algorithm to make dirs orthonormal
    
    Assumes that the number of dimensions and dirs is > 0."""
    n, h = dirs.shape

    dirs[0] /= torch.linalg.norm(dirs[0])
    for i in range(1, n):
        dirs[i] = project(dirs[i], dirs[:i])
        dirs[i] /= torch.linalg.norm(dirs[i])
    
    return dirs