from random import random
from countergen.tools.math_utils import geometric_mean
from pytest import approx


def test_geometric_mean_basic():
    """Geometric mean should behave as expected on small tests cases."""

    assert geometric_mean([3]) == approx(3)
    assert geometric_mean([1, 100]) == approx(10)
    assert geometric_mean([1, 3, 9]) == approx(3)
    assert geometric_mean([1, 1/3, 1/9]) == approx(1/3)


def test_geometric_mean_all_same():
    """Geometric mean should return the element when all elements are the same."""

    for n_elts in range(1, 4):
        for _ in range(5):
            v = random() * 2
            elts = [v] * n_elts
            assert geometric_mean(elts) == approx(v)
