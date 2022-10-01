from countergentorch.tools.math_utils import project
import torch


def test_project_cover_sends_to_zeros():
    """If dirs cover the whole space, should return 0."""

    for n_dim in range(1, 4):
        dir = torch.rand(n_dim)
        torch.testing.assert_close(project(dir, torch.eye(n_dim)), torch.zeros(n_dim))


def test_project_disjoint_does_not_affect():
    """If dirs are on different dimension that dir, it should not change dir."""

    n_dim = 4
    for dirs_span in range(1, 4):
        dir = torch.rand(n_dim)
        dirs = torch.zeros(n_dim, n_dim)
        dirs[dirs_span:, dirs_span:] = 0
        dir[:dirs_span] = 0
        torch.testing.assert_close(project(dir, dirs), dir)


def test_project_simple():
    """Project should return expected results in simple cases."""

    dir = torch.FloatTensor([1, 0.5, 2])
    dirs = torch.FloatTensor([[1, 1, 0], [1, -1, 0]])
    dirs = dirs / torch.linalg.norm(dirs, dim=-1)[:, None]
    expected = torch.FloatTensor([0, 0, 2])
    torch.testing.assert_close(project(dir, dirs), expected)

    dir = torch.FloatTensor([1, 0, 2])
    dirs = torch.FloatTensor([[1, 1, 0]])
    dirs = dirs / torch.linalg.norm(dirs, dim=-1)[:, None]
    expected = torch.FloatTensor([0.5, -0.5, 2])
    torch.testing.assert_close(project(dir, dirs), expected)

    dir = torch.FloatTensor([1, 0, 2])
    dirs = torch.FloatTensor([[1, -1, 0], [0, 0, 1]])
    dirs = dirs / torch.linalg.norm(dirs, dim=-1)[:, None]
    expected = torch.FloatTensor([0.5, 0.5, 0])
    torch.testing.assert_close(project(dir, dirs), expected)
