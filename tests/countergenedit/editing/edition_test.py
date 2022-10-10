#%%
from countergenedit import edit_model, get_edit_configs
import torch
from torch import nn


def test_edit_sequential_model():
    """Editing should return excepted result on a simple sequential model."""

    simple_model = nn.Sequential(nn.Linear(3, 3, bias=False), nn.Linear(3, 3, bias=False), nn.Linear(3, 1, bias=False))

    with torch.no_grad():
        w0 = torch.eye(3) * 2
        w1 = torch.eye(3)
        w1[2, 0] = 1
        w2 = torch.ones(1, 3)
        for i, w in enumerate([w0, w1, w2]):
            simple_model[i].weight.data = w

    layers_to_edit = {"1": simple_model[1]}

    expected_output_simple = torch.FloatTensor([[4], [2], [2]])
    torch.testing.assert_close(simple_model(torch.eye(3)), expected_output_simple)

    dirs = torch.FloatTensor([[0, 1, 0]])

    new_model = edit_model(simple_model, get_edit_configs(layers_to_edit, dirs))

    torch.testing.assert_close(simple_model(torch.eye(3)), expected_output_simple)

    expected_output_proj1 = torch.FloatTensor([[4], [0], [2]])
    torch.testing.assert_close(new_model(torch.eye(3)), expected_output_proj1)

    dirs = torch.FloatTensor([[0, 0, 1]])
    new_model = edit_model(simple_model, get_edit_configs(layers_to_edit, dirs))

    torch.testing.assert_close(simple_model(torch.eye(3)), expected_output_simple)

    expected_output_proj1 = torch.FloatTensor([[2], [2], [0]])
    torch.testing.assert_close(new_model(torch.eye(3)), expected_output_proj1)
