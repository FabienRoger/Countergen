from typing import Any, Dict, Tuple
import torch
from torch import nn

from countergen.config import VERBOSE
from countergen.tools.utils import maybe_tqdm


def get_bottlenecked_linear(input_dim: int, output_dim: int, bottleneck_dim: int = 1):
    return nn.Sequential(nn.Linear(input_dim, bottleneck_dim), nn.Linear(bottleneck_dim, output_dim))


def get_bottlenecked_mlp(input_dim: int, output_dim: int, bottleneck_dim: int = 8, hidden_dim: int = 64):
    return nn.Sequential(
        nn.Linear(input_dim, bottleneck_dim),
        nn.Linear(bottleneck_dim, hidden_dim),
        nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def fit_model(
    model: nn.Module,
    ds: torch.utils.data.Dataset,
    max_iters: int,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
    adam_kwargs: Dict[str, Any] = {"lr": 1e-4},
    dataloader_kwargs: Dict[str, Any] = {"batch_size": 256, "shuffle": True},
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
    dataloader = torch.utils.data.DataLoader(ds, **dataloader_kwargs)

    tepoch = maybe_tqdm(range(max_iters), VERBOSE >= 3)

    for _ in tepoch:
        epoch_loss = 0.0
        n_correct = 0
        n_tot = 0

        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)

            loss_val = loss_fn(out, y)
            epoch_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(out, dim=-1)
                n_correct += (preds == y).sum().item()
                n_tot += len(preds)

        if VERBOSE >= 3:
            tepoch.set_postfix(loss=epoch_loss, accuracy=n_correct / n_tot)  # type:ignore

    return {"loss": epoch_loss, "accuracy": n_correct / n_tot}
