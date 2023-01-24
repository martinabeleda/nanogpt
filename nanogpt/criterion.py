import torch
import torch.nn as nn

from nanogpt.dataset import get_batch
from nanogpt.model.gpt import GPTConfig


@torch.no_grad()
def estimate_loss(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: GPTConfig,
    model: nn.Module,
    device: str,
) -> dict[str, float]:
    out = {}
    model.eval()

    for split in ["train", "val"]:
        data = train_data if split == "train" else val_data
        losses = torch.zeros(config.eval_iters)

        for k in range(config.eval_iters):
            X, Y = get_batch(data, device=device)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out
