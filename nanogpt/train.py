from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from loguru import logger

from nanogpt.model.bigram import BigramLanguageModel


@dataclass
class Hyperparameters:
    batch_size = 32
    block_size = 8
    max_iters = 3000
    eval_interval = 300
    eval_iters = 200
    learning_rate = 1e-2
    train_split = 0.9

    @property
    def device(self) -> str:
        if torch.has_cuda:
            return "cuda"
        elif torch.has_mps:
            return "mps"
        else:
            return "cpu"


def load_dataset() -> str:
    with open("data/shakespeare/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    logger.info(f"Length of dataset in characters: {len(text)}")
    logger.info(f"Sample text: {text[:500]}")
    return text


def train_test_split(
    data: torch.Tensor,
    train_split: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(train_split * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(
    data: torch.Tensor,
    batch_size: int = 4,
    block_size: int = 8,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Hyperparameters,
    model: nn.Module,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        data = train_data if split == "train" else val_data
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(data, device=config.device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train():
    config = Hyperparameters()

    text = load_dataset()
    characters = sorted(list(set(text)))
    vocabulary_size = len(characters)

    stoi = {ch: i for i, ch in enumerate(characters)}
    itos = {i: ch for i, ch in enumerate(characters)}

    def encode(string: str) -> list[int]:
        return [stoi[c] for c in string]

    def decode(encoded: list[int]) -> str:
        return "".join([itos[i] for i in encoded])

    data = torch.tensor(encode(text), dtype=torch.long)
    logger.info(f"Loaded dataset of shape: {data.shape} and type: {data.dtype}")

    train_data, val_data = train_test_split(data, config.train_split)

    model = BigramLanguageModel(vocabulary_size)
    m = model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):

        if iter % config.eval_interval == 0:
            losses = estimate_loss(train_data, val_data, config, model)
            logger.info(
                f"Step: {iter} Train loss: {losses['train']:.4f} Val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(train_data, device=config.device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    prediction = decode(m.generate(context, max_new_tokens=500)[0].tolist())
    logger.info(f"Test generate: {prediction}")


if __name__ == "__main__":
    train()
