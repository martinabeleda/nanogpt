from dataclasses import asdict

import torch
import torch.nn as nn
from loguru import logger

from nanogpt.config import BaseConfig, GPTConfig
from nanogpt.dataset import load_dataset, get_batch, train_test_split
from nanogpt.criterion import estimate_loss


def train():
    config = GPTConfig()
    logger.info(f"Running config: {config.__class__.__name__}")
    logger.info(f"Resolved config: {asdict(config)}")

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

    model = config.model(vocabulary_size)
    m = model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):

        if iter % config.eval_interval == 0:
            losses = estimate_loss(train_data, val_data, config, model)
            logger.info(
                f"Step: {iter} Train loss: {losses['train']:.4f} Val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(train_data, device=config.device)

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    prediction = decode(m.generate(context, max_new_tokens=500)[0].tolist())
    logger.info(f"Test generate: {prediction}")


if __name__ == "__main__":
    train()
