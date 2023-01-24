from dataclasses import asdict

import torch
from loguru import logger
import wandb

from nanogpt.dataset import load_dataset, get_batch, train_test_split
from nanogpt.criterion import estimate_loss
from nanogpt.model.gpt import GPT, GPTConfig
from nanogpt.model.bigram import BigramLanguageModel, BigramModelConfig


def get_device() -> str:
    if torch.has_cuda:
        return "cuda"
    elif torch.has_mps:
        return "mps"
    else:
        return "cpu"


def train():
    config = GPTConfig()

    with wandb.init(project="nanogpt", config=asdict(config)) as run:

        logger.info(f"Running config: {config.__class__.__name__}")
        logger.info(f"Resolved config: {config.__dict__}")

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

        model = GPT(config=config, vocab_size=vocabulary_size, device=get_device())
        m = model.to(get_device())

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        for iter in range(config.max_iters):
            if iter % config.eval_interval == 0:
                losses = estimate_loss(
                    train_data, val_data, config, model, get_device()
                )
                logger.info(
                    f"Step: {iter} Train loss: {losses['train']:.4f} Val loss {losses['val']:.4f}"
                )
                run.log(losses)

            xb, yb = get_batch(train_data, device=get_device())

            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        context = torch.zeros((1, 1), dtype=torch.long, device=get_device())
        prediction = decode(m.generate(context, max_new_tokens=500)[0].tolist())
        logger.info(f"Test generate: {prediction}")


if __name__ == "__main__":
    train()
