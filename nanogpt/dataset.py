import torch
from loguru import logger


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
