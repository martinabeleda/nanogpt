"""Prepare the Shakespeare dataset with GPT-2 BPE tokenization.

Downloads tiny Shakespeare and encodes it into train.bin and val.bin using
tiktoken's GPT-2 BPE encoding.

Usage:
    $ uv run python data/shakespeare/prepare.py
"""

import os

import numpy as np
import requests
import tiktoken
from loguru import logger

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

    if not os.path.exists(input_file_path):
        logger.info(f"Downloading dataset from {DATA_URL}")
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(DATA_URL).text)

    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()

    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # Encode with tiktoken GPT-2 BPE.
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    logger.info(f"Train has {len(train_ids):,} tokens")
    logger.info(f"Val has {len(val_ids):,} tokens")

    # Export to bin files.
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    logger.info("Done. Saved train.bin and val.bin")


if __name__ == "__main__":
    main()
