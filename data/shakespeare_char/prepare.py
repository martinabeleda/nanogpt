"""Prepare the Shakespeare dataset for character-level language modeling.

Instead of encoding with GPT-2 BPE tokens, maps characters to integers.
Saves train.bin, val.bin, and meta.pkl (encoder/decoder and vocab info).

Usage:
    $ uv run python data/shakespeare_char/prepare.py
"""

import os
import pickle

import numpy as np
import requests
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

    logger.info(f"Length of dataset in characters: {len(data):,}")

    # Build character-level vocabulary.
    chars = sorted(set(data))
    vocab_size = len(chars)
    logger.info(f"Unique characters: {''.join(chars)}")
    logger.info(f"Vocab size: {vocab_size:,}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    # Train/test split.
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    train_ids = encode(train_data)
    val_ids = encode(val_data)
    logger.info(f"Train has {len(train_ids):,} tokens")
    logger.info(f"Val has {len(val_ids):,} tokens")

    # Export to bin files.
    base_dir = os.path.dirname(__file__)
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(base_dir, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(base_dir, "val.bin"))

    # Save meta information for encoding/decoding later.
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(base_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    logger.info("Done. Saved train.bin, val.bin, and meta.pkl")


if __name__ == "__main__":
    main()
