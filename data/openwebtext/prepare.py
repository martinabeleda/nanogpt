"""Prepare the OpenWebText dataset for training.

Downloads and tokenizes the OpenWebText dataset using GPT-2 BPE encoding,
saving the result as memory-mapped binary files.

Usage:
    $ uv run python data/openwebtext/prepare.py

Note: Takes ~54GB in the HuggingFace cache directory.
"""

import os

import numpy as np
import tiktoken
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

# Number of workers for parallel processing.
NUM_PROC = 8
NUM_PROC_LOAD = NUM_PROC


def main() -> None:
    enc = tiktoken.get_encoding("gpt2")

    logger.info("Loading OpenWebText dataset (~8M documents)")
    dataset = load_dataset("openwebtext", num_proc=NUM_PROC_LOAD)

    # OWT only has a 'train' split; create a small validation split.
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")
    logger.info(f"Dataset splits: {split_dataset}")

    def process(example: dict) -> dict:
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=NUM_PROC,
    )

    # Write each split to a single memory-mapped binary file.
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16  # enc.max_token_value == 50256 < 2**16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    logger.info("Done. train.bin (~17GB), val.bin (~8.5MB)")


if __name__ == "__main__":
    main()
