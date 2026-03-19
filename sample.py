"""Sample from a trained model.

Usage:
    # From a checkpoint:
    $ uv run python sample.py --out_dir=out-shakespeare-char

    # From a pretrained GPT-2 model:
    $ uv run python sample.py --init_from=gpt2-xl

    # With a custom prompt:
    $ uv run python sample.py --start="To be or not to be"
"""

import os
import pickle
from contextlib import nullcontext

import tiktoken
import torch
from loguru import logger

from nanogpt.model.gpt import GPT, GPTConfig

init_from: str = "resume"  # 'resume' (from out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir: str = "out"
start: str = "\n"  # Prompt string, or "FILE:prompt.txt" to read from file
num_samples: int = 10
max_new_tokens: int = 500
temperature: float = 0.8
top_k: int = 200
seed: int = 1337
device: str = "cuda"
dtype: str = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile: bool = False


def _parse_args() -> None:
    """Simple argument parser that overrides module-level globals."""
    import sys
    from ast import literal_eval

    for arg in sys.argv[1:]:
        if not arg.startswith("--") or "=" not in arg:
            continue
        key, val = arg.lstrip("-").split("=", 1)
        if key not in globals():
            raise ValueError(f"Unknown config key: {key}")
        try:
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            attempt = val
        globals()[key] = attempt


def main() -> None:
    _parse_args()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Load model.
    if init_from == "resume":
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith("gpt2"):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    else:
        raise ValueError(f"Unknown init_from: {init_from}")

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)

    # Set up encoder/decoder.
    load_meta = False
    if init_from == "resume" and "config" in checkpoint:
        cfg = checkpoint["config"]
        if "data_dir" in cfg and cfg["data_dir"]:
            meta_path = os.path.join(cfg["data_dir"], "meta.pkl")
        elif "dataset" in cfg:
            meta_path = os.path.join("data", cfg["dataset"], "meta.pkl")
        else:
            meta_path = ""
        load_meta = bool(meta_path) and os.path.exists(meta_path)

    if load_meta:
        logger.info(f"Loading meta from {meta_path}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]  # noqa: E731
        decode = lambda l: "".join([itos[i] for i in l])  # noqa: E731
    else:
        logger.info("No meta.pkl found, assuming GPT-2 encodings")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})  # noqa: E731
        decode = lambda l: enc.decode(l)  # noqa: E731

    # Encode the prompt.
    prompt = start
    if prompt.startswith("FILE:"):
        with open(prompt[5:], "r", encoding="utf-8") as f:
            prompt = f.read()
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate.
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(
                    x, max_new_tokens, temperature=temperature, top_k=top_k
                )
                print(decode(y[0].tolist()))
                print("---------------")


if __name__ == "__main__":
    main()
