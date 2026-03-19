# nanogpt

A production-style implementation of [nanoGPT](https://github.com/karpathy/nanoGPT) — a GPT-2 language model for training and text generation in PyTorch.

Features:

- Full GPT-2 architecture with Flash Attention
- Training from scratch, resuming from checkpoints, or fine-tuning pretrained GPT-2 weights
- Distributed Data Parallel (DDP) for multi-GPU training
- Mixed precision training (bfloat16/float16)
- Cosine learning rate schedule with warmup
- Gradient accumulation and gradient clipping
- Weights & Biases logging and artifact tracking
- Hydra configuration management
- Dataset preparation for Shakespeare (BPE + char-level) and OpenWebText

## Development

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```shell
uv sync
```

## Quick Start: Shakespeare (char-level)

Prepare the dataset:

```shell
uv run python data/shakespeare_char/prepare.py
```

Train:

```shell
uv run python train.py --config-name train_shakespeare_char
```

Sample from the trained model:

```shell
uv run python sample.py --out_dir=out-shakespeare-char
```

## Training GPT-2 on OpenWebText

Prepare the dataset (~54GB download):

```shell
uv run python data/openwebtext/prepare.py
```

Train on 8x A100 GPUs:

```shell
torchrun --standalone --nproc_per_node=8 train.py --config-name train_gpt2
```

## Training Max GPT-2 (429M) on OpenWebText — RTX 4070

A maxed-out 429M parameter model (30 layers, 16 heads, 1024 embedding) designed to fit on a single RTX 4070 12GB GPU. Trains for ~2 epochs (~551k iterations, ~14 days).

Prepare the dataset (~54GB download, if not already done):

```shell
uv run python data/openwebtext/prepare.py
```

Train from scratch:

```shell
uv run python train.py --config-name train_max_owt
```

Resume from checkpoint (loads `out-max-owt/ckpt.pt`):

```shell
uv run python train.py --config-name train_max_owt init_from=resume
```

Sample from the trained model:

```shell
uv run python sample.py --out_dir=out-max-owt
```

## Fine-tuning GPT-2 on Shakespeare

Prepare the Shakespeare dataset (BPE tokenized):

```shell
uv run python data/shakespeare/prepare.py
```

Fine-tune:

```shell
uv run python train.py --config-name finetune_shakespeare
```

## Sampling

```shell
# From checkpoint
uv run python sample.py --out_dir=out-shakespeare-char

# From pretrained GPT-2
uv run python sample.py --init_from=gpt2-xl

# With custom prompt
uv run python sample.py --init_from=gpt2 --start="To be or not to be"
```

## Configuration

Training configs are in `configs/` and use [Hydra](https://hydra.cc/).
Override any parameter from the command line:

```shell
uv run python train.py --config-name train_shakespeare_char model.dropout=0.1
```

Available configs:

- `config.yaml` — Default GPT-2 (124M) on OpenWebText
- `train_shakespeare_char.yaml` — Character-level Shakespeare (small, fast)
- `train_gpt2.yaml` — Full GPT-2 training on OpenWebText
- `train_max_owt.yaml` — Max GPT-2 (429M) on OpenWebText for RTX 4070 12GB
- `finetune_shakespeare.yaml` — Fine-tune GPT-2-XL on Shakespeare
