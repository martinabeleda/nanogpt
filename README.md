# nanogpt

Implementation of `nanogpt` from karpathy. I've modified this model to text classification
by chopping off the language model head and adding a binary classifier on top.

## Development

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

Create the virtual environment and install all dependencies:

```shell
uv sync
```

This will create a `.venv` directory with all project and dev dependencies installed.

## Training

Test run:

```shell
uv run python train.py --config-name rotten_tomatoes_binary_classification_fast
```

Full training run:

```shell
uv run python train.py --config-name rotten_tomatoes_binary_classification
```
