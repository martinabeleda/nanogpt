# nanogpt

Implementation of `nanogpt` from karpathy. I've modified this model to text classification
by chopping off the language model head and adding a binary classifier on top.

## Development

```shell
poetry install
```

## Training

Test run:

```shell
python train.py --config-name rotten_tomatoes_binary_classification_fast
```

Full training run:

```shell
python train.py --config-name rotten_tomatoes_binary_classification
```
