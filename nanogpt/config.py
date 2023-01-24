from dataclasses import dataclass
import torch

from nanogpt.model.bigram import BigramLanguageModel
from nanogpt.model.gpt import GPT


@dataclass
class BaseConfig:

    batch_size = 32
    block_size = 8

    max_iters = 3000
    eval_interval = 300
    eval_iters = 200
    learning_rate = 1e-2
    train_split = 0.9

    @property
    def device(self) -> str:
        if torch.has_cuda:
            return "cuda"
        elif torch.has_mps:
            return "mps"
        else:
            return "cpu"


@dataclass
class BigramConfig(BaseConfig):
    def model(self, vocabulary_size: int) -> BigramLanguageModel:
        model = BigramLanguageModel(vocabulary_size)
        return model


@dataclass
class GPTConfig(BaseConfig):

    batch_size = 64
    block_size = 256

    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4

    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    def model(self, vocabulary_size: int) -> GPT:
        return GPT(
            vocabulary_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
            dropout=self.dropout,
            device=self.device,
        )
