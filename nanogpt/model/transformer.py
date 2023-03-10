from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F


class Transformer(nn.Module):
    """Implements a simple Transformer

    Args:
        vocab_size: Size of the vocabulary in tokens
        n_embed: Number of positional embeddings
    """

    def __init__(
        self,
        config: DictConfig,
        vocab_size: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(
                    config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.layer_norm_f = nn.LayerNorm(config.n_embd)

        self.pre_classifier = nn.Linear(config.n_embd, config.n_embd)
        self.classifier = nn.Linear(config.n_embd, config.n_labels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (batch, time, channel)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.layer_norm_f(x)
        x = x.mean(1)

        x = self.pre_classifier(x)
        x = self.classifier(x)
        logits = self.dropout(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class Block(nn.Module):
    "Transformer block - communication followed by computation"

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(
            n_head, head_size, n_embd, block_size, dropout
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    scaling_factor = 4

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, self.scaling_factor * n_embd),
            nn.ReLU(),
            nn.Linear(self.scaling_factor * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embd: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttention(head_size, n_embd, block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class SelfAttention(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores "affinities"
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # noqa
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # Perform the weighted aggregation of the values
        v = self.value(x)
        out = weights @ v
        return out
