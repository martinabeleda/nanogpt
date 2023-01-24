from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPT(nn.Module):
    """Implements a simple GPT

    Args:
        vocab_size: Size of the vocabulary in tokens
        n_embed: Number of positional embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 8,
        n_embd: int = 32,
        n_head: int = 4,
        device: str = "cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        head_size = n_embd // n_head
        self.self_attention_heads = MultiHeadAttention(
            self.n_head, head_size, n_embd, block_size
        )
        self.language_model_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (batch, time, channel)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)
        x = token_embeddings + position_embeddings
        x = self.self_attention_heads(x)
        logits = self.language_model_head(x)

        loss = None
        if targets is not None:
            # Reshape logits to match (batch, channel) for cross entropy
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100):
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.block_size :]
            logits, _ = self(idx_context)
            # Last time step is the prediction for what comes next
            logits = logits[:, -1, :]  # (batch, time)
            probabilities = F.softmax(logits, dim=-1)  # (batch, time)
            idx_next = torch.multinomial(probabilities, num_samples=1)  # (batch, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, time + 1)
        return idx


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(head_size, n_embd, block_size) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([h(x) for h in self.heads], dim=-1)


class SelfAttentionHead(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores "affinities"
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # noqa
        weights = F.softmax(weights, dim=-1)

        # Perform the weighted aggregation of the values
        v = self.value(x)
        out = weights @ v
        return out
