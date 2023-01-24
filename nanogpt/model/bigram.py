from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class BigramModelConfig:
    batch_size: int = 32
    block_size: int = 8

    max_iters: int = 3000
    eval_interval: int = 300
    eval_iters: int = 200
    learning_rate: float = 1e-2
    train_split: float = 0.9


class BigramLanguageModel(nn.Module):
    """Implements a simple Bigram Language Model"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.token_embedding_table(idx)  # (batch, time, channel)

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
            logits, _ = self(idx)
            # Last time step is the prediction for what comes next
            logits = logits[:, -1, :]  # (batch, time)
            probabilities = F.softmax(logits, dim=-1)  # (batch, time)
            idx_next = torch.multinomial(probabilities, num_samples=1)  # (batch, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, time + 1)
        return idx
