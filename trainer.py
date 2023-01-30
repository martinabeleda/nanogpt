import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets.load import load_dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nanogpt.model.transformer import Transformer


class LitTransformerSequenceClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, config: DictConfig):
        self.model = model
        self.config = config

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.optimizer.learning_rate
        )
        return optimizer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    logger.info(f"Resolved config: {OmegaConf.to_yaml(config)}")
    pl.seed_everything(config.seed, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer)
    dataset = load_dataset(config.dataset.name)

    logger.info("Tokenizing dataset")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=config.model.block_size,
            truncation=True,
        )

    tokenized_datasets = dataset.map(tokenize, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    DataLoader(
        tokenized_datasets["train"]
        .shuffle(seed=config.seed)
        .select(range(config.train.num_examples)),
        shuffle=True,
        batch_size=config.model.batch_size,
    )
    DataLoader(
        tokenized_datasets["validation"]
        .shuffle(seed=config.seed)
        .select(range(config.val.num_examples)),
        shuffle=True,
        batch_size=config.model.batch_size,
    )
    DataLoader(
        tokenized_datasets["test"]
        .shuffle(seed=config.seed)
        .select(range(config.eval.num_examples)),
        shuffle=True,
        batch_size=config.model.batch_size,
    )

    Transformer(config=config.model, vocab_size=len(tokenizer))


if __name__ == "__main__":
    main()
