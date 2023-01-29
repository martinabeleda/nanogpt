import random

import evaluate
import hydra
import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanogpt.model.transformer import Transformer


class Trainer:
    def __init__(self, config: DictConfig):
        self.accelerator = Accelerator()
        logger.info(f"Device: {self.accelerator.device}")

        self.config = config

    @property
    def device(self) -> str:
        return self.accelerator.device

    def train(self):
        wandb.init(
            project="nanogpt",
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        logger.info(f"Resolved config: {OmegaConf.to_yaml(self.config)}")

        tokenizer = self.config.dataset.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        logger.info(f"Tokenizer {tokenizer} with {len(self.tokenizer)} tokens")

        datasets = self.load_dataset()
        logger.debug(f"Dataset: {datasets}")
        logger.info(f"Dataset sample: {datasets['train'][100]}")

        self.dataloaders = self.to_dataloader(self.tokenize_dataset(datasets))

        self.load_model()
        self.load_optimizer()
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        self.train_loop()
        self.save_model()

    def load_dataset(self):
        return load_dataset(self.config.dataset.name)

    def tokenize_dataset(self, dataset):
        logger.info("Tokenizing dataset")

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                max_length=self.config.model.block_size,
                truncation=True,
            )

        tokenized_datasets = dataset.map(tokenize, batched=True)

        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        return tokenized_datasets

    def to_dataloader(self, datasets):
        train_dataset = (
            datasets["train"]
            .shuffle(seed=self.config.seed)
            .select(range(self.config.train.num_examples))
        )
        eval_dataset = (
            datasets["test"]
            .shuffle(seed=self.config.seed)
            .select(range(self.config.eval.num_examples))
        )
        return {
            "train": DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=self.config.model.batch_size,
            ),
            "eval": DataLoader(
                eval_dataset,
                batch_size=self.config.model.batch_size,
            ),
        }

    def load_model(self):
        self.model = Transformer(
            config=self.config.model,
            vocab_size=len(self.tokenizer),
            device=self.device,
        )
        self.model.to(self.device)

    def load_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
        )

    def train_loop(self):
        metric = evaluate.load(self.config.train.metric)
        self.model.train()
        loss = 0.0
        for epoch in range(self.config.train.epochs):
            logger.info(f"Epoch: {epoch}")
            for batch in tqdm(self.dataloaders["train"]):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                inputs, targets = batch["input_ids"], batch["labels"]
                logits, loss = self.model(inputs, targets)
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

                predictions = logits.argmax(axis=1)
                metric.add_batch(predictions=predictions, references=targets)

            metrics = {
                "train/loss": loss.item(),
                "train/accuracy": metric.compute()["accuracy"],
            }
            wandb.log(metrics)
            logger.info(f"Metrics: {metrics}")

            self.eval_loop()

    def eval_loop(self):
        metric = evaluate.load(self.config.eval.metric)
        self.model.eval()
        loss = 0.0
        for batch in tqdm(self.dataloaders["eval"]):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                inputs, targets = batch["input_ids"], batch["labels"]
                logits, loss = self.model(inputs, targets)
                predictions = logits.argmax(axis=1)

            metric.add_batch(predictions=predictions, references=targets)

        metrics = {
            "eval/loss": loss.item(),
            "eval/accuracy": metric.compute()["accuracy"],
        }
        logger.info(f"Evaluation metrics: {metrics}")
        wandb.log(metrics)

        self.show_samples(inputs, predictions, targets)

    def show_samples(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        logger.info("Sample predictions:")
        inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)

        def predicted_sentiment(x):
            return "Positive" if predictions[x] == 1 else "Negative"

        def target_sentiment(x):
            return "Positive" if targets[x] == 1 else "Negative"

        for _ in range(min(self.config.eval.num_samples, len(targets))):
            index = random.randint(0, len(targets) - 1)
            msg = (
                f"Text: {inputs[index]}\n"
                f"Predicted: {predicted_sentiment(index)}\n"
                f"Target: {target_sentiment(index)}\n"
            )
            logger.info(msg)

    def save_model(self):
        model_path = "models/model.pth"
        logger.info(f"Saving model to: {model_path}")
        torch.save(self.model.state_dict(), model_path)
        wandb.save(model_path)
        artifact = wandb.Artifact("nanogpt", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
