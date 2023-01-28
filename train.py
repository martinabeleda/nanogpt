import dataclasses
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

from nanogpt.model.gpt import GPT


class Trainer:
    def __init__(self, config: DictConfig):
        self.accelerator = Accelerator()
        self.config = config

    def train(self):
        wandb.init(project="nanogpt", config=OmegaConf.to_container(self.config, resolve=True))
        logger.info(f"Resolved config: {OmegaConf.to_yaml(self.config)}")

        tokenizer = self.config.dataset.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        logger.info(f"Tokenizer {tokenizer} with {len(self.tokenizer)} tokens")

        dataset = self.load_dataset()
        logger.debug(f"Dataset: {dataset}")
        logger.info(f"Dataset sample: {dataset['train'][100]}")

        self.dataloaders = self.to_dataloader(self.tokenize_dataset(dataset))

        self.load_model()
        self.load_optimizer()
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.train_loop()
        self.save_model()

    def load_dataset(self):
        return load_dataset(self.config.dataset.name)

    def tokenize_dataset(self, dataset):
        logger.info("Tokenizing dataset")
        tokenize = lambda examples: self.tokenizer(
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
        self.model = GPT(
            config=self.config.model,
            vocab_size=len(self.tokenizer),
            device=self.accelerator.device,
        )

    def load_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
        )

    def train_loop(self):
        self.model.train()
        loss = 0.0
        for epoch in range(self.config.train.epochs):
            logger.info(f"Epoch: {epoch}")
            for batch in tqdm(self.dataloaders["train"]):
                idx, targets = batch["input_ids"], batch["labels"]
                _, loss = self.model(idx, targets)
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

            logger.info(f"Finished epoch: {epoch}, train/loss: {loss}")
            wandb.log({"train/loss": loss})

            self.eval_loop()

    def eval_loop(self):
        metric = evaluate.load(self.config.eval.metric)
        self.model.eval()
        loss = 0.0
        for batch in tqdm(self.dataloaders["eval"]):
            with torch.no_grad():
                idx, targets = batch["input_ids"], batch["labels"]
                logits, loss = self.model(idx, targets)

            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        metric = metric.compute()
        metrics = {"eval/loss": loss, "eval/accuracy": metric["accuracy"]}
        logger.info(f"Evaluation metrics: {metrics}")
        wandb.log(metrics)

        self.show_samples(idx, predictions, targets)

    def show_samples(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        logger.info(f"Sample predictions:")
        inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)
        predicted_sentiment = lambda x: "Positive" if predictions[x] == 1 else "Negative"
        target_sentiment = lambda x: "Positive" if targets[x] == 1 else "Negative"
        for _ in range(min(self.config.eval.num_samples, len(targets))):
            idx = random.randint(0, len(targets) - 1)
            logger.info(
                f"Text: {inputs[idx]} Predicted: {predicted_sentiment(idx)} Target: {target_sentiment(idx)}"
            )

    def save_model(self):
        model_path = "models/model.pth"
        logger.info(f"Saving model to: {model_path}")
        torch.save(self.model.state_dict(), model_path)
        wandb.save(model_path)
        artifact = wandb.Artifact("nanogpt", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
