from dataclasses import asdict
import random

import torch
from torch.utils.data import DataLoader
import wandb
from loguru import logger
from accelerate import Accelerator
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import evaluate

from nanogpt.criterion import estimate_loss
from nanogpt.model.gpt import GPT, GPTConfig


class Trainer:
    def __init__(self):
        self.accelerator = Accelerator()
        self.config = GPTConfig()

    def train(self):
        wandb.init(project="nanogpt", config=asdict(self.config))
        logger.info(f"Running config: {self.config.__class__.__name__}")
        logger.info(f"Resolved config: {self.config.__dict__}")

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        logger.info(f"Tokenizer {self.tokenizer} with {len(self.tokenizer)} tokens")

        dataset = self.load_dataset()
        logger.debug(f"Dataset: {dataset}")
        logger.info(f"Dataset sample: {dataset['train'][100]}")

        self.dataloaders = self.to_dataloader(self.tokenize_dataset(dataset))

        self.load_model()
        self.load_optimizer()
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        self.train_loop()
        self.eval_loop()
        self.save_model()

    def load_dataset(self):
        return load_dataset("rotten_tomatoes")

    def tokenize_dataset(self, dataset):
        logger.info("Tokenizing dataset")
        tokenize = lambda examples: self.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=self.config.block_size,
            truncation=True,
        )
        tokenized_datasets = dataset.map(tokenize, batched=True)

        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        return tokenized_datasets

    def to_dataloader(self, datasets):
        train_dataset = datasets["train"].shuffle(seed=1337).select(range(100))
        eval_dataset = datasets["test"].shuffle(seed=1337).select(range(100))
        return {
            "train": DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=self.config.batch_size,
            ),
            "eval": DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
            ),
        }

    def load_model(self):
        self.model = GPT(
            config=self.config,
            vocab_size=len(self.tokenizer),
            device=self.accelerator.device,
        )

    def load_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    def train_loop(self):
        self.model.train()
        loss = 0.0
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch: {epoch}")
            for batch in tqdm(self.dataloaders["train"]):
                idx, targets = batch["input_ids"], batch["labels"]
                _, loss = self.model(idx, targets)
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

            logger.info(f"Finished epoch: {epoch}, Loss: {loss}")
            wandb.log({"train/loss": loss})

    def eval_loop(self):
        metric = evaluate.load("accuracy")
        self.model.eval()
        loss = 0.0
        for batch in tqdm(self.dataloaders["eval"]):
            with torch.no_grad():
                idx, targets = batch["input_ids"], batch["labels"]
                logits, loss = self.model(idx, targets)

            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        metric = metric.compute()
        metric["eval/loss"] = loss
        wandb.log(metric)

        logger.info(f"Sample predictions:")
        inputs = self.tokenizer.batch_decode(idx, skip_special_tokens=True)
        predicted_sentiment = (
            lambda x: "Positive" if predictions[x] == 1 else "Negative"
        )
        target_sentiment = lambda x: "Positive" if targets[x] == 1 else "Negative"
        for _ in range(min(10, len(targets))):
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


if __name__ == "__main__":
    Trainer().train()
