"""GPT training script with DDP, mixed precision, gradient accumulation, and wandb logging.

Single GPU:
    $ uv run python train.py --config-name train_shakespeare_char

Multi-GPU with DDP:
    $ torchrun --standalone --nproc_per_node=4 train.py --config-name train_gpt2

Multi-node DDP:
    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
        --master_addr=<IP> --master_port=1234 train.py --config-name train_gpt2
"""

import math
import os
import pickle
import time
from contextlib import nullcontext

import hydra
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from nanogpt.model.gpt import GPT, GPTConfig


class Trainer:
    """Handles the full GPT training loop including DDP, mixed precision, and checkpointing."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self._setup_distributed()
        self._setup_device()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_compiler()
        self._setup_ddp_wrapper()

    def _setup_distributed(self) -> None:
        """Initialize DDP if running under torchrun."""
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            from torch.distributed import init_process_group

            init_process_group(backend=self.config.backend)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            assert self.config.gradient_accumulation_steps % self.ddp_world_size == 0
            self.gradient_accumulation_steps = (
                self.config.gradient_accumulation_steps // self.ddp_world_size
            )
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.device = self.config.device
            self.gradient_accumulation_steps = self.config.gradient_accumulation_steps

        self.tokens_per_iter = (
            self.gradient_accumulation_steps
            * self.ddp_world_size
            * self.config.batch_size
            * self.config.block_size
        )
        if self.master_process:
            logger.info(f"Tokens per iteration: {self.tokens_per_iter:,}")

    def _setup_device(self) -> None:
        """Configure device, dtype, and autocast context."""
        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }

        dtype_str = self.config.dtype
        if dtype_str == "auto":
            dtype_str = (
                "bfloat16"
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else "float16"
            )

        self.ptdtype = dtype_map[dtype_str]
        self.dtype_str = dtype_str
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        )

        if self.master_process:
            os.makedirs(self.config.out_dir, exist_ok=True)

    def _setup_data(self) -> None:
        """Locate the binary dataset files and detect vocab size from meta.pkl."""
        self.data_dir = self.config.get("data_dir") or os.path.join(
            "data", self.config.dataset
        )

        self.meta_vocab_size: int | None = None
        meta_path = os.path.join(self.data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta["vocab_size"]
            logger.info(f"Found vocab_size = {self.meta_vocab_size} (from {meta_path})")

    def _get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a random batch from the memory-mapped binary dataset."""
        filename = "train.bin" if split == "train" else "val.bin"
        data = np.memmap(
            os.path.join(self.data_dir, filename), dtype=np.uint16, mode="r"
        )
        ix = torch.randint(
            len(data) - self.config.block_size, (self.config.batch_size,)
        )
        x = torch.stack(
            [
                torch.from_numpy(data[i : i + self.config.block_size].astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    data[i + 1 : i + 1 + self.config.block_size].astype(np.int64)
                )
                for i in ix
            ]
        )
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def _setup_model(self) -> None:
        """Initialize the GPT model from scratch, checkpoint, or pretrained weights."""
        cfg = self.config
        self.model_args = dict(
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            block_size=cfg.block_size,
            bias=cfg.bias,
            vocab_size=None,
            dropout=cfg.dropout,
        )

        self.iter_num = 0
        self.best_val_loss = 1e9

        if cfg.init_from == "scratch":
            logger.info("Initializing a new model from scratch")
            if self.meta_vocab_size is None:
                logger.info(
                    "Defaulting to vocab_size of GPT-2: 50304 (50257 rounded up)"
                )
            self.model_args["vocab_size"] = self.meta_vocab_size or 50304
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)

        elif cfg.init_from == "resume":
            logger.info(f"Resuming training from {cfg.out_dir}")
            ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint["model_args"]
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                self.model_args[k] = checkpoint_model_args[k]
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
            self._resume_checkpoint = checkpoint
            # Auto-recover wandb run ID from checkpoint if not explicitly set
            if not cfg.wandb_run_id and checkpoint.get("wandb_run_id"):
                self._resumed_wandb_run_id = checkpoint["wandb_run_id"]
                logger.info(f"Found wandb run ID in checkpoint: {self._resumed_wandb_run_id}")

        elif cfg.init_from.startswith("gpt2"):
            logger.info(f"Initializing from OpenAI GPT-2 weights: {cfg.init_from}")
            override_args = dict(dropout=cfg.dropout)
            self.model = GPT.from_pretrained(cfg.init_from, override_args)
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                self.model_args[k] = getattr(self.model.config, k)
        else:
            raise ValueError(f"Unknown init_from: {cfg.init_from}")

        if cfg.block_size < self.model.config.block_size:
            self.model.crop_block_size(cfg.block_size)
            self.model_args["block_size"] = cfg.block_size

        self.model.to(self.device)

    def _setup_optimizer(self) -> None:
        """Configure the optimizer and GradScaler for mixed precision."""
        cfg = self.config
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.dtype_str == "float16")
        )
        self.optimizer = self.model.configure_optimizers(
            cfg.weight_decay,
            cfg.learning_rate,
            (cfg.beta1, cfg.beta2),
            self.device_type,
        )
        if cfg.init_from == "resume" and hasattr(self, "_resume_checkpoint"):
            self.optimizer.load_state_dict(self._resume_checkpoint["optimizer"])
            del self._resume_checkpoint

    def _setup_compiler(self) -> None:
        """Optionally compile the model with torch.compile."""
        if self.config.compile:
            logger.info("Compiling the model (takes ~1 minute)...")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)

    def _setup_ddp_wrapper(self) -> None:
        """Wrap model in DDP if running distributed."""
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    @torch.no_grad()
    def _estimate_loss(self) -> dict[str, float]:
        """Estimate loss over multiple batches for train and val splits."""
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self._get_batch(split)
                with self.ctx:
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def _get_lr(self, it: int) -> float:
        """Cosine learning rate schedule with linear warmup."""
        cfg = self.config
        if it < cfg.warmup_iters:
            return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def _save_checkpoint(self, val_loss: float) -> None:
        """Save model and optimizer state to disk and optionally to wandb."""
        raw_model = self.model.module if self.ddp else self.model
        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_args": self.model_args,
            "iter_num": self.iter_num,
            "best_val_loss": val_loss,
            "config": OmegaConf.to_container(self.config, resolve=True),
            "wandb_run_id": wandb.run.id if self.config.wandb_log and wandb.run else None,
        }
        ckpt_path = os.path.join(self.config.out_dir, "ckpt.pt")
        logger.info(f"Saving checkpoint to {self.config.out_dir}")
        torch.save(checkpoint, ckpt_path)

        if self.config.wandb_log:
            wandb.save(ckpt_path)
            artifact = wandb.Artifact(
                f"nanogpt-{self.config.dataset}",
                type="model",
                metadata={"iter_num": self.iter_num, "val_loss": val_loss},
            )
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

    def train(self) -> None:
        """Run the main training loop."""
        cfg = self.config
        resolved_config = OmegaConf.to_container(cfg, resolve=True)
        logger.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")

        if cfg.wandb_log and self.master_process:
            wandb_run_id = cfg.wandb_run_id or getattr(self, "_resumed_wandb_run_id", None)
            wandb_kwargs = dict(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=resolved_config,
            )
            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
                wandb_kwargs["resume"] = "must"
            wandb.init(**wandb_kwargs)

        raw_model = self.model.module if self.ddp else self.model
        running_mfu = -1.0

        X, Y = self._get_batch("train")
        t0 = time.time()
        local_iter_num = 0

        while True:
            # Set learning rate.
            lr = self._get_lr(self.iter_num) if cfg.decay_lr else cfg.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluate and checkpoint.
            if self.iter_num % cfg.eval_interval == 0 and self.master_process:
                losses = self._estimate_loss()
                logger.info(
                    f"Step {self.iter_num}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )
                if cfg.wandb_log:
                    wandb.log(
                        {
                            "iter": self.iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,
                        }
                    )
                if losses["val"] < self.best_val_loss or cfg.always_save_checkpoint:
                    self.best_val_loss = losses["val"]
                    if self.iter_num > 0:
                        self._save_checkpoint(self.best_val_loss)

            if self.iter_num == 0 and cfg.eval_only:
                break

            # Forward/backward with gradient accumulation.
            for micro_step in range(self.gradient_accumulation_steps):
                if self.ddp:
                    self.model.require_backward_grad_sync = (
                        micro_step == self.gradient_accumulation_steps - 1
                    )
                with self.ctx:
                    _, loss = self.model(X, Y)
                    loss = loss / self.gradient_accumulation_steps
                X, Y = self._get_batch("train")
                self.scaler.scale(loss).backward()

            if cfg.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Timing and logging.
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % cfg.log_interval == 0 and self.master_process:
                lossf = loss.item() * self.gradient_accumulation_steps
                if local_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(
                        self.config.batch_size * self.gradient_accumulation_steps, dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                logger.info(
                    f"Iter {self.iter_num}: "
                    f"loss {lossf:.4f}, "
                    f"time {dt * 1000:.2f}ms, "
                    f"mfu {running_mfu * 100:.2f}%"
                )

            self.iter_num += 1
            local_iter_num += 1

            if self.iter_num > cfg.max_iters:
                break

        if self.ddp:
            from torch.distributed import destroy_process_group

            destroy_process_group()

        if cfg.wandb_log and self.master_process:
            wandb.finish()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
