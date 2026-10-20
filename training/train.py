import inspect
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from training.data_module import DataModule
from training.lightning_module import CodecLightningModule, strip_secrets

seed = 1024
seed_everything(seed)


def _validate_init_config(path, cfg):
    """Check the architecture recorded alongside an inference export."""
    from rubatocodec.checkpoint import clean_inference_config
    from rubatocodec.firing import FiringModule

    config_path = Path(path).with_name("config.json")
    with config_path.open() as f:
        saved = json.load(f)
    current = OmegaConf.to_container(cfg.model, resolve=True)
    current.update(
        semantic_model=cfg.train.get("semantic_model", "facebook/w2v-bert-2.0"),
        semantic_layer_index=cfg.train.get("semantic_layer_index", 16),
        sr=cfg.dataset.audio.sr,
        output_sr=cfg.dataset.audio.get("output_sr", cfg.dataset.audio.sr),
    )
    firing_defaults = {
        k: v.default
        for k, v in inspect.signature(FiringModule).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    firing_defaults.update(aggregate="none")

    def normalize(config):
        firing = config.get("firing")
        if firing and firing.get("enabled", False):
            config = dict(config, firing={**firing_defaults, **firing})
        result = clean_inference_config(config)
        result.pop("inference", None)
        return json.loads(json.dumps(result))

    saved, current = normalize(saved), normalize(current)

    def differences(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            return [
                item
                for key in sorted(a.keys() | b.keys())
                for item in differences(a.get(key), b.get(key), f"{prefix}.{key}".lstrip("."))
            ]
        return [] if a == b else [f"{prefix}: checkpoint={a!r}, training={b!r}"]

    mismatch = differences(saved, current)
    if mismatch:
        raise ValueError("init_ckpt config mismatch:\n" + "\n".join(mismatch))


def _initialize_weights(module, path, cfg):
    """Initialize model weights without restoring optimizer or step state."""
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        from rubatocodec.checkpoint import INFERENCE_PREFIXES

        _validate_init_config(path, cfg)
        sd = load_file(str(path), device="cpu")
        target = module.state_dict()
        names = {
            key.replace("_orig_mod.", ""): key
            for key in target
            if key.replace("_orig_mod.", "").startswith(INFERENCE_PREFIXES)
        }
        missing = sorted(names.keys() - sd.keys())
        unexpected = sorted(sd.keys() - names.keys())
        if missing or unexpected:
            raise RuntimeError(
                f"init_ckpt codec weights mismatch: missing={missing[:10]}, "
                f"unexpected={unexpected[:10]}"
            )
        wrong_shapes = [key for key in sd if sd[key].shape != target[names[key]].shape]
        if wrong_shapes:
            raise RuntimeError(f"init_ckpt codec tensor shapes mismatch: {wrong_shapes[:10]}")
        module.load_state_dict({names[key]: value for key, value in sd.items()}, strict=False)
        print(
            f"Initialized codec weights from {path}; training-only modules retain their initialization."
        )
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = module.load_state_dict(sd, strict=False)
        bad = [key for key in missing if not key.startswith(("firing", "perceptual_loss_fn"))]
        if bad:
            raise RuntimeError(f"init_ckpt is missing base weights: {bad[:10]}")
        print(
            f"Initialized weights from {path}: missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(f"  missing keys: {missing[:8]}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")


@hydra.main(config_path="config", config_name="default", version_base=None)
def train(cfg):
    # Resolve against the launch directory (Hydra changes the working directory).
    log_dir = hydra.utils.to_absolute_path(cfg.log_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        save_top_k=-1,
        save_last=True,
        every_n_train_steps=cfg.train.get("ckpt_every_n_steps", 20000),
        monitor="val/mel_loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    datamodule = DataModule(cfg)

    # Auto-calculate max_steps from epochs if specified
    epochs = cfg.train.get("epochs", None)
    if epochs is not None:
        train_loader = datamodule.train_dataloader()
        num_devices = cfg.train.trainer.get("devices", 1)
        num_nodes = cfg.train.trainer.get("num_nodes", 1)
        world_size = num_devices * num_nodes
        steps_per_epoch = len(train_loader) // world_size
        max_steps = steps_per_epoch * epochs
        cfg.train.trainer.max_steps = max_steps
        cfg.train.trainer.min_steps = max_steps
        # Update scheduler down_step to match
        warmup = cfg.train.gen_schedule_params.warmup_step
        cfg.train.gen_schedule_params.down_step = max_steps - warmup
        cfg.train.disc_schedule_params.down_step = max_steps - warmup
        print(
            f"Epochs mode: {epochs} epoch(s) x {steps_per_epoch} steps/epoch "
            f"= {max_steps} max_steps"
        )

    lightning_module = CodecLightningModule(cfg)
    wandb_name = cfg.wandb.name or os.path.basename(os.path.normpath(log_dir))
    # Strip tokens from wandb config to avoid logging secrets
    wandb_cfg = OmegaConf.to_container(strip_secrets(cfg), resolve=True)
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=wandb_name,
        config=wandb_cfg,
    )

    ckpt_path = None
    last_ckpt = os.path.join(log_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        ckpt_path = last_ckpt
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        init_ckpt = cfg.get("init_ckpt")
        if init_ckpt:
            _initialize_weights(lightning_module, hydra.utils.to_absolute_path(init_ckpt), cfg)
        else:
            print("No checkpoint found, starting training from scratch.")

    trainer = pl.Trainer(
        **cfg.train.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=wandb_logger,
        profiler="simple",
        use_distributed_sampler=False,
    )
    torch.backends.cudnn.benchmark = True
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
    print(
        f"Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    train()
