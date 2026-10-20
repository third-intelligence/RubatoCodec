"""Learning-rate schedules for codec training."""

import math

import torch.optim as optim


def build_warmup_decay_scheduler(
    optimizer: optim.Optimizer,
    warmup_step: int = 0,
    down_step: int = 50000,
    max_lr: float = 1e-4,
    min_lr: float = 1e-5,
) -> optim.lr_scheduler.LambdaLR:
    """Build quadratic warmup followed by linear learning-rate decay.

    Args:
        optimizer: Optimizer whose parameter groups have initial ``lr=1.0``.
            The LambdaLR multiplier is the absolute learning rate.
        warmup_step: Number of warmup updates from 1e-5 to ``max_lr``.
            Zero starts directly at ``max_lr``.
        down_step: Number of decay updates from ``max_lr`` to ``min_lr``.
        max_lr: Learning rate at the end of warmup.
        min_lr: Learning rate after decay finishes.

    Returns:
        Scheduler stepped once per optimizer update.

    Raises:
        ValueError: A duration or learning rate is invalid.
    """
    if warmup_step < 0 or down_step <= 0:
        raise ValueError("warmup_step must be nonnegative and down_step must be positive")
    if not all(math.isfinite(v) and v >= 0 for v in (max_lr, min_lr)):
        raise ValueError("Learning rates must be finite and nonnegative")
    alpha = (max_lr - 1e-5) / warmup_step**2 if warmup_step else 0.0

    def lr_lambda(step):
        init_lr = 1e-5
        s1, s2 = warmup_step, warmup_step + down_step
        if step < s1:
            return init_lr + alpha * step**2
        elif s1 <= step < s2:
            return (max_lr - min_lr) / (s1 - s2) * step + (min_lr * s1 - max_lr * s2) / (s1 - s2)
        else:
            return min_lr

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
