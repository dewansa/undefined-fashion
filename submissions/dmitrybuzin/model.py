"""MLP from fashion.ipynb, wrapped in the leaderboard's train interface."""

from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    batch_size: int = 256
    epochs: int = 120
    learning_rate: float = 0.002
    min_learning_rate: float = 0.00002
    weight_decay: float = 0.01
    label_smoothing: float = 0.05
    flip_probability: float = 0.5
    shift_probability: float = 0.35


def load_all(train_set: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale the raw Fashion-MNIST pixels in one operation."""
    if hasattr(train_set, "data") and hasattr(train_set, "targets"):
        images = train_set.data.unsqueeze(1).float().mul_(1 / 255)
        return images, train_set.targets
    return next(iter(DataLoader(train_set, batch_size=len(train_set))))


def train(train_set: Dataset) -> nn.Module:
    cfg = Config()
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

    images, labels = load_all(train_set)
    images, labels = images.to(device), labels.to(device)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=cfg.min_learning_rate
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    for _ in range(cfg.epochs):
        order = torch.randperm(len(images), device=device)
        for indices in order.split(cfg.batch_size):
            inputs = images[indices]
            if torch.rand(()) < cfg.flip_probability:
                inputs = inputs.flip((-1,))
            if torch.rand(()) < cfg.shift_probability:
                top, left = torch.randint(0, 3, (2,)).tolist()
                inputs = F.pad(inputs, (1, 1, 1, 1))[:, :, top : top + 28, left : left + 28]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels[indices])
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model.eval()
