"""Baseline: a small two-layer network trained with plain SGD.

To make your own submission, copy this folder to submissions/<your-github-name>/
and change anything you like. The only rule for this file: it must define

    def train(train_set) -> nn.Module

`train_set` holds the 60,000 training images. Each item is (image, label):
the image is a 1x28x28 float tensor with values from 0 to 1, and the label is
a number from 0 to 9. Return your trained model. The scoring script then feeds
it test images in the same format and checks its answers.
"""

from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class Config:
    batch_size: int = 512
    epochs: int = 20
    learning_rate: float = 0.05


def train(train_set: Dataset) -> nn.Module:
    cfg = Config()
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

    # Keep 10,000 training images aside to check how well the model does on images
    # it did not train on. You never get the real test images.
    train_split, valid_split = random_split(train_set, [50_000, 10_000])
    train_loader = DataLoader(train_split, cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_split, cfg.batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=10),
    )
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Loss values over time, in case you want to plot them.
    history = {"loss/train": [], "loss/train_batch": [], "loss/valid": []}

    def emit_metrics(metric: str, step: int, value: float) -> None:
        history[metric].append((step, value))

    step = -1
    for epoch in range(cfg.epochs):
        train_loss = 0.0
        for images, targets in train_loader:
            step += 1
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()

            emit_metrics("loss/train_batch", step, loss.item())
            train_loss += loss.item()

            optimizer.step()
        train_loss /= len(train_loader)
        emit_metrics("loss/train", step, train_loss)

        with torch.no_grad():
            valid_loss = 0.0
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                valid_loss += loss.item()
            valid_loss /= len(valid_loader)
            emit_metrics("loss/valid", step, valid_loss)

        print(f"{epoch=} {train_loss=:.4f} {valid_loss=:.4f}")

    return model
