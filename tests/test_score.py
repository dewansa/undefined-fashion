"""Tests for the scoring script. They use tiny fake data, so they run in seconds."""

import json
import time
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from fashion.leaderboard import check_result
from fashion.score import (
    Result,
    SubmissionError,
    accuracy,
    count_flops,
    count_parameters,
    counting,
    load_train_function,
    model_bytes,
    readable_flops,
    save,
    train_with_time_limit,
)

SUBMISSIONS = sorted(p for p in Path("submissions").iterdir() if p.is_dir())


def fake_images(count: int = 20) -> TensorDataset:
    images = torch.rand(count, 1, 28, 28)
    labels = torch.randint(0, 10, (count,))
    return TensorDataset(images, labels)


def write_submission(folder: Path, code: str) -> Path:
    folder.mkdir()
    (folder / "model.py").write_text(code)
    return folder


@pytest.mark.parametrize("folder", SUBMISSIONS, ids=lambda p: p.name)
def test_every_submission_has_a_train_function(folder: Path) -> None:
    assert callable(load_train_function(folder))


def test_missing_model_file(tmp_path: Path) -> None:
    with pytest.raises(SubmissionError, match="model.py"):
        load_train_function(tmp_path)


def test_missing_train_function(tmp_path: Path) -> None:
    folder = write_submission(tmp_path / "alice", "x = 1\n")
    with pytest.raises(SubmissionError, match="def train"):
        load_train_function(folder)


def test_train_must_return_a_model() -> None:
    with pytest.raises(SubmissionError, match="nn.Module"):
        train_with_time_limit(lambda train_set: "not a model", fake_images())


def test_training_that_runs_too_long_is_stopped() -> None:
    def slow_train(train_set):
        time.sleep(5)
        return nn.Linear(1, 1)

    start = time.perf_counter()
    with pytest.raises(SubmissionError, match="longer than"):
        train_with_time_limit(slow_train, fake_images(), limit=0.5)
    assert time.perf_counter() - start < 2


def test_accuracy_and_size() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    assert count_parameters(model) == 784 * 10 + 10
    assert 0 <= accuracy(model, fake_images()) <= 1


def test_perfect_model_scores_one() -> None:
    test_set = fake_images()
    answers = torch.eye(10)[test_set.tensors[1]]

    class AlwaysRight(nn.Module):
        def forward(self, images):
            return answers[: len(images)]

    assert accuracy(AlwaysRight(), test_set) == 1.0


def one_training_step(train_set):
    images, labels = train_set.tensors
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(images.device)
    nn.functional.cross_entropy(model(images), labels).backward()
    return model


def test_flops_of_one_training_step() -> None:
    # Forward: 20 images times a 784x10 matrix, 2 FLOPs (multiply, add) per weight.
    # Backward: the same again for the weight gradient. The images need no gradient.
    forward = 2 * 20 * 784 * 10
    count = count_flops(one_training_step, fake_images(20))
    assert count is not None
    assert count.flops == 2 * forward
    assert not count.uncounted  # a plain training step has nothing heavy left uncounted


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a Mac's GPU")
def test_linear_layers_on_a_mac_gpu_count_the_same_as_on_the_cpu() -> None:
    def counted(device: str) -> int:
        images, labels = fake_images(20).tensors
        with counting() as count:
            one_training_step(TensorDataset(images.to(device), labels.to(device)))
        return count.flops

    assert counted("mps") == counted("cpu") == 2 * 2 * 20 * 784 * 10


def test_prediction_compute_is_counted() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    with counting() as count:
        accuracy(model, fake_images(20))
    assert count.flops == 2 * 20 * 784 * 10


def test_ops_without_a_formula_are_reported() -> None:
    # A matrix exponential does real work, but there is no formula for it. It is named,
    # so a reviewer can see the count is too low. Cheap ops are not named.
    with counting() as count:
        torch.linalg.matrix_exp(torch.rand(10, 10))
        torch.rand(4, 10).relu().sum()
    assert set(count.uncounted) == {"aten.linalg_matrix_exp"}


def test_model_size_counts_every_tensor_the_model_keeps() -> None:
    class Memorizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(10, 2)  # 22 numbers
            self.register_buffer("kept", torch.zeros(100))  # 100 numbers
            self.images = torch.zeros(1000)  # plain attribute: 1000 numbers
            self.extra = {"rows": [torch.zeros(50, dtype=torch.uint8)]}  # 50 bytes
            self.same = self.images[:10]  # shares memory with images: free

    assert model_bytes(Memorizer()) == 4 * (22 + 100 + 1000) + 50


def test_counting_does_not_change_the_model() -> None:
    plain, _ = train_with_time_limit(one_training_step, fake_images())
    count_flops(one_training_step, fake_images())
    counted, _ = train_with_time_limit(one_training_step, fake_images())
    assert torch.equal(plain[1].weight, counted[1].weight)


def test_flops_are_none_when_counting_takes_too_long() -> None:
    def slow_train(train_set):
        time.sleep(5)
        return nn.Linear(1, 1)

    assert count_flops(slow_train, fake_images(), limit=0.5) is None


def test_readable_flops() -> None:
    assert readable_flops(0) == "0 FLOP"
    assert readable_flops(224_900_000_000) == "225 GFLOP"
    assert readable_flops(3_272_000_000_000) == "3.3 TFLOP"


def test_save_writes_one_json_file(tmp_path: Path) -> None:
    result = Result("alice", 0.9, 1000, 12.3, "2026-01-02T03:04:05+00:00")
    path = save(result, tmp_path)
    assert path.name == "2026-01-02T03-04-05_alice.json"
    assert json.loads(path.read_text())["accuracy"] == 0.9
    saved = json.loads(path.read_text())
    assert saved["train_flops"] is None  # not counted
    assert saved["predict_flops"] is None
    assert saved["uncounted_ops"] is None


def test_pull_request_result_file_is_accepted(tmp_path: Path) -> None:
    # What the pull request check saves (no --flops) must pass the leaderboard's checks.
    result = Result("alice", 0.9, 1000, 12.3, "2026-01-02T03:04:05+00:00", model_bytes=4000)
    raw = json.loads(save(result, tmp_path).read_text())
    assert check_result(raw, {"alice"})["accuracy"] == 0.9
    assert check_result(raw, {"alice"})["model_bytes"] == 4000


def test_wrong_output_shape_is_an_error_not_a_score() -> None:
    # A [batch, 10, 1] output used to broadcast against the labels and score 400%.
    test_set = TensorDataset(torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 2, 3]))

    class ExtraDimension(nn.Module):
        def forward(self, images):
            return torch.zeros(len(images), 10, 1)

    with pytest.raises(SubmissionError, match=r"\(4, 10\)"):
        accuracy(ExtraDimension(), test_set)


class NearestCentroid(nn.Module):
    """Stores the average image of each class and picks the closest one. No parameters."""

    def __init__(self, centroids: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("centroids", centroids)

    def forward(self, images):
        return -torch.cdist(images.flatten(1), self.centroids)


def test_model_with_zero_parameters_is_scored_and_accepted() -> None:
    def train(train_set):
        images, labels = train_set.tensors
        flat = images.flatten(1)
        return NearestCentroid(torch.stack([flat[labels == c].mean(0) for c in range(10)]))

    labels = torch.arange(20) % 10
    data = TensorDataset(torch.rand(20, 1, 28, 28) + labels.view(-1, 1, 1, 1), labels)
    model, seconds = train_with_time_limit(train, data)

    raw = {
        "name": "alice",
        "accuracy": accuracy(model, data),
        "parameters": count_parameters(model),
        "train_seconds": seconds,
        "date": "2026-01-02T03:04:05+00:00",
        "code_url": None,
    }
    assert raw["parameters"] == 0
    assert raw["accuracy"] == 1.0
    assert check_result(raw, {"alice"})["parameters"] == 0
