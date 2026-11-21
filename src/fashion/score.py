"""Score a submission: train it, then test it on images it has never seen.

Run it like this:

    uv run score submissions/<your-github-name>

Add `--flops` to also count the compute (FLOPs): of training, and of labelling
each test image. That trains each submission a second time, so it takes longer.

GitHub runs the same command on pull requests and after merges, with `--save`
to keep the results for the leaderboard. After merges it also adds `--flops`.
"""

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fashion import data
from fashion.flops import Count, counting
from fashion.leaderboard import readable_bytes

TIME_LIMIT_SECONDS = 10 * 60
# Counting slows training down 2 to 6 times, so the counting run gets more time.
COUNT_LIMIT_SECONDS = 30 * 60

type TrainFunction = Callable[[Dataset], nn.Module]


class SubmissionError(Exception):
    """Something is wrong with a submission. The message says what, in plain words."""


@dataclass
class Result:
    name: str
    accuracy: float
    parameters: int
    train_seconds: float
    date: str
    code_url: str | None = None
    model_bytes: int | None = None  # memory the returned model holds on to
    train_flops: int | None = None  # None when compute was not counted
    predict_flops: int | None = None  # per test image; None when not counted
    uncounted_ops: list[str] | None = None  # heavy-looking ops the counter skipped


def load_train_function(folder: Path) -> TrainFunction:
    """Import model.py from a submission folder and return its train() function."""
    path = folder / "model.py"
    if not path.is_file():
        raise SubmissionError(f"{path} not found. Each submission needs a model.py file.")

    spec = importlib.util.spec_from_file_location(f"submission_{folder.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    train = getattr(module, "train", None)
    if not callable(train):
        raise SubmissionError(f"{path} must define: def train(train_set) -> nn.Module")
    return train


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Stop the code inside if it runs longer than `seconds`.

    This uses an alarm signal, which exists on Linux and macOS. On Windows the
    code is not stopped early, but train_with_time_limit() still rejects it.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def stop(signum, frame):
        raise SubmissionError(f"Training took longer than the limit of {seconds:.0f} seconds.")

    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def train_with_time_limit(
    train: TrainFunction, train_set: Dataset, limit: float = TIME_LIMIT_SECONDS
) -> tuple[nn.Module, float]:
    """Run the student's train() and return the model and how many seconds it took."""
    torch.manual_seed(0)  # same starting point for everyone, so runs are repeatable
    start = time.perf_counter()
    with time_limit(limit):
        model = train(train_set)
    seconds = time.perf_counter() - start

    if seconds > limit:
        raise SubmissionError(f"Training took {seconds:.0f} seconds. The limit is {limit:.0f}.")
    if not isinstance(model, nn.Module):
        raise SubmissionError("train() must return a PyTorch model (an nn.Module).")
    return model, seconds


def model_bytes(model: nn.Module) -> int:
    """How much memory the model holds on to, in bytes.

    Parameters count, and so do buffers and any other tensor the model keeps: in a
    plain attribute, a list, a dict, or another object it points to. A model that
    stores the training images is as big as those images. Tensors that share memory
    are counted once.
    """
    seen_objects: set[int] = set()
    seen_memory: set[tuple[str, int]] = set()
    total = 0
    todo: list[object] = [model]
    while todo:
        obj = todo.pop()
        if id(obj) in seen_objects:
            continue
        seen_objects.add(id(obj))
        if isinstance(obj, torch.Tensor):
            try:
                storage = obj.untyped_storage()
                key, size = (str(storage.device), storage.data_ptr()), storage.nbytes()
            except RuntimeError, NotImplementedError:  # sparse tensors have no single storage
                key, size = (str(obj.device), id(obj)), obj.numel() * obj.element_size()
            if key not in seen_memory:
                seen_memory.add(key)
                total += size
        elif isinstance(obj, dict):
            todo.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            todo.extend(obj)
        elif hasattr(obj, "__dict__") and not isinstance(
            obj, (type, ModuleType, FunctionType, MethodType, BuiltinFunctionType)
        ):
            todo.extend(vars(obj).values())
    return total


def count_parameters(model: nn.Module) -> int:
    """How many numbers the model learns by gradient descent (its parameters).

    Buffers, such as stored class averages, are not counted. A model can have 0.
    """
    return sum(p.numel() for p in model.parameters())


def count_flops(
    train: TrainFunction, train_set: Dataset, limit: float = COUNT_LIMIT_SECONDS
) -> Count | None:
    """Train again inside PyTorch's FLOP counter and return the training compute.

    The counter adds up the floating-point operations of every matrix multiply and
    convolution, forward and backward, including any validation inside train(). It
    skips cheap elementwise work such as ReLU or optimizer steps. The submission's
    code is not changed: the counter only watches the operations as they run.

    Counting slows training down, so this is a separate run from the timed one, with
    its own limit. If it does not finish, the result has no compute (None).
    """
    try:
        with counting() as count:
            train_with_time_limit(train, train_set, limit)
    except SubmissionError as error:
        print(f"Could not count the compute: {error}", file=sys.stderr)
        return None
    return count


def readable_flops(flops: int) -> str:
    """225000000000 -> "225 GFLOP", the same short form as the leaderboard page."""
    for suffix, size in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if flops >= size:
            value = flops / size
            return f"{value:.1f} {suffix}FLOP" if value < 10 else f"{value:.0f} {suffix}FLOP"
    return f"{flops} FLOP"


def accuracy(model: nn.Module, test_set: Dataset) -> float:
    """The share of test images the model labels correctly, from 0 to 1."""
    model = model.to("cpu").eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in DataLoader(test_set, batch_size=1000):
            outputs = model(images)
            if outputs.ndim != 2 or len(outputs) != len(labels):
                raise SubmissionError(
                    f"For a batch of {len(labels)} images the model returned shape "
                    f"{tuple(outputs.shape)}. It must return one row of class scores per "
                    f"image, shape ({len(labels)}, 10)."
                )
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return correct / total


def code_url(name: str) -> str | None:
    """A link to the submission's code on GitHub, when running inside GitHub Actions."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    if not (server and repo and sha):
        return None
    return f"{server}/{repo}/blob/{sha}/submissions/{name}/model.py"


def score(folder: Path, limit: float = TIME_LIMIT_SECONDS, flops: bool = False) -> Result:
    """Train one submission and measure it. With flops=True, also count its compute."""
    train = load_train_function(folder)
    # Load both sets before training. Downloading the first time is not counted as
    # training time, and the test images are already in memory before student code runs.
    train_set = data.train_set()
    test_set = data.test_set()
    model, seconds = train_with_time_limit(train, train_set, limit)
    if flops:
        # Labelling the test images is counted too: a model can be cheap to train
        # and costly to use, like one that compares each image with every training image.
        with counting() as predicted:
            test_accuracy = accuracy(model, test_set)
    else:
        test_accuracy = accuracy(model, test_set)
    result = Result(
        name=folder.name,
        accuracy=round(test_accuracy, 4),
        parameters=count_parameters(model),
        train_seconds=round(seconds, 1),
        date=datetime.now(UTC).isoformat(timespec="seconds"),
        code_url=code_url(folder.name),
        model_bytes=model_bytes(model),
    )
    if flops:
        result.predict_flops = round(predicted.flops / len(test_set))
        trained = count_flops(train, train_set)
        uncounted = predicted.uncounted
        if trained is not None:
            result.train_flops = trained.flops
            uncounted = trained.uncounted + uncounted
        result.uncounted_ops = sorted(uncounted)
    return result


def save(result: Result, results_dir: Path) -> Path:
    """Write the result as a small JSON file, one file per scored submission."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.date.replace(":", "-").removesuffix("+00-00")
    path = results_dir / f"{stamp}_{result.name}.json"
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train submissions and print their scores.")
    parser.add_argument("folders", type=Path, nargs="+", help="for example: submissions/baseline")
    parser.add_argument("--save", type=Path, metavar="DIR", help="also save the results here")
    parser.add_argument(
        "--flops",
        action="store_true",
        help="also count the compute (FLOPs) of training and of labelling each test image; "
        "trains each submission a second time",
    )
    args = parser.parse_args()

    failed = []
    for folder in args.folders:
        try:
            result = score(folder, flops=args.flops)
        except SubmissionError as error:
            print(f"Error in {folder}: {error}", file=sys.stderr)
            failed.append(folder.name)
            continue

        print()
        print(f"Submission:     {result.name}")
        print(f"Test accuracy:  {result.accuracy:.2%}")
        print(f"Parameters:     {result.parameters:,}")
        print(f"Model size:     {readable_bytes(result.model_bytes or 0)}")
        print(f"Training time:  {result.train_seconds:.0f} seconds")
        if result.train_flops is not None:
            print(f"Compute:        {readable_flops(result.train_flops)} to train")
        if result.predict_flops is not None:
            print(f"                {readable_flops(result.predict_flops)} per test image")
        if result.uncounted_ops:
            print(
                f"Not counted:    {', '.join(result.uncounted_ops)}\n"
                "                These ran but have no FLOP formula. If they do heavy math,\n"
                "                the compute above is too low."
            )
        if args.save:
            print(f"Saved to:       {save(result, args.save)}")

    if failed:
        sys.exit(f"Could not score: {', '.join(failed)}")
