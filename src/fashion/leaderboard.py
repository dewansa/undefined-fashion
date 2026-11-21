"""Keep the leaderboard's rows. Uses only the Python standard library.

There are two kinds of rows:

- Reviewed: results/<date>_<name>.json. The teacher merged the submission, and
  GitHub scored it again on main. Every merged submission keeps its own row.
- Not reviewed: results/pending/pr-<number>_<name>.json. Scored on an open pull
  request. Each new push replaces the pull request's row, and the row goes away
  when the pull request is merged or closed.

The pull request scores come from a job that ran student code, so they are
checked strictly here before they are trusted.

GitHub Actions runs this file directly (python src/fashion/leaderboard.py ...),
without installing PyTorch.
"""

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,38}")
MAX_FILES = 10
MAX_FILE_BYTES = 10_000
MAX_TRAIN_SECONDS = 10 * 60
MAX_PARAMETERS = 1_000_000_000
EXPECTED_KEYS = {"name", "accuracy", "parameters", "train_seconds", "date", "code_url"}
# Compute (FLOPs) is counted only after a merge. Pull request results may carry the
# key, but only empty: a pull request job that reports compute has been tampered with.
OPTIONAL_KEYS = {"train_flops"}


class InvalidResult(Exception):
    """A pull request result that does not look right. We skip it."""


def check_result(raw: object, allowed_names: set[str]) -> dict:
    """Check one result from a pull request and return only the trusted fields."""
    if not isinstance(raw, dict) or not EXPECTED_KEYS <= set(raw) <= EXPECTED_KEYS | OPTIONAL_KEYS:
        raise InvalidResult("the result does not have the expected fields")
    if raw.get("train_flops") is not None:
        raise InvalidResult("compute is counted after the merge, not on pull requests")

    name = raw["name"]
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise InvalidResult("the submission name is not valid")
    if name not in allowed_names:
        raise InvalidResult(f"{name} is not a folder this pull request changes")

    accuracy = raw["accuracy"]
    if type(accuracy) not in (int, float) or not 0 <= accuracy <= 1:
        raise InvalidResult("accuracy must be a number from 0 to 1")

    parameters = raw["parameters"]
    if type(parameters) is not int or not 0 <= parameters <= MAX_PARAMETERS:
        raise InvalidResult("parameters must be a whole number, 0 or more")

    seconds = raw["train_seconds"]
    is_number = type(seconds) in (int, float) and math.isfinite(seconds)
    if not is_number or not 0 <= seconds <= MAX_TRAIN_SECONDS:
        raise InvalidResult(f"training time must be from 0 to {MAX_TRAIN_SECONDS} seconds")

    return {
        "name": name,
        "accuracy": round(float(accuracy), 4),
        "parameters": parameters,
        "train_seconds": round(float(seconds), 1),
    }


def read_pr_results(folder: Path, allowed_names: set[str]) -> list[dict]:
    """Read the result files a pull request job uploaded. Bad files are skipped."""
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    results = []
    for path in files[:MAX_FILES]:
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise InvalidResult("the file is too big")
            results.append(check_result(json.loads(path.read_text()), allowed_names))
        except (InvalidResult, ValueError) as error:
            print(f"Skipping {path.name}: {error}")
    return results


def add_pr_rows(results_dir: Path, pr: int, pr_url: str, results: list[dict]) -> None:
    """Replace a pull request's not reviewed rows with new ones."""
    pending = results_dir / "pending"
    remove_pr_rows(results_dir, pr)
    pending.mkdir(parents=True, exist_ok=True)
    date = datetime.now(UTC).isoformat(timespec="seconds")
    for result in results:
        row = {**result, "date": date, "reviewed": False, "pr": pr, "code_url": pr_url}
        path = pending / f"pr-{pr}_{result['name']}.json"
        path.write_text(json.dumps(row, indent=2) + "\n")


def remove_pr_rows(results_dir: Path, pr: int) -> None:
    for path in (results_dir / "pending").glob(f"pr-{pr}_*.json"):
        path.unlink()


def remove_closed_pr_rows(results_dir: Path, open_prs: set[int]) -> None:
    """Remove not reviewed rows whose pull request was merged or closed."""
    for path in (results_dir / "pending").glob("pr-*_*.json"):
        number = int(path.name.removeprefix("pr-").split("_")[0])
        if number not in open_prs:
            path.unlink()


def all_rows(results_dir: Path) -> list[dict]:
    """Every row on the leaderboard, reviewed and not reviewed."""
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        rows.append({**json.loads(path.read_text()), "reviewed": True})
    for path in sorted((results_dir / "pending").glob("*.json")):
        rows.append({**json.loads(path.read_text()), "reviewed": False})
    return rows


def comment(results: list[dict], leaderboard_url: str) -> str:
    """The text of the bot's comment on the pull request."""
    if not results:
        return "I could not score this pull request. Open the Check run's log to see why."
    lines = ["Here is the score for this pull request:", ""]
    for r in results:
        lines.append(
            f"- **{r['name']}**: {r['accuracy']:.2%} test accuracy, "
            f"{r['parameters']:,} parameters, {r['train_seconds']:.0f} seconds of training"
        )
    lines += [
        "",
        f"It is on the [leaderboard]({leaderboard_url}) as *not reviewed*. "
        "When the teacher merges this pull request, it becomes reviewed and its "
        "compute (FLOPs) is counted.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the leaderboard's rows.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add-pr", help="add a pull request's scores as not reviewed")
    add.add_argument("--pr", type=int, required=True)
    add.add_argument("--pr-url", required=True)
    add.add_argument("--uploaded", type=Path, required=True, help="the uploaded results folder")
    add.add_argument("--names", required=True, help="submission folders the PR changes")
    add.add_argument("--leaderboard-url", required=True)
    add.add_argument("--comment-file", type=Path, required=True)

    clean = commands.add_parser("clean", help="remove rows of merged or closed pull requests")
    clean.add_argument("--open", default="", help="numbers of the open pull requests")

    build = commands.add_parser("build", help="write every row into one file for the website")
    build.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    match args.command:
        case "add-pr":
            results = read_pr_results(args.uploaded, set(args.names.split()))
            add_pr_rows(args.results, args.pr, args.pr_url, results)
            args.comment_file.write_text(comment(results, args.leaderboard_url) + "\n")
        case "clean":
            remove_closed_pr_rows(args.results, {int(n) for n in args.open.split()})
        case "build":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(all_rows(args.results), indent=2) + "\n")


if __name__ == "__main__":
    main()
