"""Tests for the leaderboard rows, above all the checks on pull request scores."""

import json
from pathlib import Path

import pytest

from fashion.leaderboard import (
    InvalidResult,
    add_pr_rows,
    all_rows,
    check_result,
    comment,
    read_pr_results,
    readable_bytes,
    remove_closed_pr_rows,
)

GOOD = {
    "name": "alice",
    "accuracy": 0.8765,
    "parameters": 1000,
    "train_seconds": 42.0,
    "date": "2026-01-02T03:04:05+00:00",
    "code_url": None,
}


def test_good_result_is_kept() -> None:
    assert check_result(GOOD, {"alice"}) == {
        "name": "alice",
        "accuracy": 0.8765,
        "parameters": 1000,
        "train_seconds": 42.0,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"name": "bob"},  # not a folder this pull request changes
        {"name": "../alice"},
        {"accuracy": 1.5},
        {"accuracy": float("nan")},
        {"accuracy": "0.9"},
        {"accuracy": True},
        {"parameters": -1},
        {"parameters": 10.5},
        {"train_seconds": 601},
        {"train_seconds": -1},
        {"extra": "field"},
        {"train_flops": 1000},  # compute is only counted after the merge
        {"predict_flops": 10},
        {"uncounted_ops": ["aten.topk"]},
        {"model_bytes": -1},
        {"model_bytes": 1.5},
        {"model_bytes": "4000"},
    ],
)
def test_bad_results_are_rejected(change: dict) -> None:
    with pytest.raises(InvalidResult):
        check_result({**GOOD, **change}, {"alice"})


@pytest.mark.parametrize("flops", [{}, {"train_flops": None}], ids=["older scorer", "empty"])
def test_pull_request_results_without_compute_are_kept(flops: dict) -> None:
    assert check_result({**GOOD, **flops}, {"alice"})["accuracy"] == 0.8765


def test_model_size_is_kept_and_shown() -> None:
    good = check_result({**GOOD, "model_bytes": 188_160_000}, {"alice"})
    assert good["model_bytes"] == 188_160_000
    assert "188 MB model" in comment([good], "https://example.com/")


def test_readable_bytes() -> None:
    assert readable_bytes(0) == "0 B"
    assert readable_bytes(203_560) == "204 kB"
    assert readable_bytes(1_250_000) == "1.2 MB"


def test_comment_says_when_compute_is_counted() -> None:
    text = comment([check_result(GOOD, {"alice"})], "https://example.com/")
    assert "87.65%" in text
    assert "compute (FLOPs) is counted" in text


def test_unreadable_and_huge_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(json.dumps(GOOD))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "huge.json").write_text(" " * 20_000 + json.dumps(GOOD))
    assert [r["name"] for r in read_pr_results(tmp_path, {"alice"})] == ["alice"]


def test_new_push_replaces_the_pull_request_row(tmp_path: Path) -> None:
    good = check_result(GOOD, {"alice"})
    add_pr_rows(tmp_path, 7, "https://example.com/pull/7", [good])
    add_pr_rows(tmp_path, 7, "https://example.com/pull/7", [{**good, "accuracy": 0.9}])
    rows = all_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["accuracy"] == 0.9
    assert rows[0]["reviewed"] is False


def test_closed_pull_request_rows_are_removed(tmp_path: Path) -> None:
    good = check_result(GOOD, {"alice"})
    add_pr_rows(tmp_path, 7, "https://example.com/pull/7", [good])
    add_pr_rows(tmp_path, 8, "https://example.com/pull/8", [good])
    remove_closed_pr_rows(tmp_path, open_prs={8})
    assert [row["pr"] for row in all_rows(tmp_path)] == [8]


def test_merged_rows_are_reviewed(tmp_path: Path) -> None:
    (tmp_path / "2026-01-02T03-04-05_alice.json").write_text(json.dumps(GOOD))
    assert all_rows(tmp_path)[0]["reviewed"] is True


def test_merged_rows_keep_their_compute(tmp_path: Path) -> None:
    row = {**GOOD, "train_flops": 562_200_000_000}
    (tmp_path / "2026-01-02T03-04-05_alice.json").write_text(json.dumps(row))
    assert all_rows(tmp_path)[0]["train_flops"] == 562_200_000_000
