"""Tests for the compute counter's extra formulas: linear algebra, distances, sorting."""

import pytest
import torch

from fashion.flops import counting

n, k, m, d = 40, 3, 100, 8
A = torch.rand(n, n)
SPD = A @ A.T + n * torch.eye(n)  # symmetric and positive definite: every solver works
B = torch.rand(n, k)
X = torch.rand(m, n)
Y = torch.rand(m, k)
v = torch.rand(n)

# Each case: what to run, and the FLOPs it should count. The setup above is not counted.
CASES = {
    "solve": (lambda: torch.linalg.solve(SPD, B), 2 * n**3 // 3 + 2 * n * n * k),
    "solve a vector": (lambda: torch.linalg.solve(SPD, v), 2 * n**3 // 3 + 2 * n * n),
    "stack of solves": (
        lambda: torch.linalg.solve(SPD.expand(3, n, n), B.expand(3, n, k)),
        3 * (2 * n**3 // 3 + 2 * n * n * k),
    ),
    "cholesky": (lambda: torch.linalg.cholesky(SPD), round(n**3 / 3)),
    "inverse": (lambda: torch.linalg.inv(SPD), 2 * n**3),
    "least squares": (lambda: torch.linalg.lstsq(X, Y), 2 * m * n * n + 4 * m * n * k),
    "eigh": (lambda: torch.linalg.eigh(SPD), 9 * n**3),
    "svd": (lambda: torch.linalg.svd(X, full_matrices=False), 6 * m * n * n + 20 * n**3),
    "qr": (lambda: torch.linalg.qr(X), 2 * (2 * m * n * n - 2 * n**3 // 3)),
    "pinv": (lambda: torch.linalg.pinv(X), 6 * m * n * n + 20 * n**3 + 2 * m * n * n),
    "distances, many rows": (
        lambda: torch.cdist(torch.rand(30, d), torch.rand(m, d)),
        2 * 30 * m * d,
    ),
    "distances, few rows": (lambda: torch.cdist(torch.rand(5, d), torch.rand(m, d)), 2 * 5 * m * d),
    "distances, p=1": (
        lambda: torch.cdist(torch.rand(30, d), torch.rand(m, d), p=1),
        3 * 30 * m * d,
    ),
    "all pairs": (lambda: torch.pdist(X), 3 * m * (m - 1) // 2 * n),
    "sort": (lambda: torch.sort(torch.rand(4, 1024), dim=1), 4 * 1024 * 10),
    "argsort": (lambda: torch.argsort(torch.rand(4, 1024)), 4 * 1024 * 10),
    "top 5": (lambda: torch.topk(torch.rand(4, 1024), 5), 0),  # one pass, like max
    "matrix times vector": (lambda: X @ v, 2 * m * n),
    "dot": (lambda: v @ v, 2 * n),
    "fft": (lambda: torch.fft.fft(torch.rand(64, dtype=torch.complex64)), 5 * 64 * 6),
}


@pytest.mark.parametrize("grad", [True, False], ids=["training", "inference"])
@pytest.mark.parametrize("case", CASES, ids=str)
def test_formula(case: str, grad: bool) -> None:
    run, expected = CASES[case]
    mode = torch.enable_grad() if grad else torch.inference_mode()
    with mode, counting() as count:
        run()
    assert count.flops == expected
    assert not count.uncounted


def test_backward_of_a_solve_is_counted() -> None:
    matrix = SPD.clone().requires_grad_()
    with counting() as count:
        torch.linalg.solve(matrix, B).sum().backward()
    # Forward: factor and solve. Backward: reuse the factors to solve once more (2n²k),
    # then one matrix multiply for the gradient of the matrix (2n²k).
    forward = 2 * n**3 // 3 + 2 * n * n * k
    assert count.flops == forward + 2 * n * n * k + 2 * n * n * k
    assert not count.uncounted


def test_ops_without_a_formula_are_named() -> None:
    with counting() as count:
        torch.linalg.matrix_exp(A)
    assert set(count.uncounted) == {"aten.linalg_matrix_exp"}
