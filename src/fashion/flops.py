"""Count the compute (FLOPs) of code as it runs, and notice what the count misses.

PyTorch's FlopCounterMode counts matrix multiplies and convolutions. FORMULAS adds
counts for linear algebra (solves, factorizations, eigenvalues), distance matrices
and sorting, so closed-form methods such as least squares or nearest neighbours are
counted too. UncountedOps names anything else that runs and may do heavy math.
"""

import math
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

# Ops the FLOP counter skips on purpose: each one does a few operations per number
# it touches, so it is cheap next to a matrix multiply. Ops tagged "pointwise" (ReLU,
# add, ...) or "reduction" (sum, max, ...) and views (reshape, ...) are cheap too.
# Anything else that runs but is not counted is reported, because it may hide heavy
# work that FORMULAS below does not cover yet.
CHEAP_OPS = {
    # Making, copying, moving and indexing tensors.
    *("_to_copy", "_unsafe_view", "alias", "arange", "as_strided", "bernoulli", "cat"),
    *("clone", "contiguous", "copy", "copy_", "detach", "empty", "empty_like"),
    *("empty_strided", "expand", "fill", "fill_", "full", "full_like", "gather"),
    *("index", "index_add", "index_add_", "index_put", "index_put_", "index_select"),
    *("lift_fresh", "masked_fill", "masked_fill_", "masked_scatter", "masked_select"),
    *("new_empty", "new_empty_strided", "new_full", "new_ones", "new_zeros", "nonzero"),
    *("normal_", "one_hot", "ones", "ones_like", "rand", "rand_like", "randint"),
    *("randn", "randn_like", "random_", "randperm", "repeat", "scalar_tensor"),
    *("scatter", "scatter_", "scatter_add", "scatter_add_", "select_backward", "set_"),
    *("slice_backward", "split", "split_with_sizes", "squeeze", "stack", "t"),
    *("transpose", "unbind", "uniform_", "unsqueeze", "view", "zero_", "zeros"),
    *("zeros_like", "_local_scalar_dense", "item", "is_nonzero", "equal"),
    *("detach_", "flip", "roll", "unfold", "unfold_backward", "pad"),
    *("eye", "linspace", "logspace", "tril", "triu", "diag_embed", "polar"),
    *("topk", "kthvalue", "median", "nanmedian", "mode"),  # one pass, like max
    *("_assert_async", "_linalg_check_errors", "bincount", "unique", "_unique2", "histc", "cumsum"),
    # Layers and losses that touch each number a few times.
    *("_log_softmax", "_log_softmax_backward_data", "_softmax", "_softmax_backward_data"),
    *("nll_loss_forward", "nll_loss_backward", "nll_loss2d_forward", "nll_loss2d_backward"),
    *("batch_norm", "native_batch_norm", "native_batch_norm_backward"),
    *("_batch_norm_with_update", "_batch_norm_with_update_functional", "batch_norm_backward"),
    *("_native_batch_norm_legit", "dropout"),
    *("_native_batch_norm_legit_functional", "_native_batch_norm_legit_no_training"),
    *("native_layer_norm", "native_layer_norm_backward", "native_group_norm"),
    *("native_group_norm_backward", "native_dropout", "native_dropout_backward"),
    *("max_pool2d_with_indices", "max_pool2d_with_indices_backward", "avg_pool2d"),
    *("avg_pool2d_backward", "_adaptive_avg_pool2d", "_adaptive_avg_pool2d_backward"),
    *("adaptive_max_pool2d", "adaptive_max_pool2d_backward", "embedding"),
    *("embedding_dense_backward", "upsample_nearest2d", "upsample_bilinear2d"),
    *("reflection_pad2d", "replication_pad2d", "constant_pad_nd", "grid_sampler_2d"),
    *("im2col", "col2im", "_cudnn_rnn"),
    # Optimizers update each parameter a few times per step.
    *("_fused_adam_", "_fused_adamw_", "_fused_sgd_"),
}


class UncountedOps(TorchDispatchMode):
    """Watch the ops that run and record those the FLOP counter has no formula for.

    It only records names; it does not change what runs.
    """

    def __init__(self, counted: dict, seen: Counter[str]) -> None:
        super().__init__()
        self.counted = counted
        self.seen = seen

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        packet = func.overloadpacket
        name = packet.__name__
        cheap = (
            packet in self.counted
            or func.is_view
            or torch.Tag.pointwise in func.tags
            or torch.Tag.reduction in func.tags
            or name in CHEAP_OPS
            or name.startswith("_foreach_")
        )
        if cheap or func.namespace == "profiler":  # profiler: markers, no math
            return func(*args, **(kwargs or {}))
        # Ops built out of other ops (such as linalg.solve) are split into those, the
        # same way the FLOP counter splits them, and the parts are checked instead.
        with self:
            result = func.decompose(*args, **(kwargs or {}))
        if result is not NotImplemented:
            return result
        self.seen[f"aten.{name}" if func.namespace == "aten" else str(packet)] += 1
        return func(*args, **(kwargs or {}))


def linear_flop(input_shape, weight_shape, *args, out_shape=None, **kwargs) -> int:
    """A linear layer: one multiply and one add per input, output and image."""
    return 2 * math.prod(input_shape[:-1]) * input_shape[-1] * weight_shape[0]


def linear_backward_flop(
    input_shape, grad_output_shape, weight_shape, output_mask, *args, out_shape=None, **kwargs
) -> int:
    """The gradients for the input and the weight each cost as much as the forward pass."""
    one = linear_flop(input_shape, weight_shape)
    return one * (int(output_mask[0]) + int(output_mask[1]))


# Linear algebra. These are the usual textbook counts for an n x n matrix A (or
# m x n for least squares, QR and SVD) and a right-hand side B with k columns. A
# stack of matrices (batch dimensions in front) costs one count per matrix.
# Eigenvalues and the SVD are found by iterating, so their counts are typical
# values rather than exact ones.


def is_shape(value) -> bool:
    return isinstance(value, (tuple, list)) and all(isinstance(d, int) for d in value)


def batch(shape) -> int:
    """How many matrices a stack of shape (..., rows, cols) holds."""
    return math.prod(shape[:-2])


def columns(a_shape, b_shape) -> int:
    """How many right-hand sides B holds for A: k for an (n, k) B, 1 for a vector."""
    return max(1, math.prod(b_shape) // max(1, batch(a_shape) * a_shape[-1]))


def solve_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """Solve A X = B: factor A (2n³/3), then substitute for each column of B (2n²k)."""
    n = a_shape[-1]
    return batch(a_shape) * (2 * n**3 // 3 + 2 * n**2 * columns(a_shape, b_shape))


def triangular_solve_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """Solve with a triangular A: n² per column of B."""
    return batch(a_shape) * a_shape[-1] ** 2 * columns(a_shape, b_shape)


def cholesky_solve_flop(b_shape, l_shape, *args, out_shape=None, **kwargs) -> int:
    """Solve with a Cholesky factor: two triangular solves."""
    return 2 * triangular_solve_flop(l_shape, b_shape)


def lu_solve_flop(lu_shape, pivots_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    return 2 * triangular_solve_flop(lu_shape, b_shape)


def cubic_flop(factor: float):
    """A formula for ops on a square matrix that cost `factor` times n³."""

    def formula(a_shape, *args, out_shape=None, **kwargs) -> int:
        return round(batch(a_shape) * factor * a_shape[-1] ** 3)

    return formula


def eigh_flop(a_shape, uplo="L", compute_v=True, *args, out_shape=None, **kwargs) -> int:
    """Symmetric eigenvalues: about 4n³/3, or 9n³ with the eigenvectors."""
    return cubic_flop(9 if compute_v else 4 / 3)(a_shape)


def lstsq_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """Least squares by QR: 2mn² to factor A, then 4mnk to apply it to B."""
    m, n = a_shape[-2:]
    long, short = max(m, n), min(m, n)
    k = max(1, math.prod(b_shape) // max(1, batch(a_shape) * m))
    return batch(a_shape) * (2 * long * short**2 + 4 * m * n * k)


def qr_flop(a_shape, mode="reduced", *args, out_shape=None, **kwargs) -> int:
    """QR: 2mn² - 2n³/3 for R, twice that when Q is formed too."""
    m, n = a_shape[-2:]
    long, short = max(m, n), min(m, n)
    r_only = 2 * long * short**2 - 2 * short**3 // 3
    return batch(a_shape) * r_only * (1 if mode == "r" else 2)


def svd_flop(a_shape, full_matrices=False, compute_uv=True, *args, out_shape=None, **kwargs):
    """SVD: about 4mn² - 4n³/3 for the values, 6mn² + 20n³ with U and V."""
    m, n = a_shape[-2:]
    long, short = max(m, n), min(m, n)
    if compute_uv:
        return batch(a_shape) * (6 * long * short**2 + 20 * short**3)
    return batch(a_shape) * (4 * long * short**2 - 4 * short**3 // 3)


# Distances between every row of x1 (m rows) and every row of x2 (n rows), d numbers
# each. The usual way is one matrix multiply (2mnd); other p-norms take a subtract,
# a power and an add per pair of numbers (3mnd).


def distance_pairs(x1_shape, x2_shape) -> int:
    """m * n * d, for each pair of stacked inputs."""
    return math.prod(x1_shape[:-2]) * x1_shape[-2] * x2_shape[-2] * x1_shape[-1]


def euclidean_dist_flop(x1_shape, x2_shape, *args, out_shape=None, **kwargs) -> int:
    return 2 * distance_pairs(x1_shape, x2_shape)


def cdist_flop(x1_shape, x2_shape, p=2.0, *args, out_shape=None, **kwargs) -> int:
    return (2 if p == 2 else 3) * distance_pairs(x1_shape, x2_shape)


def cdist_backward_flop(grad_shape, x1_shape, x2_shape, *args, out_shape=None, **kwargs) -> int:
    return 3 * distance_pairs(x1_shape, x2_shape)


def pdist_flop(self_shape, p=2.0, *args, out_shape=None, **kwargs) -> int:
    """Distances between all pairs of rows of one matrix: n(n-1)/2 pairs."""
    n, d = self_shape[-2:]
    return 3 * n * (n - 1) // 2 * d


def pinv_flop(self_shape, *args, out_shape=None, **kwargs) -> int:
    """Pseudo-inverse: an SVD (or eigh when hermitian), then 2mn² to rebuild."""
    hermitian = kwargs.get("hermitian", any(a is True for a in args))
    m, n = self_shape[-2:]
    rebuild = batch(self_shape) * 2 * m * n * min(m, n)
    if hermitian:
        return eigh_flop(self_shape) + rebuild
    return svd_flop(self_shape) + rebuild


# Matrix times vector, and vector times vector: PyTorch's counter only has the
# matrix-times-matrix versions.


def mv_flop(mat_shape, vec_shape, *args, out_shape=None, **kwargs) -> int:
    return 2 * mat_shape[0] * mat_shape[1]


def addmv_flop(self_shape, mat_shape, vec_shape, *args, out_shape=None, **kwargs) -> int:
    return mv_flop(mat_shape, vec_shape)


def dot_flop(self_shape, other_shape, *args, out_shape=None, **kwargs) -> int:
    return 2 * self_shape[0]


def addr_flop(self_shape, vec1_shape, vec2_shape, *args, out_shape=None, **kwargs) -> int:
    return 2 * vec1_shape[0] * vec2_shape[0]


def addbmm_flop(self_shape, batch1_shape, batch2_shape, *args, out_shape=None, **kwargs) -> int:
    b, n, m = batch1_shape
    return 2 * b * n * m * batch2_shape[-1]


def fft_flop(factor: float):
    """FFT of N numbers: about `factor` * N log2(N), for each transformed slice."""

    def formula(self_shape, dim, *args, out_shape=None, **kwargs) -> int:
        shape = out_shape if is_shape(out_shape) else self_shape
        size = math.prod(shape[d] for d in dim)
        return round(factor * math.prod(shape) * math.log2(max(size, 2)))

    return formula


def sort_flop(self_shape, *args, out_shape=None, **kwargs) -> int:
    """Sorting n numbers takes about n log2(n) comparisons, for each row sorted."""
    if not is_shape(self_shape) or len(self_shape) == 0:
        return 0  # sorting a plain Python list, or a single number
    dim = kwargs.get("dim", args[0] if args and isinstance(args[0], int) else -1)
    n = self_shape[dim]
    return math.prod(self_shape) * max(1, math.ceil(math.log2(max(n, 2))))


aten = torch.ops.aten
FORMULAS = {
    # On a Mac's GPU (MPS), linear layers run as one op instead of a matrix multiply,
    # and PyTorch's counter has no formula for it. Without these the count is too low.
    aten.linear: linear_flop,
    aten.linear_backward: linear_backward_flop,
    aten._linalg_solve_ex: solve_flop,
    aten.linalg_solve_triangular: triangular_solve_flop,
    aten.triangular_solve: lambda b, a, *args, out_shape=None, **kw: triangular_solve_flop(a, b),
    aten.cholesky_solve: cholesky_solve_flop,
    aten.linalg_lu_solve: lu_solve_flop,
    aten.linalg_cholesky_ex: cubic_flop(1 / 3),
    aten.cholesky_inverse: cubic_flop(1),
    aten.linalg_lu_factor_ex: cubic_flop(2 / 3),
    aten.linalg_lu: cubic_flop(2 / 3),
    aten._linalg_det: cubic_flop(2 / 3),
    aten._linalg_slogdet: cubic_flop(2 / 3),
    aten.linalg_inv_ex: cubic_flop(2),
    aten.linalg_eig: cubic_flop(25),
    aten._linalg_eigh: eigh_flop,
    aten.linalg_lstsq: lstsq_flop,
    aten.linalg_qr: qr_flop,
    aten.geqrf: lambda a, *args, out_shape=None, **kw: qr_flop(a, "r"),
    aten._linalg_svd: svd_flop,
    aten._euclidean_dist: euclidean_dist_flop,
    aten._cdist_forward: cdist_flop,
    aten._cdist_backward: cdist_backward_flop,
    aten.cdist: cdist_flop,
    aten._pdist_forward: pdist_flop,
    aten.linalg_pinv: pinv_flop,
    aten.mv: mv_flop,
    aten.addmv: addmv_flop,
    aten.dot: dot_flop,
    aten.vdot: dot_flop,
    aten.addr: addr_flop,
    aten.addbmm: addbmm_flop,
    aten._fft_r2c: fft_flop(2.5),
    aten._fft_c2r: fft_flop(2.5),
    aten._fft_c2c: fft_flop(5),
    aten.sort: sort_flop,
}


@dataclass
class Count:
    flops: int = 0
    uncounted: Counter[str] = field(default_factory=Counter)  # op name -> times it ran


@contextmanager
def counting() -> Iterator[Count]:
    """Count the FLOPs of the code inside, and record ops the count may miss."""
    count = Count()
    counter = FlopCounterMode(display=False, custom_mapping=FORMULAS)
    with counter, UncountedOps(counter.flop_registry, count.uncounted):
        yield count
    count.flops = counter.get_total_flops()
