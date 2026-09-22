"""Unit-sum-constrained MMD^2-optimal quadrature weights.

At fixed nodes ``{z_j}``, for example ``M`` i.i.d. samples from ``rho``, the
weights
that minimize the SE-MMD^2 ``w^T K w - 2 w^T mu + c_rho`` over the unit-sum
simplex ``{w : 1^T w = 1}`` are, by the Lagrange stationarity of the
equality-constrained convex QP,

    w_c = K^{-1} mu - lambda K^{-1} 1,
    lambda = (1^T K^{-1} mu - 1) / (1^T K^{-1} 1).

Theorem (per node set). ``MMD^2(z, w_c) <= MMD^2(z, 1/M)``. Proof: the feasible
set ``{w : 1^T w = 1}`` is convex, ``MMD^2`` is a convex quadratic in ``w``
(``K`` is PSD), ``w_c`` is its minimizer over that set, and the equal-weight
vector ``1/M`` is feasible. Equality holds only when ``1/M`` is already
optimal. ``w_c`` sums to one by construction (so it integrates constants
exactly, ``E_Q[1] = 1``) and the metric is invariant to a positive rescale of
the weights, so the renormalization other MMD helpers apply is a no-op here.

This is not :func:`qfield.msip.msip_weights`, which solves the unconstrained
``(K + eps I)^{-1} v0`` and then divides by the sum. That object does not sum
to one before the divide, so it mis-integrates constants, and the post-hoc
renormalization carries no ``<= floor`` guarantee. The constrained solve here
is the quadrature weight the ``<= floor`` theorem is about.

A small ``jitter`` is added to the diagonal only to keep the two linear solves
well-posed when nodes are close; it does not enter the scoring (the arms score
with the exact, un-jittered ``K`` from :func:`qfield.designed_quadrature.mmd.mmd_sq`).

Reference:
  - Belhadji, Sharp, Marzouk, "To discretize continually ...",
    arXiv:2605.14142 (the MMD-optimal designed quadrature).
  - Belhadji, Sharp, Marzouk, "Weighted quantization using MMD: From mean
    field to mean shift via gradient flows", AISTATS 2026 (arXiv:2502.10600)
    (Bayesian / kernel quadrature weights).
"""

from __future__ import annotations

import torch

from qfield.designed_quadrature.mmd import gram, gram_batched

# Environment workaround. ``torch.linalg.solve`` on a batched float64 system
# with matrix size ``m >= 256`` and more than one intra-op thread emits
# ``Intel oneMKL ERROR: Parameter 6 was incorrect on entry to DLASWP`` and
# then hangs indefinitely; with one thread the identical call returns in
# milliseconds. LU with partial pivoting is the same algorithm either way,
# and every arm in a cell uses the same setting, so no comparison is
# affected. Without this guard a large-budget run hangs rather than fails.
_SOLVE_SERIAL_M = 128


class _serial_solve_above:
    """Force single-threaded LAPACK for batched solves wider than the guard."""

    def __init__(self, m: int):
        self.serial = int(m) > _SOLVE_SERIAL_M

    def __enter__(self):
        if self.serial:
            self.prev = torch.get_num_threads()
            torch.set_num_threads(1)

    def __exit__(self, *exc):
        if self.serial:
            torch.set_num_threads(self.prev)
        return False

DEFAULT_JITTER = 1e-10


def constrained_weights_from_gram(
    K: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """The same solve, entered at the Gram rather than at the nodes.

    Split out so that a construction which already holds the (jittered)
    Gram, such as the greedy selection of
    :mod:`~qfield.designed_quadrature.sbq` or any caller supplying a kernel
    other than the squared exponential, can reuse this solve rather than
    reimplement it. :func:`constrained_weights` is this function applied to
    ``gram(z, sigma) + jitter * I``, so the two agree bit for bit.

    Args:
        K: The (already jittered) Gram at the nodes, shape ``(m, m)``,
            symmetric positive definite.
        mu: Kernel mean at the nodes, shape ``(m,)``.

    Returns:
        Constrained optimal weights ``w_c``, shape ``(m,)``; ``1^T w_c = 1``
        by construction.

    Raises:
        ValueError: On a non-square ``K`` or a mismatched ``mu``.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square (m, m); got {tuple(K.shape)}")
    if mu.shape != (K.shape[0],):
        raise ValueError(
            f"mu must be (m,) matching K; got {tuple(mu.shape)} against "
            f"{K.shape[0]} nodes"
        )
    ones = torch.ones(K.shape[0], dtype=K.dtype, device=K.device)
    rhs = torch.stack([mu, ones], dim=1)  # (m, 2): solve both at once
    sol = torch.linalg.solve(K, rhs)  # columns: K^{-1} mu, K^{-1} 1
    a, b = sol[:, 0], sol[:, 1]
    lam = (a.sum() - 1.0) / b.sum()
    return a - lam * b


def constrained_weights(
    z: torch.Tensor,
    mu: torch.Tensor,
    sigma: float,
    jitter: float = DEFAULT_JITTER,
) -> torch.Tensor:
    """Unit-sum-constrained MMD^2-optimal weights at fixed nodes.

        w_c = K^{-1} mu - lambda K^{-1} 1,
        lambda = (1^T K^{-1} mu - 1) / (1^T K^{-1} 1),

    the argmin of ``w^T K w - 2 w^T mu`` over ``{w : 1^T w = 1}`` (the
    equality-constrained convex QP). The two needed solves ``K^{-1} mu`` and
    ``K^{-1} 1`` are batched into a single ``torch.linalg.solve`` against the
    ``(m, 2)`` right-hand side ``[mu, 1]``. The result satisfies ``1^T w_c =
    1`` exactly (up to round-off) and is differentiable in ``z`` (autograd
    flows through ``solve``), which the move arm relies on.

    Args:
        z: Node positions, shape ``(m, d)``.
        mu: Exact kernel mean at the nodes ``mu_j = E_rho[k(x, z_j)]``,
            shape ``(m,)``.
        sigma: SE bandwidth.
        jitter: Small ridge added to the Gram diagonal for the solve only,
            which keeps it well-posed when nodes nearly coincide. It does
            not enter the scoring, which uses the exact ``K``.

    Returns:
        Constrained optimal weights ``w_c``, shape ``(m,)``, in ``z``'s
        dtype on ``z``'s device; ``1^T w_c = 1`` by construction.
    """
    eye = torch.eye(z.shape[0], dtype=z.dtype, device=z.device)
    return constrained_weights_from_gram(gram(z, sigma) + jitter * eye, mu)


def constrained_weights_batched(
    z: torch.Tensor,
    mu: torch.Tensor,
    sigma: float,
    jitter: float = DEFAULT_JITTER,
) -> torch.Tensor:
    """Batched unit-sum-constrained MMD^2-optimal weights over repeats.

    The vectorized-across-repeats analogue of :func:`constrained_weights`:
    solves the equality-constrained convex QP for ``R`` node sets at once.
    Per slice ``r`` the result is identical to :func:`constrained_weights`
    applied to slice ``r``: the same jittered Gram, the same stacked
    ``[mu, 1]`` right-hand side, the same ``torch.linalg.solve`` (which
    batches over the leading dimension), and the same Lagrange-multiplier
    closed form

        w_c[r] = a[r] - lambda[r] b[r],
        lambda[r] = (sum_j a[r]_j - 1) / (sum_j b[r]_j),

    with ``a[r] = K[r]^{-1} mu[r]`` and ``b[r] = K[r]^{-1} 1``. Each
    ``w_c[r]`` sums to one and is differentiable in ``z[r]`` (autograd flows
    through the batched solve), which the batched move arm relies on.

    Args:
        z: Stacked node positions, shape ``(R, m, d)``.
        mu: Stacked exact kernel means at the nodes, shape ``(R, m)``.
        sigma: SE bandwidth.
        jitter: Small ridge added to the Gram diagonal for the solve only;
            it does not enter the scoring.

    Returns:
        Stacked constrained optimal weights, shape ``(R, m)``; each row sums
        to one by construction.
    """
    R, m, _ = z.shape
    eye = torch.eye(m, dtype=z.dtype, device=z.device)
    K = gram_batched(z, sigma) + jitter * eye  # (R, m, m), broadcast eye
    ones = torch.ones(R, m, dtype=z.dtype, device=z.device)
    rhs = torch.stack([mu, ones], dim=-1)  # (R, m, 2): solve both at once
    with _serial_solve_above(m):
        sol = torch.linalg.solve(K, rhs)  # (R, m, 2): K^{-1} mu, K^{-1} 1
    a, b = sol[..., 0], sol[..., 1]  # (R, m), (R, m)
    lam = (a.sum(dim=-1) - 1.0) / b.sum(dim=-1)  # (R,)
    return a - lam.unsqueeze(-1) * b
