"""Kernel thinning: compression of a point set into an equally-weighted subset.

Compresses ``n`` input points into ``n / 2^r`` equally-weighted points taken
from the input itself, by ``r`` rounds of kernel halving followed by a swap
refinement. Its budget is restricted to the dyadic grid ``M = n / 2^r``, its
nodes must lie in the input sample, and it is a per-target computation with no
amortization across a stream of observations.

Two components:

  * :func:`kernel_halving` -- one round, ``n -> n/2``. Points are processed in
    pairs; a randomized sign keeps the running discrepancy between the kept and
    dropped halves small in the RKHS, which is the self-balancing walk the
    method rests on. The threshold used here is the delta-free form
    ``a = max(b, |alpha|)`` rather than the failure-probability schedule of the
    source; it is the same walk with a simpler variance proxy.
  * :func:`kernel_swap` -- the refinement. Each kept point is offered every
    input point and moved when the move lowers ``MMD^2`` to the input measure.
    This is where most of the practical quality lives.

Compress++ is not implemented. It is a near-linear-time accelerator whose own
guarantee is that it suffers at most a factor of four in error relative to the
thinning algorithm it wraps, so plain kernel thinning is the slower and more
accurate variant. At the budgets this package sweeps its runtime is
affordable.

Reference:
  - Dwivedi and Mackey, "Kernel thinning", COLT 2021 / JMLR 25(152), 2024
    (kernel halving, KT-SWAP, and the O_d(sqrt(log n / n)) integration error).
  - Shetty, Dwivedi and Mackey, "Distribution compression in near-linear time",
    ICLR 2022 (Compress++, the accelerator not used here, and its factor-of-four
    error concession).
"""

from __future__ import annotations

import torch

from qfield.designed_quadrature.mmd import gram

DTYPE = torch.float64


def admissible_budgets(n: int, m_min: int = 1) -> list[int]:
    """The node budgets kernel thinning can produce from ``n`` input points.

    Thinning halves, so its output size is restricted to the dyadic grid
    ``n / 2^r``. This set is a property of the method rather than of the
    tuning.

    Args:
        n: Number of input points.
        m_min: Smallest budget to report.

    Returns:
        Descending list of admissible output sizes.

    Raises:
        ValueError: If ``n < 1``.
    """
    if int(n) < 1:
        raise ValueError(f"n must be at least 1; got {n!r}")
    out: list[int] = []
    size = int(n)
    while size >= int(m_min):
        out.append(size)
        size //= 2
    return out


def kernel_halving(
    z: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
    K: torch.Tensor | None = None,
) -> torch.Tensor:
    """One kernel-halving round: keep ``floor(n/2)`` of the ``n`` input points.

    Points are consumed in consecutive pairs. For pair ``i`` with members
    ``(x, x')``, write ``f`` for the running RKHS imbalance between the kept
    and dropped halves accumulated so far. The pair contributes the direction
    ``a = Phi(x) - Phi(x')``, whose squared norm is
    ``b^2 = k(x,x) + k(x',x') - 2 k(x,x')``, and whose correlation with the
    history is ``alpha = <f, a>``. The sign is sampled with

        P(keep x) = (1 - alpha / max(b, |alpha|)) / 2,

    so the walk is self-correcting -- the sign that would grow the imbalance is
    the less likely one -- while remaining random, which is what keeps the
    construction unbiased over the pair.

    Args:
        z: Input points, shape ``(n, d)``.
        sigma: SE bandwidth.
        generator: Local ``torch.Generator`` (determinism stays explicit).
        K: Optional precomputed Gram ``(n, n)``.

    Returns:
        Long tensor of kept row indices into ``z``, length ``floor(n/2)``.

    Raises:
        ValueError: On a malformed input or a Gram of the wrong shape.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be (n, d); got {tuple(z.shape)}")
    n = z.shape[0]
    if n < 2:
        return torch.arange(0, min(n, 1), dtype=torch.long)
    G = gram(z, sigma) if K is None else K
    if G.shape != (n, n):
        raise ValueError(f"K must be (n, n); got {tuple(G.shape)}")

    n_pairs = n // 2
    kept: list[int] = []
    dropped: list[int] = []
    # f_row[j] = <f, Phi(z_j)> = sum_{kept} k(z_kept, z_j) - sum_{dropped} ...,
    # maintained incrementally so each pair costs one O(n) update.
    f_row = torch.zeros(n, dtype=G.dtype, device=G.device)

    for i in range(n_pairs):
        p, q = 2 * i, 2 * i + 1
        b_sq = float(G[p, p] + G[q, q] - 2.0 * G[p, q])
        b = float(b_sq) ** 0.5
        alpha = float(f_row[p] - f_row[q])
        thr = max(b, abs(alpha))
        if thr <= 0.0:
            # Coincident embeddings: either choice is exact, break the tie
            # deterministically rather than divide by zero.
            prob_keep_p = 0.5
        else:
            prob_keep_p = 0.5 * (1.0 - alpha / thr)
        u = float(torch.rand(1, generator=generator, dtype=G.dtype))
        if u < prob_keep_p:
            keep, drop = p, q
        else:
            keep, drop = q, p
        kept.append(keep)
        dropped.append(drop)
        f_row = f_row + G[keep] - G[drop]

    return torch.tensor(kept, dtype=torch.long)


def kernel_swap(
    z: torch.Tensor,
    idx: torch.Tensor,
    sigma: float,
    K: torch.Tensor | None = None,
    n_passes: int = 1,
) -> torch.Tensor:
    """Refine a thinned index set by swapping toward the input measure.

    Each selected slot is offered every input point in turn and moved when the
    move strictly lowers ``MMD^2`` between the equally-weighted coreset and the
    equally-weighted input. The comparison is exact, computed from the Gram
    alone, and a swap is accepted only on a strict improvement, so the returned
    set is never worse than the one it was handed.

    Args:
        z: Input points, shape ``(n, d)``.
        idx: Current selection, long tensor of length ``m``.
        sigma: SE bandwidth.
        K: Optional precomputed Gram ``(n, n)``.
        n_passes: Sweeps over the ``m`` slots.

    Returns:
        Refined long tensor of length ``m``.
    """
    n = z.shape[0]
    G = gram(z, sigma) if K is None else K
    m = int(idx.numel())
    if m == 0:
        return idx
    # MMD^2(coreset, input) = (1/m^2) sum_{ab} K_ab - (2/(mn)) sum_{a j} K_aj
    # + const. Only the two terms involving the coreset matter for a swap.
    row_mean = G.mean(dim=1)  # (n,) -- (1/n) sum_j K_ij
    sel = idx.clone()

    def _objective(s: torch.Tensor) -> float:
        sub = G[s][:, s]
        return float(sub.sum() / (m * m) - 2.0 * row_mean[s].sum() / m)

    best = _objective(sel)
    for _ in range(int(n_passes)):
        improved = False
        for slot in range(m):
            others = torch.cat([sel[:slot], sel[slot + 1:]])
            # Objective as a function of the candidate c placed in this slot:
            #   (1/m^2)[K_cc + 2 sum_{o} K_co + sum_{oo'} K_oo'] - (2/m)[
            #   row_mean[c] + sum_o row_mean[o]].
            cross = G[:, others].sum(dim=1)  # (n,)
            cand = (
                (torch.diagonal(G) + 2.0 * cross) / (m * m)
                - 2.0 * row_mean / m
            )
            c = int(torch.argmin(cand))
            trial = sel.clone()
            trial[slot] = c
            val = _objective(trial)
            if val < best - 1e-15:
                best = val
                sel = trial
                improved = True
        if not improved:
            break
    return sel


def kernel_thinning(
    z: torch.Tensor,
    sigma: float,
    M: int,
    generator: torch.Generator,
    swap_passes: int = 1,
    K: torch.Tensor | None = None,
) -> dict:
    """Thin ``z`` down to ``M`` equally-weighted points from the input.

    Halves ``r = log2(n / M)`` times, then refines by :func:`kernel_swap`. The
    budget must lie on the dyadic grid the method admits; a request off that
    grid raises rather than silently rounding, because the restricted budget
    set is a property of the method.

    Args:
        z: Input points, shape ``(n, d)``.
        sigma: SE bandwidth.
        M: Requested output size; must satisfy ``M = n // 2^r`` for an integer
            ``r >= 0`` under repeated floor-halving.
        generator: Local ``torch.Generator``.
        swap_passes: Sweeps of the swap refinement (0 disables it).
        K: Optional precomputed Gram ``(n, n)``.

    Returns:
        ``{z, w, indices, n_input, n_rounds}`` -- the selected nodes ``(M, d)``,
        their equal weights ``(M,)``, the input row indices, the input size
        consumed, and how many halving rounds were spent.

    Raises:
        ValueError: If ``M`` is not on the admissible dyadic grid for ``n``.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be (n, d); got {tuple(z.shape)}")
    n = int(z.shape[0])
    M = int(M)
    grid = admissible_budgets(n)
    if M not in grid:
        raise ValueError(
            f"kernel thinning admits only {grid} from n={n} input points; "
            f"M={M} is off the dyadic grid. The restricted budget set is a "
            "property of the method and is reported, not rounded away."
        )
    G = gram(z, sigma) if K is None else K

    idx = torch.arange(n, dtype=torch.long)
    n_rounds = 0
    while int(idx.numel()) > M:
        sub = kernel_halving(z[idx], sigma, generator, K=G[idx][:, idx])
        idx = idx[sub]
        n_rounds += 1
    if swap_passes > 0:
        idx = kernel_swap(z, idx, sigma, K=G, n_passes=int(swap_passes))

    m = int(idx.numel())
    w = torch.full((m,), 1.0 / m, dtype=z.dtype, device=z.device)
    return {
        "z": z[idx].detach().clone(),
        "w": w,
        "indices": idx,
        "n_input": n,
        "n_rounds": n_rounds,
    }
