"""Recombination: convex-weight kernel quadrature by null-space elimination.

Reduces an ``N``-point empirical measure to at most ``M`` of its own atoms with
NON-NEGATIVE unit-sum weights that reproduce the pool mean exactly in a
finite-rank feature space. This is the construction of Hayakawa, Oberhauser and
Lyons: pick a rank-``q`` truncation ``k_0`` of the kernel, recombine the pool so
the ``q`` feature means are matched exactly, and pay only the tail
``k - k_0`` plus a Monte-Carlo term in the pool size.

Two pieces:

  * :func:`nystrom_features` -- the rank-``q`` feature map built by Nystrom from
    an INDEPENDENT sample. Independence is not a nicety: the truncation must be
    sample-independent for the source's error bound to apply, so the sample
    that builds the features must be disjoint from the pool being recombined.
  * :func:`caratheodory_reduce` -- the reduction itself, by repeated null-space
    elimination. Each elimination moves along a direction that annihilates both
    the feature means and the total mass, so the reduced measure integrates
    every function in the span EXACTLY, and it is stopped by the first weight
    to reach zero, which keeps every weight non-negative.

The reduction is exact linear algebra, not an optimization: no linear program,
no dependency beyond torch. It terminates in at most ``N - q - 1`` eliminations
and leaves at most ``q + 1`` atoms, which is why a budget of ``M`` nodes calls
for a truncation of rank ``q = M - 1``.

Reference:
  - Hayakawa, Oberhauser and Lyons, "Positively weighted kernel quadrature via
    subsampling", NeurIPS 2022 (Thm. 1 and Cor. 2; the recombination
    construction and its ``8 sum_{m >= n} lambda_m + 2 c_{k,rho} / N`` bound).
  - Tchernychova and Lyons, on recombination by null-space elimination -- the
    reduction implemented here.
"""

from __future__ import annotations

import torch

from qfield.designed_quadrature.mmd import gram

DTYPE = torch.float64

# Below this, a weight is treated as exactly zero when a direction is scaled to
# drive it there; float64 recombination on unit-sum weights is well inside it.
_ZERO_TOL = 1e-14


def nystrom_features(
    z: torch.Tensor,
    z_nystrom: torch.Tensor,
    sigma: float,
    q: int,
    jitter: float = 1e-12,
) -> torch.Tensor:
    """Rank-``q`` Nystrom feature map of the SE kernel, evaluated at ``z``.

    Builds the truncation from ``z_nystrom``, which MUST be independent of the
    pool it will be used to recombine: the source's guarantee requires the
    truncation ``k_0`` to be sample-independent, and reusing the pool to build
    it silently violates that hypothesis.

    Args:
        z: Points to featurize, shape ``(N, d)``.
        z_nystrom: Independent sample defining the truncation, shape ``(L, d)``.
        sigma: SE bandwidth.
        q: Feature rank; clipped to ``L`` and to the numerical rank.
        jitter: Floor on the retained eigenvalues.

    Returns:
        Features ``(N, q_eff)`` with ``q_eff <= q``.

    Raises:
        ValueError: On a malformed input or ``q < 1``.
    """
    if z.ndim != 2 or z_nystrom.ndim != 2:
        raise ValueError(
            f"z {tuple(z.shape)} and z_nystrom {tuple(z_nystrom.shape)} must "
            "both be (n, d)"
        )
    if z.shape[1] != z_nystrom.shape[1]:
        raise ValueError(
            f"feature dimensions disagree: {z.shape[1]} vs {z_nystrom.shape[1]}"
        )
    if int(q) < 1:
        raise ValueError(f"q must be at least 1; got {q!r}")

    L = z_nystrom.shape[0]
    K_nys = gram(z_nystrom, sigma)
    evals, evecs = torch.linalg.eigh(K_nys)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    keep = int(min(int(q), L, int((evals > jitter).sum())))
    if keep < 1:
        raise ValueError(
            "the Nystrom sample carries no numerically positive eigenvalue at "
            f"jitter={jitter!r}; the truncation is empty"
        )
    evals = evals[:keep]
    evecs = evecs[:, :keep]
    # k(z_nystrom, z)^T U Lambda^{-1/2}: the standard Nystrom embedding, so
    # <phi(x), phi(y)> approximates the rank-q Mercer truncation of k.
    diff = z.unsqueeze(1) - z_nystrom.unsqueeze(0)  # (N, L, d)
    K_cross = torch.exp(-(diff**2).sum(-1) / (2.0 * sigma**2))  # (N, L)
    return K_cross @ evecs @ torch.diag(evals.rsqrt())


def caratheodory_reduce(
    features: torch.Tensor,
    weights: torch.Tensor,
    max_atoms: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce a weighted point set to ``<= q + 1`` atoms, preserving the mean.

    Repeated null-space elimination. At each step the current support carries
    ``s`` atoms with features ``Phi`` and weights ``w``; a direction ``v`` in
    the null space of ``[Phi^T ; 1^T]`` satisfies ``Phi^T v = 0`` and
    ``1^T v = 0``, so replacing ``w`` by ``w - t v`` changes neither the feature
    mean nor the total mass. Taking ``t = min_{v_i > 0} w_i / v_i`` drives the
    first weight to zero without any weight going negative, and that atom is
    dropped.

    Args:
        features: Atom features, shape ``(N, q)``.
        weights: Non-negative weights summing to one, shape ``(N,)``.
        max_atoms: Stop once the support is this small (default ``q + 1``,
            which is the smallest the argument guarantees).

    Returns:
        ``(indices, weights)`` -- surviving atom indices into the original rows
        and their non-negative unit-sum weights.

    Raises:
        ValueError: On a shape mismatch, a negative weight, or weights that do
            not sum to one.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be (N, q); got {tuple(features.shape)}")
    N, q = features.shape
    if weights.shape != (N,):
        raise ValueError(
            f"weights must be (N,) matching the atoms; got "
            f"{tuple(weights.shape)} against {N}"
        )
    if float(weights.min()) < -_ZERO_TOL:
        raise ValueError(
            f"recombination starts from a non-negative measure; the smallest "
            f"weight is {float(weights.min()):.3e}"
        )
    if abs(float(weights.sum()) - 1.0) > 1e-8:
        raise ValueError(
            f"weights must sum to one; got {float(weights.sum()):.6e}"
        )

    target = int(q + 1) if max_atoms is None else int(max_atoms)
    idx = torch.arange(N, dtype=torch.long, device=features.device)
    w = weights.clone().to(features.dtype)

    while int(idx.numel()) > target:
        s = int(idx.numel())
        Phi = features[idx]  # (s, q)
        # A = [Phi^T ; 1^T] has q + 1 rows; s > q + 1 guarantees a null vector.
        A = torch.cat(
            [Phi.T, torch.ones(1, s, dtype=Phi.dtype, device=Phi.device)],
            dim=0,
        )
        ns = torch.linalg.svd(A, full_matrices=True)[2]  # Vh, shape (s, s)
        v = ns[-1]  # smallest right-singular direction: the null vector
        pos = v > _ZERO_TOL
        if not bool(pos.any()):
            v = -v
            pos = v > _ZERO_TOL
        if not bool(pos.any()):
            # No usable elimination direction; the support is already minimal
            # for this feature set.
            break
        ratios = torch.full_like(w[idx], float("inf"))
        ratios[pos] = w[idx][pos] / v[pos]
        j = int(torch.argmin(ratios))
        t = float(ratios[j])
        w_new = w[idx] - t * v
        w_new = torch.clamp(w_new, min=0.0)
        w_new[j] = 0.0
        keep = torch.nonzero(w_new > _ZERO_TOL, as_tuple=False).reshape(-1)
        if int(keep.numel()) >= s:
            break  # no atom was eliminated; stop rather than loop forever
        w = w.clone()
        w[idx] = w_new
        idx = idx[keep]

    w_sel = w[idx]
    w_sel = w_sel / w_sel.sum()
    return idx, w_sel


def recombination_quadrature(
    z_pool: torch.Tensor,
    z_nystrom: torch.Tensor,
    sigma: float,
    M: int,
) -> dict:
    """Recombine an i.i.d. pool into ``<= M`` convex-weighted nodes.

    Builds a rank-``(M - 1)`` Nystrom truncation from an INDEPENDENT sample,
    then reduces the uniform pool measure onto at most ``M`` of its own atoms
    while matching every truncated feature mean exactly.

    Args:
        z_pool: Pool drawn i.i.d. from the reference, shape ``(N, d)``.
        z_nystrom: Independent sample defining the truncation, shape ``(L, d)``.
        sigma: SE bandwidth.
        M: Node budget; the truncation rank is ``M - 1``.

    Returns:
        ``{z, w, indices, rank, n_pool}`` -- the nodes ``(m, d)`` with
        ``m <= M``, their non-negative unit-sum weights, the pool indices, the
        realized feature rank, and the pool size consumed.

    Raises:
        ValueError: If ``M < 2`` or the pool is smaller than ``M``.
    """
    if int(M) < 2:
        raise ValueError(
            f"recombination needs M >= 2 (rank M - 1 >= 1); got {M!r}"
        )
    N = int(z_pool.shape[0])
    if N < int(M):
        raise ValueError(f"pool of {N} cannot be reduced to {M} atoms")

    feats = nystrom_features(z_pool, z_nystrom, sigma, q=int(M) - 1)
    w0 = torch.full((N,), 1.0 / N, dtype=z_pool.dtype, device=z_pool.device)
    idx, w = caratheodory_reduce(feats, w0, max_atoms=int(M))
    return {
        "z": z_pool[idx].detach().clone(),
        "w": w.detach().clone(),
        "indices": idx,
        "rank": int(feats.shape[1]),
        "n_pool": N,
    }
