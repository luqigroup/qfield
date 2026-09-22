"""Sequential Bayesian quadrature: greedy selection under RE-SOLVED weights.

At every greedy step a candidate is scored AFTER the unit-sum weights are
re-solved at the enlarged node set, so the selection is weight-aware and the
emitted rule carries signed unit-sum weights. This is the construction of
Huszar and Duvenaud: optimally reweighting a herded set dominates the fixed
schedule, and choosing the set under that reweighting dominates applying it
afterwards.

THE OBJECTIVE, IN CLOSED FORM. At a node set ``S`` with jittered Gram
``K = k(z_S, z_S) + jitter I``, kernel mean ``mu = mu_rho(z_S)`` and
self-affinity ``c_rho``, the unit-sum-constrained minimum of
``w^T K w - 2 w^T mu + c_rho`` over ``{w : 1^T w = 1}`` is obtained by
substituting the Lagrange solution of
:mod:`~qfield.designed_quadrature.weights`:

    V(S)     = c_rho - A + (B - 1)^2 / C,                        (constrained)
    V_unc(S) = c_rho - A,                                      (unconstrained)
    A = mu^T K^{-1} mu,  B = 1^T K^{-1} mu,  C = 1^T K^{-1} 1,

and ``V_unc(S) <= V(S)``, the price of integrating constants exactly. The
greedy step is ``argmin_j V(S u {j})``, which is what makes the selection
weight-aware; scoring the candidate under EQUAL weights instead and
reweighting at the end is the strictly weaker construction.

THE UPDATE IS RANK ONE. Bordering ``K`` by candidate ``j`` and applying the
Schur complement of the block inverse gives, with ``p_j = K^{-1} k_j``,

    s_j     = k(z_j, z_j) + jitter - k_j^T p_j,
    alpha_j = mu_j - p_j^T mu,     beta_j = 1 - p_j^T 1,
    A_j = A + alpha_j^2 / s_j,  B_j = B + alpha_j beta_j / s_j,
    C_j = C + beta_j^2 / s_j,

so every candidate is scored in ``O(m)`` once the triangular factor is
carried. Carrying ``G = L^{-1} k(z_S, z_pool)`` alongside the Cholesky factor
``L`` of ``K`` makes ``p_j^T u = G_{:,j}^T (L^{-1} u)`` and
``k_j^T p_j = ||G_{:,j}||^2``, and appending one row to ``G`` costs
``O(m P)``. The whole selection is therefore ``O(M^2 P)`` rather than the
``O(M^4 P)`` of re-solving each candidate from scratch.

MONOTONICITY IS A THEOREM AND IS USED AS THE CORRECTNESS CHECK. The optimum at
``S u {j}`` is at most the optimum at ``S``, because the ``S``-optimal weight
vector extended by a zero is feasible at ``S u {j}`` and has the same unit sum.
So the achieved value is non-increasing along the greedy prefix, exactly, and a
recursion that violates it is wrong.

REPEATS ARE EXCLUDED. A repeated atom enlarges
neither the span nor the feasible set, so ``V(S u {j}) = V(S)`` for
``j in S``: the greedy step would be choosing between an improvement and a
no-op. Herding permits repeats because its fixed schedule makes a repeat a
genuine reweighting of the multiset; here it is arithmetic. Candidates whose
Schur complement has collapsed relative to the diagonal are excluded for the
same reason -- they are already in the span of the selected set.

Reference:
  - Huszar and Duvenaud, "Optimally-weighted herding is Bayesian quadrature",
    UAI 2012 (arXiv:1204.1664), Sec. 4 (the greedy sequential rule).
  - O'Hagan, "Bayesian quadrature with non-normal approximating functions",
    Statistics and Computing 1991 (the Bayesian quadrature weights).
  - Briol, Oates, Girolami and Osborne, "Frank-Wolfe Bayesian quadrature",
    Advances in Neural Information Processing Systems 2015 (the greedy and
    conditional-gradient variants side by side).
  - Chen, Welling and Smola, "Super-samples from kernel herding", UAI 2010
    (the equal-weight construction this one dominates).
"""

from __future__ import annotations

import torch

from qfield.designed_quadrature.mmd import gram
from qfield.designed_quadrature.weights import (
    DEFAULT_JITTER,
    constrained_weights,
    constrained_weights_from_gram,
)

DTYPE = torch.float64

# A candidate whose Schur complement has fallen this far below the diagonal is
# in the span of the selected set to working precision; selecting it adds a
# direction that is not there. The floor is RELATIVE to the diagonal so that a
# genuinely rank-deficient kernel -- where every Schur complement is of the
# order of the jitter and the last admissible node is exactly the one that buys
# the unit-sum constraint -- is still selectable.
SCHUR_FLOOR = 1e-14

_WEIGHT_MODES = ("constrained", "unconstrained")


def _selection_state(
    K_sel: torch.Tensor,
    K_cross: torch.Tensor,
    mu_sel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild ``(L, G, g_mu, g_one)`` from scratch for a node set.

    Used to seed the greedy loop and to restart it inside a swap pass, where
    one slot is removed and a downdate would buy nothing over a factorization
    of an ``(m - 1)``-by-``(m - 1)`` matrix.

    Args:
        K_sel: Jittered Gram at the selected nodes, ``(m, m)``.
        K_cross: Gram between selected nodes and the pool, ``(m, P)``.
        mu_sel: Kernel mean at the selected nodes, ``(m,)``.

    Returns:
        ``(L, G, g_mu, g_one)`` with ``L`` the lower Cholesky factor,
        ``G = L^{-1} K_cross``, ``g_mu = L^{-1} mu_sel`` and
        ``g_one = L^{-1} 1``.
    """
    m = K_sel.shape[0]
    if m == 0:
        empty = torch.zeros(0, K_cross.shape[1], dtype=K_cross.dtype,
                            device=K_cross.device)
        vec = torch.zeros(0, dtype=K_cross.dtype, device=K_cross.device)
        return torch.zeros(0, 0, dtype=K_cross.dtype,
                           device=K_cross.device), empty, vec, vec
    L = torch.linalg.cholesky(K_sel)
    ones = torch.ones(m, 1, dtype=K_sel.dtype, device=K_sel.device)
    rhs = torch.cat([K_cross, mu_sel.reshape(m, 1), ones], dim=1)
    sol = torch.linalg.solve_triangular(L, rhs, upper=False)
    return L, sol[:, :-2], sol[:, -2], sol[:, -1]


def _candidate_values(
    G: torch.Tensor,
    g_mu: torch.Tensor,
    g_one: torch.Tensor,
    mu_pool: torch.Tensor,
    diag: torch.Tensor,
    weight_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Score every pool atom as the next node, by the rank-one update.

    Args:
        G: ``L^{-1} K_cross``, shape ``(m, P)`` (empty ``m = 0`` allowed).
        g_mu: ``L^{-1} mu_sel``, shape ``(m,)``.
        g_one: ``L^{-1} 1``, shape ``(m,)``.
        mu_pool: Kernel mean at every pool atom, shape ``(P,)``.
        diag: Jittered kernel diagonal at every pool atom, shape ``(P,)``.
        weight_mode: ``"constrained"`` (the unit-sum rule this package emits)
            or ``"unconstrained"`` (the classical Bayesian-quadrature rule,
            reported as the lower bound it is).

    Returns:
        ``(value, s, alpha, beta, value_other)`` -- the criterion under
        ``weight_mode``, the Schur complements, the two rank-one numerators,
        and the criterion under the OTHER weight mode (carried so a caller can
        report the constraint's price without a second pass).
    """
    A = float(g_mu @ g_mu) if g_mu.numel() else 0.0
    B = float(g_mu @ g_one) if g_mu.numel() else 0.0
    C = float(g_one @ g_one) if g_one.numel() else 0.0
    if G.shape[0] == 0:
        s = diag.clone()
        alpha = mu_pool.clone()
        beta = torch.ones_like(mu_pool)
    else:
        s = diag - (G * G).sum(dim=0)
        alpha = mu_pool - G.transpose(0, 1) @ g_mu
        beta = 1.0 - G.transpose(0, 1) @ g_one
    safe = s.clamp_min(torch.finfo(s.dtype).tiny)
    A_j = A + alpha * alpha / safe
    B_j = B + alpha * beta / safe
    C_j = C + beta * beta / safe
    unc = -A_j
    con = unc + (B_j - 1.0) ** 2 / C_j.clamp_min(torch.finfo(s.dtype).tiny)
    if weight_mode == "constrained":
        return con, s, alpha, beta, unc
    return unc, s, alpha, beta, con


def _greedy_select(
    K_pool: torch.Tensor,
    mu_pool: torch.Tensor,
    diag: torch.Tensor,
    M: int,
    weight_mode: str,
    schur_floor: float,
) -> tuple[list[int], list[float], list[float]]:
    """The greedy prefix: ``M`` weight-aware steps, or fewer if the span runs out.

    Returns:
        ``(indices, values, values_other)`` -- the selected pool indices in
        selection order, and the criterion at each prefix length under the
        chosen and the other weight mode. The values EXCLUDE ``c_rho``, which
        the caller adds once.
    """
    P = K_pool.shape[0]
    picked: list[int] = []
    vals: list[float] = []
    vals_other: list[float] = []
    G = torch.zeros(0, P, dtype=K_pool.dtype, device=K_pool.device)
    g_mu = torch.zeros(0, dtype=K_pool.dtype, device=K_pool.device)
    g_one = torch.zeros(0, dtype=K_pool.dtype, device=K_pool.device)
    taken = torch.zeros(P, dtype=torch.bool, device=K_pool.device)

    for _ in range(int(M)):
        value, s, alpha, beta, other = _candidate_values(
            G, g_mu, g_one, mu_pool, diag, weight_mode
        )
        # A repeat is a no-op on the objective and a collapsed Schur complement
        # means the atom is already in the span; neither is a selection.
        blocked = taken | (s <= schur_floor * diag)
        if bool(blocked.all()):
            break
        value = value.masked_fill(blocked, float("inf"))
        i = int(torch.argmin(value))
        picked.append(i)
        vals.append(float(value[i]))
        vals_other.append(float(other[i]))
        taken[i] = True

        d = torch.sqrt(s[i])
        t_i = G[:, i]
        row = (K_pool[i] - (t_i @ G if G.shape[0] else 0.0)) / d
        G = torch.cat([G, row.reshape(1, P)], dim=0)
        g_mu = torch.cat([g_mu, (alpha[i] / d).reshape(1)])
        g_one = torch.cat([g_one, (beta[i] / d).reshape(1)])
    return picked, vals, vals_other


def _swap_pass(
    K_pool: torch.Tensor,
    mu_pool: torch.Tensor,
    diag: torch.Tensor,
    picked: list[int],
    weight_mode: str,
    schur_floor: float,
    jitter: float,
) -> tuple[list[int], float, int]:
    """One weight-aware refinement sweep over the selected slots.

    Each slot in turn is vacated and refilled by the pool atom minimizing the
    same re-solved criterion, which is the greedy rule applied to the last
    position rather than only to a growing prefix. A swap is accepted only on
    a strict improvement, so the sweep is monotone by construction.

    Returns:
        ``(indices, value, n_swaps)`` -- the refined set, its criterion
        (excluding ``c_rho``), and how many slots actually moved.
    """
    idx = list(picked)
    n_swaps = 0
    for q in range(len(idx)):
        keep = [j for p, j in enumerate(idx) if p != q]
        sel = torch.tensor(keep, dtype=torch.long, device=K_pool.device)
        eye = torch.eye(len(keep), dtype=K_pool.dtype, device=K_pool.device)
        _, G, g_mu, g_one = _selection_state(
            K_pool[sel][:, sel] + jitter * eye, K_pool[sel], mu_pool[sel]
        )
        value, s, _, _, _ = _candidate_values(
            G, g_mu, g_one, mu_pool, diag, weight_mode
        )
        blocked = torch.zeros_like(value, dtype=torch.bool)
        blocked[sel] = True
        blocked |= s <= schur_floor * diag
        if bool(blocked.all()):
            continue
        value = value.masked_fill(blocked, float("inf"))
        best = int(torch.argmin(value))
        if best != idx[q]:
            n_swaps += 1
        idx[q] = best
    sel = torch.tensor(idx, dtype=torch.long, device=K_pool.device)
    eye = torch.eye(len(idx), dtype=K_pool.dtype, device=K_pool.device)
    _, _, g_mu, g_one = _selection_state(
        K_pool[sel][:, sel] + jitter * eye, K_pool[sel], mu_pool[sel]
    )
    A = float(g_mu @ g_mu)
    B = float(g_mu @ g_one)
    C = float(g_one @ g_one)
    val = -A if weight_mode == "unconstrained" else -A + (B - 1.0) ** 2 / C
    return idx, val, n_swaps


def sequential_bayesian_quadrature(
    z_pool: torch.Tensor,
    mu_pool: torch.Tensor,
    sigma: float,
    M: int,
    c_rho: torch.Tensor | float = 0.0,
    K_pool: torch.Tensor | None = None,
    weight_mode: str = "constrained",
    swap_passes: int = 0,
    jitter: float = DEFAULT_JITTER,
    schur_floor: float = SCHUR_FLOOR,
) -> dict:
    """Select ``M`` nodes greedily under the RE-SOLVED unit-sum weights.

    The weight-aware counterpart of
    :func:`~qfield.designed_quadrature.herding.kernel_herding`: where
    herding scores a candidate by the current equal-weight gradient and can
    only be reweighted afterwards, this scores it by the optimum the weight
    solve would reach once the candidate is in. The emitted weights come from
    :func:`constrained_weights`, so the rule is directly comparable with any
    other rule under :func:`~qfield.designed_quadrature.mmd.mmd_sq`.

    Args:
        z_pool: Candidate atoms, shape ``(P, d)``.
        mu_pool: Kernel mean at the pool atoms, shape ``(P,)`` -- exact where
            the reference admits it, otherwise the finite-bank estimate. This
            is the ONLY reference hook the construction reads.
        sigma: SE bandwidth. Ignored for the Gram when ``K_pool`` is given,
            which is how a caller supplies a kernel other than the SE one.
        M: Node budget.
        c_rho: Reference self-affinity. Enters the reported value as an
            additive constant only and never affects the selection; supply it
            when the returned ``value`` should be the ``MMD^2`` itself.
        K_pool: Optional precomputed pool Gram ``(P, P)``, reused across
            budgets. The dominant cost of a sweep, so passing it in matters.
            When supplied, the emitted weights are solved from its selected
            submatrix rather than from ``z_pool`` and ``sigma``.
        weight_mode: ``"constrained"`` selects and emits under the unit-sum
            constraint (the default);
            ``"unconstrained"`` selects and emits the classical Bayesian
            quadrature weights ``K^{-1} mu``, which do NOT sum to one and so
            do not integrate constants exactly. Both criteria are computed
            either way and reported.
        swap_passes: Refinement sweeps after the greedy prefix (see
            :func:`_swap_pass`). Zero keeps the construction nested in ``M``.
        jitter: Ridge on the Gram diagonal, for the selection AND for the
            emitted weight solve, so the two describe one matrix.
        schur_floor: Relative floor below which a candidate counts as being in
            the span of the selected set.

    Returns:
        ``{z, w, indices, n_selected, value, value_prefix, value_other,
        weight_mode, swap_passes, n_swaps, rank_limited, unit_sum}`` -- the
        selected nodes ``(m, d)`` and their weights ``(m,)``, the pool indices
        in selection order, the criterion the RETURNED set achieves (with
        ``c_rho`` added), the criterion at every length of the GREEDY prefix
        before any refinement (non-increasing, by the theorem in the module
        docstring), the criterion under the other weight mode, and whether the
        pool's span ran out before the budget did. With ``swap_passes = 0``
        ``value`` is the last entry of ``value_prefix``; with refinement it is
        at most that, and ``value_prefix`` still describes the unrefined
        prefix, which is what the nestedness claim of the ``M`` axis is about.

    Raises:
        ValueError: On a malformed pool, a shape mismatch, ``M < 1``, ``M``
            above the pool size, or an unknown ``weight_mode``.
    """
    if z_pool.ndim != 2:
        raise ValueError(f"z_pool must be (P, d); got {tuple(z_pool.shape)}")
    P = z_pool.shape[0]
    if mu_pool.shape != (P,):
        raise ValueError(
            f"mu_pool must be (P,) matching the pool; got "
            f"{tuple(mu_pool.shape)} against {P} atoms"
        )
    if int(M) < 1:
        raise ValueError(f"M must be at least 1; got {M!r}")
    if int(M) > P:
        raise ValueError(
            f"sequential Bayesian quadrature cannot select {M} atoms from a "
            f"pool of {P}; give it a larger candidate pool"
        )
    if weight_mode not in _WEIGHT_MODES:
        raise ValueError(
            f"weight_mode must be one of {_WEIGHT_MODES}; got {weight_mode!r}"
        )
    K = gram(z_pool, sigma) if K_pool is None else K_pool
    if K.shape != (P, P):
        raise ValueError(f"K_pool must be (P, P); got {tuple(K.shape)}")

    diag = torch.diagonal(K) + jitter
    picked, vals, vals_other = _greedy_select(
        K, mu_pool, diag, int(M), weight_mode, float(schur_floor)
    )
    if not picked:
        raise ValueError(
            "no admissible candidate: every pool atom has a collapsed Schur "
            "complement, which means the pool spans nothing"
        )
    value = vals[-1]
    value_other = vals_other[-1]
    n_swaps = 0
    for _ in range(int(swap_passes)):
        picked, swapped, moved = _swap_pass(
            K, mu_pool, diag, picked, weight_mode, float(schur_floor), jitter
        )
        n_swaps += moved
        # The reported value must describe the set that is RETURNED, so it is
        # taken from the pass rather than min-ed against the pre-pass value: a
        # refinement is monotone by construction (each slot's incumbent is
        # itself a candidate), and taking the minimum of the two would let the
        # record carry a value the emitted rule does not achieve whenever
        # round-off moves the last digit.
        value = swapped
        if moved == 0:
            break

    sel = torch.tensor(picked, dtype=torch.long, device=z_pool.device)
    m = int(sel.numel())
    mu_sel = mu_pool[sel]
    if weight_mode == "constrained":
        if K_pool is None:
            w = constrained_weights(z_pool[sel], mu_sel, sigma, jitter)
        else:
            eye = torch.eye(m, dtype=K.dtype, device=K.device)
            w = constrained_weights_from_gram(
                K[sel][:, sel] + jitter * eye, mu_sel
            )
    else:
        eye = torch.eye(m, dtype=K.dtype, device=K.device)
        w = torch.linalg.solve(K[sel][:, sel] + jitter * eye, mu_sel)

    c = float(c_rho)
    return {
        "z": z_pool[sel].detach().clone(),
        "w": w.detach().clone(),
        "indices": sel,
        "n_selected": m,
        "value": c + value,
        "value_prefix": [c + v for v in vals],
        "value_other": c + value_other,
        "weight_mode": weight_mode,
        "swap_passes": int(swap_passes),
        "n_swaps": n_swaps,
        "rank_limited": m < int(M),
        "unit_sum": float(w.sum()),
    }


def constrained_value_direct(
    K_sel: torch.Tensor,
    mu_sel: torch.Tensor,
    c_rho: torch.Tensor | float = 0.0,
) -> float:
    """The same optimum, computed from a dense solve rather than incrementally.

    The independent recomputation the incremental recursion is checked
    against: ``c_rho - A + (B - 1)^2 / C`` from one ``torch.linalg.solve``
    against ``[mu, 1]``.

    Args:
        K_sel: Jittered Gram at the selected nodes, ``(m, m)``.
        mu_sel: Kernel mean at the selected nodes, ``(m,)``.
        c_rho: Reference self-affinity.

    Returns:
        The unit-sum-constrained optimal ``MMD^2`` at those nodes.
    """
    ones = torch.ones(K_sel.shape[0], dtype=K_sel.dtype, device=K_sel.device)
    sol = torch.linalg.solve(K_sel, torch.stack([mu_sel, ones], dim=1))
    a, b = sol[:, 0], sol[:, 1]
    A = float(mu_sel @ a)
    B = float(ones @ a)
    C = float(ones @ b)
    return float(c_rho) - A + (B - 1.0) ** 2 / C
