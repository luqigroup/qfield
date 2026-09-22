"""The three arms and the M-sweep evaluation.

Scores a designed quadrature on a closed-form target. All three arms
share the SAME ``M`` i.i.d. nodes ``z0 ~ pi`` per repeat and the SAME exact
unit-sum SE-MMD^2 (:func:`qfield.designed_quadrature.mmd.mmd_sq`):

  (1) floor    -- ``(z0, w = 1/M)``: the i.i.d.-MC integration error of equal
                  weights (the baseline a plain posterior sampler pays).
  (2) reweight -- ``(z0, w_c)``: the unit-sum-constrained MMD^2 minimizer at
                  fixed nodes. THEOREM (per draw) ``reweight <= floor``.
  (3) move     -- designed nodes by direct MMD^2 descent (best-iterate seeded
                  at the reweight config). THEOREM (per draw) ``move <=
                  reweight``.

:func:`run_target` sweeps ``M`` and repeats the three arms, returning per-``M``
median/IQR triples plus the WORST per-draw ``reweight - floor`` and ``move -
reweight`` gaps so the two inequalities can be asserted to floating tolerance
over the whole sweep (a positive gap beyond tolerance is a bug in ``mu`` /
``c_rho`` / the solve). :func:`functional_spot_check` checks one bounded
moment ``E_pi[x_1^2]`` -- the constrained weights integrate it, and constants,
exactly because ``sum_j w_j = 1``. :func:`cross_check_target` validates the
metric before any sweep.

Determinism is local and reproducible: repeat ``r`` at budget ``M`` uses a
fresh ``torch.Generator`` seeded ``base_seed + 1000 * M + r`` (no global seed).

Reference:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (the MMD-optimal quadrature and
    its i.i.d.-floor comparison).
"""

from __future__ import annotations

import torch

from qfield.designed_quadrature.mmd import (
    cross_check_mmd,
    mmd_sq,
    mmd_sq_batched,
)
from qfield.designed_quadrature.move import (
    move_descend,
    move_descend_batched,
)
from qfield.designed_quadrature.target import QuadratureTarget
from qfield.designed_quadrature.weights import (
    DEFAULT_JITTER,
    constrained_weights,
    constrained_weights_batched,
)

DTYPE = torch.float64

# Tolerance for the per-draw inequality (theorem) checks; the float64 solves
# and exact-K scoring make the two theorems hold to round-off.
THEOREM_TOL = 1e-9


def cross_check_target(
    target: QuadratureTarget,
    sigma: float,
    seed: int,
    n_nodes: int = 11,
) -> float:
    """Relative gap between the explicit MMD^2 and the target's closed form.

    Samples ``n_nodes`` nodes (from the target's own sampler for the GMM, or a
    spread cloud around the Gaussian mean) and a random unit-sum weighting,
    then compares the explicit
    :func:`qfield.designed_quadrature.mmd.mmd_sq` against the target's
    independent closed form
    (:attr:`QuadratureTarget.cross_check_reference`). The caller asserts the
    returned gap is below ``~1e-8`` before any run.

    Args:
        target: The quadrature target.
        sigma: SE bandwidth.
        seed: Seed for the local generator (nodes + weights).
        n_nodes: Number of nodes for the cross-check (small).

    Returns:
        Relative gap ``|explicit - reference| / (|reference| + 1e-30)``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z = target.sampler(int(n_nodes), gen).to(DTYPE)
    w = torch.randn(int(n_nodes), generator=gen, dtype=DTYPE)
    w = w / w.sum()  # unit-sum (the metric is invariant to a rescale anyway)
    mu = target.mu_fn(z)
    return cross_check_mmd(
        z, w, mu, target.c_rho, sigma, target.cross_check_reference
    )


def score_arms_one_draw(
    z0: torch.Tensor,
    target: QuadratureTarget,
    sigma: float,
    move_lr: float,
    move_iters: int,
    jitter: float = DEFAULT_JITTER,
) -> dict:
    """Score the three arms on ONE i.i.d. node draw ``z0``.

    Returns the (clamped, reported) MMD^2 of each arm on the SAME nodes, plus
    the per-draw theorem gaps ``reweight - floor`` and ``move - reweight``.

    Args:
        z0: i.i.d. seed nodes ``(M, d)`` drawn from the target.
        target: The quadrature target (supplies ``mu_fn`` / ``c_rho``).
        sigma: SE bandwidth.
        move_lr: Adam learning rate for the move arm.
        move_iters: Adam steps for the move arm.
        jitter: Tiny ridge for the weight solves.

    Returns:
        ``{floor, reweight, move, rw_minus_floor, mv_minus_rw}`` -- the three
        reported MMD^2 values (floats, clamped at 0) and the two per-draw
        theorem gaps (un-clamped floats, so a violation is visible).
    """
    M = z0.shape[0]
    mu0 = target.mu_fn(z0)
    c_rho = target.c_rho

    w_eq = torch.full((M,), 1.0 / M, dtype=z0.dtype, device=z0.device)
    floor = float(mmd_sq(z0, w_eq, mu0, c_rho, sigma))

    w_c = constrained_weights(z0, mu0, sigma, jitter)
    reweight = float(mmd_sq(z0, w_c, mu0, c_rho, sigma))

    _, move = move_descend(
        z0, target.mu_fn, c_rho, sigma, move_lr, move_iters, jitter
    )

    return {
        "floor": max(floor, 0.0),
        "reweight": max(reweight, 0.0),
        "move": move,  # already clamped in move_descend
        # The theorem gaps use the UN-clamped reweight value so a real
        # violation (reweight > floor beyond tolerance) is not masked.
        "rw_minus_floor": reweight - floor,
        "mv_minus_rw": move - reweight,
    }


def score_arms_batched(
    z0: torch.Tensor,
    target: QuadratureTarget,
    sigma: float,
    move_lr: float,
    move_iters: int,
    jitter: float = DEFAULT_JITTER,
) -> dict:
    """Score the three arms on ``R`` STACKED i.i.d. node draws at once.

    The vectorized-across-repeats analogue of :func:`score_arms_one_draw`:
    given ``z0`` of shape ``(R, M, d)`` (one i.i.d. draw per slice), returns
    per-repeat arrays of the three arms' MMD^2 and the two per-draw theorem
    gaps. Numerically identical to looping :func:`score_arms_one_draw` over
    the slices (same row-independent ``mu``, same per-slice Gram / solve /
    Adam trajectory; see :func:`move_descend_batched`). The ``mu_fn`` is
    row-independent so it is called once on the flattened nodes.

    Args:
        z0: Stacked i.i.d. seed nodes ``(R, M, d)`` drawn from the target.
        target: The quadrature target (supplies ``mu_fn`` / ``c_rho``).
        sigma: SE bandwidth.
        move_lr: Adam learning rate for the move arm.
        move_iters: Adam steps for the move arm.
        jitter: Tiny ridge for the weight solves.

    Returns:
        ``{floor, reweight, move, rw_minus_floor, mv_minus_rw}`` -- each a
        shape-``(R,)`` float64 tensor (floor/reweight/move clamped at 0; the
        two gaps un-clamped so a violation is visible).
    """
    R, M, _ = z0.shape
    c_rho = target.c_rho
    mu0 = target.mu_fn(z0.reshape(R * M, -1)).reshape(R, M)  # row-independent

    w_eq = torch.full((R, M), 1.0 / M, dtype=z0.dtype, device=z0.device)
    floor = mmd_sq_batched(z0, w_eq, mu0, c_rho, sigma)  # (R,)

    w_c = constrained_weights_batched(z0, mu0, sigma, jitter)
    reweight = mmd_sq_batched(z0, w_c, mu0, c_rho, sigma)  # (R,)

    _, move = move_descend_batched(
        z0, target.mu_fn, c_rho, sigma, move_lr, move_iters, jitter
    )  # (R,), already clamped

    return {
        "floor": torch.clamp(floor, min=0.0),
        "reweight": torch.clamp(reweight, min=0.0),
        "move": move,
        # Un-clamped gaps so a real violation is not masked.
        "rw_minus_floor": reweight - floor,
        "mv_minus_rw": move - reweight,
    }


def _median_iqr(vals: list[float]) -> tuple[float, float, float]:
    """``(median, q25, q75)`` over the finite entries of ``vals``."""
    t = torch.tensor(vals, dtype=DTYPE)
    t = t[torch.isfinite(t)]
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


# --------------------------------------------------------------------------
# The amortizer trainers' eval row.
#
# The trainers score a fourth object the sweep above does not: the network's
# one-forward output ("warm"). The row is assembled here so the two trainers
# cannot drift apart in what they record.
# --------------------------------------------------------------------------
COND_LIMIT = 1e13
COND_SUB = 32


def seed_gram_cond(
    z0: torch.Tensor, sigma: float, n_sub: int = COND_SUB
) -> dict:
    """``cond(K)`` of the seed Gram: median, min and max.

    The RAW Gram (no jitter -- the jitter belongs to the solve, not to this
    reference), with min and max beside the median so the reader sees how far
    a value sits from the threshold. Capped at ``n_sub`` node sets because
    this is an SVD apiece and would otherwise cost more than every arm it
    describes.

    Args:
        z0: Stacked seed nodes ``(R, M, d)``.
        sigma: SE bandwidth.
        n_sub: Cap on the number of node sets inspected.

    Returns:
        ``{median, min, max, n_repeats}``.
    """
    from qfield.designed_quadrature.mmd import gram_batched
    from qfield.designed_quadrature.weights import _serial_solve_above

    n = min(int(n_sub), int(z0.shape[0]))
    with torch.no_grad():
        with _serial_solve_above(int(z0.shape[1])):
            c = torch.linalg.cond(gram_batched(z0[:n], sigma))
    return {
        "median": float(c.median()),
        "min": float(c.min()),
        "max": float(c.max()),
        "n_repeats": int(n),
    }


def eval_row(
    M: int,
    floor: torch.Tensor,
    reweight: torch.Tensor,
    warm: torch.Tensor,
    oracle: torch.Tensor,
    cond_seed: dict,
) -> dict:
    """One per-``M`` record row for the amortizer trainers.

    Each arm arrives as the per-repeat scores ``(R,)``; the row keeps the
    median triple (the reported statistic) AND the mean, because the bounds
    are statements about ``E MMD^2`` and the floor's identity ``E = (1 -
    c_rho)/M`` is exact in the mean only. The floor is a degenerate
    V-statistic, so its median sits strictly below its mean by a
    target-specific, ``M``-independent factor: ``M * floor[0]`` does NOT
    reproduce ``1 - c_rho``, and a reader checking the identity against the
    median would wrongly conclude the harness is broken.

    ``move_gain = reweight/warm`` is the learned displacement's own factor
    once the solve has run. Below one it is a net loss.
    """
    fr, rwr = _median_iqr(floor.tolist()), _median_iqr(reweight.tolist())
    wr, orc = _median_iqr(warm.tolist()), _median_iqr(oracle.tolist())
    return {
        "M": int(M),
        "floor": fr,
        "reweight": rwr,
        "warm": wr,
        "oracle": orc,
        "floor_mean": float(floor.mean()),
        "reweight_mean": float(reweight.mean()),
        "warm_mean": float(warm.mean()),
        "oracle_mean": float(oracle.mean()),
        "warm_over_floor": wr[0] / fr[0] if fr[0] > 0 else float("inf"),
        "warm_over_oracle": wr[0] / orc[0] if orc[0] > 0 else float("inf"),
        "reweight_over_floor": rwr[0] / fr[0] if fr[0] > 0 else float("inf"),
        "phi": 1.0 - rwr[0] / fr[0] if fr[0] > 0 else float("nan"),
        "move_gain": rwr[0] / wr[0] if wr[0] > 0 else float("inf"),
        "cond_seed": cond_seed,
        "cond_limited": bool(cond_seed["median"] > COND_LIMIT),
    }


def run_target(
    target: QuadratureTarget,
    m_list: list[int],
    n_repeats: int,
    sigma: float,
    base_seed: int,
    move_lr: float,
    move_iters: int,
    jitter: float = DEFAULT_JITTER,
    progress: bool = False,
) -> dict:
    """Sweep ``M`` and score the three arms over ``n_repeats`` i.i.d. draws.

    For each ``M`` in ``m_list`` and each repeat, draws ``M`` i.i.d. nodes
    (per-repeat ``torch.Generator`` seeded ``base_seed + 1000 * M + r``) and
    scores floor / reweight / move. Returns per-``M`` median/IQR triples, the
    arm ratios, and the WORST per-draw theorem gaps across the whole sweep.

    Args:
        target: The quadrature target.
        m_list: Node-budget sweep.
        n_repeats: i.i.d. node-draw repeats (median + IQR).
        sigma: SE bandwidth.
        base_seed: Base seed for the per-repeat generator offsets.
        move_lr: Adam learning rate for the move arm.
        move_iters: Adam steps for the move arm.
        jitter: Tiny ridge for the weight solves.
        progress: If True, print a per-(M) progress line.

    Returns:
        ``{rows, worst_rw_minus_floor, worst_mv_minus_rw}`` where ``rows`` is a
        list of per-``M`` dicts ``{M, floor, reweight, move, rw_ratio,
        mv_ratio, mv_over_rw}`` (each arm a ``(median, q25, q75)`` triple) and
        the two worsts are the most-positive per-draw gaps over the sweep
        (``<= THEOREM_TOL`` confirms the per-draw theorems).
    """
    rows = []
    worst_rw_minus_floor = -float("inf")
    worst_mv_minus_rw = -float("inf")
    R = int(n_repeats)
    for M in m_list:
        # Stack the R i.i.d. node sets, each with its OWN per-repeat generator
        # seeded base_seed + 1000*M + r (identical to the sequential path),
        # then score all repeats at once.
        draws = []
        for r in range(R):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(base_seed) + 1000 * int(M) + r)
            draws.append(target.sampler(int(M), gen).to(DTYPE))
        z0 = torch.stack(draws, dim=0)  # (R, M, d)

        arms = score_arms_batched(
            z0, target, sigma, move_lr, move_iters, jitter
        )
        floor_vals = arms["floor"].tolist()
        rw_vals = arms["reweight"].tolist()
        mv_vals = arms["move"].tolist()
        worst_rw_minus_floor = max(
            worst_rw_minus_floor, float(arms["rw_minus_floor"].max())
        )
        worst_mv_minus_rw = max(
            worst_mv_minus_rw, float(arms["mv_minus_rw"].max())
        )

        f_med, f_lo, f_hi = _median_iqr(floor_vals)
        r_med, r_lo, r_hi = _median_iqr(rw_vals)
        m_med, m_lo, m_hi = _median_iqr(mv_vals)
        rows.append(
            {
                "M": int(M),
                "floor": (f_med, f_lo, f_hi),
                "reweight": (r_med, r_lo, r_hi),
                "move": (m_med, m_lo, m_hi),
                "rw_ratio": r_med / f_med if f_med > 0 else float("inf"),
                "mv_ratio": m_med / f_med if f_med > 0 else float("inf"),
                "mv_over_rw": m_med / r_med if r_med > 0 else float("inf"),
            }
        )
        if progress:
            print(
                f"  [{target.name}] M={M:>3d}: floor={f_med:.3e} "
                f"reweight={r_med:.3e} move={m_med:.3e} "
                f"(rw/fl={r_med / f_med if f_med > 0 else float('inf'):.3f}, "
                f"mv/fl={m_med / f_med if f_med > 0 else float('inf'):.3f})"
            )

    return {
        "rows": rows,
        "worst_rw_minus_floor": worst_rw_minus_floor,
        "worst_mv_minus_rw": worst_mv_minus_rw,
    }


def functional_spot_check(
    target: QuadratureTarget,
    M: int,
    sigma: float,
    seed: int,
    move_lr: float,
    move_iters: int,
    jitter: float = DEFAULT_JITTER,
) -> dict:
    """Functional spot check ``|sum_j w_j z_{j,1}^2 - E_pi[x_1^2]|`` at one M.

    A single bounded-moment probe: the unit-sum
    constrained weights (and the move) should integrate ``x_1^2`` at least as
    well as the equal-weight floor on the same nodes. Because ``sum_j w_j = 1``
    for the reweight and move arms, they integrate CONSTANTS exactly, so this
    moment is the first non-trivial functional.

    Args:
        target: The quadrature target (supplies ``mu_fn`` / ``e_x1sq`` / sampler).
        M: Node budget for the check.
        sigma: SE bandwidth.
        seed: Seed for the single node draw.
        move_lr: Adam learning rate for the move arm.
        move_iters: Adam steps for the move arm.
        jitter: Tiny ridge for the weight solves.

    Returns:
        ``{e_x1sq, floor, reweight, move}`` -- the target's ``E_pi[x_1^2]`` and
        the absolute functional error of each arm.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z0 = target.sampler(int(M), gen).to(DTYPE)
    mu0 = target.mu_fn(z0)
    e_x1sq = target.e_x1sq

    w_eq = torch.full((M,), 1.0 / M, dtype=DTYPE)
    err_floor = abs(float((w_eq * z0[:, 0] ** 2).sum()) - e_x1sq)

    w_c = constrained_weights(z0, mu0, sigma, jitter)
    err_rw = abs(float((w_c * z0[:, 0] ** 2).sum()) - e_x1sq)

    z_star, _ = move_descend(
        z0, target.mu_fn, target.c_rho, sigma, move_lr, move_iters, jitter
    )
    w_mv = constrained_weights(z_star, target.mu_fn(z_star), sigma, jitter)
    err_mv = abs(float((w_mv * z_star[:, 0] ** 2).sum()) - e_x1sq)

    return {
        "e_x1sq": e_x1sq,
        "floor": err_floor,
        "reweight": err_rw,
        "move": err_mv,
    }
