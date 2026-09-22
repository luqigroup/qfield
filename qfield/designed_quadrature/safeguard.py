"""Safeguarded emission: select among the arms by the computable criterion.

Given a finite menu of candidate quadratures at one observation (the seeds
under equal weights, the constrained reweighting of those seeds, and the
network's moved-and-reweighted nodes), :func:`select_emission` returns the one
minimizing

    Jhat(Q) = w^T K w - 2 w^T mu_hat ,

the reported ``MMD^2`` with the reference self-affinity ``c_rho`` dropped. That
constant is a functional of the reference alone, independent of the node set,
the node count, and ``M``, so it is common to every arm at one observation,
cancels from the comparison, and never has to be estimated to make a selection.

What the selection buys, in two regimes:

  * **Exact estimand** (``mu_fn`` returns the true kernel mean: Gaussian,
    Gaussian-mixture, or a Nadaraya--Watson empirical conditional). The
    criterion ordering IS the ``MMD^2`` ordering, so the emitted quadrature is
    never worse in ``MMD^2`` than any menu member, in particular never worse
    than the seed arm it was handed. Pathwise, at every observation and every
    budget, including ones never seen in training.
  * **Sampled estimand** (``mu_fn`` is a finite-bank estimate, such as a
    trained conditional flow). The ordering above is in the ESTIMATED
    criterion only. Recovering never-worse costs a certification split
    independent of every quantity used to fit the arms, plus a printed
    per-emission slack. That slack lives in
    :mod:`~qfield.designed_quadrature.certsel` and this module does NOT
    compute it; see :func:`select_emission`'s ``runner_up_gap``, which is the
    quantity the slack must be compared against.

The selection is deliberately kept out of :mod:`arms`, which reports all three
arms unconditionally so the per-instance gaps stay visible. Reporting and
emitting are different jobs: a sweep wants every arm, a deployment wants one.

Ties are broken toward the LAST candidate. This is not a detail: a
zero-initialized output head makes the move arm bitwise equal to the reweight
arm, so ties are systematic rather than rare, and the reported activation rate
depends on the convention.

Reference:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (the MMD^2 identity every arm
    is scored with).
  - Owen and Zhou (2000) obtain a comparable guarantee by MIXING rather than
    by selection; the argmin form used here is elementary.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

from qfield.designed_quadrature.mmd import gram

DTYPE = torch.float64

# Absolute tolerance for the two-arm inequality.
TWOARM_TOL = 1e-9


def criterion(
    z: torch.Tensor,
    w: torch.Tensor,
    mu: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """The computable selection criterion ``w^T K w - 2 w^T mu``.

    The reported ``MMD^2`` minus the reference self-affinity ``c_rho``. That
    constant depends on the reference alone, so dropping it leaves the arm
    ordering unchanged, across candidates with different node sets and
    different budgets alike, while removing the one quantity of the pair
    ``(mu, c_rho)`` that a selection does not need.

    Note this is a criterion, not a magnitude: ``margin`` values derived from
    it are in absolute criterion units, NOT a relative gain
    ``1 - MMD^2/MMD^2(Q_0)``.

    Args:
        z: Node positions, shape ``(m, d)``.
        w: Unit-sum node weights, shape ``(m,)`` (NOT renormalized here).
        mu: Kernel mean at the nodes, shape ``(m,)`` -- exact where the
            reference admits it, otherwise the finite-bank estimate.
        sigma: SE bandwidth.

    Returns:
        Scalar criterion as a 0-dim tensor (un-clamped; it is an ``MMD^2``
        shifted by a constant and so may be negative).

    Raises:
        ValueError: If ``mu`` does not have shape ``(m,)``. A ``(m, 1)``
            estimand would otherwise broadcast silently and change the value.
    """
    if mu.shape != (z.shape[0],):
        raise ValueError(
            f"mu must be (m,) matching the nodes; got {tuple(mu.shape)} "
            f"against {z.shape[0]} nodes"
        )
    K = gram(z, sigma)
    return w @ K @ w - 2.0 * (w @ mu)


def select_emission(
    candidates: Sequence[tuple[torch.Tensor, torch.Tensor]],
    mu_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma: float,
    names: Sequence[str] | None = None,
    fitting_criterion: bool = False,
    assert_unit_sum: bool = True,
) -> dict:
    """Select the menu member minimizing the computable criterion.

    Every candidate is scored against the SAME estimand: ``mu_fn`` is called at
    each candidate's own nodes, and is required to be a deterministic pure
    function of them. A stochastic hook, such as a fresh certification sample
    per call, would score the arms against different empirical references,
    which breaks both the ``c_rho`` cancellation and the common-split
    cancellation the per-emission slack relies on.

    The first candidate is taken to be the seed arm (equal weights at the
    i.i.d. nodes) and the LAST to be the arm a deployment would otherwise emit
    (the network's). ``margin`` is measured against the first, ``active``
    against the last.

    Args:
        candidates: Menu of ``(z, w)`` pairs, ``z`` of shape ``(m, d)`` and
            ``w`` of shape ``(m,)`` summing to one. Node counts may differ
            between candidates, and the criterion is still the ``MMD^2``
            argmin, but ``margin``, ``active`` and the two-arm check assume
            the documented ordering.
        mu_fn: Deterministic kernel mean hook ``z -> E_rho[k(., z)]`` of shape
            ``(m,)``.
        sigma: SE bandwidth.
        names: Optional arm names, same length as ``candidates``.
        fitting_criterion: Declare that ``mu_fn`` IS the estimand the weights
            were solved against. Only then does the two-arm inequality hold,
            and only then is it checked. It is FALSE by default because the
            certification-split regime scores against a different estimand,
            where the seed arm CAN strictly win and the check would fire on a
            correct implementation.
        assert_unit_sum: Check each candidate's weights sum to one, which is
            the premise of the certificate (equal weights must be feasible for
            the same-node unit-sum solve).

    Returns:
        ``{z, w, index, name, values, margin, runner_up_gap, active,
        twoarm_gap}`` -- the selected nodes and weights (detached), its
        position and name, every arm's criterion value, the improvement over
        the seed arm, the gap to the runner-up (the quantity a per-emission
        slack must be compared against), whether the safeguard moved the
        answer off the last candidate, and the seed-minus-reweight gap
        (positive means the two-arm inequality is violated).

    Raises:
        ValueError: Empty menu, malformed shapes, non-unit-sum weights, a
            non-finite criterion, or a non-deterministic ``mu_fn``.
        AssertionError: The two-arm inequality is violated, when
            ``fitting_criterion`` is set.
    """
    if len(candidates) == 0:
        raise ValueError("select_emission needs a non-empty candidate menu")
    if names is not None and len(names) != len(candidates):
        raise ValueError("names must have one entry per candidate")

    with torch.no_grad():
        # One mu per DISTINCT node tensor: the seed and reweight arms share
        # their nodes, and mu_fn is O(M L) against a reference bank with
        # L >> M, so the reuse is the dominant saving on the conditional path.
        cache: dict[int, torch.Tensor] = {}

        def _mu(z: torch.Tensor) -> torch.Tensor:
            key = id(z)
            if key not in cache:
                cache[key] = mu_fn(z)
            return cache[key]

        values: list[float] = []
        for i, (z, w) in enumerate(candidates):
            if z.ndim != 2:
                raise ValueError(
                    f"candidate {i}: nodes must be (m, d); got {tuple(z.shape)}"
                )
            if w.shape != (z.shape[0],):
                raise ValueError(
                    f"candidate {i}: weights must be (m,) matching nodes; got "
                    f"{tuple(w.shape)}"
                )
            if assert_unit_sum and abs(float(w.sum()) - 1.0) > 1e-8:
                raise ValueError(
                    f"candidate {i}: weights sum to {float(w.sum()):.6e}, not "
                    "one; the certificate's premise is a unit-sum menu"
                )
            v = float(criterion(z, w, _mu(z), sigma))
            if not math.isfinite(v):
                raise ValueError(
                    f"candidate {i} has a non-finite criterion ({v}); a "
                    "non-finite arm must never be emitted"
                )
            values.append(v)

        # A stochastic mu_fn would silently un-pair the arms; catch it here
        # rather than let it corrupt a certificate downstream.
        z0 = candidates[0][0]
        if not torch.equal(cache[id(z0)], mu_fn(z0)):
            raise ValueError(
                "mu_fn is not a deterministic function of its nodes; the arms "
                "must be scored against one fixed reference"
            )

    # Ties go to the LAST candidate, the deployed arm.
    index = min(range(len(values)), key=lambda i: (values[i], -i))
    z_sel, w_sel = candidates[index]

    twoarm_gap = (values[0] - values[1]) if len(values) > 1 else float("-inf")
    if fitting_criterion and len(values) > 1:
        if not torch.equal(candidates[1][0], candidates[0][0]):
            raise ValueError(
                "fitting_criterion requires candidates[1] to be the "
                "reweighting of candidates[0]'s nodes"
            )
        if twoarm_gap < -TWOARM_TOL:
            raise AssertionError(
                f"the seed arm strictly beats the same-node reweighting "
                f"({values[0]:.6e} < {values[1]:.6e}); the solve is broken, or "
                "the weights were fitted against an estimand other than mu_fn "
                "-- sg-prop-twoarm governs the fitting criterion only"
            )

    ordered = sorted(values)
    return {
        "z": z_sel.detach(),
        "w": w_sel.detach(),
        "index": index,
        "name": (names[index] if names is not None else f"arm{index}"),
        "values": values,
        "margin": values[0] - values[index],
        "runner_up_gap": (ordered[1] - ordered[0]) if len(values) > 1
        else float("inf"),
        "active": index != len(candidates) - 1,
        "twoarm_gap": twoarm_gap,
    }
