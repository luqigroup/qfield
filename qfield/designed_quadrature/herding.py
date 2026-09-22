"""Kernel herding by conditional gradient.

Herding reads the kernel mean ``mu_rho`` and nothing else -- no density, no
score. It is a per-target greedy solve, rerun from scratch at every
observation and at every node budget.

Kernel herding is the Frank--Wolfe (conditional-gradient) algorithm on

    f(g) = 1/2 || g - mu_rho ||_H^2

over the marginal polytope, restricted here to the convex hull of a finite
candidate pool. Three step schedules are implemented, and they produce
different objects:

  * ``"herding"`` -- ``gamma_t = 1/(t+1)``, the classical schedule of Chen,
    Welling and Smola. The iterate stays the UNIFORM average of the selected
    multiset, so the emitted rule carries EQUAL weights, which is what the
    published method returns.
  * ``"fw"`` -- ``gamma_t = 2/(t+2)``, the standard Frank--Wolfe schedule of
    Bach, Lacoste-Julien and Obozinski. Weights are non-negative, sum to one,
    and are generally not uniform.
  * ``"line_search"`` -- the exact line-search variant, which attains the
    linear rate when ``mu_rho`` lies in the relative interior of the polytope.
    The search is per-step optimal along the chosen direction and is therefore
    NOT guaranteed to end below a fixed schedule after a fixed number of steps;
    greedy step-wise optimality is not optimality over the budget.

The interior condition is not generic: for a compact domain and an
infinite-dimensional RKHS -- the squared-exponential case used throughout this
package -- no interior radius exists, and the fast rate provably fails
(Bach--Lacoste-Julien--Obozinski, Proposition 2). The line-search variant is
supplied anyway.

Selection with repeats is permitted by default, as in the published algorithm.
The number of DISTINCT atoms is reported.

Reference:
  - Chen, Welling and Smola, "Super-samples from kernel herding", UAI 2010.
  - Bach, Lacoste-Julien and Obozinski, "On the equivalence between herding and
    conditional gradient algorithms", ICML 2012, Sec. 4.2 and Props. 1-2.
  - Huszar and Duvenaud, arXiv:1204.1664, on optimally reweighting a herded set.
"""

from __future__ import annotations

import torch

from qfield.designed_quadrature.mmd import gram

DTYPE = torch.float64

_SCHEDULES = ("herding", "fw", "line_search")


def herding_step_sizes(mode: str, n_steps: int) -> list[float] | None:
    """The step schedule for ``mode``, or ``None`` when it is a line search.

    Args:
        mode: One of ``"herding"``, ``"fw"``, ``"line_search"``.
        n_steps: Number of conditional-gradient steps.

    Returns:
        A list of ``n_steps`` step sizes, or ``None`` for ``"line_search"``.

    Raises:
        ValueError: If ``mode`` is not a known schedule.
    """
    if mode not in _SCHEDULES:
        raise ValueError(f"mode must be one of {_SCHEDULES}; got {mode!r}")
    if mode == "line_search":
        return None
    if mode == "herding":
        return [1.0 / (t + 1.0) for t in range(int(n_steps))]
    return [2.0 / (t + 2.0) for t in range(int(n_steps))]


def kernel_herding(
    z_pool: torch.Tensor,
    mu_pool: torch.Tensor,
    sigma: float,
    M: int,
    mode: str = "herding",
    K_pool: torch.Tensor | None = None,
) -> dict:
    """Herd ``M`` nodes out of a candidate pool by conditional gradient.

    Minimizes ``1/2 w^T K w - w^T mu`` over the simplex on the pool atoms by
    ``M`` conditional-gradient steps, each of which puts all its mass on the
    atom minimizing the current gradient. After ``M`` steps the support has at
    most ``M`` distinct atoms, and the returned rule is that support with its
    accumulated weights.

    The objective is the unit-sum ``MMD^2`` up to the constant ``c_rho``, so
    the returned rule is directly comparable with every other arm under
    :func:`~qfield.designed_quadrature.mmd.mmd_sq`.

    Args:
        z_pool: Candidate atoms, shape ``(P, d)``.
        mu_pool: Kernel mean at the pool atoms, shape ``(P,)`` -- exact where
            the reference admits it, otherwise the finite-bank estimate.
        sigma: SE bandwidth.
        M: Node budget (number of conditional-gradient steps).
        mode: Step schedule; see :func:`herding_step_sizes`.
        K_pool: Optional precomputed pool Gram ``(P, P)``, reused across
            budgets. The dominant cost of a sweep, so passing it in matters.

    Returns:
        ``{z, w, indices, n_distinct, mode, mass}`` -- the selected nodes
        ``(m, d)`` with ``m <= M`` distinct atoms, their unit-sum non-negative
        weights ``(m,)``, the pool indices selected at each of the ``M`` steps
        (with repeats, length ``M``), how many were distinct, the schedule
        used, and ``mass``, the iterate's total mass BEFORE the round-off
        renormalization. ``mass`` is reported because the division that
        produces unit-sum weights would otherwise launder an infeasible
        iterate into a feasible-looking rule: a first step that is not a full
        step leaves the conditional gradient at mass below one, and after the
        divide the returned rule is indistinguishable from the correct one at
        ``M = 1`` and merely wrong at larger budgets.

    Raises:
        ValueError: On a malformed pool, a shape mismatch, or ``M < 1``.
        RuntimeError: If the iterate is infeasible before renormalization,
            i.e. its mass differs from one by more than ``1e-8``. Conditional
            gradient keeps the iterate in the simplex by construction, so this
            is a defect in the step schedule rather than round-off.
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
            f"herding cannot select {M} atoms from a pool of {P}; give it a "
            "larger candidate pool"
        )
    if mode not in _SCHEDULES:
        raise ValueError(f"mode must be one of {_SCHEDULES}; got {mode!r}")

    M = int(M)
    K = gram(z_pool, sigma) if K_pool is None else K_pool
    if K.shape != (P, P):
        raise ValueError(f"K_pool must be (P, P); got {tuple(K.shape)}")

    steps = herding_step_sizes(mode, M)
    w = torch.zeros(P, dtype=z_pool.dtype, device=z_pool.device)
    Kw = torch.zeros(P, dtype=z_pool.dtype, device=z_pool.device)
    picked: list[int] = []

    for t in range(M):
        # grad f(w) = K w - mu; the linear minimization oracle over the simplex
        # puts all mass on the smallest gradient coordinate.
        i = int(torch.argmin(Kw - mu_pool))
        picked.append(i)
        if t == 0:
            # Frank-Wolfe must start FEASIBLE, i.e. at a vertex of the simplex.
            # A line search at t = 0 would leave the iterate at mass g < 1, so
            # the first step is a full one for every schedule (which is also
            # what both fixed schedules prescribe: 1/(0+1) = 2/(0+2) = 1).
            g = 1.0
        elif steps is None:
            # Exact line search along w -> (1 - g) w + g e_i. The objective is
            # quadratic in g with curvature ||Phi(z_i) - mu_w||^2.
            d_vec = -w.clone()
            d_vec[i] += 1.0
            Kd = K[:, i] - Kw
            quad = float(d_vec @ Kd)
            lin = float(d_vec @ (Kw - mu_pool))
            g = 1.0 if quad <= 0.0 else min(1.0, max(0.0, -lin / quad))
        else:
            g = float(steps[t])
        w = (1.0 - g) * w
        w[i] += g
        Kw = (1.0 - g) * Kw + g * K[:, i]

    support = torch.nonzero(w > 0.0, as_tuple=False).reshape(-1)
    w_sel = w[support]
    # The mass BEFORE the divide is the feasibility statement; the divide that
    # follows is for accumulated round-off only and must never be able to
    # rescue an infeasible iterate into a unit-sum one.
    mass = float(w_sel.sum())
    if abs(mass - 1.0) > 1e-8:
        raise RuntimeError(
            f"kernel herding left the simplex: iterate mass {mass:.6e} at "
            f"M = {M}, mode {mode!r}. Conditional gradient keeps the iterate "
            "feasible by construction, so the step schedule is wrong -- most "
            "often a first step that is not a full step."
        )
    w_sel = w_sel / mass
    return {
        "z": z_pool[support].detach().clone(),
        "w": w_sel.detach().clone(),
        "indices": torch.tensor(picked, dtype=torch.long),
        "n_distinct": int(support.numel()),
        "mode": mode,
        "mass": mass,
    }
