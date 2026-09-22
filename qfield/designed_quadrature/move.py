"""Direct MMD^2 descent over node positions.

The designed nodes are obtained by descending the objective

    z* = argmin_z MMD^2(z, w_c(z)),

with Adam over node positions ``z`` and the unit-sum-constrained optimal
weights ``w_c(z)`` (:func:`qfield.designed_quadrature.weights.constrained_weights`)
re-solved at every ``z``. Autograd flows through that linear solve, so the
gradient is of the fully optimized-weight objective.

The best iterate is seeded at the reweight configuration ``(z0, w_c(z0))`` and
updated only when a strictly lower value is seen, so ``MMD^2(move) <=
MMD^2(reweight)`` holds whatever path the optimizer takes (a diverging step
never raises the reported value). The candidate pool also contains the ray
probes ``{z0 - eta g : eta in H}``, with ``g`` the frozen-weight MMD^2
gradient at the reweight configuration and ``H = {2**k / L(w) :
k = 0..RAY_KMAX}`` for ``L(w)`` the uniform smoothness constant; a minimum
over a superset can only lower the returned value.

Reference:
  - Belhadji, Sharp, Marzouk, "To discretize continually ...",
    arXiv:2605.14142 (the MMD-optimal designed quadrature; node motion is the
    mean-shift fixed point of this objective).
"""

from __future__ import annotations

from typing import Callable

import torch

from qfield.designed_quadrature.mmd import (
    gram_batched,
    mmd_sq,
    mmd_sq_batched,
)
from qfield.designed_quadrature.weights import (
    DEFAULT_JITTER,
    constrained_weights,
    constrained_weights_batched,
)

# Step grid H(w) = {2**k / L(w) : k = 0..RAY_KMAX} for the ray probes added to
# the candidate pool; k = 0 is the step 1 / L(w).
RAY_KMAX = 22


def _frozen_weight_grad(
    z: torch.Tensor,
    w: torch.Tensor,
    mu_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma: float,
) -> torch.Tensor:
    """Frozen-weight MMD^2 gradient at ``(z, w)``.

    ``grad_{z_i} J(z, w) = 2 w_i [ -sigma^{-2} sum_l w_l (z_i - z_l)
    k(z_i, z_l) - grad mu_rho(z_i) ]``: the kernel-mean block by autograd of
    the row-independent ``mu_fn``, the interaction block in closed form. No
    autograd through the weight solve is involved. Accepts ``(M, d)`` /
    ``(M,)`` or batched ``(R, M, d)`` / ``(R, M)`` (``torch.cdist`` inside
    :func:`gram_batched` handles both).
    """
    zz = z.detach().clone().requires_grad_(True)
    (dmu,) = torch.autograd.grad(mu_fn(zz).sum(), zz)
    with torch.no_grad():
        K = gram_batched(z.detach(), sigma)
        wK = K * w.unsqueeze(-2)  # (..., i, l) = w_l k(z_i, z_l)
        # sum_l w_l K_il (z_i - z_l), without the (..., M, M, d) temporary.
        interact = (
            z.detach() * wK.sum(-1, keepdim=True) - wK @ z.detach()
        ) / sigma**2
        return 2.0 * w.unsqueeze(-1) * (-dmu - interact)


def move_descend(
    z0: torch.Tensor,
    mu_fn: Callable[[torch.Tensor], torch.Tensor],
    c_rho: torch.Tensor,
    sigma: float,
    lr: float,
    n_iters: int,
    jitter: float = DEFAULT_JITTER,
) -> tuple[torch.Tensor, float]:
    """Designed nodes by direct MMD^2 descent from the i.i.d. nodes ``z0``.

    Minimizes ``MMD^2(z, w_c(z))`` over node positions ``z`` (Adam), with the
    constrained optimal weights re-solved at every ``z`` (autograd through the
    solve). Returns the BEST iterate ``(z*, MMD^2*)``, initialized at the
    reweight configuration ``(z0, w_c(z0))`` so that ``MMD^2(move) <=
    MMD^2(reweight)`` holds by construction whatever the optimizer does. The
    returned value is clamped at zero to absorb round-off (the exact MMD^2 is
    non-negative).

    The candidate pool contains, besides ``z0`` and the Adam iterates, the
    ray probes ``z0 - eta g`` for ``eta in {2**k / L(w) : k = 0..RAY_KMAX}``
    (``g`` the frozen-weight gradient at the reweight configuration, ``L(w)``
    the uniform smoothness constant ``2 ||w||_inf (2 ||w||_1 + 1) /
    sigma^2``), each scored the same way as every other candidate.

    The differentiable kernel mean ``mu_fn`` is the only target hook the move
    needs (``z -> E_rho[k(., z)]``, shape ``(M,)``); ``c_rho`` is a constant
    offset (it does not affect the gradient, only the reported value).

    Args:
        z0: i.i.d. seed nodes, shape ``(M, d)`` (the reweight start point).
        mu_fn: Differentiable exact kernel mean ``z -> E_rho[k(., z)]``,
            returning shape ``(M,)``.
        c_rho: Exact self-affinity scalar (a 0-dim tensor).
        sigma: SE bandwidth.
        lr: Adam learning rate.
        n_iters: Number of Adam steps.
        jitter: Tiny ridge for the weight solve (see ``constrained_weights``).

    Returns:
        ``(z_star, mmd2_star)`` -- the best-iterate node positions (detached
        clone, shape ``(M, d)``) and its scalar MMD^2 (a Python float,
        clamped at 0).
    """
    # Best iterate seeded at the reweight configuration: if no step improves
    # it, the move arm exactly equals the reweight arm.
    with torch.no_grad():
        mu0 = mu_fn(z0)
        w0 = constrained_weights(z0, mu0, sigma, jitter)
        best_val = float(mmd_sq(z0, w0, mu0, c_rho, sigma))
    best_z = z0.detach().clone()

    # Ray probes {z0 - eta g : eta in H(w0)}, scored like every other
    # candidate (ridged solve, un-ridged value); a minimum over a superset
    # can only improve.
    g = _frozen_weight_grad(z0, w0, mu_fn, sigma)
    L = 2.0 * float(w0.abs().max()) \
        * (2.0 * float(w0.abs().sum()) + 1.0) / sigma**2
    with torch.no_grad():
        for k in range(RAY_KMAX + 1):
            zp = z0.detach() - (2.0**k / L) * g
            mu_p = mu_fn(zp)
            w_p = constrained_weights(zp, mu_p, sigma, jitter)
            cur = float(mmd_sq(zp, w_p, mu_p, c_rho, sigma))
            if cur < best_val:
                best_val = cur
                best_z = zp.clone()

    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(int(n_iters)):
        opt.zero_grad()
        mu = mu_fn(z)
        w = constrained_weights(z, mu, sigma, jitter)
        loss = mmd_sq(z, w, mu, c_rho, sigma)
        loss.backward()
        opt.step()
        # Evaluate the post-step config and keep it only if strictly better,
        # so the reported best can never exceed the reweight seed value.
        with torch.no_grad():
            mu_d = mu_fn(z)
            w_d = constrained_weights(z, mu_d, sigma, jitter)
            cur = float(mmd_sq(z, w_d, mu_d, c_rho, sigma))
            if cur < best_val:
                best_val = cur
                best_z = z.detach().clone()
    return best_z, max(best_val, 0.0)


def move_descend_batched(
    z0: torch.Tensor,
    mu_fn: Callable[[torch.Tensor], torch.Tensor],
    c_rho: torch.Tensor,
    sigma: float,
    lr: float,
    n_iters: int,
    jitter: float = DEFAULT_JITTER,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched :func:`move_descend`: descend all ``R`` repeats at once.

    The vectorized-across-repeats analogue of :func:`move_descend`. Stacks
    the ``R`` i.i.d. node sets as ``z0`` of shape ``(R, M, d)`` and descends
    them jointly with a SINGLE Adam over the stacked parameter. This is
    NUMERICALLY IDENTICAL to looping :func:`move_descend` over the slices:

      * the kernel mean ``mu_fn`` is row-independent (per-node), so calling
        it on the flattened ``(R*M, d)`` reshaped back to ``(R, M)`` equals
        the per-slice calls (no reduction couples the rows);
      * the constrained-weight solve and the MMD^2 are batched per slice
        (:func:`constrained_weights_batched` / :func:`mmd_sq_batched`),
        each slice using its OWN Gram / solve;
      * the joint loss is ``sum_r MMD^2_r``; since the slices are
        independent, ``d(sum_r loss_r)/d z[r] = d loss_r / d z[r]``, so the
        per-slice gradient is unchanged;
      * Adam's moment updates are elementwise, so one Adam over the stacked
        parameter gives the SAME per-slice trajectory as ``R`` separate Adam
        instances with the same ``lr``;
      * the ray-probe candidates are per-slice too -- per-slice ``L``,
        per-slice frozen-weight gradient, per-slice solve and score -- so
        the augmented pool matches the per-slice calls.

    Each repeat keeps its OWN best-iterate, seeded at its reweight
    configuration ``(z0[r], w_c(z0[r]))`` and updated only on strict
    per-slice improvement, so ``move[r] <= reweight[r]`` holds for every
    repeat regardless of the optimizer path.

    Args:
        z0: Stacked i.i.d. seed nodes, shape ``(R, M, d)``.
        mu_fn: Differentiable exact, ROW-INDEPENDENT kernel mean
            ``z -> E_rho[k(., z)]`` (accepts any ``(..., d)`` and returns the
            leading shape).
        c_rho: Exact self-affinity scalar (a 0-dim tensor).
        sigma: SE bandwidth.
        lr: Adam learning rate.
        n_iters: Number of Adam steps.
        jitter: Tiny ridge for the weight solves.

    Returns:
        ``(z_star, mmd2_star)`` -- the per-repeat best-iterate node positions
        (detached, shape ``(R, M, d)``) and their scalar MMD^2 values (shape
        ``(R,)``, clamped at 0).
    """
    R, M, d = z0.shape

    def _mu_rows(zz: torch.Tensor) -> torch.Tensor:
        """Row-independent ``mu_fn`` over ``(R, M, d)`` -> ``(R, M)``."""
        return mu_fn(zz.reshape(R * M, d)).reshape(R, M)

    # Per-repeat best iterate, seeded at the reweight configuration: if no
    # step improves slice r, its move equals reweight.
    with torch.no_grad():
        mu0 = _mu_rows(z0)
        w0 = constrained_weights_batched(z0, mu0, sigma, jitter)
        best_val = torch.clamp(
            mmd_sq_batched(z0, w0, mu0, c_rho, sigma), min=0.0
        )  # (R,)
    best_z = z0.detach().clone()

    # Ray probes per repeat: each slice gets its own frozen-weight gradient,
    # its own L(w0[r]) step grid and its own scored probes.
    g = _frozen_weight_grad(z0, w0, _mu_rows, sigma)
    L = 2.0 * w0.abs().amax(-1) * (2.0 * w0.abs().sum(-1) + 1.0) / sigma**2
    with torch.no_grad():
        for k in range(RAY_KMAX + 1):
            zp = z0.detach() - (2.0**k / L).view(R, 1, 1) * g
            mu_p = _mu_rows(zp)
            w_p = constrained_weights_batched(zp, mu_p, sigma, jitter)
            cur = mmd_sq_batched(zp, w_p, mu_p, c_rho, sigma)  # (R,)
            improved = cur < best_val
            if improved.any():
                best_val = torch.where(improved, cur, best_val)
                best_z = torch.where(improved.view(R, 1, 1), zp, best_z)

    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(int(n_iters)):
        opt.zero_grad()
        mu = _mu_rows(z)
        w = constrained_weights_batched(z, mu, sigma, jitter)
        # Joint loss = sum_r MMD^2_r (slice-independent => per-slice grad).
        loss = mmd_sq_batched(z, w, mu, c_rho, sigma).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu_d = _mu_rows(z)
            w_d = constrained_weights_batched(z, mu_d, sigma, jitter)
            cur = mmd_sq_batched(z, w_d, mu_d, c_rho, sigma)  # (R,)
            improved = cur < best_val
            if improved.any():
                best_val = torch.where(improved, cur, best_val)
                best_z = torch.where(
                    improved.view(R, 1, 1), z.detach(), best_z
                )
    return best_z, torch.clamp(best_val, min=0.0)
