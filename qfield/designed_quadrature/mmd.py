"""The unit-sum squared-exponential MMD^2 of a weighted quadrature.

The objective every designed-quadrature arm is scored by: for a
unit-sum quadrature ``Q = sum_j w_j delta_{z_j}`` (``sum_j w_j = 1``) against
a target ``rho`` under the SE kernel ``k(a, b) = exp(-||a - b||^2 / (2
sigma^2))`` (NO normalizing prefactor),

    MMD^2(Q, rho) = w^T K w - 2 w^T mu + c_rho,
    K_{jl} = k(z_j, z_l),
    mu_j   = E_{x ~ rho}[k(x, z_j)],
    c_rho  = E_{x, x' ~ rho}[k(x, x')].

This is ``||mu_Q - mu_rho||_H^2``, the squared worst-case RKHS integration
error of the quadrature (Belhadji-Sharp-Marzouk, arXiv:2605.14142, eqs. 4-5).
Both ``mu`` and ``c_rho`` carry the kernel-to-Gaussian factor ``Z_sigma =
(2 pi sigma^2)^{d/2}`` (because ``mu_j = int exp(-||x - z_j||^2 / 2 sigma^2)
rho(x) dx`` pulls a ``Z_sigma`` out of the SE-as-Gaussian rewrite); ``K`` does
not. The convention is consistent with the reused closed forms in
:mod:`qfield.metrics_mmd` and :meth:`AnisotropicGMM.mmd_sq`.

:func:`mmd_sq` is the literal, transparent objective: it does NOT renormalize
the weights (the caller guarantees ``sum(w) = 1`` -- the constrained weights
and the equal weights both satisfy this exactly) and does NOT clamp at zero,
so it is a clean differentiable target for the move arm's autograd descent.
Reporting code clamps the tiny negative round-off (the SE Gram is PSD, so the
exact value is non-negative). :func:`cross_check_mmd` compares the explicit
form against the independent closed forms elsewhere in the package.

Reference:
  - Belhadji, Sharp, Marzouk, "To discretize continually: Mean shift
    interacting particle systems for Bayesian inference", arXiv:2605.14142
    (squared-exponential kernel and analytic MMD, eqs. 4-5).
  - Gretton et al., "A kernel two-sample test", JMLR 2012 (the MMD).
"""

from __future__ import annotations

from typing import Callable

import torch

from qfield.kernels import se_kernel


def gram(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """SE Gram ``K_{jl} = exp(-||z_j - z_l||^2 / (2 sigma^2))``.

    A thin wrapper over :func:`qfield.kernels.se_kernel` so the rest of the
    designed-quadrature code talks about ``K`` directly. The diagonal is
    one (unit-height kernel, no prefactor).

    Args:
        z: Node positions, shape ``(m, d)``.
        sigma: SE bandwidth ``sigma > 0``.

    Returns:
        Symmetric Gram matrix, shape ``(m, m)``, with unit diagonal, in
        ``z``'s dtype on ``z``'s device.
    """
    return se_kernel(z, z, sigma).to(z.dtype)


def gram_batched(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """Batched SE Gram ``K[r]_{jl} = exp(-||z[r]_j - z[r]_l||^2 / (2 sigma^2))``.

    The vectorized-across-repeats analogue of :func:`gram`: for stacked node
    sets ``z`` of shape ``(R, m, d)`` returns the ``R`` Grams stacked as
    ``(R, m, m)``. Uses ``torch.cdist`` (which batches over leading dims)
    instead of :func:`qfield.kernels.se_kernel`, whose reshape
    flattens the batch -- the per-slice values are bit-identical to
    :func:`gram` because both compute ``exp(-||.||^2 / (2 sigma^2))`` over
    the same squared distances.

    Args:
        z: Stacked node positions, shape ``(R, m, d)``.
        sigma: SE bandwidth ``sigma > 0``.

    Returns:
        Stacked symmetric Grams, shape ``(R, m, m)``, unit diagonal, in
        ``z``'s dtype on ``z``'s device.
    """
    sq = torch.cdist(z, z) ** 2  # (R, m, m)
    return torch.exp(-sq / (2.0 * sigma**2))


def mmd_sq_batched(
    z: torch.Tensor,
    w: torch.Tensor,
    mu: torch.Tensor,
    c_rho: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Batched unit-sum SE-MMD^2 ``w^T K w - 2 w^T mu + c_rho`` over repeats.

    The vectorized-across-repeats analogue of :func:`mmd_sq`: scores ``R``
    quadratures at once. Per slice ``r`` the value is
    ``w[r] @ K[r] @ w[r] - 2 (w[r] @ mu[r]) + c_rho`` -- algebraically and
    numerically identical to :func:`mmd_sq` applied to slice ``r`` (the same
    contractions, just batched). The caller guarantees each ``w[r]`` sums to
    one; nothing is renormalized or clamped (a clean differentiable target).

    Args:
        z: Stacked node positions, shape ``(R, m, d)``.
        w: Stacked unit-sum weights, shape ``(R, m)``.
        mu: Stacked exact kernel means at the nodes, shape ``(R, m)``.
        c_rho: Exact self-affinity scalar (a 0-dim tensor; shared across
            repeats for a fixed target).
        sigma: SE bandwidth.

    Returns:
        Per-repeat ``MMD^2``, shape ``(R,)`` (un-clamped).
    """
    K = gram_batched(z, sigma)  # (R, m, m)
    quad = torch.einsum("rj,rjl,rl->r", w, K, w)  # w^T K w
    cross = (w * mu).sum(dim=-1)  # w^T mu
    return quad - 2.0 * cross + c_rho


def mmd_sq(
    z: torch.Tensor,
    w: torch.Tensor,
    mu: torch.Tensor,
    c_rho: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """The literal unit-sum SE-MMD^2 ``w^T K w - 2 w^T mu + c_rho``.

    The transparent, differentiable objective for every arm. The caller
    guarantees ``sum(w) = 1`` (the constrained weights and the equal
    weights both do, exactly); this function therefore does NOT
    renormalize and does NOT clamp, so the move arm can autograd straight
    through it. ``mu`` is the exact kernel mean ``E_rho[k(., z)]`` evaluated
    at the nodes; ``c_rho`` the exact self-affinity ``E_{x, x'}[k(x, x')]``.

    Args:
        z: Node positions, shape ``(m, d)``.
        w: Unit-sum node weights, shape ``(m,)`` (NOT renormalized here).
        mu: Exact kernel mean at the nodes ``mu_j = E_rho[k(x, z_j)]``,
            shape ``(m,)``.
        c_rho: Exact self-affinity scalar ``c_rho``, a 0-dim tensor.
        sigma: SE bandwidth.

    Returns:
        Scalar ``MMD^2`` as a 0-dim tensor (un-clamped; may be a tiny
        negative from round-off, which reporting code clamps).
    """
    K = gram(z, sigma)
    return w @ K @ w - 2.0 * (w @ mu) + c_rho


def cross_check_mmd(
    z: torch.Tensor,
    w: torch.Tensor,
    mu: torch.Tensor,
    c_rho: torch.Tensor,
    sigma: float,
    reference_fn: Callable[[torch.Tensor, torch.Tensor], float],
) -> float:
    """Relative gap between the explicit MMD^2 and an independent closed form.

    Scores a random unit-sum weighting two ways -- the explicit
    :func:`mmd_sq` (``w^T K w - 2 w^T mu + c_rho``) and an independent closed
    form (``reference_fn``, e.g.
    :func:`qfield.metrics_mmd.mmd_sq_gaussian` or
    :meth:`AnisotropicGMM.mmd_sq`) -- and returns the relative difference.
    The arms layer asserts this is below ``~1e-8`` before any run, so a bug in
    ``mu`` / ``c_rho`` / the convention is caught up front.

    Args:
        z: Node positions, shape ``(m, d)``.
        w: Unit-sum weights, shape ``(m,)``.
        mu: Exact kernel mean at the nodes, shape ``(m,)``.
        c_rho: Exact self-affinity scalar.
        sigma: SE bandwidth.
        reference_fn: ``(z, w) -> float`` the codebase closed-form MMD^2 for
            the SAME target (the independent implementation to cross-check).

    Returns:
        Relative gap ``|explicit - reference| / (|reference| + 1e-30)``.
    """
    mine = float(mmd_sq(z, w, mu, c_rho, sigma))
    theirs = float(reference_fn(z, w))
    return abs(mine - theirs) / (abs(theirs) + 1e-30)
