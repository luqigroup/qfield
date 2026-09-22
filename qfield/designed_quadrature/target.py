"""Closed-form targets ``rho = pi`` for the designed quadrature.

The reference ``rho`` is a known density that is i.i.d.-sampleable and has an
exact kernel mean ``mu`` and self-affinity ``c_rho``. There is no neural
network here.

A target is consumed by the arms layer through a narrow structural interface
(:class:`QuadratureTarget`, a ``typing.Protocol``): the arms only ever touch

  * ``mu_fn(z) -> (M,)``     -- the exact, differentiable kernel mean
                               ``mu_j = E_{x ~ pi}[k(x, z_j)]``,
  * ``c_rho``                -- the exact self-affinity ``E_{x, x'}[k(x, x')]``
                               (a 0-dim tensor; constant in the quadrature),
  * ``sampler(M, gen) -> (M, d)`` -- generator-threaded i.i.d. samples from
                               ``pi`` (the floor's nodes),
  * ``e_x1sq``               -- ``E_pi[x_1^2]`` for the functional spot check,
  * ``cross_check_reference(z, w) -> float`` -- an independent closed-form
                               MMD^2 for this target, for the metric
                               cross-check.

so the Gaussian and GMM targets are interchangeable and a flow or physics
target plugs in by satisfying the same protocol.

The two builders use the closed forms:

  Gaussian  N(m, S):
    mu_j   = |I + S/sigma^2|^{-1/2} exp(-0.5 (z_j - m)^T (S + sigma^2 I)^{-1}
             (z_j - m))   = Z_sigma N(z_j; m, S + sigma^2 I)
             -- via :func:`qfield.metrics_mmd._gaussian_vs_point_mean`,
    c_rho  = |I + 2 S/sigma^2|^{-1/2}.
  GMM  sum_k pi_k N(mu_k, S_k):
    mu_j   = Z_sigma sum_k pi_k N(z_j; mu_k, S_k + sigma^2 I)  -- via
             :meth:`AnisotropicGMM.v0`,
    c_rho  = Z_sigma sum_{kl} pi_k pi_l N(mu_k; mu_l, S_k + S_l + sigma^2 I)
             -- via :meth:`AnisotropicGMM.c_pi`.

Reference:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (SE closed forms, eqs. 4-5,
    34-42).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from qfield.dataset.gmm import AnisotropicGMM
from qfield.dataset.linear_gaussian import LinearGaussian
from qfield.designed_quadrature.mmd import mmd_sq
from qfield.metrics_mmd import (
    _gaussian_vs_point_mean,
    mmd_sq_gaussian,
)

DTYPE = torch.float64


class QuadratureTarget(Protocol):
    """Structural interface a target exposes to the arms layer.

    Any object with these attributes is a valid target; the arms depend on
    this protocol, not on the concrete Gaussian / GMM classes, so targets are
    interchangeable.
    """

    name: str
    mu_fn: Callable[[torch.Tensor], torch.Tensor]
    c_rho: torch.Tensor
    sampler: Callable[[int, torch.Generator], torch.Tensor]
    e_x1sq: float
    cross_check_reference: Callable[[torch.Tensor, torch.Tensor], float]
    meta: dict


@dataclass
class _Target:
    """Concrete :class:`QuadratureTarget` record (a plain data holder)."""

    name: str
    mu_fn: Callable[[torch.Tensor], torch.Tensor]
    c_rho: torch.Tensor
    sampler: Callable[[int, torch.Generator], torch.Tensor]
    e_x1sq: float
    cross_check_reference: Callable[[torch.Tensor, torch.Tensor], float]
    meta: dict
    # Eval estimand where it differs from the training one: a kernel mean on
    # the same bank as ``c_rho``, so scoring with the pair (mu_fn_eval,
    # c_rho) carries no cross-bank additive offset. ``None`` for closed-form
    # targets, where ``mu_fn`` / ``c_rho`` already share one exact estimand;
    # consumers fall back to ``mu_fn`` via
    # ``getattr(target, "mu_fn_eval", None) or target.mu_fn``. It is
    # optional, so it is not part of the :class:`QuadratureTarget` protocol.
    mu_fn_eval: Callable[[torch.Tensor], torch.Tensor] | None = None



def _chunked_kernel_mean(
    z: torch.Tensor, atoms: torch.Tensor, inv_2sig2: float,
    row_chunk: int = 2048,
) -> torch.Tensor:
    """``mean_l k(z_i, atom_l)`` with the query rows chunked.

    One flat ``cdist(z, atoms)`` allocates ``rows * L * 8`` bytes, which runs
    to tens of gigabytes for a large query set against a large bank.
    Row-chunking caps the transient at ``row_chunk * L * 8`` with identical
    values; autograd composes through ``cat``.
    """
    def _one(zc: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(torch.cdist(zc, atoms) ** 2) * inv_2sig2).mean(-1)

    if z.shape[0] <= row_chunk:
        return _one(z)
    # Gradient checkpointing on the chunk loop when autograd is live: without
    # it every chunk's exp/sq activations are retained for backward, so the
    # peak is the full rows x L graph again. Checkpointing recomputes each
    # chunk's forward during backward: one extra forward, a flat low peak.
    use_ckpt = torch.is_grad_enabled() and z.requires_grad
    outs = []
    for i in range(0, z.shape[0], row_chunk):
        zc = z[i : i + row_chunk]
        if use_ckpt:
            outs.append(torch.utils.checkpoint.checkpoint(
                _one, zc, use_reentrant=False))
        else:
            outs.append(_one(zc))
    return torch.cat(outs)


def _cncv_gaussian_posterior(
    d: int, sigma_obs: float, seed: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """A higher-dimensional Gaussian posterior ``N(mu_post, Sigma_post)``.

    Prior ``N(0, Sigma_pr)`` with ``Sigma_pr = Q diag(linspace(1, 2, d)) Q^T``
    (``Q`` from the QR of a ``manual_seed(42)`` random matrix),
    identity-forward likelihood ``p(y | x) = N(x, sigma_obs^2 I)``, posterior
    ``Sigma_post = (Sigma_pr^{-1} + sigma_obs^{-2} I)^{-1}``. One
    representative ``y*`` is sampled from the joint; the SE MMD is
    translation-invariant and ``Sigma_post`` is independent of ``y*``, so the
    choice of ``y*`` does not affect any MMD statistic.

    Returns ``(mu_post, Sigma_post, prior_meta)`` (all float64).
    """
    # Prior covariance Q diag(linspace(1,2,d)) Q^T, built without touching the
    # global RNG.
    diag_values = torch.linspace(1.0, 2.0, d, dtype=dtype)
    D = torch.diag(diag_values)
    if d > 1:
        gen42 = torch.Generator(device="cpu")
        gen42.manual_seed(42)
        A = torch.randn(d, d, generator=gen42, dtype=dtype)
        Q, _ = torch.linalg.qr(A)
        sigma_pr = Q @ D @ Q.T
    else:
        sigma_pr = D
    sigma_pr = 0.5 * (sigma_pr + sigma_pr.T)  # symmetrize round-off
    mu_pr = torch.zeros(d, dtype=dtype)

    eye = torch.eye(d, dtype=dtype)
    sigma_pr_inv = torch.linalg.inv(sigma_pr)
    sigma_post = torch.linalg.inv(sigma_pr_inv + (1.0 / sigma_obs**2) * eye)
    sigma_post = 0.5 * (sigma_post + sigma_post.T)

    # One representative y* from the joint y = x + eps, x ~ prior (local RNG).
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    chol_pr = torch.linalg.cholesky(sigma_pr)
    x_true = mu_pr + chol_pr @ torch.randn(d, generator=gen, dtype=dtype)
    y_star = x_true + sigma_obs * torch.randn(d, generator=gen, dtype=dtype)
    mu_post = sigma_post @ (
        sigma_pr_inv @ mu_pr + (1.0 / sigma_obs**2) * y_star
    )
    meta = {
        "kind": "cncv_gaussian_posterior",
        "sigma_obs": sigma_obs,
        "prior_cov_diag_range": [1.0, 2.0],
    }
    return mu_post.to(dtype), sigma_post.to(dtype), meta


def gaussian_target(
    sigma: float,
    d: int = 2,
    seed: int = 20260612,
    sigma_obs: float = 0.3,
    dtype: torch.dtype = DTYPE,
) -> _Target:
    """A Gaussian posterior ``rho = N(mu_post, Sigma_post)`` (any ``d``).

    For the default ``d = 2`` this is the linear-Gaussian posterior with prior
    ``N(0, [[1.5, 0.2], [0.2, 1.0]])``, forward ``A = [[1.0, 0.3], [-0.1,
    0.9]]`` and observation noise ``diag(0.15, 0.2)``. For any other ``d`` it
    is the higher-dimensional Gaussian posterior of
    :func:`_cncv_gaussian_posterior` (prior ``Q diag(linspace(1, 2, d)) Q^T``,
    identity forward, observation noise ``sigma_obs``). Either way the SE MMD
    is translation-invariant and ``Sigma_post`` is independent of ``y*``, so
    the representative ``y*`` is immaterial to every MMD statistic; ``mu`` /
    ``c_rho`` use the Gaussian closed forms.

    Args:
        sigma: SE bandwidth.
        d: Ambient dimension. ``2`` recovers the linear-Gaussian posterior;
            otherwise the higher-dimensional Gaussian at that ``d``.
        seed: Seed for the single representative ``y*`` (immaterial to any
            MMD; fixed only for reproducibility of the logged ``y*``).
        sigma_obs: Observation-noise std for the ``d != 2`` posterior.
        dtype: Floating dtype (float64 for the closed-form linear algebra).

    Returns:
        A :class:`_Target` for the Gaussian posterior at dimension ``d``.
    """
    if d == 2:
        prior_mean = torch.zeros(2, dtype=dtype)
        prior_cov = torch.tensor([[1.5, 0.2], [0.2, 1.0]], dtype=dtype)
        forward = torch.tensor([[1.0, 0.3], [-0.1, 0.9]], dtype=dtype)
        obs_cov = torch.tensor([[0.15, 0.0], [0.0, 0.2]], dtype=dtype)
        problem = LinearGaussian(
            prior_mean, prior_cov, forward, obs_cov, dtype=dtype
        )

        # One representative held-out y* (no MMD statistic depends on it).
        # ``sample_joint`` uses the global RNG, so we seed it locally and
        # restore the prior state to avoid a global side effect.
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(int(seed))
            _, y_all = problem.sample_joint(8000)
        finally:
            torch.random.set_rng_state(rng_state)
        y_star = y_all[7500].to(dtype)

        mu_post, sigma_post = problem.posterior(y_star)
        mu_post = mu_post.to(dtype)
        sigma_post = sigma_post.to(dtype)
        meta = {
            "kind": "linear_gaussian_posterior",
            "d": 2,
            "mu_post": mu_post.tolist(),
            "Sigma_post": sigma_post.tolist(),
            "y_star": y_star.tolist(),
        }
    else:
        mu_post, sigma_post, meta = _cncv_gaussian_posterior(
            d, sigma_obs, seed, dtype
        )
        meta["d"] = d

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        return _gaussian_vs_point_mean(z.to(dtype), mu_post, sigma_post, sigma)

    eye = torch.eye(d, dtype=dtype)
    _, logdet = torch.linalg.slogdet(eye + 2.0 * sigma_post / sigma**2)
    c_rho = torch.exp(-0.5 * logdet)  # |I + 2 Sigma/sigma^2|^{-1/2}

    chol = torch.linalg.cholesky(sigma_post)

    def sampler(M: int, gen: torch.Generator) -> torch.Tensor:
        z = torch.randn(int(M), d, generator=gen, dtype=dtype)
        return mu_post.unsqueeze(0) + z @ chol.T

    # E_pi[x1^2] = mu_post[0]^2 + Sigma_post[0, 0] (the functional check).
    e_x1sq = float(mu_post[0] ** 2 + sigma_post[0, 0])

    def cross_check_reference(z: torch.Tensor, w: torch.Tensor) -> float:
        return float(mmd_sq_gaussian(z, w, mu_post, sigma_post, sigma))

    return _Target(
        name="gaussian",
        mu_fn=mu_fn,
        c_rho=c_rho,
        sampler=sampler,
        e_x1sq=e_x1sq,
        cross_check_reference=cross_check_reference,
        meta=meta,
    )


def latent_gaussian_target(
    mu: torch.Tensor,
    cov: torch.Tensor,
    sigma: float,
    name: str = "latent_gaussian",
    meta: dict | None = None,
    dtype: torch.dtype = DTYPE,
) -> _Target:
    """A Gaussian target ``rho = N(mu, cov)`` from an EXPLICIT mean / covariance.

    The generic builder behind :func:`gaussian_target` (which constructs its
    own ``(mu, cov)``): given any Gaussian mean ``mu in R^d`` and SPD
    covariance ``cov in R^{d x d}``, it returns the :class:`_Target` that
    exposes the exact SE-kernel closed forms used everywhere else --
    ``mu_fn(z) = E_{x~N(mu,cov)}[k(x, z)]`` via
    :func:`qfield.metrics_mmd._gaussian_vs_point_mean`,
    ``c_rho = |I + 2 cov / sigma^2|^{-1/2}``, an i.i.d. sampler from
    ``N(mu, cov)``, and the independent closed-form cross-check
    :func:`qfield.metrics_mmd.mmd_sq_gaussian`. This lets a Gaussian whose
    ``(mu, cov)`` is computed elsewhere (e.g. a tomography posterior projected
    into the informed subspace, ``rho_z = N(W^T mu_post, W^T Sigma_post W)``)
    plug into the same arms and amortizer as the toy Gaussians.

    Args:
        mu: Gaussian mean, shape ``(d,)``.
        cov: Gaussian covariance, shape ``(d, d)``, SPD.
        sigma: SE bandwidth.
        name: Target name recorded on the :class:`_Target`.
        meta: Optional metadata dict to store on the target (e.g. the
            originating problem's geometry); ``None`` records the bare kind.
        dtype: Floating dtype (float64 for the closed-form linear algebra).

    Returns:
        A :class:`_Target` for the Gaussian ``N(mu, cov)`` at its dimension.
    """
    mu = mu.reshape(-1).to(dtype)
    cov = 0.5 * (cov.to(dtype) + cov.to(dtype).T)
    d = mu.shape[0]

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        return _gaussian_vs_point_mean(z.to(dtype), mu, cov, sigma)

    eye = torch.eye(d, dtype=dtype)
    _, logdet = torch.linalg.slogdet(eye + 2.0 * cov / sigma**2)
    c_rho = torch.exp(-0.5 * logdet)  # |I + 2 cov/sigma^2|^{-1/2}

    chol = torch.linalg.cholesky(cov)

    def sampler(M: int, gen: torch.Generator) -> torch.Tensor:
        z = torch.randn(int(M), d, generator=gen, dtype=dtype)
        return mu.unsqueeze(0) + z @ chol.T

    e_x1sq = float(mu[0] ** 2 + cov[0, 0])

    def cross_check_reference(z: torch.Tensor, w: torch.Tensor) -> float:
        return float(mmd_sq_gaussian(z, w, mu, cov, sigma))

    full_meta = {"kind": "latent_gaussian", "d": d}
    if meta is not None:
        full_meta.update(meta)
    return _Target(
        name=name,
        mu_fn=mu_fn,
        c_rho=c_rho,
        sampler=sampler,
        e_x1sq=e_x1sq,
        cross_check_reference=cross_check_reference,
        meta=full_meta,
    )


def whitened_gaussian_target(
    mu: torch.Tensor,
    cov: torch.Tensor,
    sigma: float,
    name: str = "whitened_gaussian",
    meta: dict | None = None,
    dtype: torch.dtype = DTYPE,
) -> _Target:
    """The Mahalanobis (posterior-whitened) SE-kernel target for ``N(mu, cov)``.

    Under the isotropic SE kernel with a median-heuristic bandwidth, in high
    ``d`` the pairwise squared distances between i.i.d. posterior samples
    concentrate (relative spread ``O(1 / sqrt(d))``), so every off-diagonal
    Gram entry clusters at the same value and the node Gram degenerates to
    ``K -> I``. With ``K ~ I`` the move lever has no geometry to exploit and
    the designed set degenerates. The remedy is to replace the isotropic SE
    kernel with the posterior-whitened (Mahalanobis) SE kernel

        k_M(x, z) = exp(-1/2 (x - z)^T Sigma^{-1} (x - z) / sigma^2),

    which is EXACTLY the isotropic SE kernel (bandwidth ``sigma``) in the
    WHITENED coordinate ``u = R x`` where ``Sigma = (R^T R)^{-1}`` (Cholesky
    ``Sigma = L L^T``, ``R = L^{-1}``): then ``(x - z)^T Sigma^{-1} (x - z) =
    ||u_x - u_z||^2`` and the target ``N(mu, Sigma)`` maps to ``N(R mu, I)``.
    The whitening makes the posterior isotropic in the kernel metric in every
    ``d``, so a ``d``-aware bandwidth (e.g. ``sigma = sqrt(d)``, which holds
    the typical off-diagonal Gram entry at ``exp(-1)`` since the whitened
    median squared distance is ``~2 d``) keeps the Gram well-conditioned and
    restores the move lever's high-``d`` gain.

    Because the Mahalanobis kernel is the isotropic kernel in ``u``-coords,
    this builder whitens the Gaussian (``mu_u = R mu``, ``Sigma_u = I``) and
    delegates to :func:`latent_gaussian_target` at bandwidth ``sigma``, which
    supplies the exact SE-kernel ``mu_fn`` / ``c_rho`` / cross-check in
    whitened space. The amortizer, move, weight and MMD code is unchanged and
    operates on the whitened nodes ``u``; the back-map ``x = L u`` (the
    inverse whitening, ``meta["lift"]``) recovers ambient nodes for
    reconstruction. Reported MMD values are in the Mahalanobis metric, so
    ratios against the floor are directly comparable to the isotropic runs
    (both numerator and denominator use the same kernel).

    Args:
        mu: Gaussian mean, shape ``(d,)`` (the ambient posterior mean).
        cov: Gaussian covariance ``Sigma``, shape ``(d, d)``, SPD (the ambient
            posterior covariance; the whitening metric is ``Sigma^{-1}``).
        sigma: SE bandwidth in whitened coordinates (the high-``d`` rule is
            ``sigma = sqrt(d)``; the caller picks it).
        name: Target name recorded on the :class:`_Target`.
        meta: Optional metadata dict to merge onto the target.
        dtype: Floating dtype (float64 for the closed-form linear algebra).

    Returns:
        A :class:`_Target` for the whitened Gaussian ``N(R mu, I)`` at its
        dimension, with ``meta["lift"] = L`` (the ambient back-map ``x = L u``)
        and ``meta["mu_post"]`` recorded for reconstruction.
    """
    mu = mu.reshape(-1).to(dtype)
    cov = 0.5 * (cov.to(dtype) + cov.to(dtype).T)
    d = mu.shape[0]

    # Whitening: Sigma = L L^T, R = L^{-1}; u = R x maps N(mu, Sigma) -> N(R mu, I).
    chol = torch.linalg.cholesky(cov)  # L (lower)
    r = torch.linalg.inv(chol)  # R = L^{-1}
    mu_u = r @ mu  # whitened mean
    eye = torch.eye(d, dtype=dtype)

    full_meta = {
        "kind": "whitened_gaussian",
        "d": d,
        "metric": "mahalanobis_sigma_inv",
        "sigma_whitened": float(sigma),
        # The ambient back-map x = L u (the inverse whitening) for recon, and
        # the whitening map R = L^{-1} (u = R x), recorded for downstream lift.
        "lift": chol.tolist(),
        "whiten": r.tolist(),
        "mu_post": mu.tolist(),
    }
    if meta is not None:
        full_meta.update(meta)

    # Delegate to the closed-form Gaussian target in u-coords: N(mu_u, I) with
    # bandwidth sigma.
    return latent_gaussian_target(
        mu_u, eye, sigma, name=name, meta=full_meta, dtype=dtype
    )


def _build_gmm(d: int, dtype: torch.dtype) -> AnisotropicGMM:
    """The 5-mode anisotropic GMM at dimension ``d``.

    At ``d = 2`` a fixed rotated 5-mode mixture (modes scattered out to
    ``|mu| ~ 17``); at ``d >= 5`` the axis-aligned 5-component mixture
    ``AnisotropicGMM.axis_aligned(d, alpha=5.5, cov_scale=0.5,
    anisotropy_factor=3.0)`` (each mode on its own coordinate axis, elongated
    along it). ``axis_aligned`` requires ``d >= 5``; ``d in {3, 4}`` is
    rejected.
    """
    if d == 2:
        means = 2.0 * torch.tensor(
            [[6.2, -6.0], [-4.0, 5.0], [7.0, 3.0], [-6.5, -4.5], [1.0, 7.0]],
            dtype=dtype,
        )
        covs = torch.tensor(
            [
                [[1.5, 0.1], [0.1, 0.5]],
                [[2.0, -0.6], [-0.6, 0.5]],
                [[0.7, 0.4], [0.4, 1.2]],
                [[1.3, -0.5], [-0.5, 0.9]],
                [[0.6, 0.35], [0.35, 1.6]],
            ],
            dtype=dtype,
        )
        wts = torch.full((5,), 0.2, dtype=dtype)
        return AnisotropicGMM.from_components(means, covs, wts, dtype=dtype)
    if d >= 5:
        return AnisotropicGMM.axis_aligned(
            d, alpha=5.5, cov_scale=0.5, anisotropy_factor=3.0, dtype=dtype
        )
    raise ValueError(
        f"gmm_target supports d == 2 or d >= 5 (the mc_floor battery); got "
        f"d={d}."
    )


def gmm_target(sigma: float, d: int = 2, dtype: torch.dtype = DTYPE) -> _Target:
    """The 5-mode anisotropic GMM target at dimension ``d``.

    At the default ``d = 2`` this is five rotated anisotropic modes scattered
    out to ``|mu| ~ 17`` with uniform weights. At ``d >= 5`` it is the
    axis-aligned higher-dimensional mixture (see :func:`_build_gmm`). ``mu`` /
    ``c_rho`` use :meth:`AnisotropicGMM.v0` / ``.c_pi``; the sampler
    reproduces the mixture's law generator-threaded (pick a component with
    ``multinomial``, then sample the selected Gaussian).

    Args:
        sigma: SE bandwidth.
        d: Ambient dimension (``2`` or ``>= 5``).
        dtype: Floating dtype (float64 for the closed-form linear algebra).

    Returns:
        A :class:`_Target` for the GMM at dimension ``d``.
    """
    gmm = _build_gmm(d, dtype)

    c_rho = gmm.c_pi(sigma).to(dtype)

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        return gmm.v0(z.to(dtype), sigma).to(dtype)

    # Generator-threaded mixture sampler (reproducible per repeat): pick a
    # component, then sample the selected Gaussian.
    w = gmm.weights.to(dtype)
    mu = gmm.means.to(dtype)
    chol = torch.linalg.cholesky(gmm.covariances.to(dtype))

    def sampler(M: int, gen: torch.Generator) -> torch.Tensor:
        M = int(M)
        comp = torch.multinomial(w, M, replacement=True, generator=gen)
        zz = torch.randn(M, d, generator=gen, dtype=dtype)
        return mu[comp] + torch.einsum("nij,nj->ni", chol[comp], zz)

    # E_pi[x1^2] = sum_k pi_k (mu_k[0]^2 + S_k[0, 0]).
    e_x1sq = float((w * (mu[:, 0] ** 2 + gmm.covariances[:, 0, 0])).sum())

    def cross_check_reference(z: torch.Tensor, w_in: torch.Tensor) -> float:
        return float(gmm.mmd_sq(z, w_in, sigma))

    meta = {
        "kind": "anisotropic_gmm",
        "d": d,
        "n_modes": int(gmm.K),
        "means": mu.tolist(),
    }
    return _Target(
        name="gmm",
        mu_fn=mu_fn,
        c_rho=c_rho,
        sampler=sampler,
        e_x1sq=e_x1sq,
        cross_check_reference=cross_check_reference,
        meta=meta,
    )


def gmm_cncv_target(
    sigma: float, d: int, dtype: torch.dtype = DTYPE
) -> _Target:
    """A 2-component anisotropic GMM target ``rho`` at dimension ``d``.

    Component variances ``[0.5, 2.0]`` with separation ``8 * max(std)`` along
    axis 0: component 0 is ``N(0, 0.5 I_d)``, component 1 is
    ``N(sep e_0, 2.0 I_d)`` with ``sep = 8 sqrt(max_variance) = 8 sqrt(2.0)``,
    uniform weights ``[0.5, 0.5]``. The two modes are well separated along
    axis 0. ``mu`` / ``c_rho`` use :meth:`AnisotropicGMM.v0` / ``.c_pi``; the
    sampler reproduces the mixture law generator-threaded.

    Args:
        sigma: SE bandwidth.
        d: Ambient dimension (any ``d >= 1``).
        dtype: Floating dtype (float64 for the closed-form linear algebra).

    Returns:
        A :class:`_Target` for the 2-component GMM at dimension ``d``.
    """
    variances = [0.5, 2.0]
    max_variance = max(variances)
    sep = 8.0 * math.sqrt(max_variance)

    means = torch.zeros(2, d, dtype=dtype)
    means[1, 0] = sep
    eye = torch.eye(d, dtype=dtype)
    covs = torch.stack(
        [variances[0] * eye, variances[1] * eye], dim=0
    )
    wts = torch.tensor([0.5, 0.5], dtype=dtype)
    gmm = AnisotropicGMM.from_components(means, covs, wts, dtype=dtype)

    c_rho = gmm.c_pi(sigma).to(dtype)

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        return gmm.v0(z.to(dtype), sigma).to(dtype)

    w = gmm.weights.to(dtype)
    mu = gmm.means.to(dtype)
    chol = torch.linalg.cholesky(gmm.covariances.to(dtype))

    def sampler(M: int, gen: torch.Generator) -> torch.Tensor:
        M = int(M)
        comp = torch.multinomial(w, M, replacement=True, generator=gen)
        zz = torch.randn(M, d, generator=gen, dtype=dtype)
        return mu[comp] + torch.einsum("nij,nj->ni", chol[comp], zz)

    e_x1sq = float((w * (mu[:, 0] ** 2 + gmm.covariances[:, 0, 0])).sum())

    def cross_check_reference(z: torch.Tensor, w_in: torch.Tensor) -> float:
        return float(gmm.mmd_sq(z, w_in, sigma))

    meta = {
        "kind": "cncv_gmm_2comp",
        "d": d,
        "n_modes": 2,
        "variances": variances,
        "separation": sep,
        "means": mu.tolist(),
    }
    return _Target(
        name="gmm_cncv",
        mu_fn=mu_fn,
        c_rho=c_rho,
        sampler=sampler,
        e_x1sq=e_x1sq,
        cross_check_reference=cross_check_reference,
        meta=meta,
    )


def whitened_gmm_cncv_target(
    sigma: float, d: int, dtype: torch.dtype = DTYPE
) -> _Target:
    """The 2-component GMM under the pooled-whitened (Mahalanobis) SE kernel.

    The GMM analogue of :func:`whitened_gaussian_target`. The Mahalanobis SE
    kernel ``k_M(x, z) = exp(-1/2 (x - z)^T M (x - z) / sigma^2)`` with
    ``M = Sigma_pool^{-1}`` (the mixture's pooled inverse covariance) is the
    isotropic SE kernel in the whitened coordinate ``u = R x``
    (``Sigma_pool = (R^T R)^{-1}``). Because the kernel is isotropic in
    ``u``-coords and the GMM maps to a GMM there (component ``k`` becomes
    ``N(R mu_k, R Sigma_k R^T)``), the whole target is still a GMM and is not
    replaced by a Gaussian, so the :meth:`AnisotropicGMM.v0` / ``.c_pi``
    closed forms apply unchanged in whitened space. This keeps the integration
    target the true bimodal mixture while the kernel metric removes the pooled
    anisotropy and separation scale, so a ``d``-aware whitened bandwidth holds
    the Gram O(1) and restores the move lever's high-``d`` gain (the isotropic
    median heuristic walls by ``d = 32-64`` here).

    Pooled moments (uniform weights): ``mu_pool = sum_k pi_k mu_k``,
    ``Sigma_pool = sum_k pi_k (Sigma_k + (mu_k - mu_pool)
    (mu_k - mu_pool)^T)``. The whitened mixture has means ``R mu_k`` and covs
    ``R Sigma_k R^T``. Sampling is in whitened space directly.

    Args:
        sigma: SE bandwidth in whitened coordinates (the high-``d`` rule is
            ``sigma = sqrt(d)``; the caller picks it).
        d: Ambient dimension (any ``d >= 1``).
        dtype: Floating dtype (float64 for the closed-form linear algebra).

    Returns:
        A :class:`_Target` for the whitened GMM at dimension ``d``, with
        ``meta["lift"] = L`` (ambient back-map ``x = L u``) for reconstruction.
    """
    variances = [0.5, 2.0]
    max_variance = max(variances)
    sep = 8.0 * math.sqrt(max_variance)

    means = torch.zeros(2, d, dtype=dtype)
    means[1, 0] = sep
    eye = torch.eye(d, dtype=dtype)
    covs = torch.stack([variances[0] * eye, variances[1] * eye], dim=0)
    wts = torch.tensor([0.5, 0.5], dtype=dtype)

    # Pooled mixture covariance (between + within); whiten by its inverse.
    mu_pool = (wts.unsqueeze(1) * means).sum(0)  # (d,)
    cov_pool = torch.zeros(d, d, dtype=dtype)
    for k in range(2):
        diff = (means[k] - mu_pool).unsqueeze(1)
        cov_pool = cov_pool + wts[k] * (covs[k] + diff @ diff.T)
    cov_pool = 0.5 * (cov_pool + cov_pool.T)
    chol_pool = torch.linalg.cholesky(cov_pool)  # L
    r = torch.linalg.inv(chol_pool)  # R = L^{-1}

    # Whitened mixture: means R mu_k, covs R Sigma_k R^T (still a GMM).
    means_w = means @ r.T
    covs_w = torch.einsum("ij,kjl,ml->kim", r, covs, r)
    covs_w = 0.5 * (covs_w + covs_w.transpose(-1, -2))
    gmm = AnisotropicGMM.from_components(means_w, covs_w, wts, dtype=dtype)

    c_rho = gmm.c_pi(sigma).to(dtype)

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        return gmm.v0(z.to(dtype), sigma).to(dtype)

    w = gmm.weights.to(dtype)
    mu = gmm.means.to(dtype)
    chol = torch.linalg.cholesky(gmm.covariances.to(dtype))

    def sampler(M: int, gen: torch.Generator) -> torch.Tensor:
        M = int(M)
        comp = torch.multinomial(w, M, replacement=True, generator=gen)
        zz = torch.randn(M, d, generator=gen, dtype=dtype)
        return mu[comp] + torch.einsum("nij,nj->ni", chol[comp], zz)

    e_x1sq = float((w * (mu[:, 0] ** 2 + gmm.covariances[:, 0, 0])).sum())

    def cross_check_reference(z: torch.Tensor, w_in: torch.Tensor) -> float:
        return float(gmm.mmd_sq(z, w_in, sigma))

    meta = {
        "kind": "whitened_cncv_gmm_2comp",
        "d": d,
        "n_modes": 2,
        "variances": variances,
        "separation": sep,
        "metric": "mahalanobis_pooled_inv",
        "sigma_whitened": float(sigma),
        "lift": chol_pool.tolist(),
        "whiten": r.tolist(),
        "means_whitened": means_w.tolist(),
    }
    return _Target(
        name="gmm_cncv_whitened",
        mu_fn=mu_fn,
        c_rho=c_rho,
        sampler=sampler,
        e_x1sq=e_x1sq,
        cross_check_reference=cross_check_reference,
        meta=meta,
    )


def _banana_sample(
    M: int, gen: torch.Generator, s1: float, b: float, dtype: torch.dtype
) -> torch.Tensor:
    """``M`` i.i.d. samples from the 2-D twisted-Gaussian (banana) ``rho``.

    The Rosenbrock-style banana: ``x1 ~ N(0, s1^2)``, ``x2 = t + b (x1^2 -
    s1^2)`` with ``t ~ N(0, 1)`` (the ``-b s1^2`` shift centres ``x2``).
    Generator-threaded for local determinism.
    """
    x1 = s1 * torch.randn(int(M), generator=gen, dtype=dtype)
    t = torch.randn(int(M), generator=gen, dtype=dtype)
    x2 = t + b * (x1**2 - s1**2)
    return torch.stack([x1, x2], dim=1)


def banana_target(
    sigma: float,
    d: int = 2,
    s1: float = 3.0,
    b: float = 0.1,
    n_ref: int = 20000,
    n_mu: int = 4000,
    seed: int = 20260612,
    dtype: torch.dtype = DTYPE,
) -> _Target:
    """The 2-D twisted-Gaussian (banana) target ``rho`` via Monte Carlo.

    ``rho`` is the Rosenbrock-style banana (``x1 ~ N(0, s1^2)``,
    ``x2 = t + b (x1^2 - s1^2)``, ``t ~ N(0, 1)``), which is neither Gaussian
    nor a GMM, so it has no Gaussian closed form for the SE kernel mean or
    self-affinity. Instead, ``mu`` and ``c_rho`` are estimated by Monte Carlo
    from fixed reference banks ``{x_n} ~ rho``, the same sampled-``rho``
    estimand a flow or physics ``rho`` uses:

        mu_j   = E_{x ~ rho}[k(x, z_j)] ~ (1/L) sum_l k(x_l, z_j),
        c_rho  = E_{x, x'}[k(x, x')]   ~ the V-statistic mean over the bank.

    For efficiency the estimand the gradient flows through every step,
    ``mu_fn``, integrates over a small fixed bank of ``n_mu`` points
    (``L_train``; each ``mu_fn`` call is then ``O(M n_mu)``, not
    ``O(M n_ref)``), while the constant ``c_rho``, which enters no gradient
    and is computed once at build time, and the metric cross-check use the
    larger ``n_ref``-point reference (``L_eval``). The small ``mu`` bank is
    the first ``n_mu`` rows of the same reference sample, so it is a sub-bank
    of the reference (no extra RNG, deterministic).

    ``mu_fn`` is differentiable in ``z`` (autograd flows through the SE
    kernel sum), as the move arm and the amortizer require; only the estimand
    is empirical rather than closed form. ``c_rho`` is a constant and does
    not enter any gradient. There is no independent closed form for the
    banana, so the cross-check reports the same Monte Carlo estimand.

    Args:
        sigma: SE bandwidth.
        d: Must be 2 (the twisted Gaussian is a 2-D toy).
        s1: ``x1`` prior std.
        b: Twist strength.
        n_ref: Reference-bank size ``L_eval`` for ``c_rho`` and the metric
            cross-check (the larger reference).
        n_mu: Bank size ``L_train`` the per-step ``mu_fn`` integrates over (a
            sub-bank of the reference; smaller for fast training steps).
            Clamped to at most ``n_ref``.
        seed: Seed for the fixed reference bank (so ``mu`` / ``c_rho`` are
            deterministic per build).
        dtype: Floating dtype.

    Returns:
        A :class:`_Target` for the banana at ``d = 2`` (MC estimands).
    """
    if d != 2:
        raise ValueError(f"banana_target is a 2-D toy; got d={d}.")

    inv_2sig2 = 1.0 / (2.0 * sigma**2)
    n_mu = min(int(n_mu), int(n_ref))

    # Fixed reference bank {x_n} ~ rho for the MC mu / c_rho (local RNG). The
    # small mu bank (L_train) is the first n_mu rows of this reference sample.
    ref_gen = torch.Generator(device="cpu")
    ref_gen.manual_seed(int(seed))
    x_ref = _banana_sample(int(n_ref), ref_gen, s1, b, dtype)  # (L_eval, 2)
    x_mu = x_ref[:n_mu]  # (L_train, 2) -- the per-step estimand bank

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        """MC kernel mean ``mu_j = (1/L) sum_l k(x_l, z_j)`` (autograd in z).

        Integrates over the small ``L_train`` bank ``x_mu`` so each step is
        ``O(M n_mu)``, not ``O(M n_ref)``.
        """
        z = z.to(dtype)
        return _chunked_kernel_mean(z, x_mu, inv_2sig2)

    def mu_fn_eval(z: torch.Tensor) -> torch.Tensor:
        """Kernel mean over the full reference bank (the eval estimand).

        The same-bank companion of ``c_rho``: both integrate the ``n_ref``
        reference, so a quadrature scored with the pair carries no
        cross-bank additive offset. Scoring ``mu`` on the ``n_mu`` sub-bank
        against the ``n_ref`` ``c_rho`` adds the constant
        ``c_hat_ref - c_hat_mu`` to every arm, which is large enough to
        dominate large-``M`` readings. Chunked over the bank so the
        (nodes x L_eval) kernel matrix never materializes whole;
        autograd-transparent, though eval paths call it un-tracked.
        """
        z = z.to(dtype)
        total = torch.zeros(z.shape[:-1], dtype=dtype)
        for i in range(0, x_ref.shape[0], 4096):
            for j in range(0, z.shape[0], 2048):
                sq = torch.cdist(z[j : j + 2048], x_ref[i : i + 4096]) ** 2
                total[j : j + 2048] = (
                    total[j : j + 2048]
                    + torch.exp(-sq * inv_2sig2).sum(dim=-1)
                )
        return total / x_ref.shape[0]

    # c_rho via the V-statistic mean over the larger reference bank, computed
    # once at build time in chunks to bound memory (the full N x N Gram is
    # avoided). Constant in z.
    chunk = max(256, min(2048, (32 << 20) // max(1, x_ref.shape[0])))
    total = torch.zeros((), dtype=dtype)
    n = x_ref.shape[0]
    for i in range(0, n, chunk):
        xi = x_ref[i : i + chunk]
        sq = torch.cdist(xi, x_ref) ** 2  # (chunk, L_eval)
        total = total + torch.exp(-sq * inv_2sig2).sum()
    c_rho = (total / (n * n)).to(dtype)

    chol_for_e_x1sq = s1  # E[x1^2] = s1^2 exactly.
    e_x1sq = float(chol_for_e_x1sq**2)

    def sampler(M: int, gen: torch.Generator) -> torch.Tensor:
        return _banana_sample(M, gen, s1, b, dtype)

    def cross_check_reference(z: torch.Tensor, w: torch.Tensor) -> float:
        # No independent closed form for the banana; report the same MC
        # mmd_sq.
        return float(mmd_sq(z, w, mu_fn(z), c_rho, sigma))

    meta = {
        "kind": "banana_mc",
        "d": 2,
        "s1": s1,
        "b": b,
        "n_ref": int(n_ref),
        "n_mu": int(n_mu),
        "estimand": "monte_carlo",
        "estimand_eval": "same_bank_full_ref",
    }
    return _Target(
        name="banana",
        mu_fn=mu_fn,
        c_rho=c_rho,
        sampler=sampler,
        e_x1sq=e_x1sq,
        cross_check_reference=cross_check_reference,
        meta=meta,
        mu_fn_eval=mu_fn_eval,
    )


def nw_conditional_target(
    atoms: torch.Tensor,
    alpha: torch.Tensor,
    sigma: float,
    name: str = "nw_conditional",
    meta: dict | None = None,
    dtype: torch.dtype = DTYPE,
) -> _Target:
    """A discrete Nadaraya--Watson conditional ``rho = sum_i alpha_i delta_{c_i}``.

    The target for a high-dimensional example whose posterior has no Gaussian
    closed form: the Nadaraya--Watson conditional measure in the informed
    latent. Given the bank atoms ``c_i in R^d`` (held-in latent points) and
    the responsibilities
    ``alpha_i = softmax(-||s_i - s*||^2 / 2 sigma_y^2)`` (``sum_i alpha_i = 1``),
    ``rho`` is the finite mixture of point masses ``sum_i alpha_i delta_{c_i}``.

    Because ``rho`` is discrete, the SE-kernel mean and self-affinity are
    exact finite weighted sums, with no Monte Carlo bias:

        mu_j   = E_{x ~ rho}[k(x, z_j)] = sum_i alpha_i k(c_i, z_j),
        c_rho  = E_{x, x' ~ rho}[k(x, x')] = sum_{i, i'} alpha_i alpha_i'
                 k(c_i, c_i').

    ``mu_fn`` is differentiable in ``z`` (autograd flows through the SE-kernel
    sum over the fixed atoms), as the move arm and the amortizer require.
    ``c_rho`` is a constant, enters no gradient, and is computed once at build
    time in chunks to bound memory. The i.i.d. seed nodes for the floor are
    categorical samples of the atoms under ``alpha`` (the discrete law of
    ``rho``), generator-threaded for local determinism.

    Args:
        atoms: Bank atoms ``c_i``, shape ``(n_atoms, d)`` (the latent points
            carrying the conditional's mass; typically the top-mass atoms of a
            single query, ``sum alpha`` close to one).
        alpha: Responsibilities, shape ``(n_atoms,)``; renormalized to sum to
            one inside (the conditional is a probability measure).
        sigma: SE bandwidth.
        name: Target name recorded on the :class:`_Target`.
        meta: Optional metadata dict to store on the target.
        dtype: Floating dtype.

    Returns:
        A :class:`_Target` for the discrete NW conditional at its dimension.
    """
    atoms = atoms.to(dtype)
    alpha = alpha.reshape(-1).to(dtype)
    alpha = alpha / alpha.sum()
    n_atoms, d = atoms.shape
    inv_2sig2 = 1.0 / (2.0 * sigma**2)

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        """Exact kernel mean ``mu_j = sum_i alpha_i k(c_i, z_j)`` (autograd in z)."""
        z = z.to(dtype)
        sq = torch.cdist(z, atoms) ** 2  # (..., n_atoms) per node
        return (torch.exp(-sq * inv_2sig2) * alpha).sum(dim=-1)

    # c_rho = sum_{i, i'} alpha_i alpha_i' k(c_i, c_i'), once at build time, in
    # chunks to bound memory (the full n_atoms x n_atoms Gram is avoided).
    chunk = 2048
    total = torch.zeros((), dtype=dtype)
    for i in range(0, n_atoms, chunk):
        ci = atoms[i : i + chunk]
        ai = alpha[i : i + chunk]
        sq = torch.cdist(ci, atoms) ** 2  # (chunk, n_atoms)
        total = total + (ai.unsqueeze(1) * torch.exp(-sq * inv_2sig2) * alpha).sum()
    c_rho = total.to(dtype)

    # E_rho[x1^2] = sum_i alpha_i c_i[0]^2 (the functional spot check).
    e_x1sq = float((alpha * atoms[:, 0] ** 2).sum())

    def sampler(M: int, gen: torch.Generator) -> torch.Tensor:
        """``M`` categorical samples of the atoms under ``alpha``."""
        idx = torch.multinomial(alpha, int(M), replacement=True, generator=gen)
        return atoms[idx]

    def cross_check_reference(z: torch.Tensor, w: torch.Tensor) -> float:
        # No independent closed form for a discrete measure; report the same
        # exact discrete mmd_sq.
        return float(mmd_sq(z, w, mu_fn(z), c_rho, sigma))

    full_meta = {"kind": "nw_conditional", "d": d, "n_atoms": int(n_atoms)}
    if meta is not None:
        full_meta.update(meta)
    return _Target(
        name=name,
        mu_fn=mu_fn,
        c_rho=c_rho,
        sampler=sampler,
        e_x1sq=e_x1sq,
        cross_check_reference=cross_check_reference,
        meta=full_meta,
    )


def sampled_bank_target(
    bank: torch.Tensor,
    sigma: float,
    n_mu: int | None = None,
    name: str = "sampled_bank",
    meta: dict | None = None,
    dtype: torch.dtype = DTYPE,
    replace: bool | None = None,
) -> _Target:
    """A sampled-``rho`` target from a fixed bank of i.i.d. samples ``{x_n} ~ rho``.

    For a ``rho`` that is neither Gaussian nor a GMM and has no closed form,
    for example the per-observation posterior of a trained conditional flow
    ``q_phi(. | c)`` already projected into the informed latent. Given a bank
    of i.i.d. samples ``{x_n}`` from that ``rho``, the SE kernel mean and
    self-affinity are estimated by Monte Carlo from the bank, on the same
    sampled-``rho`` estimand path the banana uses:

        mu_j   = E_{x ~ rho}[k(x, z_j)] ~ (1/L) sum_l k(x_l, z_j),
        c_rho  = E_{x, x'}[k(x, x')]   ~ the V-statistic mean over the bank.

    For efficiency the per-step estimand ``mu_fn`` integrates over a small
    fixed sub-bank of ``n_mu`` rows (``L_train``; each ``mu_fn`` call is then
    ``O(M n_mu)``), while the constant ``c_rho``, which enters no gradient and
    is computed once at build time in chunks, and the floor's i.i.d. sampler
    use the full bank (``L_eval``).

    ``mu_fn`` is differentiable in ``z`` (autograd flows through the SE kernel
    sum), as the move arm and the amortizer require; only the estimand is
    empirical. ``c_rho`` is a constant. There is no independent closed form
    for a sampled measure, so the cross-check reports the same Monte Carlo
    ``mmd_sq``.

    ``replace`` decides whether the floor is the i.i.d. floor. Sampling the
    seed nodes without replacement returns a sample that is not i.i.d. from
    the bank measure: its empirical variance carries the finite-population
    correction ``(L - M) / (L - 1)``, so the floor arm reads low by that
    factor and every margin measured against it is understated. When ``rho``
    is the bank, that is, a reference defined as its atoms such as a stored
    MCMC posterior, i.i.d. sampling from it means with replacement by
    definition, and only then does the floor reproduce the identity
    ``E MMD^2 = (1 - c_rho) / M``. Pass ``replace=True`` there. The correction
    is negligible for a bank far larger than the budget (0.16% at ``M = 32``
    of ``L = 20000``, 2.6% at ``M = 512``) and severe when it is not (51% at
    ``M = 1024`` of ``L = 2000``). ``None`` keeps without-replacement while
    the budget fits, falling back to with-replacement when it does not.

    Args:
        bank: i.i.d. samples ``{x_n}`` from ``rho``, shape ``(L_eval, d)``
            (the full reference bank; the floor samples from it).
        sigma: SE bandwidth.
        n_mu: Per-step ``mu_fn`` sub-bank size ``L_train`` (the first ``n_mu``
            rows of ``bank``; ``None`` uses the whole bank). Clamped to the
            bank size.
        name: Target name recorded on the :class:`_Target`.
        meta: Optional metadata dict to store on the target.
        dtype: Floating dtype.
        replace: Whether the i.i.d. sampler samples with replacement. ``True``
            makes the floor the genuine i.i.d. floor of the bank measure (use
            it whenever ``rho`` is defined as the bank); ``None`` samples
            without replacement while ``M <= L`` and with replacement beyond
            it. See the note above.

    Returns:
        A :class:`_Target` for the sampled bank at its dimension.
    """
    x_ref = bank.to(dtype)
    if x_ref.dim() != 2:
        raise ValueError(f"bank must be (L, d); got shape {tuple(x_ref.shape)}")
    n_ref, d = x_ref.shape
    if n_mu is None:
        n_mu = n_ref
    n_mu = min(int(n_mu), n_ref)
    x_mu = x_ref[:n_mu]  # (L_train, d) -- the per-step estimand bank
    inv_2sig2 = 1.0 / (2.0 * sigma**2)

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        """MC kernel mean ``mu_j = (1/L) sum_l k(x_l, z_j)`` (autograd in z)."""
        z = z.to(dtype)
        sq = torch.cdist(z, x_mu) ** 2  # (..., L_train) per node
        return torch.exp(-sq * inv_2sig2).mean(dim=-1)

    # c_rho via the V-statistic mean over the full bank, once at build time, in
    # chunks to bound memory (the full L x L Gram is avoided). Constant in z.
    # The chunk adapts to the bank: at a fixed 2048 rows the cdist/exp
    # transients are 2048 * L * 8 bytes twice over, several gigabytes for a
    # large bank. Capping the transient costs nothing measurable in wall time.
    chunk = max(256, min(2048, (32 << 20) // max(1, n_ref)))
    total = torch.zeros((), dtype=dtype)
    for i in range(0, n_ref, chunk):
        xi = x_ref[i : i + chunk]
        sq = torch.cdist(xi, x_ref) ** 2  # (chunk, L_eval)
        total = total + torch.exp(-sq * inv_2sig2).sum()
    c_rho = (total / (n_ref * n_ref)).to(dtype)

    # E_rho[x1^2] ~ the bank mean of x1^2 (the functional spot check).
    e_x1sq = float((x_ref[:, 0] ** 2).mean())

    def sampler(M: int, gen: torch.Generator) -> torch.Tensor:
        """``M`` i.i.d. samples from ``rho`` = rows taken from the bank.

        ``replace=True`` gives genuinely i.i.d. rows (the correct reading when
        ``rho`` is the bank). ``replace=None`` samples without replacement
        when ``M <= L_eval`` and with replacement otherwise, so any ``M``
        works.
        """
        M = int(M)
        with_repl = (M > n_ref) if replace is None else bool(replace)
        idx = torch.multinomial(
            torch.ones(n_ref, dtype=dtype), M, replacement=with_repl,
            generator=gen,
        )
        return x_ref[idx]

    def cross_check_reference(z: torch.Tensor, w: torch.Tensor) -> float:
        # No independent closed form for a sampled measure; report the same MC
        # mmd_sq.
        return float(mmd_sq(z, w, mu_fn(z), c_rho, sigma))

    full_meta = {
        "kind": "sampled_bank",
        "d": int(d),
        "n_ref": int(n_ref),
        "n_mu": int(n_mu),
        "estimand": "monte_carlo",
        # Recorded because it decides what the floor arm means; see the
        # finite-population note in the docstring.
        "floor_with_replacement": (
            "budget_dependent" if replace is None else bool(replace)
        ),
    }
    if meta is not None:
        full_meta.update(meta)
    return _Target(
        name=name,
        mu_fn=mu_fn,
        c_rho=c_rho,
        sampler=sampler,
        e_x1sq=e_x1sq,
        cross_check_reference=cross_check_reference,
        meta=full_meta,
    )


def build_target(
    name: str,
    sigma: float,
    d: int = 2,
    dtype: torch.dtype = DTYPE,
    n_ref: int = 20000,
    n_mu: int = 4000,
) -> _Target:
    """Factory: build a target by its short name and dimension.

    Args:
        name: ``"gaussian"`` (Gaussian posterior), ``"gmm"`` (5-mode GMM),
            ``"gmm_cncv"`` (the 2-component GMM at any ``d``), or
            ``"banana"`` (the 2-D twisted-Gaussian, MC estimands).
        sigma: SE bandwidth.
        d: Ambient dimension. Gaussian and ``gmm_cncv`` take any ``d``; GMM
            takes ``2`` or ``>= 5``; banana is ``d = 2`` only.
        dtype: Floating dtype.
        n_ref: Reference-bank size ``L_eval`` for the banana MC ``c_rho`` /
            cross-check (only used by the banana target; ignored otherwise).
        n_mu: Per-step ``mu_fn`` bank size ``L_train`` for the banana (only
            used by the banana target; ignored otherwise).

    Returns:
        The constructed :class:`_Target`.

    Raises:
        ValueError: if ``name`` is not a known target.
    """
    n = name.lower()
    if n == "gaussian":
        return gaussian_target(sigma, d=d, dtype=dtype)
    if n == "gmm":
        return gmm_target(sigma, d=d, dtype=dtype)
    if n == "gmm_cncv":
        return gmm_cncv_target(sigma, d=d, dtype=dtype)
    if n == "banana":
        return banana_target(sigma, d=d, n_ref=n_ref, n_mu=n_mu, dtype=dtype)
    raise ValueError(
        f"Unknown designed-quadrature target {name!r}; expected "
        f"'gaussian', 'gmm', 'gmm_cncv' or 'banana'"
    )
