"""Squared-exponential MMD^2 between a weighted node set and a target.

Two closed / empirical forms of the kernel maximum mean discrepancy under
the squared-exponential (SE) kernel ``k(a, b) = exp(-||a - b||^2 /
(2 sigma^2))``, against an arbitrary target: any Gaussian, or any reference
sample set.

  * :func:`mmd_sq_gaussian` is the Gaussian-vs-weighted-nodes closed form.
    For a target ``N(mu, Sigma)`` and a weighted node set ``nu = sum_i w_i
    delta_{z_i}`` (unit-sum weights), the SE MMD^2 is

        MMD^2 = C_target - 2 sum_i w_i mu_k(z_i) + sum_{ij} w_i w_j k(z_i, z_j),

    with the Gaussian self-affinity and Gaussian-vs-point mean

        C_target = |I + 2 Sigma / sigma^2|^{-1/2},
        mu_k(z)  = |I + Sigma / sigma^2|^{-1/2}
                   exp(-0.5 (z - mu)^T (Sigma + sigma^2 I)^{-1} (z - mu)).

    The target is an explicit ``(mu, Sigma)``.

  * :func:`mmd_sq_empirical` is the empirical ``a^T K a`` form against a
    reference sample set ``{r_l}``: exact posterior samples, or any
    Monte-Carlo reference. Writing the stacked support ``{z_i} u {r_l}``
    and the signed amplitude vector ``a = [w ; -u]`` (``w`` the unit-sum
    node weights, ``u`` the uniform ``1 / L`` reference weights), the
    biased V-statistic MMD^2 is the single quadratic form ``a^T K a`` with
    ``K`` the SE Gram over the stacked support. This mirrors the
    ``mmd_regression`` objective's empirical SE-MMD (``a^T K a``) but with a
    sample target rather than the Nadaraya-Watson conditional measure.

Both forms renormalize the node weights to unit sum, run the linear
algebra in float64 (the SE Gram / determinant are ill-conditioned at small
``sigma``), and clamp the result at zero to absorb round-off (the SE Gram
is PSD, so the exact MMD^2 is non-negative; a tiny negative is numerical).
Each public function coerces its inputs to float64 and creates all of its
internal tensors on the inputs' device, so it follows the input device
cleanly (CPU or GPU). The returned scalar is a 0-dim float64 tensor on the
inputs' device; callers that want a Python float can ``float(...)`` it.

As ``L -> infinity`` the empirical form converges to the Gaussian form
when the reference samples come from ``N(mu, Sigma)``.

Reference:
  - Belhadji, Sharp, Marzouk, "To discretize continually: Mean shift
    interacting particle systems for Bayesian inference", arXiv:2605.14142
    (squared-exponential kernel and analytic MMD, eqs. 4-5).
  - Gretton et al., "A kernel two-sample test", JMLR 2012 (the empirical
    MMD V-statistic ``a^T K a``).
"""

from __future__ import annotations

import torch


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    """Renormalize ``weights`` to unit sum in float64.

    The MSIP stationary weights are signed and need not be unit-sum; the
    MMD convention uses the normalized empirical measure. The sum is
    asserted not to be near zero, since an all-cancel weight vector has no
    well-defined normalization.
    """
    w = weights.double().reshape(-1)
    w_sum = w.sum()
    assert w_sum.abs() > 1e-6, (
        "mmd: weights sum to ~0; cannot normalize to unit mass"
    )
    return w / w_sum


def _gaussian_vs_point_mean(
    z: torch.Tensor,
    mu: torch.Tensor,
    sigma_cov: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Gaussian-vs-point kernel mean ``E_{x~N(mu,Sigma)}[k(x, z)]``.

    Equals ``|I + Sigma / sigma^2|^{-1/2} exp(-0.5 (z - mu)^T (Sigma +
    sigma^2 I)^{-1} (z - mu))`` (the SE convolution of a point with a
    Gaussian), evaluated for each row of ``z`` in float64.

    Args:
        z: Node positions, shape ``(m, d)``.
        mu: Gaussian mean, shape ``(d,)``.
        sigma_cov: Gaussian covariance ``Sigma``, shape ``(d, d)`` (SPD).
        sigma: SE bandwidth.

    Returns:
        Per-node mean ``mu_k(z_i)``, shape ``(m,)``, float64.
    """
    d = mu.shape[0]
    eye = torch.eye(d, dtype=torch.float64, device=mu.device)
    smoothed = sigma_cov + sigma**2 * eye  # (d, d)
    _, logdet = torch.linalg.slogdet(eye + sigma_cov / sigma**2)
    prefac = torch.exp(-0.5 * logdet)  # |I + Sigma/sigma^2|^{-1/2}
    diff = z - mu.unsqueeze(0)  # (m, d)
    sol = torch.linalg.solve(smoothed, diff.T).T  # (m, d)
    quad = (diff * sol).sum(dim=-1)  # (m,)
    return prefac * torch.exp(-0.5 * quad)


def mmd_sq_gaussian(
    nodes: torch.Tensor,
    weights: torch.Tensor,
    mu_post: torch.Tensor,
    Sigma_post: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Closed-form SE-kernel ``MMD^2`` between ``N(mu, Sigma)`` and nodes.

    For an arbitrary Gaussian target ``N(mu_post, Sigma_post)``:

        MMD^2 = C - 2 sum_i w_i mu_k(z_i) + sum_{ij} w_i w_j k(z_i, z_j),
        C     = |I + 2 Sigma / sigma^2|^{-1/2},

    with the Gaussian-vs-point mean ``mu_k`` of
    :func:`_gaussian_vs_point_mean`. Weights are renormalized to unit sum;
    all linear algebra is float64; the result is clamped at zero (the SE
    Gram is PSD so the exact value is non-negative).

    Args:
        nodes: Node set ``z``, shape ``(m, d)``.
        weights: Node weights, shape ``(m,)``. Renormalized internally.
        mu_post: Target Gaussian mean, shape ``(d,)``.
        Sigma_post: Target Gaussian covariance, shape ``(d, d)`` (SPD).
        sigma: SE kernel bandwidth.

    Returns:
        Scalar ``MMD^2`` as a 0-dim float64 tensor on the inputs' device.
    """
    mu = mu_post.double().reshape(-1)
    cov = Sigma_post.double()
    z = nodes.double()
    w = _normalize_weights(weights)
    d = mu.shape[0]

    eye = torch.eye(d, dtype=torch.float64, device=mu.device)
    # C = |I + 2 Sigma / sigma^2|^{-1/2} (Gaussian self-affinity).
    _, logdet_c = torch.linalg.slogdet(eye + 2.0 * cov / sigma**2)
    c_target = torch.exp(-0.5 * logdet_c)

    # Cross term: sum_i w_i E_{x~N(mu,Sigma)}[k(x, z_i)].
    mu_k = _gaussian_vs_point_mean(z, mu, cov, sigma)  # (m,)
    cross = (w * mu_k).sum()

    # Node self term: w^T K w with K_{ij} = k(z_i, z_j).
    sq_dist = torch.cdist(z, z) ** 2
    k_mat = torch.exp(-sq_dist / (2.0 * sigma**2))
    quad = w @ k_mat @ w

    mmd2 = c_target - 2.0 * cross + quad
    return torch.clamp(mmd2, min=0.0)


def mmd_sq_empirical(
    nodes: torch.Tensor,
    weights: torch.Tensor,
    ref_samples: torch.Tensor,
    sigma: float,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Empirical SE-kernel ``MMD^2`` between nodes and a reference sample set.

    The biased V-statistic MMD^2 in the single-quadratic-form
    ``a^T K a`` representation: stacking the support ``{z_i} u {r_l}`` and
    the signed amplitude vector ``a = [w ; -u]`` (``w`` the unit-sum node
    weights, ``u_l = 1 / L`` the uniform reference weights), the SE Gram
    ``K`` over the stacked support gives ``MMD^2 = a^T K a``, equivalently
    the expanded three-block form

        sum_{ij} w_i w_j k(z_i, z_j)        (node self term, w^T K_zz w)
        - 2 sum_{i,l} w_i u_l k(z_i, r_l)   (cross term,      -2 w^T K_zr u)
        + sum_{l,l'} u_l u_l' k(r_l, r_l')  (ref self term,    u^T K_rr u).

    This matches the empirical ``a^T K a`` form of the ``mmd_regression``
    objective (here the second measure is a sample set, not a
    Nadaraya-Watson conditional measure), but is computed block-wise rather
    than via a single ``(m + L) x (m + L)`` Gram so the reference count
    ``L`` never materializes an ``O(L^2)`` matrix: the node self term is the
    cheap ``m x m`` block, while the cross and reference self terms are
    accumulated in chunks of ``chunk_size`` reference rows. Peak memory is
    therefore ``O(m^2 + (m + chunk_size) * L)`` instead of ``O((m + L)^2)``
    (the ``L`` factors are the transient per-chunk SE Gram rows, freed each
    iteration), which keeps the metric usable for very large reference sets.

    Algebraically identical to ``a^T K a`` (each block uses the same
    ``exp(-cdist^2 / (2 sigma^2))`` SE evaluation as the stacked form). All
    linear algebra is float64; the result is clamped at zero (the SE Gram
    is PSD).

    Args:
        nodes: Node set ``z``, shape ``(m, d)`` (``m >= 1``).
        weights: Node weights, shape ``(m,)``. Renormalized to unit sum.
        ref_samples: Reference samples ``r``, shape ``(L, d)`` (``L >= 1``).
        sigma: SE kernel bandwidth.
        chunk_size: Number of reference rows processed per block in the
            cross and reference self terms (caps peak memory; the result is
            independent of this value).

    Returns:
        Scalar ``MMD^2`` as a 0-dim float64 tensor on the inputs' device.
    """
    z = nodes.double()
    r = ref_samples.double()
    assert z.shape[0] > 0, "mmd_sq_empirical: empty node set"
    assert r.shape[0] > 0, "mmd_sq_empirical: empty reference set"
    w = _normalize_weights(weights)
    n_ref = r.shape[0]
    u = torch.full((n_ref,), 1.0 / n_ref, dtype=torch.float64, device=r.device)
    cs = max(int(chunk_size), 1)
    two_sigma_sq = 2.0 * sigma**2

    # Node self term: w^T K_zz w (cheap m x m block).
    k_zz = torch.exp(-(torch.cdist(z, z) ** 2) / two_sigma_sq)
    node_self = w @ k_zz @ w

    # Cross term -2 w^T K_zr u and reference self term u^T K_rr u, both
    # accumulated over chunks of reference rows so no O(L^2) Gram forms.
    cross = z.new_zeros(())  # 0-dim float64 accumulator
    ref_self = z.new_zeros(())
    for start in range(0, n_ref, cs):
        r_blk = r[start : start + cs]  # (b, d)
        u_blk = u[start : start + cs]  # (b,)
        # Cross block: w^T K_{z, blk} u_blk, summed into the cross term.
        k_zr_blk = torch.exp(-(torch.cdist(z, r_blk) ** 2) / two_sigma_sq)
        cross = cross + (w @ k_zr_blk) @ u_blk
        # Reference self block: u_blk^T K_{blk, r} u (b x L row band).
        k_rr_blk = torch.exp(-(torch.cdist(r_blk, r) ** 2) / two_sigma_sq)
        ref_self = ref_self + u_blk @ (k_rr_blk @ u)

    mmd2 = node_self - 2.0 * cross + ref_self
    return torch.clamp(mmd2, min=0.0)
