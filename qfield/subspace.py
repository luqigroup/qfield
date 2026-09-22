"""Informed-subspace estimators for the latent quadrature.

The feature map ``phi`` of the latent quadrature must project onto an
INFORMED subspace -- a (near-) sufficient statistic for the posterior, the
likelihood-informed subspace (LIS) of certified dimension reduction. This
module builds that subspace two ways and validates that they agree on the
linear--Gaussian targets:

  * **H (exact closed form, score-based).** The prior-whitened
    gradient-information matrix

        H~ = Sigma_pr^{1/2} A^T Sigma_obs^{-1} A Sigma_pr^{1/2} = B^T B,
        B  = Sigma_obs^{-1/2} A Sigma_pr^{1/2} = U Sigma V^T,

    whose eigenvalues are the LIS spectrum ``lambda_k(H~) = sigma_k^2`` and
    whose eigenvectors are the right singular vectors ``V`` of the whitened
    forward map. Informed directions = leading generalized eigenvectors of
    ``(H, Sigma_pr^{-1})`` -- i.e. ``u_i = Sigma_pr^{1/2} v_i`` from the
    whitened eigendecomposition. On the linear--Gaussian targets this is
    EXACT and is the reference the other route is checked against.

  * **C (score-free, from the bank).** The covariance of the posterior
    mean across observations, in the prior metric,

        C = Cov_y( E[x | y] ),    C~ = Sigma_pr^{-1/2} C Sigma_pr^{-1/2}.

    Estimated from the joint bank with NO forward solve via the
    Nadaraya--Watson conditional mean ``mu_hat_i = E[x | y_i] =
    sum_j softmax_j(-||y_i - y_j||^2 / 2 sigma_y^2) x_j``, then ``C =
    Cov_i(mu_hat_i)`` (a SIR-slicing route is also provided). Informed
    directions = leading generalized eigenvectors of ``(C, Sigma_pr^{-1})``.

The C--H equivalence. In the linear--Gaussian case both share
the eigenvectors ``V`` and the eigenvalues are tied by the strictly
increasing bijection

    lambda_k(C~) = sigma_k^2 / (1 + sigma_k^2) = lambda_k(H~) / (1 + lambda_k(H~)),

so the leading-r subspace is identical at every r. :func:`principal_angles`
and :func:`ch_bijection_residual` check this on a real tomography instance.

The feature map. :class:`InformedSubspace` returns a callable ``phi(x) =
W^T x`` whose columns are the leading-r informed directions, optionally
PRIOR-WHITENED so the feature coordinates are ``V_r^T Sigma_pr^{-1/2} x``
(the prior-whitened informed coordinates, unit prior variance per
direction). The prior metric is what distinguishes this from plain PCA of
the parameter marginal -- PCA solves ``(Cov(x), I)`` and returns the
directions of greatest PRIOR variance, frequently the ones the data does
NOT inform.

Reference:
  - Spantini et al. 2015; Cui et al. 2014; Zahm et al. 2022 (the LIS and
    its certified bound).
  - Li 1991 (sliced inverse regression, the score-free C route).
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 / arXiv:2502.10600 (the
    quadrature the subspace feeds).
"""

from __future__ import annotations

import math

import torch


# The whiten=True branch of InformedSubspace.from_hmatvec densifies
# Sigma_pr^{1/2} (dim applies of prior_cov_half, then a (dim, dim) matrix the
# constructor inverts); a small-dim cross-check only. Refuse above this, the
# same O(N^2) guard as seismic_born.HTILDE_DX_CAP / spectral field DENSE_CAP.
DENSE_SUBSPACE_CAP: int = 4096


def _sym(m: torch.Tensor) -> torch.Tensor:
    """Symmetrize (kills tiny asymmetry before ``eigh``)."""
    return 0.5 * (m + m.transpose(-1, -2))


def _dense_from_matvec(
    matvec, dim: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Materialize a ``(dim, dim)`` matrix from a single-vector matvec.

    Applies ``matvec`` to each canonical basis vector ``e_j`` (a ``(dim,)``
    field) and stacks the results as columns. Used ONLY for the whitened
    feature-map cross-check on a SMALL ``dim`` -- it does ``dim`` applies and
    forms a dense matrix, so it must never run at seismic scale (the
    ``whiten=True`` branch of :meth:`InformedSubspace.from_hmatvec` that calls
    it is the explicitly small-``dim`` regime). Returns float64 on ``device``.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    cols = []
    for j in range(dim):
        e = torch.zeros(dim, dtype=torch.float64, device=dev)
        e[j] = 1.0
        cols.append(matvec(e).reshape(-1).to(dtype=torch.float64, device=dev))
    return torch.stack(cols, dim=1)


def _matrix_sqrt_and_inv_sqrt(
    cov: torch.Tensor, floor: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric square root and inverse square root of an SPD matrix.

    Via the eigendecomposition ``cov = V diag(w) V^T`` (clamped from below
    for numerical safety), returns ``(cov^{1/2}, cov^{-1/2})``. Used for the
    prior whitening Sigma_pr^{+/-1/2} throughout.
    """
    w, v = torch.linalg.eigh(_sym(cov))
    w = w.clamp_min(floor)
    half = (v * w.sqrt()) @ v.T
    inv_half = (v * w.rsqrt()) @ v.T
    return _sym(half), _sym(inv_half)


def participation_ratio(eigvals: torch.Tensor) -> float:
    """Participation ratio ``(sum lambda)^2 / sum lambda^2`` of a spectrum.

    The effective-rank statistic: it equals ``r`` for a
    flat top-``r`` spectrum and is small when one direction dominates. We
    round it UP to the next integer as the retained dimension (so a PR of
    4.5 keeps 5 directions). Non-negative eigenvalues only.
    """
    w = eigvals.double().clamp_min(0.0)
    s1 = w.sum()
    s2 = (w * w).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 * s1 / s2).item())


def participation_rank(eigvals: torch.Tensor, cap: int | None = None) -> int:
    """Retained rank ``r = ceil(participation_ratio)`` (capped, >= 1)."""
    pr = participation_ratio(eigvals)
    r = max(1, math.ceil(pr))
    if cap is not None:
        r = min(r, int(cap))
    return r


def spantini_reduced_posterior_cov(
    whitened_directions: torch.Tensor,
    lis_eigvals: torch.Tensor,
    prior_cov: torch.Tensor,
    r: int,
) -> torch.Tensor:
    """The Spantini-optimal rank-``r`` reduced posterior covariance.

    The optimal $\\bphi$-reduced posterior of~\\citet{spantini2015optimal,
    zahm2022certified}: FULL conditional along the leading-``r``
    informed directions, PRIOR along the discarded complement. For the
    linear--Gaussian target the whitened posterior covariance is
    ``Sigma~_post = (I + H~)^{-1} = V (I + Sigma^2)^{-1} V^T`` with ``H~ = V
    Sigma^2 V^T``, and the rank-``r`` reduced covariance keeps
    the prior (identity in whitened coordinates) off the leading-``r`` span,

        Sigma~_r = I - sum_{i<=r} (sigma_i^2 / (1 + sigma_i^2)) v_i v_i^T
                 = I - V_r diag(c_i) V_r^T,   c_i = sigma_i^2 / (1 + sigma_i^2),

    so the only contraction below the prior happens along ``V_r``. Un-whitened,

        Sigma_r = Sigma_pr^{1/2} Sigma~_r Sigma_pr^{1/2}.

    The per-direction coefficient ``c_i = sigma_i^2 / (1 + sigma_i^2)`` is the
    prior-to-posterior variance reduction in direction ``i`` -- and, by
    the C--H bijection, EXACTLY the whitened-``C`` eigenvalue
    ``lambda_i(C~)``. So the
    SCORE-FREE ``C`` (whose ``eigvals`` ARE these ``c_i``) and the
    SCORE-BASED ``H`` (whose ``eigvals`` are ``sigma_i^2``, mapped through
    ``c_i = lambda_H / (1 + lambda_H)``) build the IDENTICAL reduced
    covariance: feed this with ``C``'s ``(whitened_directions, eigvals)`` and
    the coefficients are read off directly; feed it with ``H``'s and pass
    ``lis_eigvals = sigma_i^2`` (the conversion is applied here).

    At ``r = rank(B)`` (the full informed rank) ``Sigma_r`` equals the EXACT
    posterior covariance ``Sigma_post`` (the prior is kept only along the
    null directions, where the posterior already equals the prior); below it,
    the approximation error is bounded by the certified discarded-eigenvalue
    tail :func:`certified_eigenvalue_tail`.

    Args:
        whitened_directions: The orthonormal whitened informed directions
            ``v_i`` as columns, shape (dx, k) (``InformedSubspace.
            whitened_directions``); only the leading ``r`` are used.
        lis_eigvals: The DESCENDING informed spectrum. For ``H`` these are the
            LIS eigenvalues ``sigma_i^2`` (``lambda_H``); for ``C`` they are
            already ``c_i = sigma_i^2 / (1 + sigma_i^2)`` (``lambda_C``). The
            two are auto-distinguished: any eigenvalue ``> 1`` cannot be a
            ``C`` eigenvalue (those live in ``[0, 1)``), so a spectrum with a
            leading value ``> 1`` is read as ``lambda_H`` and converted via
            ``c = lambda / (1 + lambda)``; a spectrum bounded by ``1`` is read
            as ``lambda_C`` and used directly. (Equivalent up to the bijection;
            the guard only picks the right reading.)
        prior_cov: Prior covariance ``Sigma_pr`` (SPD), for the un-whitening.
        r: Retained rank (clamped to ``[1, k]``).

    Returns:
        The rank-``r`` reduced posterior covariance ``Sigma_r`` (dx, dx),
        float64, SPD.
    """
    half, _ = _matrix_sqrt_and_inv_sqrt(prior_cov.double())
    v = whitened_directions.double()
    dx = v.shape[0]
    r = max(1, min(int(r), v.shape[1]))
    vr = v[:, :r]
    lam = lis_eigvals.double()[:r]
    # Read the spectrum: lambda_C in [0, 1) is the coefficient directly;
    # lambda_H = sigma^2 (can exceed 1) maps through c = lambda / (1 + lambda).
    if lam.numel() > 0 and float(lam.max()) > 1.0 + 1e-9:
        coeff = lam / (1.0 + lam)  # H-spectrum -> variance-reduction coeff
    else:
        coeff = lam  # already lambda_C = sigma^2 / (1 + sigma^2)
    eye = torch.eye(dx, dtype=torch.float64)
    tilde = eye - (vr * coeff) @ vr.T  # I - sum c_i v_i v_i^T (whitened)
    return _sym(half @ tilde @ half)


def certified_eigenvalue_tail(lis_eigvals: torch.Tensor, r: int) -> float:
    """The certified discarded-eigenvalue tail ``½ Σ_{i>r} λ_i``.

    The Zahm 2022 / Spantini 2015 bound on the rank-``r`` reduced posterior,
    ``KL(pi || pi_phi) <= ½ Σ_{i>r} λ_i(H~)``, the half-sum of
    the DISCARDED LIS eigenvalues. ``lis_eigvals`` must be the ``H``-spectrum
    ``λ_i = σ_i^2`` (the certificate is stated on ``H``); if a ``C``-spectrum
    (values in ``[0, 1)``) is passed it is first mapped back through the
    inverse bijection ``σ^2 = λ_C / (1 - λ_C)`` so the certificate is computed
    on the right spectrum either way. ``r >= len`` gives ``0`` (nothing
    discarded). Returns a Python float.
    """
    lam = lis_eigvals.double().sort(descending=True).values
    if lam.numel() > 0 and float(lam.max()) <= 1.0 + 1e-9:
        # C-spectrum in [0, 1): invert the C--H bijection to sigma^2.
        lam = lam.clamp(max=1.0 - 1e-12) / (1.0 - lam.clamp(max=1.0 - 1e-12))
    r = max(0, int(r))
    if r >= lam.numel():
        return 0.0
    return 0.5 * float(lam[r:].sum())


def gaussian_kl_same_mean(
    cov_p: torch.Tensor, cov_q: torch.Tensor
) -> float:
    """Exact ``KL(N(m, cov_p) || N(m, cov_q))`` for a shared mean.

    The shared-mean Gaussian Kullback--Leibler divergence,

        KL = ½ [ tr(cov_q^{-1} cov_p) - d + log det cov_q - log det cov_p ],

    the LHS of the certified bound when ``cov_p = Sigma_post``
    (the EXACT posterior covariance) and ``cov_q = Sigma_r`` (the rank-``r``
    reduced covariance of :func:`spantini_reduced_posterior_cov`). Both
    covariances are ``Sigma_post``-shaped and the posterior mean is the same
    for the exact and reduced Gaussians (the reduction is a covariance-only
    update), so this is exactly the divergence the discarded tail bounds.
    Computed in float64. Returns a Python float (clamped at 0 for round-off).
    """
    sp = _sym(cov_p.double())
    sq = _sym(cov_q.double())
    d = sp.shape[0]
    _, logdet_p = torch.linalg.slogdet(sp)
    _, logdet_q = torch.linalg.slogdet(sq)
    sol = torch.linalg.solve(sq, sp)  # cov_q^{-1} cov_p
    kl = 0.5 * (float(torch.trace(sol)) - d + float(logdet_q) - float(logdet_p))
    return max(0.0, kl)


def diag_relative_error(
    cov_exact: torch.Tensor, cov_approx: torch.Tensor
) -> float:
    """Relative error of the marginal (per-pixel) variances, ``||.||_2``-wise.

    ``||diag(cov_approx) - diag(cov_exact)||_2 / ||diag(cov_exact)||_2`` -- the
    interpretable companion to :func:`gaussian_kl_same_mean`: how far the
    rank-``r`` reduced PER-PIXEL posterior uncertainty is from the exact one
    (the quantity the uncertainty MAP visualizes). Returns a Python float.
    """
    da = torch.diag(cov_approx.double())
    de = torch.diag(cov_exact.double())
    denom = float(de.norm())
    if denom <= 0:
        return 0.0
    return float((da - de).norm()) / denom


def _eig_from_whitened(
    m_tilde: torch.Tensor, sigma_pr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Descending eigenpairs of an ALREADY-whitened informed matrix ``M~``.

    The informed subspace is read in the prior-whitened coordinates
    ``x~ = Sigma_pr^{-1/2} x``: ``M~`` (either ``H~`` or ``C~``) is
    diagonalized, ``M~ = sum_i lambda_i v_i
    v_i^T``, and the AMBIENT informed direction is ``u_i = Sigma_pr^{1/2}
    v_i`` so that the whitened feature ``v_i^T Sigma_pr^{-1/2} x`` is what
    the latent kernel measures.

    Passing the whitened matrix directly (rather than whitening a generic
    ``M`` here) is deliberate: the closed-form ``H`` and ``C`` carry
    DIFFERENT powers of ``Sigma_pr`` (``H~ = Sigma_pr^{1/2} H
    Sigma_pr^{1/2}`` but ``C~ = Sigma_pr^{-1/2} C Sigma_pr^{-1/2}``), and it
    is the whitened ``H~``/``C~`` that share the eigenvectors ``V`` and obey
    the C--H bijection. Whitening both with the same fixed power would
    misalign their subspaces; the dedicated ``whitened_*`` builders below
    supply the right ``M~`` for each.

    Args:
        m_tilde: The whitened informed matrix ``M~`` (SPD), shape (d, d).
        sigma_pr: Prior covariance (for the un-whitening to ``u_i``).

    Returns:
        (lambdas [d] descending, U [d, d] = ambient u_i columns, V [d, d]
        = whitened v_i columns), all descending. ``V`` (orthonormal) is the
        right basis for the prior-metric subspace comparison.
    """
    half, _ = _matrix_sqrt_and_inv_sqrt(sigma_pr.double())
    w, v = torch.linalg.eigh(_sym(m_tilde.double()))  # ascending
    order = torch.argsort(w, descending=True)
    w = w[order]
    v = v[:, order]
    u = half @ v  # un-whiten: u_i = Sigma_pr^{1/2} v_i
    return w, u, v


def randomized_eigh_spd(
    hmatvec,
    dim: int,
    ell: int,
    n_iter: int = 2,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized eigendecomposition of an SPD operator from its action only.

    The Halko--Martinsson--Tropp (HMT, arXiv:0909.4061) randomized range
    finder + Rayleigh--Ritz projection, for the symmetric positive
    semidefinite operator ``H~`` supplied ONLY through the matrix-free
    callable ``hmatvec`` (no ``(dim, dim)`` matrix is ever formed). This is
    how the leading informed directions of the seismic Born ``H~ = B^T B``
    (``dx = nx nz ~ 3.3 x 10^4``) are reached -- the dense whitened LIS
    matrix cannot be materialized, but ``H~ v = B^T(B v)`` is a forward +
    adjoint Born solve away.

    The algorithm (subspace / randomized power iteration on the SPD ``H~``):

      1. Gaussian test matrix ``Omega = randn(dim, ell)`` (seeded by
         ``generator``), ``Y = H~ Omega``.
      2. ``n_iter`` power iterations, RE-ORTHONORMALIZING each pass for
         numerical stability on the SPD operator: ``Q, _ = qr(Y)``;
         ``Y = H~ Q``. (Without the QR the columns collapse onto the top
         eigenvector at the rate of the spectral gap; the QR keeps the basis
         well-conditioned, the standard HMT subspace-iteration safeguard.)
      3. ``Q, _ = qr(Y)`` -> ``Q`` is ``(dim, ell)`` orthonormal, an
         approximate basis for the leading-``ell`` eigenspace of ``H~``.
      4. Rayleigh--Ritz: ``T = Q^T (H~ Q)`` (``ell x ell``), symmetrized and
         diagonalized ``T = Vecs diag(w) Vecs^T`` (``eigh``, ascending ->
         reversed to DESCENDING). The Ritz values ``w`` approximate the
         leading eigenvalues of ``H~`` and ``U = Q Vecs`` its eigenvectors.

    All linear algebra is float64 (the small ``ell x ell`` ``eigh`` and the
    ``dim x ell`` QR), regardless of the precision ``hmatvec`` works in.

    Args:
        hmatvec: Callable ``V (dim, m) -> (dim, m)`` applying the SPD ``H~``
            to a batch of ``m`` column vectors. Treated as a black box.
        dim: Ambient dimension ``d`` (the side of ``H~``).
        ell: Sketch size ``ell = r + oversample`` (``<= dim``); the number of
            random probes / Ritz pairs computed.
        n_iter: Number of subspace power iterations ``q`` (``>= 0``). Two is
            ample when the spectrum decays; more sharpens a flat tail.
        generator: Optional torch RNG for ``Omega`` (reproducibility).
            ``generator.device`` must equal ``device`` (the device ``Omega``
            is drawn on).
        dtype: Floating dtype for the sketch and projection (float64).
        device: Device for ``Omega`` (and hence the sketch). ``None`` -> CPU.

    Returns:
        ``(w, U)`` with ``w`` the ``ell`` DESCENDING Ritz values (float64,
        on ``device``) and ``U`` the ``(dim, ell)`` Ritz eigenvectors
        (orthonormal columns), the approximate leading eigenpairs of ``H~``.

    Raises:
        ValueError: If ``ell <= 0``, ``ell > dim``, ``n_iter < 0``, or
            ``hmatvec`` returns the wrong shape / non-finite values.
    """
    if ell <= 0 or ell > dim:
        raise ValueError(
            f"sketch size ell must satisfy 0 < ell <= dim={dim}; got {ell}"
        )
    if n_iter < 0:
        raise ValueError(f"n_iter must be >= 0; got {n_iter}")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if generator is not None and generator.device != dev:
        raise ValueError(
            f"generator.device ({generator.device}) must equal the draw "
            f"device ({dev}); Omega is drawn on the latter."
        )

    def _apply(v: torch.Tensor) -> torch.Tensor:
        """Apply ``hmatvec`` and validate its shape / finiteness."""
        out = hmatvec(v)
        if out.shape != v.shape:
            raise ValueError(
                "hmatvec must map (dim, m) -> (dim, m); got input "
                f"{tuple(v.shape)} -> output {tuple(out.shape)}"
            )
        out = out.to(dtype=dtype, device=dev)
        if not torch.isfinite(out).all():
            raise ValueError("hmatvec returned non-finite values.")
        return out

    # 1. Gaussian sketch and its image under H~.
    omega = torch.randn(dim, ell, generator=generator, dtype=dtype, device=dev)
    y = _apply(omega)
    # 2. Subspace power iterations, re-orthonormalizing each pass.
    for _ in range(n_iter):
        q, _ = torch.linalg.qr(y)
        y = _apply(q)
    # 3. Orthonormal basis for the leading-ell eigenspace.
    q, _ = torch.linalg.qr(y)
    # 4. Rayleigh--Ritz projection and its (descending) eigendecomposition.
    t = _sym(q.transpose(-1, -2) @ _apply(q))  # (ell, ell)
    w, vecs = torch.linalg.eigh(t)  # ascending
    order = torch.argsort(w, descending=True)
    w = w[order]
    vecs = vecs[:, order]
    u = q @ vecs  # (dim, ell) Ritz eigenvectors of H~
    return w, u


def whitened_lis_matrix_H(
    forward: torch.Tensor,
    obs_cov: torch.Tensor,
    prior_cov: torch.Tensor,
) -> torch.Tensor:
    """Prior-whitened gradient-information matrix ``H~ = B^T B``.

    ``H~ = Sigma_pr^{1/2} A^T Sigma_obs^{-1} A Sigma_pr^{1/2}`` with
    eigenvalues the LIS spectrum ``sigma_k^2``. Returned in float64.
    """
    a = forward.double()
    spr_half, _ = _matrix_sqrt_and_inv_sqrt(prior_cov.double())
    sobs_inv = torch.linalg.inv(obs_cov.double())
    return _sym(spr_half @ a.T @ sobs_inv @ a @ spr_half)


def whitened_C_matrix_closed_form(
    forward: torch.Tensor,
    obs_cov: torch.Tensor,
    prior_cov: torch.Tensor,
) -> torch.Tensor:
    """Closed-form prior-whitened ``C~``, the exact reference for C.

    ``C~ = Sigma_pr^{-1/2} C Sigma_pr^{-1/2} = B^T (B B^T + I)^{-1} B`` with
    eigenvalues ``sigma_k^2 / (1 + sigma_k^2)``. This is the exact ``C`` the
    score-free estimator targets; the bank estimate is checked against it.
    Returned in float64.
    """
    a = forward.double()
    spr_half, _ = _matrix_sqrt_and_inv_sqrt(prior_cov.double())
    sobs_half, sobs_inv_half = _matrix_sqrt_and_inv_sqrt(obs_cov.double())
    b = sobs_inv_half @ a @ spr_half  # whitened forward map B
    dy = b.shape[0]
    inner = torch.linalg.inv(b @ b.T + torch.eye(dy, dtype=torch.float64))
    return _sym(b.T @ inner @ b)


def whitened_y_information_matrix(
    forward: torch.Tensor,
    obs_cov: torch.Tensor,
    prior_cov: torch.Tensor,
) -> torch.Tensor:
    """Noise-whitened y-side information matrix ``B B^T``.

    The y-side twin of :func:`whitened_lis_matrix_H`. From the whitened
    forward map ``B = Sigma_obs^{-1/2} A Sigma_pr^{1/2} = U Sigma V^T``
    this is

        B B^T = Sigma_obs^{-1/2} A Sigma_pr A^T Sigma_obs^{-1/2}
              = U Sigma^2 U^T,

    whose LEADING eigenvectors are the LEFT singular vectors ``U`` -- the
    informed y-directions in the noise-whitened data space -- and whose
    eigenvalues are the SAME LIS spectrum ``sigma_k^2`` as ``H~ = B^T B``.
    The sufficient statistic the NW conditioning should act on is
    ``s = U_{r_y}^T Sigma_obs^{-1/2} (y - y_bar)``, which discards the
    noise-only data directions that NW otherwise has to average over.
    Returned in float64.
    """
    a = forward.double()
    spr_half, _ = _matrix_sqrt_and_inv_sqrt(prior_cov.double())
    _, sobs_inv_half = _matrix_sqrt_and_inv_sqrt(obs_cov.double())
    b = sobs_inv_half @ a @ spr_half  # whitened forward map B
    return _sym(b @ b.T)


def whitened_cross_cov_from_bank(
    x_bank: torch.Tensor,
    y_bank: torch.Tensor,
    obs_cov: torch.Tensor,
    prior_cov: torch.Tensor,
) -> torch.Tensor:
    """Score-free whitened cross-covariance ``Sigma_obs^{-1/2} Cov(y,x) Sigma_pr^{-1/2}``.

    The score-free route to the informed y-subspace, symmetric to the
    score-free ``C`` on the x-side. It reads ONLY the bank pairs
    ``(x_i, y_i)`` (no forward solve): the empirical cross-covariance
    ``Cov_hat(y, x)`` whitened on the left by ``Sigma_obs^{-1/2}`` and on the
    right by ``Sigma_pr^{-1/2}``. In the linear--Gaussian model the
    population cross-covariance is ``Cov(y, x) = A Sigma_pr``, so the whitened
    object is

        Sigma_obs^{-1/2} (A Sigma_pr) Sigma_pr^{-1/2} = Sigma_obs^{-1/2} A
        Sigma_pr^{1/2} = B,

    the whitened forward map itself -- hence its LEFT singular vectors
    recover ``U`` exactly (the y-loadings of the inverse regression / CCA
    between x and y). Returned in float64, shape (dy, dx).
    """
    x = x_bank.double()
    y = y_bank.double()
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    n = x.shape[0]
    cross = yc.T @ xc / max(1, n - 1)  # (dy, dx) ~ Cov(y, x)
    _, spr_inv_half = _matrix_sqrt_and_inv_sqrt(prior_cov.double())
    _, sobs_inv_half = _matrix_sqrt_and_inv_sqrt(obs_cov.double())
    return sobs_inv_half @ cross @ spr_inv_half


def nw_conditional_means(
    x_bank: torch.Tensor,
    y_bank: torch.Tensor,
    sigma_y: float,
    chunk: int = 4096,
) -> torch.Tensor:
    """Nadaraya--Watson posterior means ``mu_hat_i = E[x | y_i]``.

    For each bank observation ``y_i``, the softmax-weighted average of the
    bank parameters, ``mu_hat_i = sum_j softmax_j(-||y_i - y_j||^2 /
    2 sigma_y^2) x_j`` -- the same conditioning, at bandwidth ``sigma_y``,
    that the quadrature uses. Reads only ``(x_i, y_i)``; no forward solve.

    The pairwise distances are formed in FLOAT32 and in row CHUNKS so the
    full ``N x N`` Gram is never materialized: it grows quadratically (a
    dense float64 Gram is ~0.5 GB at ``N = 8000`` and tens of gigabytes by
    ``N ~ 5 x 10^4``), whereas chunked float32 caps the peak at
    ``O(chunk * N)``. Each chunk's softmax-weighted average is accumulated
    into the output, which is cast back to float64 (the covariance
    eigendecomposition downstream wants the higher precision on the small
    ``d_x x d_x`` matrix, not on the ``N``-sized kernel sums). Shape
    (N, dx), float64.
    """
    x = x_bank.float()
    y = y_bank.float()
    n = x.shape[0]
    out = torch.empty_like(x)
    two_sig2 = 2.0 * sigma_y**2
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sq = torch.cdist(y[s:e], y) ** 2  # (b, N)
        p = torch.softmax(-sq / two_sig2, dim=1)  # (b, N), rows sum to 1
        out[s:e] = p @ x  # (b, dx)
    return out.double()


def estimate_C_from_bank(
    x_bank: torch.Tensor,
    y_bank: torch.Tensor,
    sigma_y: float,
    route: str = "nw",
    n_slices: int = 20,
) -> torch.Tensor:
    """Score-free estimate of ``C = Cov_y(E[x | y])`` from the joint bank.

    Two interchangeable routes, both reading only ``(x_i, y_i)``:

      * ``"nw"`` (default): the Nadaraya--Watson route -- the empirical
        covariance of the per-observation conditional means
        :func:`nw_conditional_means`.
      * ``"sir"``: sliced inverse regression -- sort by the first principal
        component of ``y``, bin into ``n_slices`` equal-count slices, and
        take the (count-weighted) between-slice covariance of the
        per-slice ``x`` means. A coarse first-moment estimator, provided as
        a cross-check.

    Args:
        x_bank: Joint x-samples, shape (N, dx).
        y_bank: Joint y-samples, shape (N, dy).
        sigma_y: NW bandwidth (the conditioning bandwidth; ``"nw"`` route).
        route: ``"nw"`` or ``"sir"``.
        n_slices: Number of slices for the ``"sir"`` route.

    Returns:
        ``C`` of shape (dx, dx), float64.
    """
    x = x_bank.double()
    if route == "nw":
        mu = nw_conditional_means(x, y_bank.double(), sigma_y)  # (N, dx)
        mu_c = mu - mu.mean(dim=0, keepdim=True)
        n = mu_c.shape[0]
        return _sym(mu_c.T @ mu_c / max(1, n - 1))
    if route == "sir":
        y = y_bank.double()
        # Project y onto its leading PC to get a 1-D slicing variable.
        yc = y - y.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(yc, full_matrices=False)
        score = yc @ vh[0]
        order = torch.argsort(score)
        n = x.shape[0]
        x_glob = x.mean(dim=0, keepdim=True)
        slices = torch.tensor_split(order, n_slices)
        c = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
        for sl in slices:
            if sl.numel() == 0:
                continue
            xm = x[sl].mean(dim=0, keepdim=True) - x_glob  # (1, dx)
            c += (sl.numel() / n) * (xm.T @ xm)
        return _sym(c)
    raise ValueError(f"Unknown route {route!r}; use 'nw' or 'sir'.")


def ch_bijection_residual(
    lam_C: torch.Tensor, lam_H: torch.Tensor
) -> float:
    """Max ``|lambda_C - lambda_H / (1 + lambda_H)|`` over matched directions.

    The empirical check of the C--H bijection: feeds the descending
    C-eigenvalues and
    H-eigenvalues (same ordering of directions) and returns the largest
    absolute deviation from the bijection ``lambda_C = lambda_H /
    (1 + lambda_H)``. Near zero confirms the derivation on a real instance.
    """
    k = min(lam_C.numel(), lam_H.numel())
    lc = lam_C.double()[:k]
    lh = lam_H.double()[:k]
    pred = lh / (1.0 + lh)
    return float((lc - pred).abs().max().item())


def principal_angles(
    u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Principal angles (radians, ascending) between two subspaces.

    ``u`` and ``v`` hold basis vectors of the two subspaces as COLUMNS
    (shapes (d, r1), (d, r2)); the angles are ``arccos`` of the singular
    values of ``Qu^T Qv`` with ``Qu, Qv`` orthonormal bases (QR). Near-zero
    angles mean the subspaces coincide -- the C--H subspace check. Returns
    ``min(r1, r2)`` angles, float64.

    IMPORTANT (prior metric). The angles are measured in the Euclidean
    inner product of the supplied columns. For the C--H equivalence the
    subspaces live in the PRIOR metric, so pass the WHITENED directions
    (``InformedSubspace.whitened_directions``, the orthonormal ``v_i``),
    NOT the un-whitened ``directions`` -- comparing un-whitened generalized
    eigenvectors with plain QR uses the wrong inner product and reports
    spurious non-zero angles even when the (prior-metric) subspaces
    coincide. :meth:`InformedSubspace.subspace_angles_to` does this
    correctly.
    """
    qu, _ = torch.linalg.qr(u.double())
    qv, _ = torch.linalg.qr(v.double())
    s = torch.linalg.svdvals(qu.T @ qv).clamp(-1.0, 1.0)
    return torch.arccos(s)


class InformedSubspace:
    """A fixed informed-subspace feature map ``phi(x)`` for the latent kernel.

    Holds the leading-``r`` informed directions (columns of ``U``, the
    generalized eigenvectors of ``(M, Sigma_pr^{-1})``) and exposes the
    callable ``phi(x)`` the latent quadrature consumes. With
    ``whiten=True`` (default) the features are the PRIOR-WHITENED informed
    coordinates ``V_r^T Sigma_pr^{-1/2} x`` (unit prior variance per
    direction, so the SE bandwidth is comparable across directions); with
    ``whiten=False`` they are the raw projections ``U_r^T x``.

    Build it from the closed-form ``H``/``C`` (the exact / validation
    routes) via :meth:`from_H` / :meth:`from_C_closed_form`, or
    from the score-free bank estimate via :meth:`from_bank`. The retained
    rank ``r`` is either supplied or set by the participation ratio of the
    spectrum (reported in ``self.rank``).

    Attributes:
        directions: The leading-r informed directions ``u_i`` as columns,
            shape (dx, r), float64.
        eigvals: The full descending spectrum used to pick ``r``, float64.
        rank: The retained dimension ``r``.
        sigma_pr: The prior covariance (for the whitening), or ``None`` when
            built MATRIX-FREE with no prior metric (the whitened-only / raw
            :meth:`from_hmatvec` regime). ``None`` means "identity metric":
            the dense ``(dim, dim)`` prior is NEVER formed, so only the
            metric-free attributes / methods are available (see below).
        whiten: Whether ``phi`` prior-whitens.
    """

    def __init__(
        self,
        directions: torch.Tensor,
        eigvals: torch.Tensor,
        sigma_pr: torch.Tensor | None,
        whiten: bool = True,
        whitened_directions: torch.Tensor | None = None,
    ):
        # sigma_pr=None means "no prior metric / identity": the matrix-free
        # builder reaches the leading subspace WITHOUT ever forming the dense
        # (dim, dim) prior (which is ~8.7 GB at seismic dim ~ 3.3e4), so the
        # whitened-only / raw object stores a None sentinel instead of a dense
        # identity and never takes the whiten=True dense-inverse path. Methods
        # that genuinely need the metric (the whitened feature map _w, the
        # whitened_directions un-whitening fallback, residual_fraction_off_
        # subspace) raise a clear ValueError when it is absent.
        if whiten and sigma_pr is None:
            raise ValueError(
                "whiten=True needs the prior metric Sigma_pr (its inverse "
                "defines the feature map W = Sigma_pr^{-1} U), but sigma_pr "
                "is None (built matrix-free without a prior metric); pass "
                "whiten=False for the raw U_r^T x map."
            )
        if whitened_directions is None and sigma_pr is None:
            raise ValueError(
                "whitened_directions must be supplied when sigma_pr is None: "
                "the un-whitening v_i = Sigma_pr^{-1/2} u_i needs the prior "
                "metric, which is absent (matrix-free build)."
            )
        self.eigvals = eigvals.double()
        self.sigma_pr = sigma_pr.double() if sigma_pr is not None else None
        self.rank = directions.shape[1]
        self.whiten = whiten
        self.directions = directions.double()  # un-whitened u_i (dx, r)
        # The whitened eigenvectors v_i = Sigma_pr^{-1/2} u_i (orthonormal);
        # the natural basis to compare subspaces in the prior metric (so the
        # C--H principal-angle check is well-posed even under un-whitening).
        if whitened_directions is not None:
            self.whitened_directions = whitened_directions.double()
        else:
            _, inv_half = _matrix_sqrt_and_inv_sqrt(self.sigma_pr)
            self.whitened_directions = inv_half @ self.directions
        # Projection matrix W (dx, r) so phi(x) = x @ W. Whitened coords are
        # V_r^T Sigma_pr^{-1/2} x = u_i^T Sigma_pr^{-1} x (since v_i =
        # Sigma_pr^{-1/2} u_i), i.e. W = Sigma_pr^{-1} U. Raw coords: W = U.
        if whiten:
            sigma_pr_inv = torch.linalg.inv(self.sigma_pr)
            self._w = sigma_pr_inv @ self.directions  # (dx, r)
        else:
            self._w = self.directions

    @classmethod
    def from_whitened(
        cls,
        m_tilde: torch.Tensor,
        sigma_pr: torch.Tensor,
        r: int | None = None,
        whiten: bool = True,
        rank_cap: int | None = None,
    ) -> "InformedSubspace":
        """Build from an ALREADY-whitened informed matrix ``M~``.

        Diagonalizes ``M~`` (``H~`` or ``C~``), picks ``r`` (the
        participation ratio of its spectrum if ``r`` is None), and keeps the
        leading ``r`` directions. The ambient directions ``u_i =
        Sigma_pr^{1/2} v_i`` and the whitened ``v_i`` are both retained.
        """
        lam, u, v = _eig_from_whitened(m_tilde, sigma_pr)
        if r is None:
            r = participation_rank(lam, cap=rank_cap)
        r = max(1, min(r, u.shape[1]))
        return cls(
            u[:, :r], lam, sigma_pr, whiten=whiten,
            whitened_directions=v[:, :r],
        )

    @classmethod
    def from_H(
        cls,
        forward: torch.Tensor,
        obs_cov: torch.Tensor,
        prior_cov: torch.Tensor,
        r: int | None = None,
        whiten: bool = True,
        rank_cap: int | None = None,
    ) -> "InformedSubspace":
        """Gold-standard subspace from the exact ``H = A^T Sigma_obs^{-1} A``.

        Leading eigenvectors of the whitened LIS matrix ``H~ = Sigma_pr^{1/2}
        H Sigma_pr^{1/2}``; the ambient directions ``u_i =
        Sigma_pr^{1/2} v_i`` feed the feature map.
        """
        h_tilde = whitened_lis_matrix_H(forward, obs_cov, prior_cov)
        return cls.from_whitened(
            h_tilde, prior_cov, r=r, whiten=whiten, rank_cap=rank_cap
        )

    @classmethod
    def from_hmatvec(
        cls,
        hmatvec,
        dim: int,
        r: int | None = None,
        *,
        prior_cov_half=None,
        oversample: int = 10,
        n_iter: int = 2,
        rank_cap: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        whiten: bool = True,
    ) -> "InformedSubspace":
        """MATRIX-FREE exact subspace from the action of ``H~ = B^T B``.

        The large-scale twin of :meth:`from_H`: it builds the SAME informed
        subspace -- the leading eigenvectors ``v_i`` of the whitened LIS
        matrix ``H~ = Sigma_pr^{1/2} A^T Sigma_obs^{-1} A Sigma_pr^{1/2}``
        -- but reaches them WITHOUT forming the ``(dim, dim)``
        matrix, through a callable that applies ``H~`` to a batch of vectors.
        For the seismic Born problem ``dim = nx nz`` can be ~3.3 x 10^4, so the
        dense ``H~`` of :func:`whitened_lis_matrix_H` / ``form_H_tilde`` is out
        of reach, but ``H~ v = whitened_adjoint(whitened_forward(v))`` (a Born
        forward + adjoint pair) is not; this method consumes exactly that. The
        seismic ``whitened_forward`` / ``whitened_adjoint`` / ``cov_half`` act
        on 2-D ``(nx, nz)`` fields and are NOT batched over ``m``, so the
        caller supplies a thin adapter that reshapes each of the ``m`` flat
        ``(dim,)`` columns to ``(nx, nz)``, applies the field-space op, and
        flattens the result back to ``(dim,)`` (a ``(dim, m)`` column loop).

        The leading-``r`` eigenpairs come from :func:`randomized_eigh_spd`
        (HMT with ``n_iter`` power iterations on a sketch of size ``ell = r +
        oversample``). The ``ell`` Ritz values double as the spectrum the
        participation-ratio rule reads when ``r`` is ``None`` (same rule as
        the other ``from_*``). The Ritz eigenvectors ``U`` are the WHITENED
        informed directions ``v_i`` (eigenvectors of ``H~``, orthonormal); the
        AMBIENT directions are ``u_i = Sigma_pr^{1/2} v_i``, reusing the SAME
        un-whitening relation as :func:`_eig_from_whitened` /
        :meth:`from_whitened` -- here applied MATRIX-FREE through
        ``prior_cov_half`` (no dense ``Sigma_pr^{1/2}``).

        Metric / un-whitening. ``hmatvec`` acts in the prior-WHITENED
        coordinates ``x~ = Sigma_pr^{-1/2} x``; its eigenvectors are the
        whitened ``v_i``, which is what :attr:`whitened_directions` (and hence
        :meth:`subspace_angles_to`, well-posed in the prior metric) wants
        directly. If ``prior_cov_half`` is given (a callable ``v -> Sigma_pr^
        {1/2} v``, e.g. the spectral field's ``cov_half`` adapted to flat
        ``(dim,)`` vectors), the ambient :attr:`directions` are produced
        column-by-column to match :meth:`from_H`; otherwise the object is
        WHITENED-ONLY (``directions = whitened_directions = v_i``, the
        documented degenerate case -- valid for the prior-metric subspace
        diagnostics, which read only ``whitened_directions``).

        Feature-map plumbing. The whitened callable ``phi`` (``whiten=True``)
        needs the DENSE precision action ``Sigma_pr^{-1} U``, which is exactly
        what the matrix-free regime avoids; it is therefore only built when a
        dense ``prior_cov_half`` is supplied AND ``dim`` is small enough to
        realize ``Sigma_pr^{1/2}`` as a matrix (GUARDED by
        :data:`DENSE_SUBSPACE_CAP`; a ``ValueError`` above it directs the
        caller to ``whiten=False``). At seismic scale the subspace
        (``whitened_directions`` / ``directions`` / ``eigvals`` / ``rank``) is
        the deliverable and the projection is applied matrix-free elsewhere;
        pass ``whiten=False`` to get the raw ``U_r^T x`` map (``_w = U``, no
        dense inverse, ``sigma_pr=None``). The subspace attributes are
        identical either way and no dense ``(dim, dim)`` prior is ever formed.

        Args:
            hmatvec: Callable ``V (dim, m) -> (dim, m)`` applying the WHITENED
                SPD ``H~`` to ``m`` whitened-coordinate columns (black box).
            dim: Ambient / whitened dimension ``d`` (the side of ``H~``).
            r: Retained rank; the deliverable at seismic scale (where ``r +
                oversample <= dim`` is enforced). ``None`` probes the FULL
                spectrum (``ell = dim``) and sets ``r`` from its participation
                ratio (capped by ``rank_cap``) -- a small-``dim`` convenience
                only, since it costs ``dim`` applies of ``H~``; supply ``r``
                explicitly in the matrix-free / large-``dim`` regime.
            prior_cov_half: Optional callable ``v -> Sigma_pr^{1/2} v`` on a
                single ``(dim,)`` vector, for the ambient un-whitening. If
                ``None``, the subspace is whitened-only.
            oversample: HMT oversampling ``p``; sketch size ``ell = r + p``.
            n_iter: Subspace power iterations (``>= 0``; 2 is ample).
            rank_cap: Optional cap on the auto-selected ``r``.
            generator: Optional torch RNG for the Gaussian sketch.
            dtype: Floating dtype for the randomized solve (float64).
            device: Device for the sketch / returned directions.
            whiten: Whether the feature map ``phi`` prior-whitens (see above).

        Returns:
            An :class:`InformedSubspace` whose subspace matches :meth:`from_H`
            (a drop-in alongside it).

        Raises:
            ValueError: If ``dim <= 0``, the resolved ``ell = r + oversample``
                exceeds ``dim``, ``oversample < 0``, ``prior_cov_half``
                returns the wrong shape, or ``whiten=True`` with ``dim >``
                :data:`DENSE_SUBSPACE_CAP` (the densify path is small-``dim``
                only; pass ``whiten=False`` at scale).
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        if oversample < 0:
            raise ValueError(f"oversample must be >= 0; got {oversample}")
        # Resolve the sketch size ell, guarding ell <= dim BEFORE any sketch
        # is drawn. With an EXPLICIT r the documented contract is r +
        # oversample <= dim, so an over-ask is an error (the caller asked for
        # more directions than the operator has). With r = None we PROBE the
        # full spectrum (ell = dim) and let the participation ratio pick
        # r <= dim -- a small-dim convenience only (it costs dim applies).
        if r is not None:
            ell = int(r) + int(oversample)
            if ell > dim:
                raise ValueError(
                    f"r + oversample = {r} + {oversample} = {ell} exceeds "
                    f"dim = {dim}; reduce r/oversample."
                )
            ell = max(1, ell)
        else:
            ell = dim  # probe the full spectrum to read its participation ratio

        w_ell, u_ell = randomized_eigh_spd(
            hmatvec, dim, ell, n_iter=n_iter,
            generator=generator, dtype=dtype, device=device,
        )  # w_ell descending (ell,), u_ell (dim, ell) whitened v_i

        # Retained rank: supplied, else participation ratio of the Ritz
        # spectrum (the same rule as the dense from_*),
        # clamped to the ell pairs actually computed.
        if r is None:
            r = participation_rank(w_ell, cap=rank_cap)
        r = max(1, min(int(r), u_ell.shape[1]))

        eigvals = w_ell  # full ell-spectrum used to pick r (descending)
        v = u_ell[:, :r]  # whitened informed directions v_i (dim, r)

        if prior_cov_half is None:
            # Whitened-only: no Sigma_pr^{1/2}, so ambient == whitened. Pass
            # sigma_pr=None (NOT a dense (dim, dim) identity, ~8.7 GB at
            # seismic dim) and whiten=False -- the constructor never forms the
            # prior here (whitened_directions is supplied explicitly, so the
            # un-whitening fallback is bypassed, and whiten=False skips the
            # dense inverse). This is the documented "never forms (dim, dim)"
            # path; only the metric-free attributes / subspace_angles_to work.
            return cls(
                v, eigvals, None, whiten=False, whitened_directions=v,
            )

        # Ambient directions u_i = Sigma_pr^{1/2} v_i, MATRIX-FREE: apply the
        # callable column-by-column (each column is a single (dim,) field),
        # reusing the exact un-whitening relation of _eig_from_whitened
        # without ever forming Sigma_pr^{1/2}.
        cols = []
        for j in range(r):
            uj = prior_cov_half(v[:, j])
            uj = uj.reshape(-1)
            if uj.shape != (dim,):
                raise ValueError(
                    "prior_cov_half must map a (dim,) vector to a (dim,) "
                    f"vector; got output of {uj.numel()} elements for dim="
                    f"{dim}"
                )
            cols.append(uj.to(dtype=torch.float64, device=v.device))
        u = torch.stack(cols, dim=1)  # (dim, r) ambient u_i

        # Construct via the standard __init__ (the same path from_whitened
        # uses): whitened_directions=v is supplied so the dense inv-sqrt
        # fallback is skipped; sigma_pr is only touched for the whitened _w.
        # The dense whitened feature map needs Sigma_pr^{-1}; realize it only
        # when whiten is requested AND prior_cov_half is densifiable (the
        # small cross-check regime). Otherwise return the raw (whiten=False)
        # map with sigma_pr=None -- the subspace attributes are identical
        # either way and no dense (dim, dim) prior is formed.
        if whiten:
            # Densifying Sigma_pr^{1/2} is dim applies of prior_cov_half and a
            # (dim, dim) matrix the constructor inverts -- a small-dim cross-
            # check ONLY (the docstring's "realizable as a matrix" regime).
            if dim > DENSE_SUBSPACE_CAP:
                raise ValueError(
                    f"whiten=True densifies Sigma_pr^{{1/2}} ({dim}x{dim}) and "
                    f"costs {dim} prior_cov_half applies; pass whiten=False at "
                    f"this scale (dim={dim} > cap {DENSE_SUBSPACE_CAP})"
                )
            spr_half = _dense_from_matvec(prior_cov_half, dim, v.device)
            sigma_pr = _sym(spr_half @ spr_half.transpose(-1, -2))
            return cls(
                u, eigvals, sigma_pr, whiten=True, whitened_directions=v,
            )
        return cls(u, eigvals, None, whiten=False, whitened_directions=v)

    @classmethod
    def from_C_closed_form(
        cls,
        forward: torch.Tensor,
        obs_cov: torch.Tensor,
        prior_cov: torch.Tensor,
        r: int | None = None,
        whiten: bool = True,
        rank_cap: int | None = None,
    ) -> "InformedSubspace":
        """Subspace from the closed-form ``C``, the exact reference for C.

        Leading eigenvectors of the whitened ``C~ = Sigma_pr^{-1/2} C
        Sigma_pr^{-1/2} = B^T(B B^T + I)^{-1} B``. By the C--H bijection
        this shares
        the leading-r subspace of :meth:`from_H` exactly (the eigenvalues
        are tied by ``lambda_C = sigma^2 / (1 + sigma^2)``).
        """
        c_tilde = whitened_C_matrix_closed_form(forward, obs_cov, prior_cov)
        return cls.from_whitened(
            c_tilde, prior_cov, r=r, whiten=whiten, rank_cap=rank_cap
        )

    @classmethod
    def from_bank(
        cls,
        x_bank: torch.Tensor,
        y_bank: torch.Tensor,
        prior_cov: torch.Tensor,
        sigma_y: float,
        r: int | None = None,
        route: str = "nw",
        whiten: bool = True,
        rank_cap: int | None = None,
        n_slices: int = 20,
    ) -> "InformedSubspace":
        """Score-free subspace from the joint bank.

        Estimates ``C`` from ``(x_i, y_i)`` with NO forward solve (the
        ``"nw"`` or ``"sir"`` route), prior-whitens it as ``C~ =
        Sigma_pr^{-1/2} C Sigma_pr^{-1/2}`` (the closed-form convention, so
        the bank subspace lands on the same footing as the closed-form C and
        H), and takes the leading eigenvectors. This is the practical,
        sunk-cost-only subspace the method uses by default.
        """
        c = estimate_C_from_bank(
            x_bank, y_bank, sigma_y, route=route, n_slices=n_slices
        )
        _, inv_half = _matrix_sqrt_and_inv_sqrt(prior_cov.double())
        c_tilde = _sym(inv_half @ c @ inv_half)
        return cls.from_whitened(
            c_tilde, prior_cov, r=r, whiten=whiten, rank_cap=rank_cap
        )

    @classmethod
    def from_bank_lowrank(
        cls,
        x_bank: torch.Tensor,
        y_bank: torch.Tensor,
        *,
        sigma_y: float,
        prior_cov_inv_half,
        prior_cov_half=None,
        r: int | None = None,
        rank_cap: int | None = None,
        oversample: int = 10,
        chunk: int = 4096,
        eig_floor: float = 1e-12,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        whiten: bool = True,
    ) -> "InformedSubspace":
        """MATRIX-FREE, low-rank score-free subspace from the joint bank.

        The large-scale twin of :meth:`from_bank` -- the score-free dual of
        the matrix-free :meth:`from_hmatvec` -- and the only route to the
        score-free ``C`` at seismic scale. It builds the SAME leading-``r``
        informed subspace as :meth:`from_bank` (the leading eigenvectors of
        the whitened ``C~ = Sigma_pr^{-1/2} C Sigma_pr^{-1/2}``, in the
        closed-form convention) WITHOUT ever forming the ``(dx, dx)``
        covariance the dense :func:`estimate_C_from_bank` materializes
        (~8.7 GB at ``dx = nx nz ~ 3.3 x 10^4``). It exploits the LOW RANK of
        the empirical ``C``: the score-free ``C = Cov_y(E[x | y])`` is
        estimated as the covariance of the ``N`` per-observation conditional
        means, so its column space (and every leading eigenvector) lies in the
        span of those ``N`` means;
        with ``N << dx`` the leading-``r`` modes are reached by a
        SAMPLE-SPACE (``N x N``) eigendecomposition.

        The algorithm (a thin-SVD route):

          1. NW conditional means ``mu_hat_i = E[x | y_i] = sum_j
             softmax_j(-||y_i - y_j||^2 / 2 sigma_y^2) x_j``, REUSING
             :func:`nw_conditional_means` verbatim
             (chunked float32 kernel sums, no ``N x N`` Gram materialized).
             ``(N, dx)``.
          2. Prior-whiten each conditional mean, ``m_i = Sigma_pr^{-1/2}
             mu_hat_i``, applying the matrix-free callable ``prior_cov_inv_
             half`` ROW BY ROW (each row a single ``(dx,)`` field), then center
             to ``M_w = [m_i - m_bar]`` of shape ``(N, dx)`` (the centered
             whitened conditional means as ROWS).
          3. The whitened score-free matrix is ``C~ = (1/N) M_w^T M_w``
             (``(dx, dx)``, NEVER formed); its leading-``r`` eigenpairs come
             from a thin (economy) SVD of ``M_w`` DIRECTLY, ``M_w = W S_f
             V_f^T`` -- no ``(N, N)`` Gram and no condition-number squaring. The
             eigenvalues of ``C~`` are exactly ``lambda_k = S_f^2 / N``
             (descending) and the whitened directions ARE the RIGHT singular
             vectors ``v_k = (V_f)_k`` (``(dx,)``, unit-norm); the right singular
             vectors of ``M_w`` are by definition the eigenvectors of ``C~ =
             M_w^T M_w / N``. Near-zero modes ``lambda_k <= eig_floor`` are
             DROPPED (they sit in the numerical null space of ``M_w``), so ``r``
             never exceeds the numerical rank of ``M_w``.
          4. Wrap as an :class:`InformedSubspace`: ``whitened_directions =
             [v_k]``, ``eigvals = [lambda_k]``, and -- if ``prior_cov_half`` is
             given -- the AMBIENT ``directions = Sigma_pr^{1/2} v_k`` produced
             column-by-column MATRIX-FREE, reusing the EXACT un-whitening
             relation of :func:`_eig_from_whitened` / :meth:`from_hmatvec`
             (``u_i = Sigma_pr^{1/2} v_i``). With no ``prior_cov_half`` the
             object is WHITENED-ONLY (``sigma_pr=None``, the documented
             degenerate case valid for the prior-metric subspace diagnostics,
             which read only ``whitened_directions``).

        Equivalence. The thin SVD of ``M_w`` gives exactly the same factor
        ``M_w = W S_f V_f^T`` whether one diagonalizes the ``(dx, dx)`` Gram
        ``C~ = M_w^T M_w / N`` (right singular vectors ``V_f``, eigenvalues
        ``S_f^2 / N``) or its ``(N, N)`` twin ``M_w M_w^T / N`` (left singular
        vectors ``W``, same nonzero spectrum). Taking the SVD of ``M_w``
        directly returns the right singular vectors / ``S_f^2 / N`` -- the TRUE
        leading eigenpairs of the SAME empirical ``C~`` that :meth:`from_bank`
        diagonalizes densely -- exactly and deterministically, no randomized
        approximation (cf. :meth:`from_hmatvec`'s HMT sketch); the only error is
        the finite-sample / NW-bandwidth estimation error shared with
        :meth:`from_bank`. In the linear--Gaussian case the population
        eigenvalues obey the bijection ``lambda_C = lambda_H / (1 + lambda_H)``
        on the LIS spectrum.

        Retained rank. ``r`` is supplied, else the participation ratio of the
        ``C~`` spectrum (the same rule as the other ``from_*``),
        capped by ``rank_cap`` and -- since the empirical ``C~`` has rank at
        most ``N - 1`` after centering -- by the number of above-floor singular
        modes of ``M_w``. ``oversample`` is accepted ONLY for signature symmetry
        with :meth:`from_hmatvec` and is a NO-OP on this exact thin-SVD route: it
        enters no computation (the rank is resolved by
        :func:`participation_rank` of the ``C~`` spectrum, capped as above,
        without it). It does not enlarge any probe and does not pad the
        auto-rank headroom -- there is no sketch to oversample (the thin-SVD
        route is EXACT). It is kept (with its ``oversample < 0`` check) purely
        for signature/error parity with :meth:`from_hmatvec`.

        Peak memory. NO ``(dx, dx)`` or ``(N, N)`` Gram object is ever formed.
        The largest array is ``M_w`` of shape ``(N, dx)`` (the centered whitened
        conditional means, ~264 MB in float32 at ``dx ~ 3.3 x 10^4``, ``N ~ 2 x
        10^3``; ~528 MB in float64), of which the thin SVD is taken directly; the
        ``mu_hat`` kernel sums are chunked to ``O(chunk * N)`` inside
        :func:`nw_conditional_means`. The right singular vectors of ``M_w`` ARE
        the directions, so no ``(dx, N) x (N, r)`` reconstruction product is
        formed; if even ``M_w`` is a concern at larger ``N`` its rows could be
        chunked, but ``(N, dx)`` is the documented working set.

        Args:
            x_bank: Joint x-samples (flattened parameters), shape (N, dx).
            y_bank: Joint y-samples (flattened observations), shape (N, dy).
            sigma_y: NW bandwidth (the conditioning bandwidth; ``> 0``).
            prior_cov_inv_half: Callable ``v -> Sigma_pr^{-1/2} v`` on a single
                ``(dx,)`` vector -- the prior whitener applied to each
                conditional mean (step 2). Matrix-free (e.g. the spectral
                field's ``cov_inv_half`` adapted to flat ``(dx,)`` fields).
            prior_cov_half: Optional callable ``v -> Sigma_pr^{1/2} v`` on a
                single ``(dx,)`` vector, for the ambient un-whitening ``u_k =
                Sigma_pr^{1/2} v_k`` (step 4). If ``None``, the subspace is
                whitened-only (``sigma_pr=None``).
            r: Retained rank; the deliverable at seismic scale. ``None`` sets
                ``r`` from the participation ratio of the Gram spectrum.
            rank_cap: Optional cap on the auto-selected ``r``.
            oversample: NO-OP here (``>= 0``); accepted only for signature
                symmetry with :meth:`from_hmatvec` and never used. See
                "Retained rank".
            chunk: Row-chunk for the NW kernel sums (passed through to
                :func:`nw_conditional_means`).
            eig_floor: ``C~`` eigenvalues ``lambda_k = S_f^2 / N <= eig_floor``
                are treated as zero (their directions are dropped from the
                numerical null space of ``M_w``).
            dtype: Floating dtype for the sample-space linear algebra
                (float64; the thin SVD of ``M_w`` and the directions).
            device: Device for the returned directions. ``None`` -> the
                device of ``x_bank``.
            whiten: Whether the feature map ``phi`` prior-whitens. ``True``
                needs a dense ``Sigma_pr^{-1}`` (the densify path), so it is
                only available with a ``prior_cov_half`` AND ``dx <=``
                :data:`DENSE_SUBSPACE_CAP` (a ValueError above it directs the
                caller to ``whiten=False``, the seismic-scale default).

        Returns:
            An :class:`InformedSubspace` whose subspace matches
            :meth:`from_bank` to the finite-sample tolerance (a drop-in
            alongside it at scale).

        Raises:
            ValueError: If the bank shapes are inconsistent, ``sigma_y <= 0``,
                ``oversample < 0``, ``r + ...`` would exceed ``N``, the
                callables return the wrong shape, every ``C~`` eigenvalue is
                below ``eig_floor``, or ``whiten=True`` with ``dx >``
                :data:`DENSE_SUBSPACE_CAP`.
        """
        x = x_bank.double()
        y = y_bank.double()
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError(
                "x_bank and y_bank must be 2-D (N, dx) / (N, dy); got "
                f"{tuple(x.shape)} and {tuple(y.shape)}"
            )
        n, dx = x.shape
        if y.shape[0] != n:
            raise ValueError(
                f"x_bank and y_bank must share N; got N_x={n}, N_y="
                f"{y.shape[0]}"
            )
        if not sigma_y > 0:
            raise ValueError(f"sigma_y must be > 0; got {sigma_y!r}")
        if oversample < 0:
            raise ValueError(f"oversample must be >= 0; got {oversample}")
        # The centered empirical C~ has rank <= N - 1, so at most N - 1
        # directions are defined. Guard an EXPLICIT over-ask up front (the
        # documented contract, mirroring from_hmatvec's r + oversample <= dim).
        if r is not None and int(r) >= n:
            raise ValueError(
                f"r = {r} must be < N = {n} (the centered empirical C~ has "
                f"rank at most N - 1); reduce r or grow the bank."
            )
        dev = (
            torch.device(device) if device is not None else x.device
        )

        # 1. NW conditional means -- REUSE the existing chunked estimator
        #    (no N x N Gram materialized). (N, dx), float64.
        mu = nw_conditional_means(x, y, sigma_y, chunk=chunk)  # (N, dx)

        # 2. Prior-whiten each conditional mean m_i = Sigma_pr^{-1/2} mu_i,
        #    MATRIX-FREE row by row, then center -> M_w of shape (N, dx)
        #    (rows are the centered whitened conditional means).
        rows = []
        for i in range(n):
            mi = prior_cov_inv_half(mu[i]).reshape(-1)
            if mi.shape != (dx,):
                raise ValueError(
                    "prior_cov_inv_half must map a (dx,) vector to a (dx,) "
                    f"vector; got output of {mi.numel()} elements for dx={dx}"
                )
            rows.append(mi.to(dtype=dtype, device=dev))
        m_w = torch.stack(rows, dim=0)  # (N, dx) whitened conditional means
        m_w = m_w - m_w.mean(dim=0, keepdim=True)  # center

        # 3. Thin (economy) SVD of the centered whitened conditional means
        #    M_w (N, dx) DIRECTLY, M_w = W S_f V_f^T, to reach the leading
        #    eigenpairs of C~ = M_w^T M_w / N (the (dx, dx) C~ is NEVER formed)
        #    without forming the (N, N) Gram or squaring the condition number.
        #    The eigenvalues of C~ are exactly lam_k = S_f^2 / N (descending)
        #    and the whitened directions ARE the RIGHT singular vectors V_f
        #    (columns, unit-norm) -- the right singular vectors of M_w are by
        #    definition the eigenvectors of C~.
        _, s_f, vh_f = torch.linalg.svd(m_w, full_matrices=False)  # descending
        lam_all = s_f**2 / n  # eigenvalues of C~ = M_w^T M_w / N
        v_f = vh_f.transpose(0, 1)  # (dx, .) right singular vectors as columns
        # Drop near-zero modes (the centered C~ rank deficiency + noise floor):
        # they sit in the numerical null space of M_w and carry no direction.
        keep = lam_all > eig_floor
        n_keep = int(keep.sum())
        if n_keep == 0:
            raise ValueError(
                "all whitened-conditional-mean C~ eigenvalues are <= eig_floor="
                f"{eig_floor:g}; the whitened conditional means are (near-) "
                "constant -- the bank carries no informed signal (check "
                "sigma_y / the whitener / the bank)."
            )
        lam = lam_all[:n_keep]  # (n_keep,) descending, > floor

        # Retained rank: supplied (already < N), else participation ratio of
        # the C~ spectrum (same rule as the dense from_*),
        # clamped to the above-floor modes actually defined.
        if r is None:
            r = participation_rank(lam, cap=rank_cap)
        r = max(1, min(int(r), n_keep))

        # Whitened directions: the leading-r RIGHT singular vectors of M_w ==
        # leading eigenvectors of C~ = M_w^T M_w / N, already unit-norm.
        v = v_f[:, :r]  # (dx, r)
        eigvals = lam  # full above-floor spectrum used to pick r (descending)

        if prior_cov_half is None:
            # Whitened-only: ambient == whitened, sigma_pr=None (NEVER a dense
            # (dx, dx) prior), whiten=False -- the constructor forms no prior
            # (whitened_directions supplied, dense inverse skipped). Only the
            # metric-free attributes / subspace_angles_to are available.
            return cls(
                v, eigvals, None, whiten=False, whitened_directions=v,
            )

        # Ambient directions u_k = Sigma_pr^{1/2} v_k, MATRIX-FREE column by
        # column (each column a single (dx,) field), reusing the exact
        # un-whitening relation of _eig_from_whitened / from_hmatvec without
        # ever forming Sigma_pr^{1/2}.
        cols = []
        for j in range(r):
            uj = prior_cov_half(v[:, j]).reshape(-1)
            if uj.shape != (dx,):
                raise ValueError(
                    "prior_cov_half must map a (dx,) vector to a (dx,) "
                    f"vector; got output of {uj.numel()} elements for dx={dx}"
                )
            cols.append(uj.to(dtype=torch.float64, device=dev))
        u = torch.stack(cols, dim=1)  # (dx, r) ambient u_k

        # The dense whitened feature map needs Sigma_pr^{-1}; realize it only
        # when whiten is requested AND dx is small enough to densify
        # Sigma_pr^{1/2} (the same small-dim cross-check guard as from_hmatvec).
        if whiten:
            if dx > DENSE_SUBSPACE_CAP:
                raise ValueError(
                    f"whiten=True densifies Sigma_pr^{{1/2}} ({dx}x{dx}) and "
                    f"costs {dx} prior_cov_half applies; pass whiten=False at "
                    f"this scale (dx={dx} > cap {DENSE_SUBSPACE_CAP})"
                )
            spr_half = _dense_from_matvec(prior_cov_half, dx, dev)
            sigma_pr = _sym(spr_half @ spr_half.transpose(-1, -2))
            return cls(
                u, eigvals, sigma_pr, whiten=True, whitened_directions=v,
            )
        return cls(u, eigvals, None, whiten=False, whitened_directions=v)

    @classmethod
    def from_bank_regression(
        cls,
        x_bank: torch.Tensor,
        m_bank: torch.Tensor,
        *,
        prior_cov_inv_half,
        prior_cov_half=None,
        r: int | None = None,
        rank_cap: int | None = None,
        rank_floor: float = 1e-8,
        eig_floor: float = 1e-12,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        whiten: bool = True,
    ) -> "InformedSubspace":
        """MATRIX-FREE, low-rank, SCORE-FREE subspace by REDUCED-RANK REGRESSION.

        The regression / sliced-inverse-regression (SIR) twin of the kernel
        :meth:`from_bank_lowrank` -- INTERCHANGEABLE with the
        Nadaraya--Watson estimator, and the one that
        survives the curse of dimensionality where the NW kernel does not. It
        builds the SAME informed subspace (the leading eigenvectors of the
        whitened central-mean / score-free matrix ``C~ = Sigma_pr^{-1/2}
        Cov_m(E[x | m]) Sigma_pr^{-1/2}``, closed-form convention) but reaches
        ``E[x | m]`` by a LINEAR (reduced-rank) REGRESSION of the prior-whitened
        ``x`` on a CONDITIONING VARIABLE ``m`` -- NO kernel, NO distance, NO
        softmax. For a linear--Gaussian model ``E[x | m]`` is EXACTLY linear in
        ``m``, so the regression is unbiased: the leading-``r`` regressed
        directions equal the central-mean subspace with no kernel/bandwidth and
        -- crucially -- no DISTANCE CONCENTRATION. :meth:`from_bank_lowrank`'s
        softmax conditions on ``||m_i - m_j||`` in the high-dimensional ``m``
        space (for the seismic migration ``m = A^T y``, ``dm = dx ~ 3 x 10^3``),
        where pairwise distances concentrate, the responsibilities cannot
        discriminate, and the recovered ``C`` is tens of degrees off ``H``; the
        regression replaces that nearest-neighbour average with a global linear
        fit that is INVARIANT to ``dm``.

        Choice of ``m``. ``m`` is any conditioning variable for which the
        regressed ``E[x | m]`` spans the informed subspace -- in the seismic
        application the MIGRATION ``m = A^T y`` (or ``A^T Sigma_obs^{-1} y``),
        a ``dx``-dimensional summary of the data in PARAMETER space. (It need
        not be the raw ``y``; any sufficient linear readout works.) The method
        is agnostic to its meaning: it regresses ``x`` on whatever ``m`` is
        supplied.

        The algorithm (reduced-rank regression by two thin SVDs):

          1. Prior-whiten and center ``x``: ``xt_i = Sigma_pr^{-1/2} x_i``
             (apply ``prior_cov_inv_half`` ROW BY ROW, each row a single
             ``(dx,)`` field), then subtract the column mean -> ``Xt`` of shape
             ``(N, dx)``. Center ``m`` -> ``M`` of shape ``(N, dm)``.
          2. REDUCED-RANK LINEAR REGRESSION of ``Xt`` on ``M``. ``M^T M`` is
             rank ``<= N << dm``, so the fit is done in the SAMPLE / low-rank
             space -- NO ``(dm, dm)`` or ``(dx, dx)`` inverse. Thin (economy)
             SVD of ``M`` DIRECTLY, ``M = Q S P^T`` with ``Q`` the ``(N, k)``
             LEFT singular vectors and ``S`` the descending singular values,
             ``k = rank(M) <= N`` -- no ``N x N`` Gram ``M M^T`` is formed (the
             thin SVD is exact and does not square the condition number).
             Directions with ``S_j <= rank_floor * S_max`` are dropped
             (ridge-free reduced rank, the regularization of an ill-conditioned
             ``M``). The FITTED whitened conditional means are the projection of
             ``Xt`` onto the row-span of ``M``, ``Mu_fit = Q (Q^T Xt)`` of shape
             ``(N, dx)`` -- ``Q`` orthonormal, so ``Q Q^T`` is the orthogonal
             projector onto ``span`` of the ``m``-directions and ``Mu_fit_i =
             E[xt | m_i]`` for the linear model (each whitened-``x`` coordinate
             regressed on the ``k`` retained ``m``-coordinates).
          3. The whitened central-mean matrix is ``C~ = (1/N) Mu_fit^T Mu_fit``
             (``(dx, dx)``, NEVER formed); its leading-``r`` eigenpairs come from
             a thin (economy) SVD of ``Mu_fit`` DIRECTLY, ``Mu_fit = W S_f
             V_f^T`` -- no ``(N, N)`` Gram and no condition-number squaring. The
             eigenvalues of ``C~`` are exactly ``lambda_k = S_f^2 / N``
             (descending) and the whitened directions ARE the RIGHT singular
             vectors ``v_k = (V_f)_k`` (``(dx,)``, unit-norm); the right singular
             vectors of ``Mu_fit`` are by definition the eigenvectors of ``C~ =
             Mu_fit^T Mu_fit / N``, so NO ``Mu_fit^T w / sqrt(N lambda)``
             reconstruction is needed. Near-zero modes ``lambda_k <= eig_floor``
             are DROPPED, so ``r`` never exceeds the numerical rank of ``Mu_fit``
             (itself ``<= k = rank(M)``).
          4. Wrap as an :class:`InformedSubspace`: ``whitened_directions =
             [v_k]``, ``eigvals = [lambda_k]``, and -- if ``prior_cov_half`` is
             given -- the AMBIENT ``directions = Sigma_pr^{1/2} v_k`` produced
             column-by-column MATRIX-FREE, reusing the EXACT un-whitening
             relation ``u_k = Sigma_pr^{1/2} v_k`` of :func:`_eig_from_whitened`
             / :meth:`from_bank_lowrank`. With no ``prior_cov_half`` the object
             is WHITENED-ONLY (``sigma_pr=None``, the documented degenerate case
             valid for the prior-metric subspace diagnostics, which read only
             ``whitened_directions``).

        Why it recovers ``H`` in the linear--Gaussian case. With ``E[x | m]``
        linear in ``m``, ``Cov_m(E[x | m])`` is exactly the regression part of
        ``Cov(x)`` explained by ``m``; whitened, its leading eigenvectors are
        the central-mean (= central-subspace = LIS) directions ``V``. Unlike
        :meth:`from_bank_lowrank`, the only approximation is the FINITE-SAMPLE
        estimate of the (linear) regression -- there is no kernel-bandwidth bias
        and no distance concentration, so the recovered subspace matches the
        exact :meth:`from_H` to a small principal angle at moderate ``N`` even
        when ``dm`` is large (the regime where the NW route's softmax fails).

        Reduced-rank fit -- exactness. ``Q`` carries the FULL column space of
        ``M`` (every above-floor singular direction), so ``Q (Q^T Xt)`` is the
        EXACT least-squares fit of ``Xt`` on ``M`` (the ordinary multivariate
        regression, no truncation of the regressor rank beyond dropping ``M``'s
        numerical null space). The "reduced rank" is in the OUTPUT: the
        leading-``r`` eigendecomposition of the fitted-mean covariance keeps the
        ``r`` most-explained directions, exactly the reduced-rank-regression /
        SIR estimand.

        Peak memory ``O(N dx + N dm)``. NO ``(dx, dx)``, ``(dm, dm)``, or
        ``(N, N)`` Gram object is ever formed -- both factorizations are thin
        SVDs of the ``(N, .)`` data matrices directly. The largest arrays are
        ``Xt`` and ``Mu_fit`` of shape ``(N, dx)`` and the centered regressor
        ``M`` of shape ``(N, dm)``; the right singular vectors of ``Mu_fit`` ARE
        the directions, so no ``(dx, N) x (N, r)`` reconstruction product is
        formed. At seismic scale (``dx, dm ~ 3 x 10^3``, ``N ~ 2 x 10^3``) this
        is a few hundred MB, the same working set as :meth:`from_bank_lowrank`.

        Args:
            x_bank: Joint x-samples (flattened parameters), shape (N, dx).
            m_bank: Conditioning-variable samples (the regressor; e.g. the
                seismic migration ``m_i = A^T y_i``), shape (N, dm). ``dm`` need
                NOT equal ``dx``.
            prior_cov_inv_half: Callable ``v -> Sigma_pr^{-1/2} v`` on a single
                ``(dx,)`` vector -- the prior whitener applied to each ``x``
                row (step 1). Matrix-free (e.g. the spectral field's
                ``cov_inv_half`` adapted to flat ``(dx,)`` fields).
            prior_cov_half: Optional callable ``v -> Sigma_pr^{1/2} v`` on a
                single ``(dx,)`` vector, for the ambient un-whitening ``u_k =
                Sigma_pr^{1/2} v_k`` (step 4). If ``None``, the subspace is
                whitened-only (``sigma_pr=None``).
            r: Retained rank; the deliverable at seismic scale. ``None`` sets
                ``r`` from the participation ratio of the fitted-mean Gram
                spectrum (the same rule as the other ``from_*``).
            rank_cap: Optional cap on the auto-selected ``r``.
            rank_floor: RELATIVE floor on the regressor singular values
                ``S_j / S_max``: directions at or below it are dropped from the
                regression basis ``Q`` (ridge-free reduced rank, guarding an
                ill-conditioned ``M``). ``>= 0``.
            eig_floor: Fitted-mean ``C~`` eigenvalues ``lambda_k = S_f^2 / N``
                ``<= eig_floor`` are treated as zero (their directions are
                dropped from the numerical null space of ``Mu_fit``).
            dtype: Floating dtype for the sample-space linear algebra (float64;
                the whitening, the two thin SVDs, and the directions).
            device: Device for the returned directions. ``None`` -> the
                device of ``x_bank``.
            whiten: Whether the feature map ``phi`` prior-whitens. ``True``
                needs a dense ``Sigma_pr^{-1}`` (the densify path), so it is
                only available with a ``prior_cov_half`` AND ``dx <=``
                :data:`DENSE_SUBSPACE_CAP` (a ValueError above it directs the
                caller to ``whiten=False``, the seismic-scale default).

        Returns:
            An :class:`InformedSubspace` whose subspace matches the score-free
            :meth:`from_bank` / exact :meth:`from_H` to the finite-sample
            tolerance (a drop-in alongside :meth:`from_bank_lowrank`, the kernel
            route it is interchangeable with -- and superior to at large ``dm``).

        Raises:
            ValueError: If the bank shapes are inconsistent, ``r >= N`` (the
                centered fitted-mean ``C~`` has rank at most ``N - 1``),
                ``rank_floor < 0``, the callables return the wrong shape, every
                regressor singular value is below ``rank_floor * S_max`` or every
                ``C~`` eigenvalue is below ``eig_floor``, or ``whiten=True`` with
                ``dx >`` :data:`DENSE_SUBSPACE_CAP`.
        """
        x = x_bank.double()
        m = m_bank.double()
        if x.dim() != 2 or m.dim() != 2:
            raise ValueError(
                "x_bank and m_bank must be 2-D (N, dx) / (N, dm); got "
                f"{tuple(x.shape)} and {tuple(m.shape)}"
            )
        n, dx = x.shape
        if m.shape[0] != n:
            raise ValueError(
                f"x_bank and m_bank must share N; got N_x={n}, N_m="
                f"{m.shape[0]}"
            )
        if rank_floor < 0:
            raise ValueError(f"rank_floor must be >= 0; got {rank_floor!r}")
        # The centered fitted-mean C~ has rank <= N - 1, so at most N - 1
        # directions are defined. Guard an EXPLICIT over-ask up front (the
        # documented contract, mirroring from_bank_lowrank).
        if r is not None and int(r) >= n:
            raise ValueError(
                f"r = {r} must be < N = {n} (the centered fitted-mean C~ has "
                f"rank at most N - 1); reduce r or grow the bank."
            )
        dev = torch.device(device) if device is not None else x.device

        # 1. Prior-whiten each x row, xt_i = Sigma_pr^{-1/2} x_i, MATRIX-FREE
        #    row by row, then center -> Xt (N, dx). Center m -> M (N, dm).
        rows = []
        for i in range(n):
            xi = prior_cov_inv_half(x[i]).reshape(-1)
            if xi.shape != (dx,):
                raise ValueError(
                    "prior_cov_inv_half must map a (dx,) vector to a (dx,) "
                    f"vector; got output of {xi.numel()} elements for dx={dx}"
                )
            rows.append(xi.to(dtype=dtype, device=dev))
        xt = torch.stack(rows, dim=0)  # (N, dx) whitened x
        xt = xt - xt.mean(dim=0, keepdim=True)  # center
        mc = m.to(dtype=dtype, device=dev)
        mc = mc - mc.mean(dim=0, keepdim=True)  # (N, dm) centered regressor

        # 2. Reduced-rank regression of Xt on M, done in the N x N sample space.
        #    Thin (economy) SVD of M DIRECTLY -- M = Q S P^T with Q the (N, k)
        #    LEFT singular vectors and S the descending singular values -- no
        #    (N, N) Gram formed and no condition-number squaring. A RELATIVE
        #    floor on S drops M's numerical null space (ridge-free reduced rank).
        #    Q (N, k) is an orthonormal basis for span of the m-directions; Q Q^T
        #    projects onto it, so the fitted whitened means are Mu_fit = Q (Q^T
        #    Xt) (N, dx).
        q_all, s_all, _ = torch.linalg.svd(mc, full_matrices=False)  # descending
        s_max = float(s_all[0]) if s_all.numel() > 0 else 0.0
        keep_m = s_all > rank_floor * s_max
        k = int(keep_m.sum())
        if k == 0:
            raise ValueError(
                "all regressor singular values are <= rank_floor * S_max "
                f"(rank_floor={rank_floor:g}); the centered conditioning "
                "variable M is (near-) constant -- no regression direction "
                "(check m_bank / rank_floor)."
            )
        q = q_all[:, :k]  # (N, k) left singular vectors of M (orthonormal)
        mu_fit = q @ (q.transpose(0, 1) @ xt)  # (N, dx) = Q Q^T Xt = E[xt | m]

        # 3. Leading-r eigenpairs of C~ = (1/N) Mu_fit^T Mu_fit from a thin
        #    (economy) SVD of Mu_fit DIRECTLY (no (N, N) Gram, no condition-number
        #    squaring): Mu_fit = W S_f V_f^T. The eigenvalues of C~ are exactly
        #    lam_k = S_f^2 / N (descending) and the whitened directions ARE the
        #    RIGHT singular vectors V_f (columns, unit-norm) -- no
        #    Mu_fit^T w / sqrt(N lambda) reconstruction needed.
        _, s_f, vh_f = torch.linalg.svd(mu_fit, full_matrices=False)  # descending
        lam_all = s_f**2 / n  # eigenvalues of C~ = Mu_fit^T Mu_fit / N
        v_f = vh_f.transpose(0, 1)  # (dx, .) right singular vectors as columns
        keep = lam_all > eig_floor
        n_keep = int(keep.sum())
        if n_keep == 0:
            raise ValueError(
                "all fitted-mean C~ eigenvalues are <= eig_floor="
                f"{eig_floor:g}; the regressed whitened means are (near-) "
                "constant -- the regression of x on m carries no informed "
                "signal (check m_bank / the whitener / rank_floor)."
            )
        lam = lam_all[:n_keep]  # (n_keep,) descending, > floor

        # Retained rank: supplied (already < N), else participation ratio of the
        # fitted-mean C~ spectrum (same rule as the other
        # from_*), clamped to the above-floor modes actually defined.
        if r is None:
            r = participation_rank(lam, cap=rank_cap)
        r = max(1, min(int(r), n_keep))

        # Whitened directions: the leading-r RIGHT singular vectors of Mu_fit ==
        # leading eigenvectors of C~ = Mu_fit^T Mu_fit / N, already unit-norm.
        v = v_f[:, :r]  # (dx, r)
        eigvals = lam  # full above-floor spectrum used to pick r (descending)

        if prior_cov_half is None:
            # Whitened-only: ambient == whitened, sigma_pr=None (NEVER a dense
            # (dx, dx) prior), whiten=False -- the constructor forms no prior
            # (whitened_directions supplied, dense inverse skipped). Only the
            # metric-free attributes / subspace_angles_to are available.
            return cls(
                v, eigvals, None, whiten=False, whitened_directions=v,
            )

        # Ambient directions u_k = Sigma_pr^{1/2} v_k, MATRIX-FREE column by
        # column (each column a single (dx,) field), reusing the exact
        # un-whitening relation of _eig_from_whitened / from_bank_lowrank
        # without ever forming Sigma_pr^{1/2}.
        cols = []
        for j in range(r):
            uj = prior_cov_half(v[:, j]).reshape(-1)
            if uj.shape != (dx,):
                raise ValueError(
                    "prior_cov_half must map a (dx,) vector to a (dx,) "
                    f"vector; got output of {uj.numel()} elements for dx={dx}"
                )
            cols.append(uj.to(dtype=torch.float64, device=dev))
        u = torch.stack(cols, dim=1)  # (dx, r) ambient u_k

        # The dense whitened feature map needs Sigma_pr^{-1}; realize it only
        # when whiten is requested AND dx is small enough to densify
        # Sigma_pr^{1/2} (the same small-dim cross-check guard as from_hmatvec
        # / from_bank_lowrank).
        if whiten:
            if dx > DENSE_SUBSPACE_CAP:
                raise ValueError(
                    f"whiten=True densifies Sigma_pr^{{1/2}} ({dx}x{dx}) and "
                    f"costs {dx} prior_cov_half applies; pass whiten=False at "
                    f"this scale (dx={dx} > cap {DENSE_SUBSPACE_CAP})"
                )
            spr_half = _dense_from_matvec(prior_cov_half, dx, dev)
            sigma_pr = _sym(spr_half @ spr_half.transpose(-1, -2))
            return cls(
                u, eigvals, sigma_pr, whiten=True, whitened_directions=v,
            )
        return cls(u, eigvals, None, whiten=False, whitened_directions=v)

    def subspace_angles_to(self, other: "InformedSubspace") -> torch.Tensor:
        """Principal angles (radians) to another subspace, in the prior metric.

        Compares the two subspaces through their WHITENED directions ``v_i``
        (orthonormal in the prior metric), so the C--H equivalence
        reads near-zero angles when it should. Returns
        ``min(r_self, r_other)`` angles, ascending.
        """
        return principal_angles(
            self.whitened_directions, other.whitened_directions
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map ``phi(x)``, shape (..., r), in the dtype of ``x``.

        ``phi(x) = x @ W`` with ``W = Sigma_pr^{-1} U`` (whitened) or
        ``W = U`` (raw). Accepts (B, dx) or (dx,); preserves the input dtype
        so it slots into the float32 embedding path.
        """
        w = self._w.to(x.dtype).to(x.device)
        single = x.dim() == 1
        xb = x.unsqueeze(0) if single else x
        out = xb @ w  # (B, r)
        return out.squeeze(0) if single else out

    def residual_fraction_off_subspace(
        self, sigma_post: torch.Tensor, h_vec: torch.Tensor
    ) -> float:
        """``rho_perp``: the off-subspace posterior variance fraction of ``h``.

        For a LINEAR functional ``h(x) = h_vec^T x``, the posterior variance
        ``h_vec^T Sigma_post h_vec`` splits along the informed subspace and
        its complement; ``rho_perp`` is the fraction carried by the
        complement (the directions ``phi`` discards). With ``P`` the
        Sigma_pr-orthogonal projector onto the informed subspace
        ``span(U)`` (``P = U (U^T Sigma_pr^{-1} U)^{-1} U^T Sigma_pr^{-1}``),
        ``h_perp = (I - P)^T h_vec`` and

            rho_perp = (h_perp^T Sigma_post h_perp) / (h^T Sigma_post h).

        It is near 0 for a wedge-aligned functional and large for an
        off-wedge one. Returns a Python float.
        """
        if self.sigma_pr is None:
            raise ValueError(
                "residual_fraction_off_subspace needs the prior metric "
                "Sigma_pr (the Sigma_pr-orthogonal projector onto span(U)), "
                "but sigma_pr is None (built matrix-free without a prior "
                "metric); build this subspace with a prior_cov_half / dense "
                "Sigma_pr to use this diagnostic."
            )
        spost = sigma_post.double()
        hv = h_vec.double().reshape(-1)
        u = self.directions  # (dx, r)
        spr_inv = torch.linalg.inv(self.sigma_pr)
        # Sigma_pr-orthogonal projector onto span(U).
        g = u.T @ spr_inv @ u  # (r, r)
        p = u @ torch.linalg.inv(g) @ u.T @ spr_inv  # (dx, dx)
        h_perp = hv - p.T @ hv  # complement component of the functional
        denom = float(hv @ spost @ hv)
        if denom <= 0:
            return 0.0
        num = float(h_perp @ spost @ h_perp)
        return max(0.0, min(1.0, num / denom))


class InformedYSummary:
    """An informed y-summary map ``y -> U_r^T Sigma_obs^{-1/2} (y - y_bar)``.

    The y-side twin of :class:`InformedSubspace`. The Nadaraya--Watson
    conditioning of
    :class:`ConditionalDataDrivenEmbeddings` measures ``||y_i - y*||`` in the
    RAW data space, whose dimension ``d_y`` includes the noise-only
    directions the data does not inform; the NW responsibilities therefore
    average over irrelevant coordinates and the conditional-mean bias walls
    with ``d_y``. Conditioning instead on the informed y-summary

        s(y) = U_{r_y}^T Sigma_obs^{-1/2} (y - y_bar),    s in R^{r_y},

    collapses the conditioning onto the ``r_y`` noise-whitened data
    directions the parameters actually drive -- the LEFT singular vectors
    ``U`` of the whitened forward map ``B = Sigma_obs^{-1/2} A Sigma_pr^{1/2}
    = U Sigma V^T``. Plugged into ``y_feature_map`` of the
    conditional embeddings it makes the NW kernel act on ``s(y)``.

    Two construction routes, exactly mirroring the x-side exact/score-free
    split (and recovering the SAME ``U`` in the linear--Gaussian case):

      * :meth:`from_forward` (EXACT, closed form): ``U`` = leading
        eigenvectors of the y-side information matrix
        ``B B^T = Sigma_obs^{-1/2} A Sigma_pr A^T Sigma_obs^{-1/2}``
        (:func:`whitened_y_information_matrix`), eigenvalues the LIS spectrum
        ``sigma_k^2``.
      * :meth:`from_bank` (SCORE-FREE, from the bank): ``U`` = leading LEFT
        singular vectors of the whitened cross-covariance
        ``Sigma_obs^{-1/2} Cov_hat(y, x) Sigma_pr^{-1/2}``
        (:func:`whitened_cross_cov_from_bank`), the y-loadings of the inverse
        regression between x and y. Reads only ``(x_i, y_i)`` -- no forward
        solve.

    The retained rank ``r_y`` is supplied or set by the participation ratio
    of the y-side spectrum (the same rule as the x-side,
    reported in ``self.rank``). The shift ``y_bar`` is the population data
    mean ``A mu_pr`` when available (``from_forward`` with a prior mean) or
    the empirical bank mean (``from_bank``); it is a harmless constant offset
    in a translation-invariant SE kernel but is kept so ``s`` is centered.

    Attributes:
        directions: The leading-``r_y`` informed y-directions ``U`` as
            columns in the NOISE-WHITENED data space, shape (dy, r_y),
            float64. (These are orthonormal: ``U^T U = I``.)
        eigvals: The descending y-side spectrum used to pick ``r_y``.
        rank: The retained y-summary dimension ``r_y``.
        sobs_inv_half: ``Sigma_obs^{-1/2}`` (the noise whitening).
        y_bar: The data-mean shift ``y_bar``, shape (dy,).
    """

    def __init__(
        self,
        directions: torch.Tensor,
        eigvals: torch.Tensor,
        sobs_inv_half: torch.Tensor,
        y_bar: torch.Tensor,
    ):
        self.directions = directions.double()  # U (dy, r_y), orthonormal
        self.eigvals = eigvals.double()
        self.rank = directions.shape[1]
        self.sobs_inv_half = sobs_inv_half.double()
        self.y_bar = y_bar.double().reshape(-1)
        # Summary map matrix W_y (dy, r_y) so s(y) = (y - y_bar) @ W_y, with
        # W_y = Sigma_obs^{-1/2} U (since s = U^T Sigma_obs^{-1/2} (y-y_bar)
        # and Sigma_obs^{-1/2} is symmetric).
        self._w = self.sobs_inv_half @ self.directions  # (dy, r_y)

    @classmethod
    def from_forward(
        cls,
        forward: torch.Tensor,
        obs_cov: torch.Tensor,
        prior_cov: torch.Tensor,
        prior_mean: torch.Tensor | None = None,
        r: int | None = None,
        rank_cap: int | None = None,
    ) -> "InformedYSummary":
        """EXACT informed y-summary from the forward operator (closed form).

        ``U`` = leading eigenvectors of the y-side information matrix
        ``B B^T`` (:func:`whitened_y_information_matrix`); exact on the
        linear--Gaussian targets. ``r`` defaults to the
        participation ratio of the y-side spectrum.
        """
        m_y = whitened_y_information_matrix(forward, obs_cov, prior_cov)
        w, v = torch.linalg.eigh(m_y)  # ascending
        order = torch.argsort(w, descending=True)
        w = w[order]
        v = v[:, order]  # columns = U (orthonormal)
        if r is None:
            r = participation_rank(w, cap=rank_cap)
        r = max(1, min(r, v.shape[1]))
        _, sobs_inv_half = _matrix_sqrt_and_inv_sqrt(obs_cov.double())
        a = forward.double()
        if prior_mean is None:
            y_bar = torch.zeros(a.shape[0], dtype=torch.float64)
        else:
            y_bar = a @ prior_mean.double().reshape(-1)
        return cls(v[:, :r], w, sobs_inv_half, y_bar)

    @classmethod
    def from_bank(
        cls,
        x_bank: torch.Tensor,
        y_bank: torch.Tensor,
        obs_cov: torch.Tensor,
        prior_cov: torch.Tensor,
        r: int | None = None,
        rank_cap: int | None = None,
    ) -> "InformedYSummary":
        """SCORE-FREE informed y-summary from the joint bank (no forward solve).

        ``U`` = leading LEFT singular vectors of the whitened cross-covariance
        ``Sigma_obs^{-1/2} Cov_hat(y, x) Sigma_pr^{-1/2}``
        (:func:`whitened_cross_cov_from_bank`); the y-loadings of the inverse
        regression. The shift ``y_bar`` is the empirical bank mean of ``y``.
        Recovers the exact ``U`` of :meth:`from_forward` in the
        linear--Gaussian case. ``r`` defaults to the participation ratio of
        the singular-value-squared spectrum.
        """
        b_hat = whitened_cross_cov_from_bank(
            x_bank, y_bank, obs_cov, prior_cov
        )
        # Left singular vectors of B_hat (the y-loadings); s_k^2 is the
        # y-side spectrum, matching B B^T's eigenvalues.
        u_left, s, _ = torch.linalg.svd(b_hat, full_matrices=False)
        spectrum = s.double() ** 2
        if r is None:
            r = participation_rank(spectrum, cap=rank_cap)
        r = max(1, min(r, u_left.shape[1]))
        _, sobs_inv_half = _matrix_sqrt_and_inv_sqrt(obs_cov.double())
        y_bar = y_bank.double().mean(dim=0)
        return cls(u_left[:, :r], spectrum, sobs_inv_half, y_bar)

    def subspace_angles_to(self, other: "InformedYSummary") -> torch.Tensor:
        """Principal angles (radians) between the two y-subspaces ``U``.

        Both ``directions`` are orthonormal bases in the noise-whitened data
        space, so the plain Euclidean :func:`principal_angles` is the correct
        comparison (no metric reweighting needed, unlike the x-side prior
        metric). Returns ``min(r_self, r_other)`` angles, ascending.
        """
        return principal_angles(self.directions, other.directions)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """The y-summary ``s(y) = U^T Sigma_obs^{-1/2} (y - y_bar)``.

        ``s(y) = (y - y_bar) @ W_y`` with ``W_y = Sigma_obs^{-1/2} U``.
        Accepts (B, dy) or (dy,); preserves the input dtype so it slots into
        the float32 conditioning path.
        """
        w = self._w.to(y.dtype).to(y.device)
        y_bar = self.y_bar.to(y.dtype).to(y.device)
        single = y.dim() == 1
        yb = y.unsqueeze(0) if single else y
        out = (yb - y_bar.unsqueeze(0)) @ w  # (B, r_y)
        return out.squeeze(0) if single else out
