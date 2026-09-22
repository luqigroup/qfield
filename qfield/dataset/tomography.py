"""Stylized limited-angle tomography as a linear-Gaussian inverse problem.

The unknown is a discretized image ``x in R^{dx}`` (an ``n x n`` pixel field,
``dx = n^2``) under a smoothing Gaussian field prior; the forward model is the
parallel-beam Radon transform (line integrals of the image) restricted to a
limited angular range ``[-Phi, Phi]`` with ``Phi < 90 deg``, the missing
wedge. With Gaussian observation noise the posterior is an exact Gaussian, so
the whole construction is handed to :class:`LinearGaussian` (joint sampling,
the closed-form posterior, the analytic MMD) by supplying ``(prior_mean=0,
prior_cov=Sigma_pr, forward=A_Phi, obs_cov=Sigma_obs)``.

Why limited angle. By the Fourier-slice theorem each projection at angle
``theta`` samples the image's Fourier transform along the radial line at
angle ``theta``; a limited angular range therefore informs only a wedge of
frequency space and leaves the complementary frequencies essentially
unconstrained. The likelihood-informed subspace
``A_Phi^T Sigma_obs^{-1} A_Phi`` is thus geometrically explicit, the
visible-wedge image modes, and its rank, the informed dimension, is a knob:
tightening the wedge (smaller ``Phi``), using fewer angles, or raising the
noise all shrink it. The structure emerges from the geometry rather than
being planted.

The forward operator is a self-contained sparse parallel-beam projector.
``A_Phi`` is built directly as the ``(dy, dx)`` matrix of line-integral
weights: for each projection angle ``theta`` in ``[-Phi, Phi]`` and each
detector offset ``s`` on a 1-D detector array, the ray
``{ center + s * n_hat + t * d_hat : t }`` (with ``n_hat`` perpendicular and
``d_hat`` along the beam) is marched in small steps and each sample point
deposits a bilinear (area) weight onto its four surrounding pixels. Summing
the per-step contributions and scaling by the step length gives a discrete
line integral; row ``(theta, s)`` of ``A_Phi`` is the resulting pixel-weight
vector. It is a basic discretization of the Radon transform, and it is dense
because the grids here are small (``n in {12..32}``) and the closed-form
posterior machinery of :class:`LinearGaussian` wants dense ``A``,
``Sigma_pr`` and ``Sigma_obs`` anyway.

The smoothing prior is a Laplacian / Whittle-Matern-type field:
``Sigma_pr = tau^2 (delta I - Delta)^{-alpha}`` with ``Delta`` the 5-point
discrete Laplacian on the grid (Neumann / reflecting boundary). This is the
standard Whittle--Matern stationary GP field prior: ``(delta I -
Delta)^{-alpha}`` penalizes roughness (high spatial frequencies), ``delta
> 0`` sets the correlation length (``ell ~ 1/sqrt(delta)`` in pixel units),
``alpha`` the smoothness, and ``tau^2`` the marginal scale. It is SPD for any
``delta > 0`` and concentrates prior mass on smooth (low-frequency) images,
complementary to the high-frequency missing wedge.

References:
  - Cui, Law, Marzouk, "Dimension-independent likelihood-informed MCMC"
    and Zahm et al., "Certified dimension reduction" (the likelihood-informed
    subspace this target instantiates).
  - Belhadji, Sharp, Marzouk, "To discretize continually: ...",
    arXiv:2605.14142.
"""

from __future__ import annotations

import math

import torch

from qfield.dataset.linear_gaussian import LinearGaussian
from qfield.dataset.spectral_gaussian_field import (
    SpectralGaussianField,
    _flip_both_axes,
)


def stationary_grf_field(
    n: int,
    corr_len: float = 6.0,
    alpha: float = 2.0,
    tau: float = 1.0,
    floor: float = 1e-3,
    dtype: torch.dtype = torch.float64,
) -> SpectralGaussianField:
    """Stationary isotropic Whittle--Matern GRF on the periodic n x n grid.

    Returns a :class:`SpectralGaussianField` whose per-mode variance is the
    PERIODIC (circulant) isotropic Whittle--Matern spectrum

        S(k) = 1 / (delta + |k|^2)^alpha,    delta = (n / corr_len)^2,

    with ``|k|^2 = m1^2 + m2^2`` the squared INTEGER wavenumber (the DFT mode
    index folded to ``[-n/2, n/2)`` on each axis, so a mode of index ``m`` has
    spatial wavelength ``n / |m|`` pixels). Choosing ``delta = (n /
    corr_len)^2`` rolls the spectrum off around the mode ``|m| ~ n / corr_len``,
    i.e. a correlation length of ``corr_len`` PIXELS; ``alpha`` sets the
    smoothness. The spectrum is even-symmetrized to machine precision (so the
    covariance is real), floored for conditioning, and then RESCALED so the
    field's marginal (per-pixel) variance equals ``tau^2`` exactly --
    ``Var(x_p) = mean_k S_k`` for the orthonormal-DFT diagonal covariance, and
    it is the SAME at every pixel because the field is stationary (this is
    precisely what removes the Neumann-Laplacian prior's boundary
    variance artifact).

    Unlike :func:`laplacian_field_prior_cov` (a Neumann / reflecting-boundary
    Whittle--Matern, whose marginal variance swells at the grid boundary), this
    is the genuinely SHIFT-INVARIANT stationary field: its per-pixel prior
    variance is UNIFORM, so the resulting posterior uncertainty map reflects
    the DATA's limited-angle resolution rather than a boundary effect. It is the
    same in-repo construction used for the seismic reflectivity prior
    (:class:`SpectralGaussianField`), here with a PARAMETRIC isotropic spectrum
    rather than a data-driven one.

    Args:
        n: Grid side length (the image is ``n x n``, ``dx = n^2``).
        corr_len: Correlation length in PIXELS (sets ``delta = (n /
            corr_len)^2``); larger -> smoother / longer-range fields.
        alpha: Smoothness exponent ``alpha > 0`` (the spectral decay power).
        tau: Marginal standard deviation (the field is rescaled so every
            pixel has prior variance ``tau^2``).
        floor: Relative spectral floor ``S = max(S, floor * S.max())`` for
            conditioning / invertibility; must be in ``(0, 1)``.
        dtype: Field dtype for samples / operator outputs.

    Returns:
        The configured :class:`SpectralGaussianField`; call
        :meth:`~SpectralGaussianField.dense_prior_cov` for the dense
        ``Sigma_pr`` and :meth:`~SpectralGaussianField.sample` for a prior
        draw consistent with that covariance.
    """
    if corr_len <= 0.0 or alpha <= 0.0 or tau <= 0.0:
        raise ValueError(
            f"corr_len, alpha, tau must be > 0; got {corr_len}, {alpha}, {tau}"
        )
    if not (0.0 < floor < 1.0):
        raise ValueError(f"floor must be in (0, 1); got {floor!r}")
    idx = torch.arange(n)
    # DFT mode index folded to [-n/2, n/2) per axis (fftfreq * n), so |k|^2 is
    # even in the mode index and the spectrum is even-symmetric by parity.
    m = ((idx + n // 2) % n) - n // 2
    m1 = m.reshape(n, 1).double()
    m2 = m.reshape(1, n).double()
    delta = (n / corr_len) ** 2
    s = 1.0 / (delta + m1 * m1 + m2 * m2) ** alpha
    # Establish exact even symmetry S[-k] = S[k] (machine-precision; makes the
    # covariance real), then floor for conditioning.
    s = 0.5 * (s + _flip_both_axes(s))
    s = s.clamp_min(floor * s.max())
    # Rescale so the marginal per-pixel variance is exactly tau^2 (uniform,
    # since the field is stationary). mean_k S_k = trace(C)/N = Var(x_p).
    s = s * (tau * tau / s.mean())
    return SpectralGaussianField(s, dtype=dtype)


def dense_cov_from_field(
    field: SpectralGaussianField, dim: int | None = None
) -> torch.Tensor:
    """Dense ``Sigma_pr`` of a :class:`SpectralGaussianField`, NO size cap.

    The cap-free companion to
    :meth:`SpectralGaussianField.dense_prior_cov`. It builds the SAME dense
    covariance ``C = Finv diag(S) F`` -- column ``m`` is the matrix-free
    ``field.cov(e_m)`` applied to the ``m``-th canonical pixel basis vector --
    by batching ``cov`` over the ``dim x dim`` identity reshaped to ``(dim,
    n1, n2)`` (``dim`` cheap exact FFT applications, ``O(dim^2 log dim)``
    time / ``O(dim^2)`` memory, NO ``O(dim^3)`` eig and NO dense factorization).
    It is symmetric PSD with eigenvalues the ``S_k``.

    Unlike :meth:`~SpectralGaussianField.dense_prior_cov` (hard-capped at
    :attr:`SpectralGaussianField.DENSE_CAP` = 4096 = 64x64), this is
    unguarded, for callers that need the dense ``Sigma_pr`` on a grid finer
    than 64x64 (for example ``n = 96``, ``dim = 9216``) to feed the dense
    :class:`~qfield.dataset.linear_gaussian.LinearGaussian` /
    :mod:`~qfield.subspace` machinery; forming it via ``cov`` is exact and
    fast. The per-pixel variance ``diag(C)`` is exactly uniform
    (``= tau^2``), since the field is stationary.

    Args:
        field: The stationary :class:`SpectralGaussianField`.
        dim: Expected flat dimension ``n1 * n2`` (validated if given; defaults
            to the field's own ``n1 * n2``).

    Returns:
        ``Sigma_pr`` of shape ``(dim, dim)``, float64, on the field device.
    """
    n = field.n1 * field.n2
    if dim is not None and int(dim) != n:
        raise ValueError(
            f"dim={dim} does not match field grid n1*n2={n}"
        )
    # Apply cov to all N basis columns at once (batched): identity reshaped to
    # (N, n1, n2). cov is linear, so row m is (C e_m) flattened = column m of C.
    eye = torch.eye(n, dtype=torch.float64, device=field.device)
    basis = eye.reshape(n, field.n1, field.n2)
    cols = field.cov(basis).reshape(n, n)  # row m = (C e_m) flat = col m of C
    return 0.5 * (cols + cols.T)  # symmetrize round-off (C is exactly symmetric)


def laplacian_field_prior_cov(
    n: int,
    delta: float = 0.5,
    alpha: float = 2.0,
    tau: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Whittle--Matern-type smoothing field prior covariance on an n x n grid.

    Returns ``Sigma_pr = tau^2 (delta I - Delta)^{-alpha}`` of shape
    ``(n^2, n^2)``, with ``Delta`` the 5-point discrete Laplacian under
    Neumann (reflecting) boundary conditions (so the precision is a proper
    SPD operator for ``delta > 0``). Larger ``delta`` -> shorter
    correlation length; larger ``alpha`` -> smoother samples; ``tau``
    scales the marginal standard deviation.

    The Laplacian is assembled densely (small grids), symmetrized for
    numerical safety, and raised to ``-alpha`` through its eigendecomposition
    so non-integer ``alpha`` is allowed. Computed in float64.

    Args:
        n: Grid side length (the image is ``n x n``, ``dx = n^2``).
        delta: Zeroth-order term ``delta > 0`` (sets the correlation length).
        alpha: Smoothness exponent ``alpha > 0`` (the matrix power).
        tau: Marginal scale ``tau > 0``.
        dtype: Working dtype (float64 recommended for the matrix power).

    Returns:
        SPD covariance ``Sigma_pr`` of shape ``(n^2, n^2)``.
    """
    if delta <= 0.0 or alpha <= 0.0 or tau <= 0.0:
        raise ValueError(
            f"delta, alpha, tau must be > 0; got {delta}, {alpha}, {tau}"
        )
    d = n * n
    # 5-point Laplacian with Neumann BC: a pixel's diagonal counts its
    # number of in-grid neighbours; off-diagonals are -1 for 4-adjacency.
    lap = torch.zeros(d, d, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            p = i * n + j
            deg = 0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    lap[p, ii * n + jj] = -1.0
                    deg += 1
            lap[p, p] = float(deg)
    # Precision A0 = delta I + Laplacian (note -Delta = +lap here, since
    # ``lap`` is the positive-semidefinite graph Laplacian = -Delta).
    a0 = delta * torch.eye(d, dtype=torch.float64) + lap
    a0 = 0.5 * (a0 + a0.T)
    evals, evecs = torch.linalg.eigh(a0)
    evals = evals.clamp_min(1e-12)
    # Sigma_pr = tau^2 A0^{-alpha} = tau^2 V diag(evals^{-alpha}) V^T.
    cov = (evecs * (tau**2) * evals.pow(-alpha)) @ evecs.T
    cov = 0.5 * (cov + cov.T)
    return cov.to(dtype)


def parallel_beam_radon_operator(
    n: int,
    n_angles: int = 12,
    phi_deg: float = 60.0,
    n_detectors: int | None = None,
    n_steps: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-contained sparse parallel-beam Radon operator on an n x n grid.

    Builds the dense ``(dy, dx)`` matrix ``A_Phi`` whose row ``(theta, s)``
    holds the pixel weights of the line integral along the ray at angle
    ``theta`` and detector offset ``s``. Angles are ``n_angles`` equally
    spaced values in the LIMITED range ``[-phi_deg, +phi_deg]`` degrees, so
    ``dy = n_angles * n_detectors``.

    Discretization. The image is the unit square ``[-1, 1]^2`` with pixel
    centers on a regular grid. For angle ``theta`` let ``d_hat =
    (cos theta, sin theta)`` (beam direction) and ``n_hat = (-sin theta,
    cos theta)`` (detector axis). The ray for detector offset ``s`` is
    ``s n_hat + t d_hat``; we march ``t`` across the square in ``n_steps``
    steps, and at each sample deposit a bilinear weight onto the four
    surrounding pixels. The row is the step-length-scaled sum of those
    contributions (a Riemann approximation of the line integral). Detector
    offsets span ``[-1, 1]`` so every ray crosses the inscribed disk.

    Args:
        n: Grid side length.
        n_angles: Number of projection angles in ``[-phi_deg, phi_deg]``.
        phi_deg: Half-range of the limited angular wedge, in degrees
            (``< 90``); the informed-dimension knob.
        n_detectors: Detector bins per angle (default ``n``).
        n_steps: Ray-march steps (default ``2 n``, fine enough for these
            grids).
        dtype: Output dtype.

    Returns:
        (A [dy, dx], angles_rad [n_angles]): the Radon matrix and the
        angle grid (radians) used, for diagnostics / wedge-aligned bases.
    """
    if not (0.0 < phi_deg < 90.0):
        raise ValueError(f"phi_deg must be in (0, 90); got {phi_deg}")
    if n_detectors is None:
        n_detectors = n
    if n_steps is None:
        n_steps = 2 * n

    d_x = n * n
    d_y = n_angles * n_detectors

    # Pixel-center coordinates on [-1, 1]^2 (row i -> y, col j -> x), and the
    # grid spacing so we can map a continuous point to fractional indices.
    coords = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    h = float(coords[1] - coords[0]) if n > 1 else 2.0  # pixel pitch
    if n_angles > 1:
        angles = torch.linspace(
            -math.radians(phi_deg), math.radians(phi_deg), n_angles,
            dtype=torch.float64,
        )
    else:
        angles = torch.zeros(1, dtype=torch.float64)
    detectors = torch.linspace(-1.0, 1.0, n_detectors, dtype=torch.float64)
    # March t across the square diagonal so rays fully traverse it.
    t_grid = torch.linspace(
        -math.sqrt(2.0), math.sqrt(2.0), n_steps, dtype=torch.float64
    )
    dt = float(t_grid[1] - t_grid[0]) if n_steps > 1 else 1.0

    a = torch.zeros(d_y, d_x, dtype=torch.float64)
    row = 0
    for theta in angles:
        d_hat = torch.tensor(
            [math.cos(theta), math.sin(theta)], dtype=torch.float64
        )
        n_hat = torch.tensor(
            [-math.sin(theta), math.cos(theta)], dtype=torch.float64
        )
        for s in detectors:
            base = s * n_hat  # (2,)
            # Sample points along the ray: (n_steps, 2).
            pts = base.unsqueeze(0) + t_grid.unsqueeze(1) * d_hat.unsqueeze(0)
            # Map physical (px, py) to fractional grid indices.
            fx = (pts[:, 0] - coords[0]) / h  # column index (x)
            fy = (pts[:, 1] - coords[0]) / h  # row index (y)
            x0 = torch.floor(fx).long()
            y0 = torch.floor(fy).long()
            wx = fx - x0.double()
            wy = fy - y0.double()
            # Bilinear deposit onto the 4 surrounding pixels (row = y, col = x).
            for dy_, dx_, weight in (
                (0, 0, (1 - wy) * (1 - wx)),
                (0, 1, (1 - wy) * wx),
                (1, 0, wy * (1 - wx)),
                (1, 1, wy * wx),
            ):
                yy = y0 + dy_
                xx = x0 + dx_
                inside = (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
                if inside.any():
                    flat = (yy[inside] * n + xx[inside])
                    a[row].index_add_(0, flat, dt * weight[inside])
            row += 1
    return a.to(dtype), angles.to(dtype)


def wedge_functionals(
    n: int,
    forward: torch.Tensor,
    obs_cov: torch.Tensor,
    prior_cov: torch.Tensor,
    blob_width: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A visible-wedge-aligned and an off-wedge LINEAR functional ``h``.

    Both are linear functionals ``h(x) = h_vec^T x`` on the ``n x n`` image,
    one whose energy lies in the visible wedge and one carried by the missing
    wedge. The aligned one is a smooth region-average; the off-wedge one is
    read from the operator's own likelihood-informed subspace, so its
    placement in the prior-dominated complement is geometrically exact:

      * ``h_wedge`` -- a centered smooth Gaussian-blob region-average,
        ``h(x) = sum_pixels g(pixel) x_pixel`` with ``g`` a normalized
        Gaussian of width ``blob_width`` (in the ``[-1, 1]^2`` image
        coordinates). It is a LOW-FREQUENCY image functional, hence its
        energy sits in the visible wedge: ``rho_perp`` is small and the
        latent quadrature should integrate it well (the aligned arm).
      * ``h_offwedge`` -- the least-informed generalized direction
        ``u_d = Sigma_pr^{1/2} v_d`` (the trailing eigenvector of the
        whitened likelihood-informed matrix ``H~``), which lives in the
        missing wedge, the prior-dominated complement. Its posterior variance
        is carried almost entirely by the directions the feature map discards,
        so ``rho_perp -> 1`` and the latent quadrature shows no advantage.

    Args:
        n: Grid side (the image is ``n x n``, ``dx = n^2``).
        forward: Limited-angle Radon operator ``A_Phi``, shape (dy, dx).
        obs_cov: Observation-noise covariance ``Sigma_obs``, (dy, dy).
        prior_cov: Prior covariance ``Sigma_pr``, (dx, dx).
        blob_width: Gaussian-blob width of the smooth region-average in the
            ``[-1, 1]^2`` image coordinates (larger -> lower frequency ->
            more wedge-aligned).

    Returns:
        (h_wedge [dx], h_offwedge [dx]) as float64 unit-norm vectors.
    """
    from qfield.subspace import _eig_from_whitened, whitened_lis_matrix_H

    coords = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    g = torch.exp(-(gx**2 + gy**2) / (2.0 * blob_width**2)).reshape(-1)
    h_wedge = g / g.norm()

    h_tilde = whitened_lis_matrix_H(forward, obs_cov, prior_cov)
    _, u, _ = _eig_from_whitened(h_tilde, prior_cov.double())
    h_offwedge = u[:, -1]
    h_offwedge = h_offwedge / h_offwedge.norm()
    return h_wedge, h_offwedge


def limited_angle_tomography(
    n: int = 16,
    n_angles: int = 12,
    phi_deg: float = 60.0,
    sigma_obs: float = 0.05,
    n_detectors: int | None = None,
    prior: str = "laplacian",
    prior_delta: float = 0.5,
    prior_alpha: float = 2.0,
    prior_tau: float = 1.0,
    prior_corr_len: float | None = None,
    prior_floor: float = 1e-3,
    dtype: torch.dtype = torch.float32,
) -> tuple[LinearGaussian, dict]:
    """Build the stylized limited-angle tomography linear-Gaussian problem.

    Assembles the limited-angle Radon operator ``A_Phi`` and the smoothing
    field prior ``Sigma_pr``, sets ``Sigma_obs = sigma_obs^2 I``, and
    returns a :class:`LinearGaussian` (zero prior mean) plus a small ``meta``
    dict (the angle grid, the assembled ``A``/``Sigma_pr``, the knob
    settings) for the subspace estimators and diagnostics. The angular
    range ``phi_deg`` (and ``n_angles`` / ``sigma_obs``) is the
    informed-dimension KNOB.

    Two priors are available (``prior``):

      * ``"laplacian"`` (default) -- the Neumann / reflecting-boundary
        Whittle--Matern (:func:`laplacian_field_prior_cov`). It is SPD and
        smoothing, but its marginal variance swells at the grid boundary;
        ``meta["prior_field"]`` is ``None``. It is the default, for
        compatibility with callers tuned to its scale and conditioning.
      * ``"stationary_grf"`` -- the genuinely SHIFT-INVARIANT periodic
        Whittle--Matern GRF (:func:`stationary_grf_field`), whose per-pixel
        prior variance is uniform (``tau^2`` everywhere). With this prior the
        posterior uncertainty map reflects the data's limited-angle
        resolution rather than a boundary artifact; the field is returned in
        ``meta["prior_field"]`` so a ground-truth sample is consistent with
        ``Sigma_pr``.
        ``prior_corr_len`` (pixels, default ``n / 4``) sets the correlation
        length, ``prior_alpha`` the smoothness, ``prior_tau`` the marginal
        std, ``prior_floor`` the spectral floor.

    Args:
        n: Grid side (image is ``n x n``, ``dx = n^2``).
        n_angles: Number of projection angles in ``[-phi_deg, phi_deg]``.
        phi_deg: Half-range of the limited wedge in degrees (``< 90``).
        sigma_obs: Observation-noise standard deviation (isotropic).
        n_detectors: Detector bins per angle (default ``n``).
        prior: Which field prior to build -- ``"stationary_grf"`` (uniform
            marginal variance) or ``"laplacian"`` (Neumann boundary).
        prior_delta: Laplacian-prior ``delta`` (correlation-length knob;
            only used when ``prior="laplacian"``).
        prior_alpha: Field-prior smoothness exponent (both priors).
        prior_tau: Field-prior marginal scale (both priors).
        prior_corr_len: Stationary-GRF correlation length in PIXELS (only
            used when ``prior="stationary_grf"``; default ``n / 4``).
        prior_floor: Stationary-GRF relative spectral floor (only used when
            ``prior="stationary_grf"``).
        dtype: Output dtype for the :class:`LinearGaussian` (float32 to
            match the experiments; the closed forms run in float64 inside).

    Returns:
        (problem, meta): ``problem`` is the :class:`LinearGaussian`;
        ``meta`` holds ``{"A", "Sigma_pr", "Sigma_obs", "angles_rad", "n",
        "dx", "dy", "phi_deg", "n_angles", "sigma_obs", "prior",
        "prior_field"}`` (matrices in float64; ``prior_field`` is the
        :class:`SpectralGaussianField` for the stationary GRF, else ``None``).
    """
    a64, angles = parallel_beam_radon_operator(
        n, n_angles=n_angles, phi_deg=phi_deg, n_detectors=n_detectors,
        dtype=torch.float64,
    )
    if prior == "stationary_grf":
        corr_len = float(n) / 4.0 if prior_corr_len is None else prior_corr_len
        prior_field = stationary_grf_field(
            n, corr_len=corr_len, alpha=prior_alpha, tau=prior_tau,
            floor=prior_floor, dtype=torch.float64,
        )
        # dense_prior_cov is capped at SpectralGaussianField.DENSE_CAP (4096 =
        # 64x64); for a finer grid (n > 64, dx > 4096) build the SAME dense
        # covariance cap-free via the matrix-free field.cov on the canonical
        # basis (dense_cov_from_field). Both give diag(Sigma_pr) exactly
        # uniform; this only lifts the size guard for the dense prototype.
        if n * n <= SpectralGaussianField.DENSE_CAP:
            sigma_pr64 = prior_field.dense_prior_cov()
        else:
            sigma_pr64 = dense_cov_from_field(prior_field, n * n)
    elif prior == "laplacian":
        prior_field = None
        sigma_pr64 = laplacian_field_prior_cov(
            n, delta=prior_delta, alpha=prior_alpha, tau=prior_tau,
            dtype=torch.float64,
        )
    else:
        raise ValueError(
            f"prior must be 'stationary_grf' or 'laplacian'; got {prior!r}"
        )
    d_y, d_x = a64.shape
    sigma_obs64 = (sigma_obs**2) * torch.eye(d_y, dtype=torch.float64)
    prior_mean = torch.zeros(d_x, dtype=torch.float64)

    problem = LinearGaussian(
        prior_mean=prior_mean,
        prior_cov=sigma_pr64,
        forward=a64,
        obs_cov=sigma_obs64,
        dtype=dtype,
    )
    meta = {
        "A": a64,
        "Sigma_pr": sigma_pr64,
        "Sigma_obs": sigma_obs64,
        "angles_rad": angles,
        "n": n,
        "dx": d_x,
        "dy": d_y,
        "phi_deg": phi_deg,
        "n_angles": n_angles,
        "sigma_obs": sigma_obs,
        "prior": prior,
        "prior_field": prior_field,
    }
    return problem, meta
