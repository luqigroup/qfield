"""Stationary Gaussian random field with an empirical 2D power-spectrum prior.

The linear-Gaussian reflectivity prior of the seismic Born-imaging
experiment. The unknown is a discretized ``n1 x n2`` reflectivity
image; the prior is a zero-mean, stationary
(shift-invariant) Gaussian random field whose covariance is DIAGONAL in
the 2D Fourier basis, with the per-mode variance ``S`` read off a
reference image's empirical 2D power spectrum (periodogram). A stationary
GP has a covariance that is a convolution operator, hence diagonalized by
the Fourier transform: this class is the discrete, data-driven instance of
that fact.

ONE FFT convention, everywhere. We use the ORTHONORMAL (unitary) 2D DFT
throughout -- ``F = fft2(., norm="ortho")``, ``Finv = ifft2(., norm="ortho")``
-- so ``F^H = F^{-1} = Finv`` and Parseval is an isometry (``||F x|| =
||x||``). With ``S`` a real, nonnegative, EVEN-SYMMETRIC (``S[-k] = S[k]``)
array of per-mode variances, the prior covariance is

    C = Finv diag(S) F   (real symmetric PSD; eigenvalues are the S_k),

and -- because ``S`` is even-symmetric -- every operator below maps a real
field to a real field (the tiny imaginary part is numerical and asserted
negligible). The covariance square root and its inverse are the SAME real
symmetric operators evaluated with ``sqrt(S)`` / ``1/sqrt(S)`` in place of
``S``:

    Sigma_pr        = C        : cov(x)        = Finv( S * F(x) ).real
    Sigma_pr^{1/2}  = C^{1/2}  : cov_half(w)   = Finv( sqrt(S) * F(w) ).real
    Sigma_pr^{-1}   = C^{-1}   : cov_inv(x)    = Finv( F(x) / S ).real
    Sigma_pr^{-1/2} = C^{-1/2} : cov_inv_half(x) = Finv( F(x) / sqrt(S) ).real

The math the class implements (single zero-mean field ``x``, ``u = F(x)``).

    sample:     x = Finv( sqrt(S) * F(w) ).real,   w ~ N(0, I) real white,
    score:      grad_x log p(x) = -C^{-1} x = -Finv( F(x) / S ).real,
    log_prob:   log p(x) = -0.5 sum_k |u_k|^2 / S_k - 0.5 sum_k log(2 pi S_k).

The KEY correctness invariant (mutual consistency of sampler and density).
``cov_half`` is the symmetric square root ``Sigma_pr^{1/2}`` of the SAME
``C`` whose inverse appears in ``score`` and ``log_prob``: because
``sqrt(S)`` is real even-symmetric, ``Sigma_pr^{1/2} = Finv diag(sqrt S) F``
is real SYMMETRIC, so ``Sigma_pr^{1/2} (Sigma_pr^{1/2})^T = Finv diag(S) F
= C`` exactly, and ``Cov(x) = C`` for ``x = Sigma_pr^{1/2} w``. Sampling
and the density therefore agree by construction: the field is sampled at
EXACTLY the variance its own ``score`` / ``log_prob`` assume.

Spectrum estimation + regularization (:meth:`from_image`). From a reference
image we (1) center it (subtract the mean) and form the periodogram ``P =
|F(image_centered)|^2``; (2) SMOOTH ``P`` with a symmetric Gaussian blur
(blur the ``fftshift``-ed periodogram with a symmetric separable kernel, then
``ifftshift`` back), to tame the periodogram's order-one variance -- on an
even-sized grid this is only an APPROXIMATE smoother of an even-symmetric
array (``fftshift`` does not land the DC at the geometric center for even
lengths), so it does not by itself guarantee ``S[-k] = S[k]``; (3) enforce
even symmetry EXACTLY with the explicit projector ``S = 0.5 (S +
flip_both_axes(S))`` -- THIS step is what ESTABLISHES ``S[-k] = S[k]`` to
machine precision (so ``C`` is real), the blur merely making it a small
correction;
(4) FLOOR for invertibility / conditioning, ``S = max(S, floor * S.max())``,
and pin the DC (``k = 0``) bin to the floor too -- the field is zero-mean,
so the giant raw DC power is discarded rather than baked into the prior.

This is the seismic analogue of the smoothing field prior in
:mod:`qfield.dataset.tomography`: there the covariance is the
Whittle--Matern operator ``tau^2 (delta I - Delta)^{-alpha}`` (a PARAMETRIC
isotropic spectrum); here the spectrum is the DATA's own, so the samples
inherit the reference image's anisotropy / layering and its mid-band
structure rather than an isotropic power law.

Dense interface. For the small grids that plug into the dense
:class:`~qfield.dataset.linear_gaussian.LinearGaussian` /
:mod:`~qfield.subspace` machinery, :meth:`dense_prior_cov`
materializes ``C`` as an ``(N, N)`` matrix (``N = n1 n2``) by applying
``cov`` to the identity columns; it is GUARDED by a size cap (it is only
meant for the small dense-formable grid, not the full image).

References:
  - Whittle (1954); Rue & Held, "Gaussian Markov Random Fields" (stationary
    GP <-> spectral / circulant covariance, the Fourier diagonalization).
  - the tomography smoothing prior this mirrors,
    :mod:`qfield.dataset.tomography`.
"""

from __future__ import annotations

import math

import torch


def _flip_both_axes(s: torch.Tensor) -> torch.Tensor:
    """Even-symmetrization partner ``s[-k]`` of a DFT-indexed array.

    For an array indexed by the 2D DFT frequencies ``k = (k1, k2)`` in the
    standard ``0..n-1`` (``fftfreq``) order, the spectral partner of mode
    ``k`` is ``-k mod n`` on each axis. That index reversal is ``roll(flip)``
    by one: ``flip`` sends index ``j -> n-1-j`` and the unit ``roll`` shifts
    it to ``j -> (n-j) mod n = -j mod n`` (so index ``0`` -- the DC / Nyquist
    fixed points -- maps to itself). Hence ``_flip_both_axes(s)[k] = s[-k]``,
    which is exactly what the even-symmetry projector
    ``0.5 (S + S[-k])`` needs. Operates on the last two dims.
    """
    return torch.roll(torch.flip(s, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))


def _gaussian_blur_even(p: torch.Tensor, sigma: float) -> torch.Tensor:
    """Symmetric Gaussian blur of a periodogram (an APPROXIMATE even smoother).

    Smooths the periodogram ``P`` (DFT-indexed, last two dims ``n1 x n2``)
    with a separable Gaussian. We blur the ``fftshift``-ed array with a kernel
    that is symmetric about its own center, then ``ifftshift`` back. This
    APPROXIMATELY respects even symmetry, but does NOT guarantee it: on an
    even-sized axis ``fftshift`` does not place the DC at the geometric center
    of the array (the even-length shift leaves DC off-center by half a bin), so
    the blur is not exactly centered on the even-symmetry fixed point and can
    introduce a small ``S[-k] != S[k]`` asymmetry. Exact even symmetry is
    ESTABLISHED separately and explicitly by the :func:`_flip_both_axes`
    projector ``0.5 (S + S[-k])`` applied in :meth:`from_image`; this blur is
    only the variance-reduction step. The convolution is done in float64 with
    REFLECT padding (no wrap-around across the spectrum edges). ``sigma <= 0``
    returns ``P`` unchanged.

    Args:
        p: Periodogram, real nonnegative, shape (..., n1, n2), float64.
        sigma: Gaussian blur std in BIN units (per axis); ``<= 0`` -> no-op.

    Returns:
        The blurred periodogram, same shape / dtype as ``p``.
    """
    if sigma <= 0.0:
        return p
    n1, n2 = p.shape[-2], p.shape[-1]
    # 1D Gaussian kernels (separable). Radius ~ 3 sigma, capped to the grid so
    # padding never exceeds the array. Symmetric about its center by
    # construction (arange centered); this only APPROXIMATELY respects even
    # symmetry (exact S[-k]=S[k] is set later by the _flip_both_axes projector).
    dev = p.device
    rad1 = min(int(3.0 * sigma) + 1, n1 - 1) if n1 > 1 else 0
    rad2 = min(int(3.0 * sigma) + 1, n2 - 1) if n2 > 1 else 0
    inv_two_s2 = 1.0 / (2.0 * sigma * sigma)

    def _kernel(rad: int) -> torch.Tensor:
        t = torch.arange(-rad, rad + 1, device=dev, dtype=torch.float64)
        k = torch.exp(-(t * t) * inv_two_s2)
        return k / k.sum()

    # Center DC for the symmetric blur, smooth each axis, then shift back.
    ps = torch.fft.fftshift(p, dim=(-2, -1))
    out = ps
    if rad1 > 0:
        k1 = _kernel(rad1).reshape(-1, 1)  # (2 rad1 + 1, 1)
        out = _separable_reflect_conv(out, k1, axis=-2)
    if rad2 > 0:
        k2 = _kernel(rad2).reshape(1, -1)  # (1, 2 rad2 + 1)
        out = _separable_reflect_conv(out, k2, axis=-1)
    return torch.fft.ifftshift(out, dim=(-2, -1))


def _separable_reflect_conv(
    x: torch.Tensor, kernel: torch.Tensor, axis: int
) -> torch.Tensor:
    """1D reflect-padded convolution of ``x`` along ``axis`` with ``kernel``.

    A small, dependency-light separable smoothing step (no ``F.conv2d`` /
    channel reshaping needed). ``kernel`` is a 1D weight vector (oriented as
    a row or column); REFLECT padding avoids spectral wrap-around. Operates
    on the last two dims and returns the same shape, float64.
    """
    klen = kernel.numel()
    rad = klen // 2
    pad = [0, 0, 0, 0]  # (left, right, top, bottom) for the last two dims.
    if axis == -2:
        pad[2] = pad[3] = rad
    else:
        pad[0] = pad[1] = rad
    # ``F.pad(mode="reflect")`` needs a rank >= 3 tensor for a 4-element (2D)
    # pad: a bare ``(n1, n2)`` periodogram is rank-2 and raises in torch 2.9.
    # Insert a singleton channel dim so the pad always sees ``(..., 1, n1, n2)``
    # (works for a single field AND a leading batch), then drop it again.
    x3 = x.reshape(x.shape[:-2] + (1,) + x.shape[-2:])  # (..., 1, n1, n2)
    xp = torch.nn.functional.pad(x3, pad, mode="reflect").reshape(
        x.shape[:-2]
        + (x.shape[-2] + pad[2] + pad[3], x.shape[-1] + pad[0] + pad[1])
    )
    out = torch.zeros_like(x)
    kflat = kernel.reshape(-1)
    for j in range(klen):
        if axis == -2:
            out = out + kflat[j] * xp[..., j : j + x.shape[-2], :]
        else:
            out = out + kflat[j] * xp[..., :, j : j + x.shape[-1]]
    return out


class SpectralGaussianField:
    """Stationary GRF whose covariance is an empirical 2D power spectrum ``S``.

    Zero-mean Gaussian field on an ``n1 x n2`` grid with covariance ``C =
    Finv diag(S) F`` DIAGONAL in the orthonormal 2D Fourier basis. ``S`` is
    real, nonnegative, even-symmetric (``S[-k] = S[k]``), so ``C`` is real
    symmetric PSD; the sampler is the symmetric square root ``Sigma_pr^{1/2}``
    of EXACTLY this ``C``, so sampling and the density (``score`` /
    ``log_prob``) are mutually consistent by construction.

    Build it from a reference image with :meth:`from_image` (the data-driven
    spectrum + regularization); the four diagonal-in-Fourier operators
    :meth:`cov`, :meth:`cov_half`, :meth:`cov_inv`, :meth:`cov_inv_half`
    realize ``Sigma_pr``, ``Sigma_pr^{1/2}``, ``Sigma_pr^{-1}``,
    ``Sigma_pr^{-1/2}`` (each returns a real field), matching the actions the
    :mod:`~qfield.subspace` machinery consumes. The spectrum is stored
    in float64; the operators run their FFTs in float64 and cast the (real)
    result back to the field dtype.

    Attributes:
        n1, n2: Grid dimensions (the field is ``n1 x n2``, ``N = n1 n2``).
        S: The per-mode variance spectrum, real nonnegative even-symmetric,
            shape ``(n1, n2)``, float64 (on the configured device).
        dtype: Floating dtype of fields produced / accepted (default
            float32, matching the experiments).
        device: Device the spectrum / outputs live on.
        DENSE_CAP: Hard cap on ``N`` for :meth:`dense_prior_cov`.
    """

    # Dense covariance is O(N^2) memory; only meant for the small dense grid
    # that feeds the LinearGaussian / subspace machinery. Refuse above this.
    DENSE_CAP: int = 4096

    def __init__(
        self,
        spectrum: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        """Construct directly from a per-mode variance spectrum ``S``.

        Prefer :meth:`from_image`; this low-level constructor takes an
        already-formed ``S`` (e.g. for tests). ``S`` MUST be real,
        finite, strictly positive (floor applied upstream), and
        even-symmetric; the constructor validates these defensively and
        stores ``S`` in float64 on ``device``.

        Args:
            spectrum: Per-mode variance ``S``, shape ``(n1, n2)``.
            dtype: Field dtype for samples / operator outputs.
            device: Device for the stored spectrum and outputs (defaults to
                the spectrum's device).
        """
        if spectrum.dim() != 2:
            raise ValueError(
                f"spectrum must be 2D (n1, n2); got shape {tuple(spectrum.shape)}"
            )
        if device is None:
            device = spectrum.device
        self.device = torch.device(device)
        self.dtype = dtype
        s = spectrum.to(device=self.device, dtype=torch.float64)
        if not torch.isfinite(s).all():
            raise ValueError("spectrum has non-finite entries")
        if (s <= 0.0).any():
            raise ValueError(
                "spectrum must be strictly positive (apply a floor); got "
                f"min {float(s.min())!r}"
            )
        sym_err = (s - _flip_both_axes(s)).abs().max()
        if float(sym_err) > 1e-9 * float(s.max()):
            raise ValueError(
                "spectrum is not even-symmetric (S[-k] != S[k]); max "
                f"asymmetry {float(sym_err)!r}"
            )
        self.S = s
        self.n1, self.n2 = int(s.shape[0]), int(s.shape[1])
        # Cached real square roots (even-symmetric, so their C-operators are
        # real symmetric); float64, on-device. 1/sqrt(S) is well-defined
        # because S is strictly positive (floored).
        self._sqrt_S = s.sqrt()
        self._inv_sqrt_S = s.rsqrt()

    # ------------------------------------------------------------------
    # Construction from a reference image (data-driven spectrum + reg).
    # ------------------------------------------------------------------
    @classmethod
    def from_image(
        cls,
        image: torch.Tensor,
        *,
        smooth_sigma: float = 2.0,
        floor: float = 1e-3,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "SpectralGaussianField":
        """Build the field from a reference image's empirical power spectrum.

        Pipeline (see the module docstring): center the image, form the
        periodogram ``P = |F(image_centered)|^2`` with the orthonormal FFT,
        smooth ``P`` with a symmetric Gaussian blur (:func:`_gaussian_blur_even`,
        std ``smooth_sigma`` bins; an APPROXIMATE even smoother only), then
        enforce even symmetry EXACTLY with the ``0.5 (S + flip_both_axes(S))``
        projector -- it is THIS step, not the blur, that makes ``S[-k] = S[k]``
        to machine precision -- and finally floor for conditioning and pin the
        DC bin to the floor (zero-mean field -> discard the raw DC power). The
        spectrum is stored float64.

        Args:
            image: Reference image, shape ``(n1, n2)`` (the field grid).
            smooth_sigma: Gaussian-blur std in spectral-bin units; larger ->
                smoother spectrum. ``<= 0`` keeps the raw periodogram.
            floor: Relative floor ``S = max(S, floor * S.max())`` (and the DC
                value); guarantees ``C`` is invertible / well-conditioned.
                Must be in ``(0, 1)``.
            dtype: Field dtype for samples / operator outputs.
            device: Device for the stored spectrum and outputs (defaults to
                the image's device).

        Returns:
            The constructed :class:`SpectralGaussianField`.
        """
        if image.dim() != 2:
            raise ValueError(
                f"image must be 2D (n1, n2); got shape {tuple(image.shape)}"
            )
        if not (0.0 < floor < 1.0):
            raise ValueError(f"floor must be in (0, 1); got {floor!r}")
        if not math.isfinite(smooth_sigma):
            raise ValueError(f"smooth_sigma must be finite; got {smooth_sigma!r}")
        if device is None:
            device = image.device
        device = torch.device(device)

        img = image.to(device=device, dtype=torch.float64)
        if not torch.isfinite(img).all():
            raise ValueError("image has non-finite entries")

        # (1) center + periodogram with the ORTHONORMAL FFT (one convention).
        img_c = img - img.mean()
        u = torch.fft.fft2(img_c, norm="ortho")
        p = (u.real**2 + u.imag**2)  # |F(image_centered)|^2, real >= 0

        # (2) Gaussian smoothing of the periodogram (variance reduction). This
        #     is only an APPROXIMATE even smoother -- on even-sized grids the
        #     fftshift-ed blur is not exactly centered on the even-symmetry
        #     fixed point, so a small S[-k] != S[k] asymmetry can remain.
        s = _gaussian_blur_even(p, smooth_sigma)

        # (3) ESTABLISH even symmetry S[-k] = S[k] EXACTLY with the explicit
        #     projector (removes the periodogram's own asymmetry AND any
        #     introduced by the off-center blur); makes C real to machine eps.
        s = 0.5 * (s + _flip_both_axes(s))
        s = s.clamp_min(0.0)  # blur of a nonneg array stays >= 0; guard anyway.

        # (4) floor for invertibility / conditioning, and PIN the DC bin to the
        #     floor: the field is zero-mean, so the (enormous) raw DC power is
        #     dropped rather than baked into the prior.
        s_max = s.max()
        if float(s_max) <= 0.0:
            raise ValueError(
                "periodogram is identically zero (constant image?); cannot "
                "build a spectrum"
            )
        floor_val = floor * s_max
        s[0, 0] = floor_val  # DC at index (0, 0) under fftfreq ordering.
        s = s.clamp_min(floor_val)
        return cls(s, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Internal: shape handling + the shared diagonal-in-Fourier kernel.
    # ------------------------------------------------------------------
    def _as_batched_field(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """Validate ``x`` is a single field or a leading-batch of fields.

        Accepts ``(n1, n2)`` (single) or ``(..., n1, n2)`` (leading batch);
        returns ``(x_b, single)`` with ``x_b`` flattened to ``(B, n1, n2)``
        on the field device, and ``single`` True iff the input was a single
        2D field. The trailing two dims must equal the grid.
        """
        if x.shape[-2:] != (self.n1, self.n2):
            raise ValueError(
                f"field trailing dims must be ({self.n1}, {self.n2}); got "
                f"shape {tuple(x.shape)}"
            )
        single = x.dim() == 2
        xb = x.reshape(-1, self.n1, self.n2)
        return xb, single

    def _apply_diag(
        self, x: torch.Tensor, factor: torch.Tensor, *, name: str
    ) -> torch.Tensor:
        """Apply a diagonal-in-Fourier operator ``Finv( factor * F(x) )``.

        The single shared kernel behind all four covariance operators (and the
        score): forward orthonormal FFT, multiply by the real even-symmetric
        ``factor`` (one of ``S``, ``sqrt(S)``, ``1/S``, ``1/sqrt(S)``), inverse
        orthonormal FFT, and return the REAL part. Because ``factor`` is even
        symmetric and ``x`` real, the result is real up to round-off; the
        imaginary part is asserted negligible (a guard against a convention or
        symmetry slip). Runs the FFTs in float64 and casts the real output back
        to the input dtype; preserves the leading batch shape and device.

        Args:
            x: Real field(s), shape ``(n1, n2)`` or ``(..., n1, n2)``.
            factor: Real even-symmetric per-mode multiplier, shape
                ``(n1, n2)`` (float64).
            name: Operator name, for error messages.

        Returns:
            The transformed real field(s), same shape / dtype / device as
            ``x``.
        """
        in_dtype = x.dtype
        in_device = x.device
        xb, single = self._as_batched_field(x)
        xb64 = xb.to(device=self.device, dtype=torch.float64)
        u = torch.fft.fft2(xb64, norm="ortho")
        v = torch.fft.ifft2(factor * u, norm="ortho")
        # Imag part must be numerical only (even-symmetric factor, real input).
        imag_scale = v.real.abs().max().clamp_min(1e-300)
        imag_err = v.imag.abs().max() / imag_scale
        if float(imag_err) > 1e-6:
            raise AssertionError(
                f"{name}: result has non-negligible imaginary part (relative "
                f"{float(imag_err):.3e}); the spectrum may not be even-"
                "symmetric or the FFT convention is inconsistent"
            )
        out = v.real.to(device=in_device, dtype=in_dtype)
        return out[0] if single else out.reshape(x.shape)

    # ------------------------------------------------------------------
    # The four covariance operators (each returns a real field).
    # ------------------------------------------------------------------
    def cov(self, x: torch.Tensor) -> torch.Tensor:
        """``Sigma_pr x = Finv( S * F(x) ).real`` (the covariance action)."""
        return self._apply_diag(x, self.S, name="cov")

    def cov_half(self, w: torch.Tensor) -> torch.Tensor:
        """``Sigma_pr^{1/2} w = Finv( sqrt(S) * F(w) ).real``.

        The symmetric square root: for real white ``w``, ``cov_half(w)`` is a
        prior sample (this is exactly what :meth:`sample` uses). It is the
        square root of the SAME ``C`` whose inverse :meth:`score` /
        :meth:`log_prob` use, so the two are mutually consistent.
        """
        return self._apply_diag(w, self._sqrt_S, name="cov_half")

    def cov_inv(self, x: torch.Tensor) -> torch.Tensor:
        """``Sigma_pr^{-1} x = Finv( F(x) / S ).real`` (the precision action)."""
        return self._apply_diag(x, 1.0 / self.S, name="cov_inv")

    def cov_inv_half(self, x: torch.Tensor) -> torch.Tensor:
        """``Sigma_pr^{-1/2} x = Finv( F(x) / sqrt(S) ).real`` (prior whitening)."""
        return self._apply_diag(x, self._inv_sqrt_S, name="cov_inv_half")

    # ------------------------------------------------------------------
    # Sampling, score, log-density.
    # ------------------------------------------------------------------
    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw ``n`` i.i.d. prior fields, shape ``(n, n1, n2)``.

        Real white noise ``w ~ N(0, I)`` pushed through the symmetric square
        root: ``x = Sigma_pr^{1/2} w = Finv( sqrt(S) * F(w) ).real``. Since
        this is the square root of ``C``, ``Cov(x) = C`` -- the field is drawn
        at exactly the variance its own ``score`` / ``log_prob`` assume.

        Args:
            n: Number of fields.
            generator: Optional ``torch.Generator`` (on the field device) for
                reproducible draws; ``None`` uses the global RNG.

        Returns:
            Samples of shape ``(n, n1, n2)`` in the configured dtype / device.
        """
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}")
        # White noise in float64 for the (float64) transform; cast at the end.
        w = torch.randn(
            n, self.n1, self.n2,
            generator=generator, device=self.device, dtype=torch.float64,
        )
        return self.cov_half(w).to(dtype=self.dtype)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """``grad_x log p(x) = -Sigma_pr^{-1} x = -Finv( F(x) / S ).real``.

        The exact spatial-domain gradient of :meth:`log_prob` (a centered
        Gaussian log-density, ``log p = -0.5 x^T C^{-1} x + const``). Accepts a
        single field ``(n1, n2)`` or a leading batch ``(..., n1, n2)`` and
        returns the gradient of matching shape.
        """
        return -self.cov_inv(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian log-density ``log p(x)`` (single field or leading-batch).

        With ``u = F(x)`` (orthonormal, so ``x^T C^{-1} x = sum_k |u_k|^2 /
        S_k`` and ``log det(2 pi C) = sum_k log(2 pi S_k)``),

            log p(x) = -0.5 sum_k |u_k|^2 / S_k - 0.5 sum_k log(2 pi S_k),

        the second (normalizing) term INCLUDED. Computed in float64 and cast
        back to the input dtype.

        Args:
            x: Field(s), shape ``(n1, n2)`` or ``(..., n1, n2)``.

        Returns:
            Log-density: a 0-dim tensor for a single field, else shape equal
            to the leading batch dims, in the input dtype / device.
        """
        in_dtype = x.dtype
        in_device = x.device
        xb, single = self._as_batched_field(x)
        xb64 = xb.to(device=self.device, dtype=torch.float64)
        u = torch.fft.fft2(xb64, norm="ortho")
        umag2 = u.real**2 + u.imag**2  # |F(x)_k|^2, (B, n1, n2)
        quad = (umag2 / self.S).sum(dim=(-2, -1))  # sum_k |u_k|^2 / S_k
        # Normalizing constant: 0.5 sum_k log(2 pi S_k) (mode-count from S).
        log_norm = 0.5 * torch.log(2.0 * torch.pi * self.S).sum()
        lp = -0.5 * quad - log_norm  # (B,)
        lp = lp.to(device=in_device, dtype=in_dtype)
        if single:
            return lp.reshape(())
        return lp.reshape(x.shape[:-2])

    # ------------------------------------------------------------------
    # Dense materialization (small grids only).
    # ------------------------------------------------------------------
    def dense_prior_cov(self) -> torch.Tensor:
        """Materialize ``Sigma_pr`` as a dense ``(N, N)`` matrix, ``N = n1 n2``.

        Builds ``C`` column by column by applying :meth:`cov` to the standard
        basis: column ``m`` is ``cov(e_m)`` flattened, with ``e_m`` the field
        that is ``1`` at the ``m``-th pixel (row-major ``reshape(n1, n2)``) and
        ``0`` elsewhere. The result is real symmetric PSD (the eigenvalues are
        the ``S_k``) and is exactly the ``Sigma_pr`` the dense
        :class:`~qfield.dataset.linear_gaussian.LinearGaussian` /
        :mod:`~qfield.subspace` machinery consumes. Returned float64 on
        the field device.

        GUARDED by :attr:`DENSE_CAP`: this is ``O(N^2)`` memory and is only
        meant for the small dense-formable grid, NOT the full reflectivity
        image. Raises ``ValueError`` when ``N`` exceeds the cap.

        Returns:
            ``Sigma_pr`` of shape ``(N, N)``, float64, on the field device.
        """
        n = self.n1 * self.n2
        if n > self.DENSE_CAP:
            raise ValueError(
                f"dense_prior_cov refuses N = n1*n2 = {n} > DENSE_CAP = "
                f"{self.DENSE_CAP}: forming the dense covariance is O(N^2) "
                "memory and is only for the small dense-formable grid. Use the "
                "matrix-free operators (cov / cov_half / cov_inv / "
                "cov_inv_half) on the full image instead."
            )
        # Apply cov to all N basis columns at once (batched): identity reshaped
        # to (N, n1, n2). cov is linear, so this is C e_m for each m.
        eye = torch.eye(n, dtype=torch.float64, device=self.device)
        basis = eye.reshape(n, self.n1, self.n2)
        cols = self.cov(basis)  # (N, n1, n2), real
        c = cols.reshape(n, n)  # row m = (C e_m) flattened = column m of C
        # Symmetrize to kill round-off asymmetry (C is symmetric exactly).
        return 0.5 * (c + c.T)
