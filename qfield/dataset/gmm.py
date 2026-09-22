"""Anisotropic Gaussian-mixture target with MSIP closed forms.

Mixture pi(x) = sum_k m_k N(x; mu_k, Sigma_k) with general (rotated,
anisotropic) component covariances. Beyond the usual log_prob / score /
sample, this class supplies the squared-exponential closed forms that
MSIP-exact and the analytic MMD need:

    Z_sigma           = (2 pi sigma^2)^(d/2),
    Sigma_tilde_k     = Sigma_k + sigma^2 I,
    v0(y)             = Z_sigma sum_k m_k N(y; mu_k, Sigma_tilde_k),
    grad_log_v0(y)    = sum_k r_k(y) (-Sigma_tilde_k^{-1} (y - mu_k)),
    v1(y)             = v0(y) (y + sigma^2 grad_log_v0(y)),
    C_pi(sigma)       = Z_sigma sum_{k,l} m_k m_l
                        N(mu_k; mu_l, Sigma_k + Sigma_l + sigma^2 I),
    MMD^2(pi, mu)     = C_pi - 2 sum_i w_i v0(y_i) + sum_{ij} w_i w_j
                        k(y_i, y_j),   mu = sum_i w_i delta_{y_i}.

v0 (hence v1, C_pi) is assembled in log-space to stay overflow-safe in d
(the Z_sigma prefactor blows up); only differences are exponentiated.

Reference: Belhadji, Sharp, Marzouk, "To discretize continually: Mean
shift interacting particle systems for Bayesian inference",
arXiv:2605.14142 (mixtures of Gaussians, eqs. 34-41; MMD, eqs. 4-5).
"""

import math

import torch
from torch.distributions import (
    Categorical,
    MixtureSameFamily,
    MultivariateNormal,
)

from qfield.kernels import log_z_sigma, se_kernel


class AnisotropicGMM:
    """Gaussian mixture with general (anisotropic) covariances in R^d.

    Components are specified by their means, covariance matrices, and
    mixture weights. Covariances are typically built by rotating a
    diagonal of squared standard deviations, R_k diag(sd^2) R_k^T, so the
    modes can be tilted. All tensors are kept in float64 for the linear
    algebra in the closed forms to remain accurate.

    Args:
        means: Component means, shape (K, d).
        covariances: Component covariance matrices, shape (K, d, d). Must
            be symmetric positive definite.
        weights: Mixture weights, shape (K,). Normalized to sum to one;
            uniform if None.
        dtype: Floating dtype for all stored tensors. Defaults to
            float32.
    """

    def __init__(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        weights: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        means = means.to(dtype)
        covariances = covariances.to(dtype)
        self.dtype = dtype
        self.K, self.d = means.shape
        self.means = means
        self.covariances = covariances
        self.precisions = torch.linalg.inv(covariances)

        if weights is None:
            weights = torch.full((self.K,), 1.0 / self.K, dtype=dtype)
        else:
            weights = weights.to(dtype)
            weights = weights / weights.sum()
        self.weights = weights

        mix = Categorical(probs=weights)
        comp = MultivariateNormal(means, covariance_matrix=covariances)
        self.dist = MixtureSameFamily(mix, comp)

    @classmethod
    def from_components(
        cls,
        means: torch.Tensor,
        covariances: torch.Tensor,
        weights: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "AnisotropicGMM":
        """Build a mixture from explicit component means/covs/weights.

        Thin convenience constructor matching the reference benchmark
        target (``gmm_2d_benchmark.py``), which specifies each component's
        mean and covariance explicitly rather than placing them on a
        circle.

        Args:
            means: Component means, shape (K, d).
            covariances: Component covariances, shape (K, d, d).
            weights: Mixture weights, shape (K,); uniform if None.
            dtype: Floating dtype.

        Returns:
            The constructed AnisotropicGMM.
        """
        return cls(means, covariances, weights, dtype=dtype)

    @classmethod
    def axis_aligned(
        cls,
        d: int,
        n_components: int = 5,
        alpha: float = 5.5,
        cov_scale: float = 0.5,
        anisotropy_factor: float = 3.0,
        dtype: torch.dtype = torch.float64,
    ) -> "AnisotropicGMM":
        """Five-component axis-aligned GMM in R^d (reference make_axis_gmm).

        Component k (k = 0, ..., K-1) has mean ``alpha * e_k`` and a
        DIAGONAL covariance ``cov_scale * I_d`` whose k-th diagonal entry
        is inflated to ``anisotropy_factor * cov_scale``, so each mode sits
        on its own coordinate axis and is elongated along that axis.
        Requires ``d >= n_components`` (the means need ``n_components``
        distinct axes); coordinates ``n_components, ..., d-1`` have zero
        mean and the un-inflated variance ``cov_scale`` for every
        component. Weights are uniform.

        This mirrors ``make_axis_gmm`` in
        ``nak_torch/examples/gmm/gmm_high-d_benchmark.py`` (lines 122-146)
        with its defaults ``alpha = MODE_SEPARATION_ALPHA = 5.5``,
        ``cov_scale = TARGET_COV_SCALE = 0.5``, ``anisotropy_factor =
        ANISOTROPY_FACTOR = 3.0``.

        The reference paper's text (App. A.3) describes an isotropic
        mixture ``N(.; mu_k, sigma^2 I_d)`` with ``mu_k = 7.5``,
        ``sigma^2 = 0.5``, but its Table-1 numbers were produced by this
        anisotropic ``make_axis_gmm`` target. The default here is therefore
        the anisotropic target; ``alpha`` / ``cov_scale`` /
        ``anisotropy_factor`` are exposed so the literal App-A.3 target
        (``alpha = 7.5``, ``anisotropy_factor = 1.0``) is a one-call
        change.

        Args:
            d: Dimensionality; must be ``>= n_components``.
            n_components: Number of mixture components K (default 5).
            alpha: Mode separation; mean of component k is ``alpha * e_k``.
            cov_scale: Base diagonal variance ``cov_scale`` on every axis.
            anisotropy_factor: Multiplier on the active-axis variance of
                each component (``1.0`` recovers an isotropic mixture).
            dtype: Floating dtype.

        Returns:
            The constructed axis-aligned ``AnisotropicGMM``.
        """
        if d < n_components:
            raise ValueError(
                f"axis_aligned requires d >= n_components ({n_components}); "
                f"got d={d}."
            )
        means = torch.zeros(n_components, d, dtype=dtype)
        for k in range(n_components):
            means[k, k] = alpha
        covariances = cov_scale * torch.eye(d, dtype=dtype).unsqueeze(
            0
        ).repeat(n_components, 1, 1)
        for k in range(n_components):
            covariances[k, k, k] = anisotropy_factor * cov_scale
        return cls(means, covariances, dtype=dtype)

    @classmethod
    def on_circle(
        cls,
        n_components: int,
        radius: float,
        sd_major: float,
        sd_minor: float,
        dtype: torch.dtype = torch.float64,
    ) -> "AnisotropicGMM":
        """Build a 2D mixture with modes on a circle, tangential tilt.

        Component k sits at angle 2 pi k / K on a circle of the given
        radius; its covariance is R_k diag(sd_major^2, sd_minor^2) R_k^T
        with R_k rotating the major axis tangentially (perpendicular to
        the radial direction), so each blob is elongated along the
        circle. Weights are uniform.

        Args:
            n_components: Number of mixture components K.
            radius: Circle radius for the means.
            sd_major: Standard deviation along the (tangential) major
                axis.
            sd_minor: Standard deviation along the (radial) minor axis.
            dtype: Floating dtype.

        Returns:
            The constructed AnisotropicGMM.
        """
        angles = 2.0 * math.pi * torch.arange(n_components, dtype=dtype)
        angles = angles / n_components
        means = radius * torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=-1
        )

        covariances = torch.zeros(n_components, 2, 2, dtype=dtype)
        diag = torch.diag(
            torch.tensor([sd_major**2, sd_minor**2], dtype=dtype)
        )
        for k in range(n_components):
            # Tangential major axis: rotate the (major, minor) frame so
            # the major axis is perpendicular to the radial direction.
            theta = angles[k] + math.pi / 2.0
            c, s = torch.cos(theta), torch.sin(theta)
            rot = torch.tensor([[c, -s], [s, c]], dtype=dtype)
            covariances[k] = rot @ diag @ rot.T
        return cls(means, covariances, dtype=dtype)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density log pi(x).

        Args:
            x: Points, shape (n, d).

        Returns:
            Log-densities, shape (n,).
        """
        return self.dist.log_prob(x.to(self.dtype))

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Score grad_x log pi(x) of the (normalized) mixture.

        Computed in closed form as a responsibility-weighted sum of the
        per-component scores, avoiding autograd. For component k the
        score is -Sigma_k^{-1} (x - mu_k); responsibilities are the
        softmax of the per-component log weighted densities.

        Args:
            x: Points, shape (n, d).

        Returns:
            Scores, shape (n, d).
        """
        x = x.to(self.dtype)
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (n, K, d)
        # Per-component log weighted density for responsibilities.
        comp = MultivariateNormal(
            self.means, covariance_matrix=self.covariances
        )
        log_w = torch.log(self.weights)
        log_resp = log_w.unsqueeze(0) + comp.log_prob(x.unsqueeze(1))
        resp = torch.softmax(log_resp, dim=-1)  # (n, K)
        # Per-component score -Sigma_k^{-1} (x - mu_k).
        prec = torch.linalg.inv(self.covariances)  # (K, d, d)
        comp_score = -torch.einsum("kij,nkj->nki", prec, diff)
        return torch.einsum("nk,nki->ni", resp, comp_score)

    def post_log_dens(self, x: torch.Tensor) -> torch.Tensor:
        """UNNORMALIZED mixture log-density (reference ``post_log_dens``).

        log pi(x) = logsumexp_k[ log w_k - 0.5 (x - mu_k)^T Sigma_k^{-1}
        (x - mu_k) ], WITHOUT the per-component Gaussian normalization
        constants. This is the target used by the Fredholm / SVGD paths
        and by the cross-entropy metric (matching ``gmm_2d_benchmark.py``);
        the missing constant offset is identical for every method, so it
        cancels in all method-vs-method comparisons and in the MSIP map.

        Args:
            x: Points, shape (n, d).

        Returns:
            Unnormalized log-densities, shape (n,).
        """
        x = x.to(self.dtype)
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (n, K, d)
        # quad[n, k] = (x - mu_k)^T Sigma_k^{-1} (x - mu_k).
        quad = torch.einsum("nki,kij,nkj->nk", diff, self.precisions, diff)
        log_w = torch.log(self.weights)
        return torch.logsumexp(log_w.unsqueeze(0) - 0.5 * quad, dim=-1)

    def post_score(self, x: torch.Tensor) -> torch.Tensor:
        """Score of the UNNORMALIZED density (eq. matching post_log_dens).

        grad_x log pi(x) = sum_k r_k(x) (-Sigma_k^{-1} (x - mu_k)) with
        responsibilities r_k from the unnormalized component log-weights.
        Because the omitted Gaussian normalizers are constants in x, this
        equals the score of the normalized mixture; it is provided as the
        score paired with ``post_log_dens`` for the Fredholm/SVGD paths.

        Args:
            x: Points, shape (n, d).

        Returns:
            Scores, shape (n, d).
        """
        x = x.to(self.dtype)
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (n, K, d)
        quad = torch.einsum("nki,kij,nkj->nk", diff, self.precisions, diff)
        log_w = torch.log(self.weights)
        resp = torch.softmax(log_w.unsqueeze(0) - 0.5 * quad, dim=-1)  # (n,K)
        comp_score = -torch.einsum("kij,nkj->nki", self.precisions, diff)
        return torch.einsum("nk,nki->ni", resp, comp_score)

    def sample(self, n: int) -> torch.Tensor:
        """Draw n i.i.d. samples from the mixture, shape (n, d)."""
        return self.dist.sample((n,))

    def _tilde_dists(self, sigma: float) -> MultivariateNormal:
        """Per-component smoothed Gaussians N(.; mu_k, Sigma_k +
        sigma^2 I) used throughout the closed forms."""
        eye = torch.eye(self.d, dtype=self.dtype, device=self.means.device)
        tilde = self.covariances + sigma**2 * eye
        return MultivariateNormal(self.means, covariance_matrix=tilde)

    def log_v0(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Log kernel mean embedding log v0(y) (eqs. 36-38).

        log v0(y) = log Z_sigma + logsumexp_k[ log m_k + log
        N(y; mu_k, Sigma_tilde_k) ]. Computed in log-space for overflow
        safety.

        Args:
            y: Nodes, shape (m, d).
            sigma: Kernel bandwidth.

        Returns:
            log v0 at each node, shape (m,).
        """
        y = y.to(self.dtype)
        tilde = self._tilde_dists(sigma)
        log_w = torch.log(self.weights)
        # log_comp[m, k] = log m_k + log N(y_m; mu_k, Sigma_tilde_k).
        log_comp = log_w.unsqueeze(0) + tilde.log_prob(y.unsqueeze(1))
        return log_z_sigma(sigma, self.d) + torch.logsumexp(log_comp, dim=-1)

    def v0(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Kernel mean embedding v0(y) = exp(log_v0(y)), shape (m,)."""
        return torch.exp(self.log_v0(y, sigma))

    def grad_log_v0(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Gradient of log v0, grad_log_v0(y) (eqs. 39-40).

        grad_log_v0(y) = sum_k r_k(y) (-Sigma_tilde_k^{-1} (y - mu_k)),
        with smoothed responsibilities r_k(y) = m_k N(y; mu_k,
        Sigma_tilde_k) / sum_j m_j N(y; mu_j, Sigma_tilde_j). This is the
        score of v0 viewed as an unnormalized smoothed mixture.

        Args:
            y: Nodes, shape (m, d).
            sigma: Kernel bandwidth.

        Returns:
            Gradient of log v0, shape (m, d).
        """
        y = y.to(self.dtype)
        tilde = self._tilde_dists(sigma)
        log_w = torch.log(self.weights)
        log_resp = log_w.unsqueeze(0) + tilde.log_prob(y.unsqueeze(1))
        resp = torch.softmax(log_resp, dim=-1)  # (m, K)

        eye = torch.eye(self.d, dtype=self.dtype, device=self.means.device)
        prec_tilde = torch.linalg.inv(self.covariances + sigma**2 * eye)
        diff = y.unsqueeze(1) - self.means.unsqueeze(0)  # (m, K, d)
        comp_grad = -torch.einsum("kij,mkj->mki", prec_tilde, diff)
        return torch.einsum("mk,mki->mi", resp, comp_grad)

    def v1(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Kernelized first moment V1(y) (eqs. 36-41).

        Uses the identity v1(y) = v0(y) (y + sigma^2 grad_log_v0(y)).

        Args:
            y: Nodes, shape (m, d).
            sigma: Kernel bandwidth.

        Returns:
            First-moment embedding, shape (m, d).
        """
        y = y.to(self.dtype)
        v0 = self.v0(y, sigma).unsqueeze(-1)  # (m, 1)
        return v0 * (y + sigma**2 * self.grad_log_v0(y, sigma))

    def c_pi(self, sigma: float) -> torch.Tensor:
        """Self-affinity constant C_pi(sigma) (eqs. 5, 42).

        C_pi = Z_sigma sum_{k,l} m_k m_l N(mu_k; mu_l, Sigma_k + Sigma_l
        + sigma^2 I). This is E_{X,X'~pi}[k(X, X')] for normalized pi and
        is the configuration-independent term of the MMD. Assembled in
        log-space.

        Args:
            sigma: Kernel bandwidth.

        Returns:
            Scalar C_pi as a (0-dim) tensor.
        """
        eye = torch.eye(self.d, dtype=self.dtype, device=self.means.device)
        # Pairwise covariances Sigma_k + Sigma_l + sigma^2 I.
        cov_sum = (
            self.covariances.unsqueeze(1)
            + self.covariances.unsqueeze(0)
            + sigma**2 * eye
        )  # (K, K, d, d)
        diff = self.means.unsqueeze(1) - self.means.unsqueeze(0)  # (K,K,d)
        pair = MultivariateNormal(
            torch.zeros(self.d, dtype=self.dtype, device=self.means.device),
            covariance_matrix=cov_sum,
        )
        log_n = pair.log_prob(diff)  # (K, K)
        log_w = torch.log(self.weights)
        log_terms = log_w.unsqueeze(1) + log_w.unsqueeze(0) + log_n
        log_c = log_z_sigma(sigma, self.d) + torch.logsumexp(
            log_terms.reshape(-1), 0
        )
        return torch.exp(log_c)

    def mmd_sq(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Analytic squared MMD between pi and mu = sum_i w_i delta_{y_i}.

        MMD^2 = C_pi - 2 sum_i w_i v0(y_i) + sum_{ij} w_i w_j k(y_i, y_j),
        using the same kernel bandwidth sigma. Weights are first
        normalized to sum to one, which makes the embedding term use the
        normalized-pi embedding v0. The
        v0 cross term is formed as exp(log_v0 - log Z_sigma) * Z_sigma via
        log_v0 for stability.

        Args:
            y: Nodes, shape (m, d).
            w: Weights, shape (m,). Normalized internally.
            sigma: Kernel bandwidth.

        Returns:
            Scalar MMD^2 as a (0-dim) tensor (clamped at 0 to absorb tiny
            negative round-off).
        """
        y = y.to(self.dtype)
        w = w.to(self.dtype)
        # Guard the unit-sum normalization: signed weights can sum to ~0,
        # which would silently produce NaN on the divide. The threshold is
        # loose enough for float32 round-off on near-unit-sum weights.
        w_sum = w.sum()
        assert w_sum.abs() > 1e-6, (
            "mmd_sq: weights sum to ~0; cannot normalize to unit sum"
        )
        w = w / w_sum

        c_pi = self.c_pi(sigma)
        v0 = self.v0(y, sigma)  # (m,)
        cross = (w * v0).sum()
        k_mat = se_kernel(y, y, sigma).to(self.dtype)
        quad = w @ k_mat @ w
        mmd2 = c_pi - 2.0 * cross + quad
        return torch.clamp(mmd2, min=0.0)


def cross_entropy(
    gmm: "AnisotropicGMM",
    pts: torch.Tensor,
    wts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy CE = -E_mu[log pi] of a particle measure.

    The primary benchmark metric of ``gmm_2d_benchmark.py``: for uniform
    weights CE = -(1/N) sum_i log pi(y_i); for weighted measures (MSIP)
    CE = -(log pi evals) @ w with w renormalized to sum 1. Smaller is
    better. Uses the UNNORMALIZED ``post_log_dens``; the omitted constant
    offset is identical for every method, so method-vs-method differences
    are unaffected.

    Args:
        gmm: Target ``AnisotropicGMM``.
        pts: Particles, shape (m, d).
        wts: Optional weights, shape (m,). Uniform if None; renormalized
            to sum to one when given.

    Returns:
        Scalar cross-entropy as a (0-dim) tensor.
    """
    log_dens = gmm.post_log_dens(pts)  # (m,)
    if wts is None:
        return -log_dens.mean()
    wts = wts.to(log_dens.dtype)
    wts = wts / wts.sum()
    return -(log_dens @ wts)
