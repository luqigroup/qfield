"""Linear-Gaussian inverse problem with a closed-form posterior.

The canonical sanity-check target for conditional inference: a Gaussian
prior and a linear-Gaussian forward map give a Gaussian posterior in
closed form, so the conditional data-driven MSIP quadrature can be
checked against an exact mean / covariance and an analytic posterior MMD.

    prior     x ~ N(mu_pr, Sigma_pr),
    forward   y = A x + eps,   eps ~ N(0, Sigma_obs),
    joint     (x, y) ~ N( [mu_pr; A mu_pr],
                          [[Sigma_pr,         Sigma_pr A^T            ],
                           [A Sigma_pr,       A Sigma_pr A^T + Sigma_obs]] ),
    posterior p(x | y*) = N(mu_post, Sigma_post) with
              Sigma_post = (Sigma_pr^{-1} + A^T Sigma_obs^{-1} A)^{-1},
              mu_post    = Sigma_post (Sigma_pr^{-1} mu_pr
                           + A^T Sigma_obs^{-1} y*).

Posterior MMD. For the squared-exponential kernel k(a, b) =
exp(-||a - b||^2 / (2 sigma^2)) and a weighted node set nu = sum_i w_i
delta_{z_i} (unit-sum weights),

    MMD^2(post, nu) = C_post - 2 sum_i w_i mu_k(z_i) + sum_{ij} w_i w_j
                      k(z_i, z_j),

using the Gaussian-vs-Gaussian and Gaussian-vs-point closed forms (same
posterior mean, so all mean differences against the posterior collapse):

    mu_k(z) = E_{x~N(mu_post,Sigma_post)}[k(x, z)]
            = |I + Sigma_post / sigma^2|^{-1/2}
              exp(-0.5 (z - mu_post)^T (Sigma_post + sigma^2 I)^{-1}
                  (z - mu_post)),
    C_post  = E_{x,x'~post}[k(x, x')] = |I + 2 Sigma_post / sigma^2|^{-1/2}.

The closed-form linear algebra (inverses, determinants, the Cholesky for
sampling) is carried out in float64 for accuracy and the results are
returned in the requested ``dtype`` (float32 for the experiments),
mirroring ``AnisotropicGMM``.

Reference:
  - Belhadji, Sharp, Marzouk, "To discretize continually: Mean shift
    interacting particle systems for Bayesian inference", arXiv:2605.14142
    (squared-exponential kernel and analytic MMD, eqs. 4-5).
  - Belhadji, Sharp, Marzouk, "Mean-shift interacting particles" (data-driven
    MSIP), arXiv:2502.10600.
"""

import torch
from torch.distributions import MultivariateNormal


class LinearGaussian:
    """Linear-Gaussian inverse problem y = A x + eps with x ~ N(mu, Sigma).

    Provides joint sampling, the analytic Gaussian posterior, and the
    analytic squared MMD between that posterior and a weighted node set
    under the squared-exponential kernel. All stored tensors use the
    requested ``dtype``; the internal closed-form linear algebra is done
    in float64 and cast back, so the float32 path stays accurate.

    Args:
        prior_mean: Prior mean mu_pr, shape (dx,).
        prior_cov: Prior covariance Sigma_pr, shape (dx, dx) (SPD).
        forward: Forward operator A, shape (dy, dx).
        obs_cov: Observation-noise covariance Sigma_obs, shape (dy, dy)
            (SPD).
        dtype: Floating dtype for stored tensors and outputs (defaults to
            float32, matching the experiments).
    """

    def __init__(
        self,
        prior_mean: torch.Tensor,
        prior_cov: torch.Tensor,
        forward: torch.Tensor,
        obs_cov: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        self.dtype = dtype
        # Keep a float64 copy for the closed forms; expose float32 tensors.
        self._mu_pr = prior_mean.reshape(-1).double()
        self._sigma_pr = prior_cov.double()
        self._a = forward.double()
        self._sigma_obs = obs_cov.double()
        self.dy, self.dx = self._a.shape

        self.prior_mean = self._mu_pr.to(dtype)
        self.prior_cov = self._sigma_pr.to(dtype)
        self.forward = self._a.to(dtype)
        self.obs_cov = self._sigma_obs.to(dtype)

        # Joint Gaussian (x, y) blocks (float64).
        self._mu_y = self._a @ self._mu_pr  # (dy,)
        self._cov_yy = self._a @ self._sigma_pr @ self._a.T + self._sigma_obs
        self._cov_xy = self._sigma_pr @ self._a.T  # (dx, dy)

        # Constant posterior precision / covariance (y* independent).
        prec_pr = torch.linalg.inv(self._sigma_pr)
        prec_obs = torch.linalg.inv(self._sigma_obs)
        self._post_prec = prec_pr + self._a.T @ prec_obs @ self._a
        self._sigma_post = torch.linalg.inv(self._post_prec)
        self._prec_pr = prec_pr
        self._prec_obs = prec_obs

    def sample_joint(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return n i.i.d. joint samples (x_i, y_i) ~ p(x, y).

        Samples x ~ N(mu_pr, Sigma_pr) from the prior and y = A x + eps
        with eps ~ N(0, Sigma_obs), the generative process itself (no need
        to form the joint covariance for sampling).

        Args:
            n: Number of joint samples.

        Returns:
            (X [n, dx], Y [n, dy]) in the configured dtype.
        """
        x_dist = MultivariateNormal(self._mu_pr, covariance_matrix=self._sigma_pr)
        eps_dist = MultivariateNormal(
            torch.zeros(self.dy, dtype=torch.float64),
            covariance_matrix=self._sigma_obs,
        )
        x = x_dist.sample((n,))  # (n, dx) float64
        eps = eps_dist.sample((n,))  # (n, dy) float64
        y = x @ self._a.T + eps  # (n, dy)
        return x.to(self.dtype), y.to(self.dtype)

    def posterior(
        self, y_star: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytic Gaussian posterior p(x | y*) = N(mu_post, Sigma_post).

        Sigma_post = (Sigma_pr^{-1} + A^T Sigma_obs^{-1} A)^{-1} (constant
        in y*); mu_post = Sigma_post (Sigma_pr^{-1} mu_pr + A^T
        Sigma_obs^{-1} y*).

        Args:
            y_star: Query observation, shape (dy,) or (1, dy).

        Returns:
            (mu_post [dx], Sigma_post [dx, dx]) in the configured dtype.
        """
        y = y_star.reshape(-1).double()
        rhs = self._prec_pr @ self._mu_pr + self._a.T @ (self._prec_obs @ y)
        mu_post = self._sigma_post @ rhs
        return mu_post.to(self.dtype), self._sigma_post.to(self.dtype)

    def posterior_mean_batch(self, y: torch.Tensor) -> torch.Tensor:
        """Vectorized analytic posterior mean ``mu_post(y)`` for a batch.

        For a batch ``y`` of shape ``(B, dy)`` returns the analytic
        posterior mean for each row, shape ``(B, dx)``:

            mu_post(y) = Sigma_post (Sigma_pr^{-1} mu_pr
                                     + A^T Sigma_obs^{-1} y)

        The closed-form precision / covariance matrices are precomputed
        once at construction, so this is a single matmul per call --
        cheap enough to drop inside the CNCV training loop for the
        variance QoI ``h(x; y) = (x - mu_post(y))^2``.

        Args:
            y: Observation batch, shape (B, dy) (or (dy,), which is
                treated as a single row).

        Returns:
            Batched posterior means, shape (B, dx) in the configured
            dtype, on the device / dtype of ``y``.
        """
        if y.dim() == 1:
            y_batch = y.unsqueeze(0)
        else:
            y_batch = y
        device = y_batch.device
        dtype = y_batch.dtype
        prec_pr = self._prec_pr.to(device=device, dtype=dtype)
        prec_obs = self._prec_obs.to(device=device, dtype=dtype)
        a = self.forward.to(device=device, dtype=dtype)
        mu_pr = self.prior_mean.to(device=device, dtype=dtype)
        sigma_post = self._sigma_post.to(device=device, dtype=dtype)
        # rhs: (B, dx).
        rhs = (prec_pr @ mu_pr).unsqueeze(0) + y_batch @ (prec_obs @ a)
        mu_post_batch = rhs @ sigma_post.T
        if y.dim() == 1:
            return mu_post_batch.squeeze(0)
        return mu_post_batch

    def log_evidence(self, y_star: torch.Tensor) -> torch.Tensor:
        """Closed-form log marginal likelihood ``log p(y*)``.

        The y-marginal of the joint Gaussian is ``N(_mu_y, _cov_yy)`` with
        ``_mu_y = A mu_pr`` and ``_cov_yy = A Sigma_pr A^T + Sigma_obs``
        (both precomputed at construction), so the evidence is just the
        Gaussian log-density of ``y*`` under that marginal. It is the exact
        evidence, not the empirical Nadaraya-Watson normalizer. Computed in
        float64 and cast back to the configured dtype.

        Args:
            y_star: Query observation, shape (dy,) or (1, dy).

        Returns:
            Scalar ``log p(y*)`` as a (0-dim) tensor in the configured
            dtype.
        """
        y = y_star.reshape(-1).double()
        dist = MultivariateNormal(self._mu_y, covariance_matrix=self._cov_yy)
        return dist.log_prob(y).to(self.dtype)

    def sample_posterior(self, y_star: torch.Tensor, n: int) -> torch.Tensor:
        """Return n i.i.d. samples from the analytic posterior p(x | y*).

        The raw-posterior-samples baseline: a same-budget Monte-Carlo
        reference the quadrature is compared against.

        Args:
            y_star: Query observation, shape (dy,).
            n: Number of samples.

        Returns:
            Samples, shape (n, dx), in the configured dtype.
        """
        mu_post, _ = self.posterior(y_star)
        dist = MultivariateNormal(
            mu_post.double(), covariance_matrix=self._sigma_post
        )
        return dist.sample((n,)).to(self.dtype)

    def posterior_mmd_sq(
        self,
        nodes: torch.Tensor,
        weights: torch.Tensor,
        y_star: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Analytic squared MMD between p(x | y*) and sum_i w_i delta_{z_i}.

        MMD^2 = C_post - 2 sum_i w_i mu_k(z_i) + sum_{ij} w_i w_j
        k(z_i, z_j), with the Gaussian-vs-point mean mu_k and the
        Gaussian self-affinity C_post = |I + 2 Sigma_post / sigma^2|^{-1/2}
        (see the module docstring). Weights are renormalized to unit sum;
        the result is clamped at 0 to absorb tiny float round-off.

        Thin wrapper over :func:`qfield.metrics_mmd.mmd_sq_gaussian`
        with this problem's closed-form posterior ``(mu_post, Sigma_post)``;
        the float64 algebra and the unit-sum and clamp conventions live in
        that target-agnostic helper, and the result is cast to the
        configured ``dtype``.

        Args:
            nodes: Node set z, shape (m, dx).
            weights: Node weights, shape (m,). Renormalized internally.
            y_star: Query observation, shape (dy,).
            sigma: Kernel bandwidth (the MSIP bandwidth sigma_x).

        Returns:
            Scalar MMD^2 as a (0-dim) tensor in the configured dtype.
        """
        from qfield.metrics_mmd import mmd_sq_gaussian

        mu_post, sigma_post = self.posterior(y_star)
        mmd2 = mmd_sq_gaussian(nodes, weights, mu_post, sigma_post, sigma)
        return mmd2.to(self.dtype)
