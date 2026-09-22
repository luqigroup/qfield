"""Conditional, data-driven MSIP embeddings from joint samples.

A conditional posterior quadrature built purely from joint samples
``{(x_i, y_i)}_{i=1..N} ~ p(x, y)`` and a query ``y*``, with no density, no
score, and no inner quadrature, only kernel sums (the data-driven MSIP of
arXiv:2502.10600, conditioned via Nadaraya-Watson).

Data-driven MSIP for a weighted empirical measure
``pi = sum_i alpha_i delta_{x_i}`` (alpha_i >= 0, sum_i alpha_i = 1) has
the kernel-sum embeddings (arXiv:2502.10600)

    v0(x')      = sum_i alpha_i k_x(x_i, x'),
    v1(x')      = sum_i alpha_i x_i k_x(x_i, x'),

with the squared-exponential kernel k_x(x, x') = exp(-||x - x'||^2 /
(2 sigma_x^2)). The MSIP estimator convention (see
``qfield.estimators``) is the pair (log_v0, sigma_x^2
grad_log_v0), where sigma_x^2 grad_log_v0(x') = v1(x') / v0(x') - x' is
the weighted mean-shift vector. Writing these in the log-stable form the
map consumes:

    log_v0(x')              = logsumexp_i( log alpha_i
                              - ||x_i - x'||^2 / (2 sigma_x^2) ),
    sigma_x^2 grad_log_v0(x') = (sum_i p_i(x') x_i) - x',
    p_i(x')                 = softmax_i( log alpha_i
                              - ||x_i - x'||^2 / (2 sigma_x^2) ).

The global kernel constant Z_{sigma_x} cancels in the MSIP map
(Prop. 3.2), so it is dropped from log_v0 (matching ``FredholmEmbeddings``
and ``MCEmbeddings``); the softmax weights p_i are normalization-free in
alpha by construction.

CONDITIONING (Nadaraya-Watson). For the posterior p(x | y*), the
data-driven weights are the y-kernel-smoothed responsibilities

    alpha_i(y*) = k_y(y_i, y*) / sum_j k_y(y_j, y*),
    k_y(y, y')  = exp(-||y - y'||^2 / (2 sigma_y^2)),

so ``log alpha_i(y*) = log_softmax_i( -||y_i - y*||^2 / (2 sigma_y^2) )``.
This realizes the data-driven estimate p(x | y*) ~ sum_i alpha_i(y*)
delta_{x_i}, which MSIP then quantizes into M weighted nodes.

LATENT (FEATURE-SPACE) KERNEL. An optional feature map
``phi : R^{dx} -> R^{r}`` (r << dx) moves the MSIP kernel onto an informed
subspace while the mean shift still moves the ambient node. With
``feature_map`` set, the x-distance in the embedding is measured in feature
space, ``||phi(x_i) - phi(x')||^2``, so the responsibilities, the v0
embedding, and (downstream) the node Gram are all computed on phi(x):

    v0(x')   = sum_i alpha_i(y*) k_phi(x_i, x'),
    p_i(x')  = softmax_i( log alpha_i(y*) - ||phi(x_i) - phi(x')||^2
               / (2 sigma_x^2) ),

with k_phi(x, x') = exp(-||phi(x) - phi(x')||^2 / 2 sigma_x^2) the
feature-space squared-exponential kernel. The single quantity the feature
map does not enter is the mean-shift vector, which keeps moving the node
in the ambient space toward the responsibility-weighted ambient samples,

    sigma_x^2 grad_log_v0(x') = (sum_i p_i(x') x_i) - x',   x' in R^{dx},

so the returned quadrature {z_j, w_j} is ambient and integrates an ordinary
functional h : R^{dx} -> R. The latent acts as a metric on responsibilities
and weights, not as a coordinate the nodes live in. The same ``feature_map``
must be threaded into the MSIP node Gram (the ``msip.py`` weight solve and
the SE map dispatch take a matching ``feature_map`` argument). With
``feature_map=None`` (the default) the ambient path is recovered bit for
bit. The feature-space kernel is positive semi-definite but not strictly
positive definite (it is constant along the fibers of phi), so the node
Gram is restored to invertibility by the existing ``kernel_diag_infl``
diagonal inflation rather than by strict positive-definiteness.

The whole pipeline is log-stable (logsumexp / softmax / log_softmax) so
it stays finite in float32 even when the bandwidths make individual
kernel weights underflow.

Reference:
  - Belhadji, Sharp, Marzouk, "Weighted Quantization Using MMD: From Mean
    Field to Mean Shift via Gradient Flows", arXiv:2502.10600 (the original
    DATA-DRIVEN MSIP from samples alone; kernel-sum embeddings).
  - Belhadji, Sharp, Marzouk, "To discretize continually: Mean shift
    interacting particle systems for Bayesian inference", arXiv:2605.14142
    (regularized MSIP map; estimator convention, eqs. 21-30).
"""

import math

import torch

from qfield.msip import run_msip


class DataDrivenEmbeddings:
    """Unconditional data-driven MSIP embeddings from target samples.

    The score-free, density-free quantization of a single target known
    only through i.i.d. samples ``{x_i}_{i=1..N} ~ pi`` (the original
    data-driven MSIP of arXiv:2502.10600). With uniform weights ``alpha_i
    = 1 / N`` the kernel-sum embeddings are

        v0(x')      = (1 / N) sum_i k_x(x_i, x'),
        v1(x')      = (1 / N) sum_i x_i k_x(x_i, x'),

    and this class returns the reference pair ``(log_v0, sigma_x^2
    grad_log_v0)`` the MSIP map consumes (see the module docstring),

        log_v0(x')              = logsumexp_i(-||x_i - x'||^2 /
                                  (2 sigma_x^2)) - log N,
        sigma_x^2 grad_log_v0(x') = (sum_i p_i(x') x_i) - x',
        p_i(x')                 = softmax_i(-||x_i - x'||^2 /
                                  (2 sigma_x^2)).

    The global kernel constant ``Z_{sigma_x}`` and the ``- log N`` offset
    cancel in the MSIP map (Prop. 3.2), so log_v0 may carry an arbitrary
    additive constant; the ``- log N`` is kept only so log_v0 is a
    bona-fide log-density estimate. This is the unconditional
    (degenerate-conditioning) case of
    :class:`ConditionalDataDrivenEmbeddings` -- equivalently, the
    conditional class with uniform ``alpha_i`` -- factored out as its own
    minimal estimator for a single target (no ``y`` samples, no query
    ``y*``).

    In the source paper (Belhadji, Sharp, Marzouk, "Weighted Quantization
    Using MMD: From Mean Field to Mean Shift via Gradient Flows",
    arXiv:2502.10600) these sums are eq. 5 with ``pi`` the empirical
    measure -- equivalently the KDE of their eq. 11 -- and the kernelized
    first moment ``v1`` of eq. 23; the damped SE fixed point is their
    eq. 30 and the sample-only listing their Algorithm 2. With these
    embeddings, ``qfield.msip.run_msip`` on the SE path with
    ``bounds=None`` is their Algorithm 2 (``Kbar = K / sigma_x^2``
    collapses the corrected map), up to three differences: the map solves
    the ``kernel_diag_infl``-inflated Gram (their listing carries no
    inflation; pass a tiny value); the returned stationary weights are
    normalized to unit sum (theirs are the raw ``K^{-1} v0``); and
    ``run_msip``'s default ``bounds=(-100, 100)`` box-clamps every
    iterate (their listing has no box, hence the ``bounds=None``
    qualifier above). ``msip_map`` also holds a node fixed for the step
    when its mapped position is non-finite, a numerical safety net their
    listing does not carry.

    Memory: each ``__call__`` holds ~3 coexisting dense ``(m, N)``
    float64 buffers (squared distances, log-kernel weights,
    responsibilities) -- ~25 MB at m = 256 nodes over N = 4096 atoms --
    so no bank chunking is implemented; chunk over the bank (streaming
    logsumexp / running moment sums) before raising these scales by
    orders of magnitude.

    Iterate-magnitude instrumentation (arXiv:2605.14142 clamps iterates
    to (-1e3, 1e3), its Table 3; arXiv:2502.10600's Algorithm 2 carries
    no clamp). The object tracks the running maximum absolute coordinate
    over every query batch it is evaluated at, in ``self.max_abs_query``
    (with ``self.n_queries`` the number of query points seen). Over one
    ``run_msip(..., bounds=None)`` call the embeddings are evaluated at
    every iterate ``Y^(0) .. Y^(T)`` (T map calls plus the final
    stationary-weight solve; the ``record_fn`` path only revisits the
    same iterates), so after the run ``max_abs_query`` equals the
    trajectory-wide max |coordinate|. Read it destructively with
    :meth:`pop_query_stats` immediately after the run: any later
    evaluation of the embeddings off the trajectory (a KDE grid probe,
    scoring at the bank atoms, a plotting diagnostic) folds into the same
    running max and contaminates a later read. Non-finite query batches
    raise ``ValueError`` before the tracker updates, so ``max(x, nan)``
    can never silently skip a batch. :meth:`reset_query_stats` re-arms it
    explicitly.

    Args:
        x_samples: Target samples, shape (N, d) with N >= 1, finite.
        sigma_x: x-kernel (MSIP) bandwidth (finite, > 0); ~ the target
            length scale.
        weights: Optional non-negative sample weights ``alpha_i``, shape
            (N,), finite with positive sum; uniform ``1 / N`` if None.
            Normalized in log-space.
        dtype: Dtype of the public ``self.x_samples`` / ``self.log_alpha``
            attributes (defaults to float32). All kernel sums run in
            float64 internally regardless of ``dtype`` (on a float64 copy
            of the original ``x_samples``, so no input precision is lost),
            and the outputs are cast back to the query's dtype.
    """

    def __init__(
        self,
        x_samples: torch.Tensor,
        sigma_x: float,
        weights: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if x_samples.ndim != 2 or x_samples.shape[0] < 1:
            raise ValueError(
                "DataDrivenEmbeddings: x_samples must be (N, d) with "
                f"N >= 1; got shape {tuple(x_samples.shape)}"
            )
        if not torch.isfinite(x_samples).all():
            raise ValueError(
                "DataDrivenEmbeddings: x_samples contains non-finite "
                "entries"
            )
        self.sigma_x = float(sigma_x)
        if not (self.sigma_x > 0.0 and math.isfinite(self.sigma_x)):
            raise ValueError(
                "DataDrivenEmbeddings: sigma_x must be finite and > 0; "
                f"got {sigma_x}"
            )
        self.dtype = dtype
        self.x_samples = x_samples.to(dtype)  # (N, d) public, interface dtype
        # Every kernel sum runs on a float64 copy of the original bank (full
        # input precision); the outputs are cast back to the query's dtype in
        # __call__.
        self._x64 = x_samples.detach().double()  # (N, d)
        self.n, self.dx = self.x_samples.shape

        if weights is None:
            # Uniform: log alpha_i = -log N (a constant, dropped by the map
            # but kept so log_v0 is a genuine log-density estimate).
            log_alpha64 = torch.full(
                (self.n,), -math.log(float(self.n)), dtype=torch.float64
            )
        else:
            if weights.shape != (self.n,):
                raise ValueError(
                    "DataDrivenEmbeddings: weights must have shape "
                    f"({self.n},); got {tuple(weights.shape)}"
                )
            w64 = weights.detach().double()
            if (
                not torch.isfinite(w64).all()
                or (w64 < 0).any()
                or not w64.sum() > 0
            ):
                raise ValueError(
                    "DataDrivenEmbeddings: weights must be finite and "
                    "non-negative with a positive sum"
                )
            log_alpha64 = torch.log_softmax(torch.log(w64), dim=0)
        self._log_alpha64 = log_alpha64  # (N,) float64 compute copy
        self.log_alpha = log_alpha64.to(dtype)  # public, interface dtype

        # Tracker state (see the class docstring).
        self.max_abs_query: float = float("-inf")
        self.n_queries: int = 0

    def reset_query_stats(self) -> None:
        """Zero the max-|coordinate| tracker (call before each run)."""
        self.max_abs_query = float("-inf")
        self.n_queries = 0

    def pop_query_stats(self) -> tuple[float, int]:
        """Destructive read: return ``(max_abs_query, n_queries)``, reset.

        Consume the trajectory statistics through this method, immediately
        after the ``run_msip`` call: every later evaluation of the
        embeddings -- an off-trajectory diagnostic such as a KDE grid
        probe, scoring at the bank atoms, or a plotting pass -- folds into
        the same running max and would contaminate a later read. Popping
        closes that window; the next run starts from a clean tracker.

        Returns:
            (max_abs_query, n_queries): the running max |coordinate| over
            all query points seen since the last reset (``-inf`` if none
            were seen) and the number of query points seen.
        """
        stats = (self.max_abs_query, self.n_queries)
        self.reset_query_stats()
        return stats

    def __call__(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Data-driven embeddings at the nodes (float64-internal).

        Args:
            nodes: Query nodes x', shape (m, dx), finite.

        Returns:
            (log_v0 [m], sigma_x^2 grad_log_v0 [m, dx]), cast to
            ``nodes.dtype``.
        """
        if nodes.ndim != 2 or nodes.shape[1] != self.dx:
            raise ValueError(
                f"DataDrivenEmbeddings: query must be (m, {self.dx}); "
                f"got shape {tuple(nodes.shape)}"
            )
        yd = nodes.double()
        if self._x64.device != yd.device:
            self._x64 = self._x64.to(yd.device)
            self._log_alpha64 = self._log_alpha64.to(yd.device)
        if yd.numel() > 0:
            # Mirror the bank's isfinite check on every query batch
            # before the tracker updates, so max(x, nan) can never
            # silently skip a batch.
            if not torch.isfinite(yd).all():
                raise ValueError(
                    "DataDrivenEmbeddings: query contains non-finite "
                    "entries"
                )
            self.max_abs_query = max(
                self.max_abs_query, yd.abs().max().item()
            )
            self.n_queries += int(yd.shape[0])
        # Pairwise squared x-distances ||x_i - x'_j||^2, shape (m, N) --
        # the first of the ~3 coexisting (m, N) float64 buffers of the
        # class docstring's memory note.
        sq_dist_x = torch.cdist(yd, self._x64) ** 2
        # log w_ji = log alpha_i - ||x_i - x'_j||^2 / (2 sigma_x^2).
        log_w = self._log_alpha64.unsqueeze(0) - sq_dist_x / (
            2.0 * self.sigma_x**2
        )  # (m, N)
        # log_v0(x') = logsumexp_i log w_ji (Z_sigma_x dropped; it cancels).
        log_v0 = torch.logsumexp(log_w, dim=-1)  # (m,)
        # p_ji = softmax_i log w_ji ; weighted x-mean minus the node gives
        # the mean-shift vector sigma_x^2 grad_log_v0 = v1 / v0 - x'.
        p = torch.softmax(log_w, dim=-1)  # (m, N)
        weighted_mean = p @ self._x64  # (m, dx)
        sigma_sq_grad_log_v0 = weighted_mean - yd
        return (
            log_v0.to(nodes.dtype),
            sigma_sq_grad_log_v0.to(nodes.dtype),
        )


class ConditionalDataDrivenEmbeddings:
    """Conditional data-driven MSIP embeddings from joint samples.

    Builds the embeddings of the Nadaraya-Watson conditional empirical
    measure p(x | y*) ~ sum_i alpha_i(y*) delta_{x_i} from joint samples
    ``{(x_i, y_i)}`` and a query ``y*``, returning the reference pair
    (log_v0, sigma_x^2 grad_log_v0) the MSIP map consumes (see the module
    docstring for the formulas). The y-conditioning log-weights
    ``log alpha_i(y*) = log_softmax_i(-||y_i - y*||^2 / (2 sigma_y^2))``
    are precomputed once at construction; each ``__call__`` evaluates only
    the x-kernel sums at the given nodes.

    Args:
        x_samples: Joint x-samples, shape (N, dx).
        y_samples: Joint y-samples, shape (N, dy).
        y_star: Query observation, shape (dy,) or (1, dy).
        sigma_x: x-kernel (MSIP) bandwidth; ~ posterior length scale.
        sigma_y: y-kernel (Nadaraya-Watson) bandwidth; controls the
            conditional-estimate bias/variance (too small -> few effective
            samples / high variance; too large -> over-smoothed posterior /
            bias).
        feature_map: Optional callable ``phi : (B, dx) -> (B, r)`` moving
            the MSIP x-kernel onto an informed subspace. When set, the
            responsibilities and the v0 embedding measure x-distances as
            ``||phi(x_i) - phi(x')||^2`` (the latent kernel); the mean-shift
            vector still moves the ambient node toward the
            responsibility-weighted ambient samples. The cached features
            ``phi(x_i)`` of the joint x-bank are precomputed once at
            construction. ``None`` (default) recovers the ambient path bit
            for bit. The same ``feature_map`` must be passed to
            ``msip_weights`` / ``msip_map`` so the node Gram is the latent
            Gram ``K_phi``.
        y_feature_map: Optional callable ``psi : (B, dy) -> (B, r_y)`` moving
            the Nadaraya-Watson conditioning onto an informed y-summary. When
            set, the conditioning responsibilities ``alpha_i(y*)`` are
            computed in the summary space, ``||psi(y_i) - psi(y*)||^2``,
            instead of the raw ``||y_i - y*||^2``, so the NW kernel no longer
            averages over the noise-only data directions whose dimension
            ``d_y`` otherwise walls the conditional-mean bias.
            ``y_feature_map`` is independent of the x-side ``feature_map``
            (either, both, or neither may be set); ``None`` (default)
            conditions on raw ``y`` bit for bit. The same ``sigma_y`` is
            interpreted as the bandwidth in the summary space.
        dtype: Floating dtype for the stored samples and the embeddings
            (defaults to float32, matching the experiments).
    """

    def __init__(
        self,
        x_samples: torch.Tensor,
        y_samples: torch.Tensor,
        y_star: torch.Tensor,
        sigma_x: float,
        sigma_y: float,
        feature_map: "callable | None" = None,
        y_feature_map: "callable | None" = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.dtype = dtype
        self.x_samples = x_samples.to(dtype)  # (N, dx)
        self.y_samples = y_samples.to(dtype)  # (N, dy)
        self.y_star = y_star.reshape(1, -1).to(dtype)  # (1, dy)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.feature_map = feature_map
        self.y_feature_map = y_feature_map
        self.n, self.dx = self.x_samples.shape

        # Latent kernel: cache the features phi(x_i) of the joint x-bank
        # once (the responsibilities/v0 measure x-distances in feature
        # space). With feature_map=None the cache is unused and the ambient
        # cdist path runs unchanged.
        if feature_map is not None:
            self.phi_x_samples = feature_map(self.x_samples).to(dtype)  # (N, r)
        else:
            self.phi_x_samples = None

        # Nadaraya-Watson conditioning log-weights, normalized over i in
        # log-space: log alpha_i(y*) = log_softmax_i(-||y_i - y*||^2 /
        # (2 sigma_y^2)). The global k_y(., y*) normalizer cancels in the
        # softmax, and log_softmax is the float32-stable route. With a
        # y_feature_map the conditioning acts on the informed y-summary
        # s(y) = psi(y) instead of raw y: the NW distances are
        # ||s(y_i) - s(y*)||^2, removing the noise-only data directions
        # whose dimension walls the raw-y conditioning. y_feature_map=None
        # reproduces the raw-y conditioning bit for bit.
        if y_feature_map is not None:
            s_samples = y_feature_map(self.y_samples).to(dtype)  # (N, r_y)
            s_star = y_feature_map(self.y_star).reshape(1, -1).to(dtype)
            sq_dist_y = ((s_samples - s_star) ** 2).sum(dim=-1)  # (N,)
        else:
            sq_dist_y = ((self.y_samples - self.y_star) ** 2).sum(dim=-1)  # (N,)
        self.log_alpha = torch.log_softmax(
            -sq_dist_y / (2.0 * sigma_y**2), dim=0
        )  # (N,)

    def effective_sample_size(self) -> float:
        """Nadaraya-Watson effective sample size 1 / sum_i alpha_i^2.

        A Kish-style diagnostic for the conditioning: it is N when the
        weights are uniform (y* far from every sample relative to sigma_y,
        or a huge bandwidth) and collapses toward 1 when a single sample
        dominates (tiny sigma_y). It should be kept comfortably above M so
        the M-node MSIP quadrature is informed by enough effective joint
        samples (rule of thumb >= 5-10 M).

        Returns:
            Effective sample size as a Python float.
        """
        log_ess = -torch.logsumexp(2.0 * self.log_alpha, dim=0)
        return torch.exp(log_ess).item()

    def __call__(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditional embeddings at the nodes.

        Args:
            nodes: Query nodes x', shape (m, dx).

        Returns:
            (log_v0 [m], sigma_x^2 grad_log_v0 [m, dx]).
        """
        nodes = nodes.to(self.dtype)
        # Pairwise squared x-distances. In the ambient path this is
        # ||x_i - x'_j||^2; with a feature map it is the latent distance
        # ||phi(x_i) - phi(x'_j)||^2, which is what makes v0 / p_i the
        # feature-space kernel sums.
        if self.feature_map is not None:
            phi_nodes = self.feature_map(nodes).to(self.dtype)  # (m, r)
            sq_dist_x = torch.cdist(phi_nodes, self.phi_x_samples) ** 2
        else:
            sq_dist_x = torch.cdist(nodes, self.x_samples) ** 2

        # log w_ji = log alpha_i - ||.||^2 / (2 sigma_x^2).
        log_w = self.log_alpha.unsqueeze(0) - sq_dist_x / (
            2.0 * self.sigma_x**2
        )  # (m, N)

        # log_v0(x') = logsumexp_i log w_ji (Z_sigma_x dropped; it cancels).
        log_v0 = torch.logsumexp(log_w, dim=-1)  # (m,)

        # p_ji = softmax_i log w_ji ; weighted ambient x-mean minus the node
        # gives the mean-shift vector sigma_x^2 grad_log_v0 = v1 / v0 - x'.
        # The feature map never enters this vector: the node moves in the
        # ambient space toward sum_i p_i x_i.
        p = torch.softmax(log_w, dim=-1)  # (m, N)
        weighted_mean = p @ self.x_samples  # (m, dx)
        sigma_sq_grad_log_v0 = weighted_mean - nodes
        return log_v0, sigma_sq_grad_log_v0


def conditional_quadrature(
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    y_star: torch.Tensor,
    y_init: torch.Tensor,
    sigma_x: float,
    sigma_y: float,
    lr: float,
    n_iters: int,
    kernel_diag_infl: float = 1e-6,
    bounds: tuple[float, float] | None = (-100.0, 100.0),
    feature_map: "callable | None" = None,
    progress: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conditional posterior quadrature via data-driven MSIP.

    Builds ``ConditionalDataDrivenEmbeddings`` for the query ``y*`` and
    runs MSIP (Algorithm 1) from ``y_init`` to produce M weighted nodes
    approximating p(x | y*). Convenience wrapper around ``run_msip``.

    Args:
        x_samples: Joint x-samples, shape (N, dx).
        y_samples: Joint y-samples, shape (N, dy).
        y_star: Query observation, shape (dy,).
        y_init: Initial nodes, shape (M, dx).
        sigma_x: x-kernel (MSIP) bandwidth.
        sigma_y: y-kernel (Nadaraya-Watson) bandwidth.
        lr: MSIP damping / step size in (0, 1].
        n_iters: Number of MSIP iterations.
        kernel_diag_infl: Diagonal inflation added to the MSIP kernel
            matrix.
        bounds: (lo, hi) box the nodes are clamped to each step; None
            disables clamping.
        feature_map: Optional informed-subspace feature map ``phi``. Threaded
            into both the embeddings and the MSIP node Gram so the whole
            quadrature acts in the latent; ``None`` (default) is the ambient
            path.
        progress: Whether to show a tqdm progress bar.
        dtype: Floating dtype.

    Returns:
        (nodes [M, dx], weights [M]): the final MSIP nodes and their
        stationary (unit-sum) weights.
    """
    embeddings = ConditionalDataDrivenEmbeddings(
        x_samples, y_samples, y_star, sigma_x, sigma_y,
        feature_map=feature_map, dtype=dtype,
    )
    nodes, weights, _ = run_msip(
        y_init.to(dtype),
        embeddings,
        lr=lr,
        kernel_diag_infl=kernel_diag_infl,
        sigma=sigma_x,
        n_iters=n_iters,
        bounds=bounds,
        feature_map=feature_map,
        progress=progress,
    )
    return nodes, weights
