"""Regularized mean-shift interacting particles (MSIP) and CMSIP.

This module implements two related map iterations on a node set Y:

  * **MSIP (SE kernel).** Belhadji, Sharp, Marzouk (arXiv:2605.14142): the
    map ``Psi(Y) = K^{-1} V1 / K^{-1} v0`` of the squared-exponential
    kernel ``K``, assembled in a float32-robust log-stable form (see
    ``_weighted_average``). This is the original MSIP map.

  * **CMSIP (general distance-based kernel).** Belhadji, Sharp, Marzouk
    (arXiv:2502.10600, sec. 3.1, eqs. 21-25): the *Corrected* MSIP map
    that uses the bar-kernel ``Kbar`` on the LHS of the position update,

        Psi_i = (Kbar^{-1} v1_full)_i / w_i,
        w     = K^{-1} v0,
        v1_full[j] = grad_v0[j] + (Kbar w)_j y_j.

    For the SE kernel ``Kbar = K / sigma^2``, so CMSIP collapses to MSIP
    exactly. For Matern-5/2 and IMQ ``Kbar`` differs nontrivially from
    ``K`` and CMSIP is the principled choice.

The shared embeddings interface (see ``qfield.estimators``)
returns ``(log_v0, sigma_sq_grad)`` where ``sigma_sq_grad`` equals
``v0 * (y + sigma^2 grad_log_v0) / v0 - y`` for SE -- i.e. ``sigma^2
grad_log_v0`` -- and the CMSIP path divides by ``sigma^2`` to recover
``grad_log_v0`` for use in ``grad_v0 = v0 * grad_log_v0``. This keeps the
estimator API unchanged: the same Fredholm / GF / exact estimator drives
both SE-MSIP and non-SE-CMSIP.

The map is normalization-invariant in pi (Prop. 3.2): scaling pi by a
constant shifts log_v0 by a constant, which the per-row ``- max`` removes,
leaving Psi unchanged.

References:
  - Belhadji, Sharp, Marzouk, "To discretize continually: Mean shift
    interacting particle systems for Bayesian inference", arXiv:2605.14142
    (regularized MSIP, eqs. 13-20; Algorithm 1).
  - Belhadji, Sharp, Marzouk, "Corrected MSIP for non-SE kernels",
    arXiv:2502.10600 (Sec. 3.1, eqs. 21-25).
"""

import torch
from tqdm import tqdm

from qfield.kernels import Kernel, SEKernel


def _weighted_average(
    y: torch.Tensor,
    alpha: torch.Tensor,
    log_v: torch.Tensor,
) -> torch.Tensor:
    """Log-stable signed weighted averages, all output rows at once.

    Vectorized form of the reference ``recursive_weighted_average_alpha_v``
    over the rows of ``alpha``. For each row i,

        out_i = sum_j sign(alpha_ij) exp(log|alpha_ij| + log_v_j - z_i)
                y_j
              / sum_j sign(alpha_ij) exp(log|alpha_ij| + log_v_j - z_i),

    with z_i = max_j (log|alpha_ij| + log_v_j). The per-row shift z_i
    cancels in the ratio but keeps every exponential at most O(1), which
    is what makes the divide float32-robust.

    Args:
        y: Vectors being averaged, shape (n, d).
        alpha: Arbitrary (signed) weights, shape (m, n) -- row i holds
            the weights for output i (a row of K^{-1}).
        log_v: Positive-weight logs, shape (n,).

    Returns:
        Weighted averages, shape (m, d).
    """
    # log|w_ij| = log|alpha_ij| + log_v_j ; sign from alpha.
    log_abs = torch.log(alpha.abs()) + log_v.unsqueeze(0)  # (m, n)
    sign = alpha.sign()  # (m, n)
    z = log_abs.amax(dim=-1, keepdim=True)  # (m, 1)
    weighted_signs = sign * torch.exp(log_abs - z)  # (m, n)
    denom = weighted_signs.sum(dim=-1, keepdim=True)  # (m, 1)
    numer = weighted_signs @ y  # (m, d)
    return numer / denom


def _resolve_kernel(
    kernel: Kernel | None,
    sigma: float,
) -> Kernel:
    """Default ``kernel`` argument: ``None`` -> SE kernel at ``sigma``.

    Calls that omit ``kernel`` get the squared-exponential kernel at the
    supplied ``sigma``.
    """
    return SEKernel(sigma) if kernel is None else kernel


def _gram(
    kern: Kernel,
    y: torch.Tensor,
    feature_map: "callable | None",
) -> torch.Tensor:
    """Node Gram matrix, ambient or latent (feature-space).

    With ``feature_map=None`` this is the ordinary node Gram
    ``kern.kernel(y, y)``. With a feature map ``phi`` it is the latent Gram
    ``K_phi = kern.kernel(phi(y), phi(y))``: the
    same SE kernel evaluated on the r-dimensional features, so distances
    and the conditioning of the Gram live in the informed subspace. The
    feature map is applied to the nodes only here (the embeddings cache the
    bank's features separately); ``feature_map=None`` reproduces the
    ambient Gram bit for bit.
    """
    if feature_map is None:
        return kern.kernel(y, y)
    phi_y = feature_map(y).to(y.dtype)
    return kern.kernel(phi_y, phi_y)


def _kernel_and_inverse(
    y: torch.Tensor,
    sigma: float,
    kernel_diag_infl: float,
    kernel: Kernel | None = None,
    feature_map: "callable | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inflated Gram matrix and its inverse for the MSIP map.

    The kernel diagonal is inflated by ``kernel_diag_infl``; the inverse
    is ``torch.linalg.inv`` when the inflation is positive (a regular
    system) and the pseudo-inverse ``torch.linalg.pinv`` otherwise (the
    bare SE Gram can be singular), matching the reference.

    Args:
        y: Nodes, shape (m, d).
        sigma: SE bandwidth (used when ``kernel`` is ``None``).
        kernel_diag_infl: Diagonal inflation added to the Gram matrix.
        kernel: Optional ``Kernel``; ``None`` -> SE at ``sigma``.
        feature_map: Optional latent feature map ``phi``; ``None`` (default)
            is the ambient Gram, otherwise the latent Gram ``K_phi``. The
            feature-space Gram is only PSD, so the diagonal inflation is what
            restores invertibility.

    Returns:
        (k_infl [m, m], k_inv [m, m]).
    """
    kern = _resolve_kernel(kernel, sigma)
    m = y.shape[0]
    k = _gram(kern, y, feature_map).clone()
    idx = torch.arange(m, device=y.device)
    k[idx, idx] += kernel_diag_infl
    if kernel_diag_infl > 0.0:
        k_inv = torch.linalg.inv(k)
    else:
        k_inv = torch.linalg.pinv(k)
    return k, k_inv


def _kernel_bar_inflated(
    y: torch.Tensor,
    kernel: Kernel,
    kernel_diag_infl: float,
) -> torch.Tensor:
    """Inflated bar-kernel Gram matrix (CMSIP LHS).

    Mirrors the diagonal-inflation strategy of ``_kernel_and_inverse`` but
    for the bar-kernel ``Kbar``, which is the LHS of the CMSIP position
    update. We return the matrix and let the caller decide between
    ``solve`` (kept as the default when ``kernel_diag_infl > 0``) and
    ``pinv`` (the singular fallback); we never form the explicit inverse
    here because it is numerically inferior to ``solve``.

    For SE this matrix is ``K / sigma^2`` so CMSIP reduces to MSIP; for
    Matern / IMQ it is a distinct positive-definite Gram matrix.

    Args:
        y: Nodes, shape (m, d).
        kernel: ``Kernel`` instance.
        kernel_diag_infl: Diagonal inflation.

    Returns:
        Inflated bar-kernel Gram ``kbar`` of shape ``(m, m)``.
    """
    kbar = kernel.kernel_bar(y, y).clone()
    m = y.shape[0]
    idx = torch.arange(m, device=y.device)
    kbar[idx, idx] += kernel_diag_infl
    return kbar


def msip_weights(
    y: torch.Tensor,
    embeddings,
    kernel_diag_infl: float,
    sigma: float,
    kernel: Kernel | None = None,
    feature_map: "callable | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stationary MSIP weights at a configuration.

    Returns ``w = solve(K_infl, v0) / sum``, with ``v0 = exp(log_v0 -
    max log_v0)``, together with the embeddings ``(log_v0, sigma^2
    grad_log_v0)`` so callers can reuse them. The system is built with
    the kernel of ``kernel`` (defaults to SE at ``sigma``); CMSIP uses
    the *same* weight definition as MSIP (only the position update
    changes).

    Args:
        y: Nodes, shape (m, d).
        embeddings: Callable Y -> (log_v0 [m], sigma^2 grad_log_v0 [m, d]).
        kernel_diag_infl: Diagonal inflation added to the kernel matrix.
        sigma: SE bandwidth (used when ``kernel`` is ``None``).
        kernel: Optional ``Kernel``; ``None`` -> SE at ``sigma``.
        feature_map: Optional latent feature map ``phi``; ``None`` (default)
            builds the ambient Gram, otherwise the weight system uses the
            latent Gram ``K_phi = kern.kernel(phi(y), phi(y))``. It must
            match the ``feature_map`` used to build ``embeddings`` for the v0
            vector and the Gram to live in the same metric.

    Returns:
        (w [m], log_v0 [m], sigma_sq_grad_log_v0 [m, d]).
    """
    kern = _resolve_kernel(kernel, sigma)
    log_v0, sigma_sq_grad = embeddings(y)
    m = y.shape[0]
    k = _gram(kern, y, feature_map).clone()
    idx = torch.arange(m, device=y.device)
    k[idx, idx] += kernel_diag_infl
    v0 = torch.exp(log_v0 - log_v0.max())
    w = torch.linalg.solve(k, v0)
    w = w / w.sum()
    return w, log_v0, sigma_sq_grad


def msip_map(
    y: torch.Tensor,
    embeddings,
    kernel_diag_infl: float,
    sigma: float,
    kernel: Kernel | None = None,
    feature_map: "callable | None" = None,
) -> torch.Tensor:
    """(Corrected) MSIP map ``Psi(Y)`` at the given configuration.

    With ``kernel`` left ``None`` (or set to an ``SEKernel``) this is the
    standard SE MSIP map of arXiv:2605.14142, computed in log-stable form:

        Psi_i = wavg(Y; K^{-1}_i, log_v0)
              + wavg(sigma^2 grad_log_v0; K^{-1}_i, log_v0),

    which equals ``K^{-1} V1 / K^{-1} v0`` with ``V1 = v0 (y + sigma^2
    grad_log_v0)``.

    For a non-SE distance-based kernel the map switches to the *Corrected*
    MSIP form of arXiv:2502.10600 (eqs. 21-25):

        Psi_i = (Kbar^{-1} v1_full)_i / w_i,
        w     = K^{-1} v0,
        v1_full[j] = v0[j] grad_log_v0(y_j) + (Kbar w)_j y_j,

    with the LHS bar-kernel ``Kbar``. For SE this reduces to the standard
    map exactly (``Kbar = K / sigma^2`` cancels through), so we can flag
    SE via ``kernel.is_se`` and dispatch to the log-stable form.

    The map is normalization-invariant in pi: scaling pi by a constant
    shifts ``log_v0`` by a constant, which the per-row ``- max`` removes
    in the MSIP form, and which cancels in the CMSIP form's
    ``grad_v0 / w`` ratio (the ``exp(log_v0 - max)`` rescaling is applied
    on both the numerator and the weight system).

    Args:
        y: Nodes, shape (m, d).
        embeddings: Callable Y -> (log_v0 [m], sigma^2 grad_log_v0 [m, d]).
        kernel_diag_infl: Diagonal inflation added to the kernel matrix.
        sigma: SE bandwidth (used when ``kernel`` is ``None``).
        kernel: Optional ``Kernel`` (SE / Matern / IMQ); ``None`` -> SE.
        feature_map: Optional latent feature map ``phi``. Supported only on
            the SE path (the latent quadrature is defined for the SE
            feature-space kernel): the node
            Gram becomes ``K_phi = kern.kernel(phi(y), phi(y))`` while the
            ambient mean-shift in ``sigma_sq_grad`` (already moved in the
            ambient node by the embeddings) is averaged with the latent
            Gram's inverse. ``None`` (default) is the ambient path. Combining
            a feature map with a non-SE (CMSIP) kernel is rejected.

    Returns:
        Psi(Y), shape (m, d).
    """
    kern = _resolve_kernel(kernel, sigma)
    log_v0, sigma_sq_grad = embeddings(y)

    if kern.is_se:
        # SE log-stable path: the ambient Gram when feature_map is None,
        # and the latent Gram K_phi through _kernel_and_inverse when a
        # feature map is supplied.
        _, k_inv = _kernel_and_inverse(
            y, sigma, kernel_diag_infl, kern, feature_map=feature_map
        )
        term_v0 = _weighted_average(y, k_inv, log_v0)
        term_v1 = _weighted_average(sigma_sq_grad, k_inv, log_v0)
        psi = term_v0 + term_v1
        finite = torch.isfinite(psi).all(dim=-1, keepdim=True)
        return torch.where(finite, psi, y)

    if feature_map is not None:
        raise ValueError(
            "feature_map (the latent kernel) is supported only on the SE "
            "MSIP path; it is not defined for the corrected (CMSIP) map of "
            "non-SE kernels."
        )

    # CMSIP form for non-SE distance-based kernels.
    # v0 is rescaled by exp(- max log_v0) to keep numbers at unit scale
    # without changing the map (an overall positive scale on v0 cancels
    # through w and grad_v0 = v0 * grad_log_v0 in the ratio v1_full / w).
    v0 = torch.exp(log_v0 - log_v0.max())  # (m,)
    # grad_log_v0 = (sigma^2 grad_log_v0) / sigma^2 from the embedding's
    # second output; this is the kernel-agnostic score-like quantity.
    grad_log_v0 = sigma_sq_grad / (sigma**2)  # (m, d)
    grad_v0 = v0.unsqueeze(-1) * grad_log_v0  # (m, d)

    # Standard MSIP weights w (unnormalized solve; K-bar is applied to it
    # in the position update below).
    k = kern.kernel(y, y).clone()
    m = y.shape[0]
    idx = torch.arange(m, device=y.device)
    k[idx, idx] += kernel_diag_infl
    w = torch.linalg.solve(k, v0)  # (m,)

    # Bar-kernel Gram and its action on w. We solve the K-bar system via
    # ``torch.linalg.solve`` rather than forming the explicit inverse,
    # which is more numerically stable when ``kernel_diag_infl`` is small.
    kbar = _kernel_bar_inflated(y, kern, kernel_diag_infl)
    kbar_w = kbar @ w  # (m,)

    # v1_full = grad_v0 + diag(kbar_w) Y (the CMSIP RHS).
    v1_full = grad_v0 + kbar_w.unsqueeze(-1) * y  # (m, d)

    # Solve Kbar Psi_unscaled = v1_full ; then divide each row by w_i.
    if kernel_diag_infl > 0.0:
        psi_unscaled = torch.linalg.solve(kbar, v1_full)
    else:
        # No inflation: kbar may be singular; fall back to pinv.
        psi_unscaled = torch.linalg.pinv(kbar) @ v1_full
    psi = psi_unscaled / w.unsqueeze(-1)  # (m, d)

    # Float32 numerical safety net (matches the SE-path policy): if any
    # row is non-finite, hold that particle fixed for the step.
    finite = torch.isfinite(psi).all(dim=-1, keepdim=True)
    return torch.where(finite, psi, y)


def run_msip(
    y_init: torch.Tensor,
    embeddings,
    lr: float,
    kernel_diag_infl: float,
    sigma: float,
    n_iters: int,
    bounds: tuple[float, float] | None = (-100.0, 100.0),
    record_fn=None,
    progress: bool = True,
    kernel: Kernel | None = None,
    feature_map: "callable | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Run the damped (C)MSIP iteration for n_iters steps (Algorithm 1).

    Iterates ``Y <- (1 - lr) Y + lr Psi(Y)`` then clamps Y to ``bounds``.
    There is no trust-region cap and no zero-weight freeze.

    Embedding-evaluation count: both :func:`msip_map` and
    :func:`msip_weights` call ``embeddings`` exactly once, so over
    ``n_iters = T`` steps this function calls it

      * ``T + 1`` times with ``record_fn=None`` -- ``T`` inside the map, plus
        one for the stationary weights returned at the end;
      * ``2 T + 2`` times with a ``record_fn`` -- the above plus one weight
        solve at the initial configuration and one after every step, which is
        the logging path, not the algorithm.

    The doubling is a property of the recording path alone.

    Args:
        y_init: Initial nodes, shape (m, d).
        embeddings: Callable Y -> (log_v0 [m], sigma^2 grad_log_v0 [m, d]).
        lr: Damping / step size in (0, 1].
        kernel_diag_infl: Diagonal inflation added to the kernel matrix.
        sigma: SE bandwidth (used when ``kernel`` is ``None``).
        n_iters: Number of iterations T.
        bounds: (lo, hi) box the particles are clamped to each step; None
            disables clamping.
        record_fn: Optional callable (Y, w) -> float logged per iteration
            into ``history["record"]`` (the initial configuration is
            logged first, giving n_iters + 1 entries).
        progress: Whether to show a tqdm progress bar.
        kernel: Optional ``Kernel`` (SE / Matern / IMQ); ``None`` -> SE.
        feature_map: Optional latent feature map ``phi``, threaded into both
            the map and the weight solve so the node Gram is the latent
            ``K_phi``; ``None`` (default) is the ambient path. It must match
            the ``feature_map`` used to build ``embeddings``.

    Returns:
        (y_final [m, d], w_final [m], history): w_final are the
        stationary weights at the final configuration; history is a dict
        with key "record" (possibly empty).
    """
    y = y_init.clone()
    history: dict = {"record": []}

    if record_fn is not None:
        w0, _, _ = msip_weights(
            y, embeddings, kernel_diag_infl, sigma, kernel=kernel,
            feature_map=feature_map,
        )
        history["record"].append(record_fn(y, w0))

    iterator = range(n_iters)
    if progress:
        iterator = tqdm(
            iterator,
            desc="MSIP",
            colour="#B5F2A9",
            dynamic_ncols=True,
        )
    for _ in iterator:
        psi = msip_map(
            y, embeddings, kernel_diag_infl, sigma, kernel=kernel,
            feature_map=feature_map,
        )
        y = (1.0 - lr) * y + lr * psi
        if bounds is not None:
            y = y.clamp(bounds[0], bounds[1])
        if record_fn is not None:
            w, _, _ = msip_weights(
                y, embeddings, kernel_diag_infl, sigma, kernel=kernel,
                feature_map=feature_map,
            )
            history["record"].append(record_fn(y, w))

    w_final, _, _ = msip_weights(
        y, embeddings, kernel_diag_infl, sigma, kernel=kernel,
        feature_map=feature_map,
    )
    return y, w_final, history
