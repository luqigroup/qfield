"""Distance-based kernels for MSIP and CMSIP.

This module provides a small kernel family abstraction used by the MSIP
map. Every kernel is *distance-based*, ``k(x, y) = phi(r)`` with ``r =
||x - y||``, and C^1 in ``r``. The ``Kernel`` base class exposes three
quantities that the (corrected) MSIP map consumes:

  * ``kernel(x, y)``       -- pairwise kernel values ``k[i, j] = phi(r_ij)``,
  * ``kernel_bar(x, y)``   -- the bar-kernel ``kbar(x, y)`` defined by
                              ``grad_y k(x, y) = (x - y) * kbar(x, y)``,
  * ``log_kernel(x, y)``   -- log of ``kernel(x, y)`` (used by the
                              log-stable map's weighted average; for SE
                              this is exact, for Matern / IMQ it is just
                              ``log(kernel(x, y))`` clamped from below).

For a distance-based kernel ``k(x, y) = phi(r)`` we have
``grad_y k = phi'(r) (y - x) / r = -(phi'(r) / r) (x - y)``, so the
bar-kernel is ``kbar(r) = -phi'(r) / r`` (with the well-defined limit at
``r = 0``).

  * SE kernel: ``phi(r) = exp(-r^2 / (2 sigma^2))``, so
    ``kbar = phi / sigma^2 = kernel / sigma^2``.
  * Matern-5/2: ``phi(r) = (1 + s + s^2 / 3) exp(-s)`` with
    ``s = sqrt(5) r / rho``; chain rule gives
    ``kbar(r) = (5 / (3 rho^2)) (1 + s) exp(-s)``.
  * IMQ: ``phi(r) = (c^2 + r^2)^(-beta)``, so
    ``kbar(r) = 2 beta (c^2 + r^2)^(-beta - 1)``.

For the SE kernel ``kbar(x, y) = k(x, y) / sigma^2`` is a *scaling* of
``k``, so MSIP and CMSIP coincide. For the other kernels ``kbar`` differs
nontrivially from ``k``, and the rigorous (Corrected) MSIP map of
arXiv:2502.10600 (sec. 3.1, eqs. 21-25) uses ``Kbar`` on the LHS of the
linear system for the ``v1``-weighted average.

The module-level helpers (``se_kernel``, ``kernel_matrix``,
``log_z_sigma``, ``svgd_kernel_and_grad``, ``median_heuristic_h_sq``) are
wired to the SE kernel.

References:
  - Belhadji, Sharp, Marzouk, "To discretize continually: Mean shift
    interacting particle systems for Bayesian inference", arXiv:2605.14142
    (squared exponential kernel, eqs. 4, 21-30).
  - Belhadji, Sharp, Marzouk, "Corrected MSIP for non-SE kernels",
    arXiv:2502.10600 (Sec. 3.1, eqs. 21-25).
"""

import math
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Kernel family abstraction
# ---------------------------------------------------------------------------


class Kernel:
    """Base class for distance-based kernels ``k(x, y) = phi(||x - y||)``.

    Subclasses implement ``kernel`` and ``kernel_bar``. ``log_kernel``
    falls back to ``log(kernel + floor)`` (clamped to avoid ``log(0)``),
    which is what the log-stable MSIP map needs.

    The boolean attribute ``is_se`` flags the squared-exponential kernel,
    for which ``kbar = k / sigma^2`` is a positive scaling of ``k`` and
    CMSIP reduces to vanilla MSIP. For non-SE kernels CMSIP differs.
    """

    is_se: bool = False

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pairwise kernel values ``k(x_i, y_j)``, shape ``(n, m)``."""
        raise NotImplementedError

    def kernel_bar(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pairwise bar-kernel values ``kbar(x_i, y_j)``, shape ``(n, m)``.

        Defined by ``grad_y k(x, y) = (x - y) kbar(x, y)``.
        """
        raise NotImplementedError

    def log_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log of ``kernel(x, y)`` (clamped from below for safety)."""
        k = self.kernel(x, y)
        # Floor protects ``log`` against ``+0`` from underflow / from
        # heavy-tail kernels with vanishing values at long distances.
        # The floor is well below f32 underflow so it never biases values
        # that are merely small.
        floor = torch.finfo(k.dtype).tiny
        return torch.log(k.clamp_min(floor))


class SEKernel(Kernel):
    """Squared-exponential (Gaussian) kernel.

    ``k(x, y) = exp(-||x - y||^2 / (2 sigma^2))``,
    ``kbar(x, y) = k(x, y) / sigma^2``.

    Args:
        sigma: Bandwidth ``sigma > 0``.
    """

    is_se: bool = True

    def __init__(self, sigma: float):
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)

    def _pairwise_sq_dists(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        return torch.cdist(x, y) ** 2

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sq = self._pairwise_sq_dists(x, y)
        return torch.exp(-sq / (2.0 * self.sigma**2))

    def kernel_bar(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y) / (self.sigma**2)

    def log_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sq = self._pairwise_sq_dists(x, y)
        return -sq / (2.0 * self.sigma**2)


class MaternKernel(Kernel):
    """Matern-5/2 kernel.

    ``phi(r) = (1 + sqrt(5) r / rho + 5 r^2 / (3 rho^2)) exp(-sqrt(5) r /
    rho)`` with bar-kernel

    ``kbar(r) = (5 / (3 rho^2)) (1 + sqrt(5) r / rho) exp(-sqrt(5) r /
    rho)``,

    obtained by chain rule from ``-phi'(r) / r``: letting ``s = sqrt(5) r
    / rho`` and ``phi = (1 + s + s^2 / 3) exp(-s)``, ``dphi/ds = -(s / 3)
    (1 + s) exp(-s)`` and ``-phi'(r) / r = (sqrt(5) / rho) (s / 3) (1 +
    s) exp(-s) / r = (5 / (3 rho^2)) (1 + s) exp(-s)``. At ``r = 0`` this
    evaluates to ``5 / (3 rho^2)``.

    Args:
        rho: Length-scale ``rho > 0``.
    """

    is_se: bool = False

    def __init__(self, rho: float):
        if rho <= 0.0:
            raise ValueError(f"rho must be > 0, got {rho}")
        self.rho = float(rho)

    def _pairwise_dists(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        return torch.cdist(x, y)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = self._pairwise_dists(x, y)
        s = math.sqrt(5.0) * r / self.rho
        return (1.0 + s + (s * s) / 3.0) * torch.exp(-s)

    def kernel_bar(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = self._pairwise_dists(x, y)
        s = math.sqrt(5.0) * r / self.rho
        return (5.0 / (3.0 * self.rho**2)) * (1.0 + s) * torch.exp(-s)


class Matern32Kernel(Kernel):
    """Matern-3/2 kernel (smoothness ``nu = 3/2``).

    ``phi(r) = (1 + sqrt(3) r / rho) exp(-sqrt(3) r / rho)`` with
    bar-kernel

    ``kbar(r) = (3 / rho^2) exp(-sqrt(3) r / rho)``,

    obtained by chain rule from ``-phi'(r) / r``: letting ``s = sqrt(3) r
    / rho`` and ``phi = (1 + s) exp(-s)``, ``dphi/ds = -s exp(-s)`` and
    ``-phi'(r) / r = (sqrt(3) / rho) (s exp(-s)) / r = (3 / rho^2)
    exp(-s)``. At ``r = 0`` this evaluates to ``3 / rho^2``.

    Matern-3/2 has *heavier tails* than Matern-5/2 (lower smoothness
    ``nu``). The interface mirrors :class:`MaternKernel` (5/2) exactly:
    ``is_se = False`` so the (corrected) MSIP map dispatches to the
    CMSIP path, which consumes ``kernel`` and ``kernel_bar`` (and the
    base-class ``log_kernel`` fallback).

    Args:
        rho: Length-scale ``rho > 0``.
    """

    is_se: bool = False

    def __init__(self, rho: float):
        if rho <= 0.0:
            raise ValueError(f"rho must be > 0, got {rho}")
        self.rho = float(rho)

    def _pairwise_dists(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        return torch.cdist(x, y)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = self._pairwise_dists(x, y)
        s = math.sqrt(3.0) * r / self.rho
        return (1.0 + s) * torch.exp(-s)

    def kernel_bar(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = self._pairwise_dists(x, y)
        s = math.sqrt(3.0) * r / self.rho
        return (3.0 / (self.rho**2)) * torch.exp(-s)


class IMQKernel(Kernel):
    """Inverse Multi-Quadric kernel.

    ``phi(r) = (c^2 + r^2)^(-beta)``,
    ``kbar(r) = 2 beta (c^2 + r^2)^(-beta - 1)``.

    Args:
        c: Length-scale ``c > 0`` (controls the kernel's effective range).
        beta: Power ``beta > 0`` (Stein-discrepancy literature commonly
            uses ``beta = 1/2`` so the kernel is c-IPM; here we keep it
            free).
    """

    is_se: bool = False

    def __init__(self, c: float, beta: float = 0.5):
        if c <= 0.0:
            raise ValueError(f"c must be > 0, got {c}")
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.c = float(c)
        self.beta = float(beta)

    def _pairwise_sq_dists(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        return torch.cdist(x, y) ** 2

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sq = self._pairwise_sq_dists(x, y)
        return (self.c**2 + sq) ** (-self.beta)

    def kernel_bar(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sq = self._pairwise_sq_dists(x, y)
        return 2.0 * self.beta * (self.c**2 + sq) ** (-self.beta - 1.0)

    def log_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sq = self._pairwise_sq_dists(x, y)
        # log (c^2 + r^2)^(-beta) = -beta log(c^2 + r^2).
        return -self.beta * torch.log(self.c**2 + sq)


def default_kernel(name: str, **hp: Any) -> Kernel:
    """Factory: build a kernel from its short name and hyperparameters.

    Names:
      * ``"se"``       -- ``SEKernel(sigma=hp["sigma"])``.
      * ``"matern52"`` -- ``MaternKernel(rho=hp["rho"])``. If only
                          ``sigma`` is supplied we read it as ``rho``.
      * ``"matern32"`` -- ``Matern32Kernel(rho=hp["rho"])``. If only
                          ``sigma`` is supplied we read it as ``rho``.
      * ``"imq"``      -- ``IMQKernel(c=hp["c"], beta=hp.get("beta", 0.5))``.
                          If only ``sigma`` is supplied we read it as ``c``.

    The ``sigma`` fallback is for callers that already have a single
    bandwidth knob and want to compare SE / Matern / IMQ at comparable
    length-scales.

    Args:
        name: Short kernel name.
        **hp: Hyperparameters (see above).

    Returns:
        A ``Kernel`` instance.
    """
    n = name.lower()
    if n == "se":
        sigma = hp.get("sigma")
        if sigma is None:
            raise ValueError("SE kernel requires 'sigma'")
        return SEKernel(sigma=sigma)
    if n == "matern52":
        rho = hp.get("rho", hp.get("sigma"))
        if rho is None:
            raise ValueError("Matern-5/2 kernel requires 'rho' (or 'sigma')")
        return MaternKernel(rho=rho)
    if n == "matern32":
        rho = hp.get("rho", hp.get("sigma"))
        if rho is None:
            raise ValueError("Matern-3/2 kernel requires 'rho' (or 'sigma')")
        return Matern32Kernel(rho=rho)
    if n == "imq":
        c = hp.get("c", hp.get("sigma"))
        beta = hp.get("beta", 0.5)
        if c is None:
            raise ValueError("IMQ kernel requires 'c' (or 'sigma')")
        return IMQKernel(c=c, beta=beta)
    raise ValueError(f"Unknown kernel name: {name!r}")


# ---------------------------------------------------------------------------
# SE helpers
# ---------------------------------------------------------------------------


def se_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Squared-exponential kernel value(s) k(x, y).

    Args:
        x: Points, shape (n, d) (or (d,) for a single point).
        y: Points, shape (m, d) (or (d,) for a single point).
        sigma: Kernel bandwidth.

    Returns:
        Pairwise kernel values, shape (n, m).
    """
    return SEKernel(sigma).kernel(x, y)


def kernel_matrix(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gram matrix K(Y)_{ij} = k(y_i, y_j) of the node set (SE kernel).

    Args:
        y: Nodes, shape (m, d).
        sigma: Kernel bandwidth.

    Returns:
        Symmetric Gram matrix, shape (m, m), with unit diagonal.
    """
    return se_kernel(y, y, sigma)


def log_z_sigma(sigma: float, d: int) -> float:
    """Log of Z_sigma = (2 * pi * sigma^2)^(d/2) = (sqrt(2 pi) sigma)^d.

    This is the kernel-to-Gaussian normalizing factor omega_{sigma, d}.
    It is returned in log-space because (sqrt(2 pi) sigma)^d overflows in
    moderate-to-high d; downstream code only ever needs differences of
    such logs (the factor cancels in the MSIP map and unit-sum weights).

    Args:
        sigma: Kernel bandwidth.
        d: Dimensionality.

    Returns:
        log Z_sigma, a Python float.
    """
    return 0.5 * d * math.log(2.0 * math.pi * sigma**2)


def svgd_kernel_and_grad(
    z: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SVGD kernel matrix and its gradient w.r.t. the first argument.

    Uses the SAME squared-exponential kernel as MSIP (reference
    ``sqexp_kernel_elem``):

        k(z_j, z_i) = exp(-||z_j - z_i||^2 / (2 sigma^2)),

    so the SVGD baseline shares the MSIP bandwidth ``sigma`` rather than
    using a median-heuristic h. The gradient w.r.t. the first slot is
    grad_{z_j} k(z_j, z_i) = k(z_j, z_i) (z_i - z_j) / sigma^2. We return
    the per-pair vector difference scaled by the kernel so the caller can
    assemble the SVGD field directly.

    Args:
        z: Particles, shape (m, d).
        sigma: Kernel bandwidth (the same sigma used by MSIP).

    Returns:
        (k_mat, grad_k): k_mat is (m, m) with k_mat[j, i] = k(z_j, z_i);
        grad_k is (m, m, d) with grad_k[j, i] = grad_{z_j} k(z_j, z_i).
    """
    diff = z.unsqueeze(1) - z.unsqueeze(0)  # diff[j, i] = z_j - z_i
    sq_dists = (diff**2).sum(dim=-1)
    k_mat = torch.exp(-sq_dists / (2.0 * sigma**2))
    # grad_{z_j} k = k * (z_i - z_j) / sigma^2 = -k * diff / sigma^2.
    grad_k = -k_mat.unsqueeze(-1) * diff / (sigma**2)
    return k_mat, grad_k


def median_heuristic_sigma_sq(z: torch.Tensor) -> float:
    """Median-heuristic squared bandwidth for an MMD kernel.

    ``sigma^2 = median(pairwise squared distances)``, equivalently
    ``sigma = median(||x - y||)`` -- the median commutes with the square, so
    the two statements coincide. The median is over OFF-DIAGONAL pairs only;
    the ``m`` zero self-distances are not genuine pairwise distances and
    including them biases the bandwidth down, markedly so at small ``m``.

    THIS IS NOT :func:`median_heuristic_h_sq`, AND THE DIFFERENCE IS LARGE.
    That function is Liu & Wang's SVGD rule, which divides by
    ``2 log(m + 1)`` to balance the repulsive force against the drift. That
    divisor is a property of SVGD's dynamics and has no place in the
    bandwidth of a kernel used as a METRIC: it is 10.6 at ``m = 200`` and
    16.6 at ``m = 4000``, so using it as an MMD ruler makes the bandwidth
    roughly three to four times too small AND makes it depend on how many
    samples happened to be used to compute it -- which is not a property of
    the target being measured.

    Use this for anything that scores a quadrature. Use
    :func:`median_heuristic_h_sq` for the SVGD sampler, where it is correct.

    Args:
        z: Points, shape ``(m, d)``, ``m >= 2``.

    Returns:
        Squared bandwidth ``sigma^2`` as a Python float, floored away from
        zero so a collapsed configuration cannot produce a degenerate kernel.
    """
    m = z.shape[0]
    if m < 2:
        raise ValueError(
            f"the median heuristic needs at least two points, got {m}"
        )
    sq_dists = torch.cdist(z, z) ** 2
    iu, iv = torch.triu_indices(m, m, offset=1)
    med = torch.median(sq_dists[iu, iv])
    return max(med.item(), 1e-8)


def median_heuristic_h_sq(z: torch.Tensor) -> float:
    """Median-heuristic squared bandwidth h^2 for SVGD.

    h^2 = median(pairwise squared distances) / (2 * log(M + 1)), the
    standard SVGD choice (Liu & Wang, 2016). The median is taken over the
    OFF-diagonal pairwise distances only: the M zero self-distances on the
    diagonal are not genuine pairwise distances and including them biases
    h^2 downward (markedly so at small M). A small floor avoids a zero
    bandwidth when particles collapse.

    Args:
        z: Particles, shape (m, d).

    Returns:
        Squared bandwidth h^2 as a Python float.
    """
    m = z.shape[0]
    sq_dists = torch.cdist(z, z) ** 2
    # Median over off-diagonal entries only (strict upper triangle), so
    # the M zero self-distances do not bias h^2 down.
    iu, iv = torch.triu_indices(m, m, offset=1)
    off_diag = sq_dists[iu, iv]
    med = torch.median(off_diag)
    h_sq = med.item() / (2.0 * math.log(m + 1.0))
    return max(h_sq, 1e-8)
