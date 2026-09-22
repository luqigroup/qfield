"""The public surface of the package.

Everything here delegates to the implementation modules. Import from
``qfield`` rather than from this module.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from qfield.designed_quadrature.mmd import gram, gram_batched, mmd_sq, mmd_sq_batched
from qfield.designed_quadrature.move import move_descend
from qfield.designed_quadrature.net import QuadratureAmortizer
from qfield.designed_quadrature.cond_net import ConditionalQuadratureAmortizer
from qfield.designed_quadrature.safeguard import criterion, select_emission
from qfield.dataset.linear_gaussian import LinearGaussian
from qfield.designed_quadrature.target import (
    _chunked_kernel_mean,
    banana_target,
    build_target,
    gaussian_target,
    gmm_target,
    latent_gaussian_target,
    sampled_bank_target,
)
from qfield.designed_quadrature.weights import (
    DEFAULT_JITTER,
    constrained_weights,
    constrained_weights_batched,
)
from qfield.kernels import median_heuristic_sigma_sq

solve_weights = constrained_weights
solve_weights_batched = constrained_weights_batched

MuFn = Callable[[torch.Tensor], torch.Tensor]


def median_bandwidth(z: torch.Tensor) -> float:
    """The kernel bandwidth implied by a sample set.

    The median heuristic: the median distance between distinct samples. This
    is the rule every experiment in the paper uses.

    Args:
        z: Samples, shape ``(M, d)`` with ``M >= 2``.

    Returns:
        The bandwidth.
    """
    return math.sqrt(median_heuristic_sigma_sq(z))


def gaussian_reference(d: int = 2, m: int = 64, seed: int = 0,
                       dtype: torch.dtype = torch.float64):
    """A Gaussian reference posterior and ``m`` samples of it.

    The bandwidth is taken from the samples, so nothing has to be chosen by
    hand. The sampler does not depend on it.

    Args:
        d: Dimension.
        m: Number of samples.
        seed: Seed for the sampler.
        dtype: Working precision.

    Returns:
        The reference and its samples.
    """
    probe = gaussian_target(1.0, d=d, dtype=dtype)
    z0 = probe.sampler(m, torch.Generator().manual_seed(seed))
    return gaussian_target(median_bandwidth(z0), d=d, dtype=dtype), z0


def conditional_gaussian_reference(d: int = 2, dy: int = 2, m: int = 64,
                                   seed: int = 0,
                                   dtype: torch.dtype = torch.float64):
    """One member of a Gaussian family, its observation, and samples of it.

    The observation indexes the family, and the posterior for it is Gaussian,
    so its kernel mean is exact and a conditional field can be exercised
    without training anything. The bandwidth is taken from the samples.

    Args:
        d: Dimension of the unknown.
        dy: Dimension of the observation.
        m: Number of samples.
        seed: Seed for the family and the sampler.
        dtype: Working precision.

    Returns:
        The reference, its samples, and the observation.
    """
    gen = torch.Generator().manual_seed(seed)
    prior_cov = torch.eye(d, dtype=dtype) + 0.2 * torch.ones(d, d, dtype=dtype)
    forward = torch.randn(dy, d, generator=gen).to(dtype) / d ** 0.5
    problem = LinearGaussian(
        prior_mean=torch.zeros(d, dtype=dtype),
        prior_cov=prior_cov,
        forward=forward,
        obs_cov=0.3 ** 2 * torch.eye(dy, dtype=dtype),
    )
    _, y_all = problem.sample_joint(64)
    y = y_all[-1].to(dtype)
    mu_post, cov_post = problem.posterior(y)
    probe = latent_gaussian_target(mu_post, cov_post, 1.0, dtype=dtype)
    z0 = probe.sampler(m, gen)
    sigma = median_bandwidth(z0)
    return latent_gaussian_target(mu_post, cov_post, sigma, dtype=dtype), z0, y


def kernel_mean(z: torch.Tensor, samples: torch.Tensor, sigma: float) -> torch.Tensor:
    """The kernel mean at ``z``, estimated from samples of the reference.

    Args:
        z: Nodes, shape ``(M, d)``.
        samples: Samples of the reference, shape ``(n, d)``.
        sigma: Kernel bandwidth.

    Returns:
        The estimate, shape ``(M,)``.
    """
    return _chunked_kernel_mean(z, samples, 1.0 / (2.0 * sigma ** 2))


def emit(
    field: torch.nn.Module,
    samples: torch.Tensor,
    y: torch.Tensor | None = None,
    m: int | None = None,
    sigma: float | None = None,
    mu_fn: MuFn | None = None,
    jitter: float = DEFAULT_JITTER,
) -> dict:
    """One forward pass: displace the samples, solve the weights, choose.

    Three candidates are scored by the same criterion and the best is
    returned: the samples under equal weights, the samples under solved
    weights, and the displaced nodes under the weights solved there. The
    samples are among the candidates, so what is returned is never worse than
    them when the kernel mean is exact.

    Args:
        field: A :class:`QuadratureField` or
            :class:`ConditionalQuadratureField`.
        samples: Samples of the reference at this observation, shape
            ``(n, d)``. The first ``m`` seed the nodes and the rest estimate
            the kernel mean. With ``m`` unset all of them seed the nodes and
            the estimate is read off the same set.
        y: The observation, for a conditional field. Required by one, refused
            by the other.
        m: The node count. Defaults to every sample given.
        sigma: Kernel bandwidth. Taken from the seeds by the median heuristic
            when it is not given.
        mu_fn: An exact kernel mean, where the reference supplies one. The
            estimate from ``samples`` is used when it is not given.
        jitter: Ridge on the weight solve.

    Returns:
        The dictionary :func:`select_emission` returns, plus the ``sigma``
        used. Its ``z`` and ``w`` are the returned quadrature's nodes and
        weights, ``name`` says which candidate won, and ``margin`` is how much
        the criterion improved on the samples.
    """
    if samples.dim() != 2:
        raise ValueError(f"samples must be (n, d); got {tuple(samples.shape)}")
    conditional = isinstance(field, ConditionalQuadratureAmortizer)
    if conditional and y is None:
        raise ValueError("a conditional field needs the observation y")
    if y is not None and not conditional:
        raise ValueError("this field takes no observation")

    m = samples.shape[0] if m is None else m
    if m > samples.shape[0]:
        raise ValueError(f"asked for {m} nodes from {samples.shape[0]} samples")
    z0 = samples[:m]
    bank = samples[m:] if samples.shape[0] > m else samples
    if sigma is None:
        sigma = median_bandwidth(z0)
    if mu_fn is None:
        def mu_fn(z: torch.Tensor) -> torch.Tensor:
            return kernel_mean(z, bank, sigma)

    with torch.no_grad():
        if conditional:
            z = field(z0.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
        else:
            z = field(z0.unsqueeze(0)).squeeze(0)
    w_equal = torch.full((m,), 1.0 / m, dtype=z0.dtype, device=z0.device)
    w_reweighted = constrained_weights(z0, mu_fn(z0), sigma, jitter)
    w_moved = constrained_weights(z, mu_fn(z), sigma, jitter)
    out = select_emission(
        [(z0, w_equal), (z0, w_reweighted), (z, w_moved)],
        mu_fn,
        sigma,
        names=("samples", "reweighted", "moved"),
    )
    out["sigma"] = sigma
    return out


class QuadratureField(QuadratureAmortizer):
    """The quadrature field for a reference that carries no observation.

    A stack of permutation-equivariant blocks that displaces a set of samples
    into nodes. The weights are solved, not learned.
    """

    def emit(self, samples: torch.Tensor, m: int | None = None,
             sigma: float | None = None, mu_fn: MuFn | None = None,
             jitter: float = DEFAULT_JITTER) -> dict:
        """See :func:`qfield.emit`."""
        return emit(self, samples, None, m, sigma, mu_fn, jitter)


class ConditionalQuadratureField(ConditionalQuadratureAmortizer):
    """The quadrature field for a family of posteriors.

    Adds cross-attention from the samples to tokens of the observation, so one
    trained field serves every observation of the family.
    """

    def emit(self, samples: torch.Tensor, y: torch.Tensor,
             m: int | None = None, sigma: float | None = None,
             mu_fn: MuFn | None = None,
             jitter: float = DEFAULT_JITTER) -> dict:
        """See :func:`qfield.emit`."""
        return emit(self, samples, y, m, sigma, mu_fn, jitter)


__all__ = [
    "ConditionalQuadratureField",
    "DEFAULT_JITTER",
    "QuadratureField",
    "banana_target",
    "build_target",
    "criterion",
    "emit",
    "conditional_gaussian_reference",
    "gaussian_reference",
    "gaussian_target",
    "gmm_target",
    "gram",
    "kernel_mean",
    "gram_batched",
    "median_bandwidth",
    "mmd_sq",
    "mmd_sq_batched",
    "move_descend",
    "sampled_bank_target",
    "select_emission",
    "solve_weights",
    "solve_weights_batched",
]
