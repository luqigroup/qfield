"""Amortized quadrature for posterior expectations in inverse problems.

The public surface is re-exported here. The implementation is imported on
first use, so naming the package costs nothing.
"""

from __future__ import annotations

__version__ = "1.0.0"

_API = (
    "ConditionalQuadratureField", "DEFAULT_JITTER", "QuadratureField",
    "banana_target", "build_target", "conditional_gaussian_reference",
    "criterion", "emit", "gaussian_target",
    "gaussian_reference", "gmm_target", "gram", "gram_batched", "kernel_mean",
    "median_bandwidth", "mmd_sq",
    "mmd_sq_batched", "move_descend", "sampled_bank_target",
    "select_emission", "solve_weights", "solve_weights_batched",
)

__all__ = list(_API) + ["__version__"]


def __getattr__(name: str):
    if name in _API:
        from qfield import api
        return getattr(api, name)
    raise AttributeError(f"module 'qfield' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
