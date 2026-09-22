"""Designed quadrature of a closed-form posterior (Stage 1).

Given a posterior ``rho``, take ``M`` i.i.d. samples and MOVE + REWEIGHT them
into an ``M``-node weighted quadrature ``Q`` whose ``MMD(Q, rho)`` is BELOW the
i.i.d. set's -- it integrates ``rho`` better than ``M`` i.i.d. draws, for free.
This subpackage is Stage 1: the CLOSED-FORM toys where ``rho`` is the TRUE
``pi`` (known density, exact kernel mean + self-affinity), proving the core
claim. There is NO neural network here -- the set-equivariant amortizer is
Stage 3 (out of scope).

The pieces (one concept per file):
  * :mod:`~qfield.designed_quadrature.mmd` -- the unit-sum SE-MMD^2
    objective and the explicit-vs-closed-form cross-check;
  * :mod:`~qfield.designed_quadrature.weights` -- the
    unit-sum-constrained MMD^2-optimal weights (the ``<= floor`` lever);
  * :mod:`~qfield.designed_quadrature.move` -- direct MMD^2 descent
    over node positions with best-iterate (the ``<= reweight`` lever);
  * :mod:`~qfield.designed_quadrature.target` -- the
    :class:`QuadratureTarget` structural interface and the Gaussian / GMM
    closed-form builders;
  * :mod:`~qfield.designed_quadrature.arms` -- the three arms and the
    ``M``-sweep evaluation with per-draw theorem checks.

Reference:
  - Belhadji, Sharp, Marzouk, "To discretize continually ...",
    arXiv:2605.14142 (the MMD-optimal designed quadrature; SE closed forms).
  - Belhadji, Sharp, Marzouk, "Weighted quantization using MMD: From mean
    field to mean shift via gradient flows", AISTATS 2026 (arXiv:2502.10600)
    (kernel-quadrature weights; the i.i.d.-floor benchmark).
"""

from __future__ import annotations

from qfield.designed_quadrature.arms import (
    THEOREM_TOL,
    cross_check_target,
    functional_spot_check,
    run_target,
    score_arms_batched,
    score_arms_one_draw,
)
from qfield.designed_quadrature.cost import (
    CostLedger,
    CountingHook,
    CountingSampler,
    FitCost,
    break_even_curve,
    break_even_queries,
    n_points,
    target_equivalents,
)
from qfield.designed_quadrature.herding import (
    herding_step_sizes,
    kernel_herding,
)
from qfield.designed_quadrature.mmd import (
    cross_check_mmd,
    gram,
    gram_batched,
    mmd_sq,
    mmd_sq_batched,
)
from qfield.designed_quadrature.sbq import (
    SCHUR_FLOOR,
    constrained_value_direct,
    sequential_bayesian_quadrature,
)
from qfield.designed_quadrature.recombination import (
    caratheodory_reduce,
    nystrom_features,
    recombination_quadrature,
)
from qfield.designed_quadrature.thinning import (
    admissible_budgets,
    kernel_halving,
    kernel_swap,
    kernel_thinning,
)
from qfield.designed_quadrature.move import (
    move_descend,
    move_descend_batched,
)
from qfield.designed_quadrature.net import QuadratureAmortizer
from qfield.designed_quadrature.certsel import (  # noqa: F401
    COINCIDENCE_D,
    menu_slack,
    pair_scale,
    pairwise_slack,
    selection_verdict,
    slack_from_witness,
    slack_value,
    witness_evaluations,
)
from qfield.designed_quadrature.safeguard import (
    TWOARM_TOL,
    criterion,
    select_emission,
)
from qfield.designed_quadrature.target import (
    QuadratureTarget,
    banana_target,
    build_target,
    gaussian_target,
    gmm_target,
    latent_gaussian_target,
    nw_conditional_target,
    sampled_bank_target,
    whitened_gaussian_target,
    whitened_gmm_cncv_target,
)
from qfield.designed_quadrature.weights import (
    DEFAULT_JITTER,
    constrained_weights,
    constrained_weights_batched,
    constrained_weights_from_gram,
)

__all__ = [
    "DEFAULT_JITTER",
    "SCHUR_FLOOR",
    "THEOREM_TOL",
    "TWOARM_TOL",
    "CostLedger",
    "CountingHook",
    "CountingSampler",
    "FitCost",
    "QuadratureAmortizer",
    "QuadratureTarget",
    "admissible_budgets",
    "banana_target",
    "break_even_curve",
    "break_even_queries",
    "build_target",
    "caratheodory_reduce",
    "constrained_value_direct",
    "constrained_weights",
    "constrained_weights_batched",
    "constrained_weights_from_gram",
    "cross_check_mmd",
    "criterion",
    "cross_check_target",
    "functional_spot_check",
    "gaussian_target",
    "gmm_target",
    "gram",
    "gram_batched",
    "herding_step_sizes",
    "kernel_halving",
    "kernel_herding",
    "kernel_swap",
    "kernel_thinning",
    "latent_gaussian_target",
    "mmd_sq",
    "mmd_sq_batched",
    "move_descend",
    "n_points",
    "nw_conditional_target",
    "nystrom_features",
    "recombination_quadrature",
    "sampled_bank_target",
    "sequential_bayesian_quadrature",
    "move_descend_batched",
    "run_target",
    "score_arms_batched",
    "score_arms_one_draw",
    "select_emission",
    "COINCIDENCE_D",
    "menu_slack",
    "pair_scale",
    "pairwise_slack",
    "selection_verdict",
    "slack_from_witness",
    "slack_value",
    "witness_evaluations",
    "target_equivalents",
    "whitened_gaussian_target",
    "whitened_gmm_cncv_target",
]
