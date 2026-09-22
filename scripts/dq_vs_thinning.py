"""Comparison against compression at an equal reference budget.

Evaluation only, on CPU: no training and no GPU.

The peers:
  * ``herding``     -- kernel herding. It reads the same oracle this
    construction reads, the kernel mean, and no density and no score.
    Arbitrary budget, equal weights.
  * ``sbq``         -- sequential Bayesian quadrature: the same greedy search
    over the same candidate pool, but every candidate scored after the
    unit-sum weights are re-solved at the enlarged node set, and the same
    signed unit-sum weights emitted.
  * ``thinning``    -- kernel thinning. Consumes ``n = M^2`` samples, equal
    weights, nodes restricted to the input sample, budget restricted to the
    dyadic grid. Its own input sample's floor is recorded per cell.
  * ``recombine``   -- recombination on a rank-``(M-1)`` Nystrom truncation
    built from an independent sample. Convex weights on pool atoms.
  * ``msip_dd``     -- the data-driven quantizer of arXiv:2502.10600,
    Algorithm 2: empirical-bank KDE embeddings
    (:class:`~qfield.conditional.DataDrivenEmbeddings`) driven through
    ``run_msip`` at its published settings (SE bandwidth 0.6, step 0.5, 1000
    steps, no box clamp), fed the same candidate pool the other peers consume
    as its bank. Both weightings its own paper admits are scored: the
    unit-sum normalization ``mmd_sq``, and ``mmd_sq_raw``, the raw
    ``K^{-1} v0`` its Algorithm 2's final line emits.
  * ``msip_dd_scaled`` -- the same construction at the case's own trained
    ``sigma_eval``. The literal 0.6 is a unit-scale number and its source
    scales the bandwidth per problem (their A.2), so on a ``d = 16`` latent
    of trained scale 5.54 the literal column measures bandwidth transfer as
    much as the construction.

Each equal-weight or non-negative-weight peer additionally gets a ``_rw``
variant carrying the constrained weights on its own nodes, which separates
node placement from weighting. ``sbq`` already carries those weights, so it
is reported instead with ``sbq_eq``, its own nodes under equal weights.

``ours`` is one forward pass of the trained amortizer at the budget, followed
by the constrained weight solve at its displaced nodes and the safeguarded
selection; ``ours_move_raw`` is the moved arm before that selection.
``oracle_move`` is the per-instance MMD^2 descent of
``designed_quadrature.move``, which is what the map amortizes.
``ours_samples_only`` is the same forward pass on the same seeds, but the
weight solve and the safeguarded selection read a kernel-mean estimate over
the same ``n = M^2`` samples thinning consumes, regenerated from thinning's
own seed stream; its reported MMD^2 is still scored against the exact
reference.

``wedge`` is a family of exact latent Gaussian posteriors whose shape moves
with a per-observation acquisition, served by one trained
:class:`~qfield.designed_quadrature.cond_net.ConditionalQuadratureAmortizer`.
Checkpoints are resolved by config identity through
``scripts/_ckpt.checkpoint_path``, never by glob sort: several directories
share the name prefix and differ only in the trained bandwidth. The
per-observation references are staged by reusing ``dq_wedge_conditional``'s
own machinery (``_build_prior`` / ``_build_basis`` / ``_build_family`` /
``_obs_target``) on that run's held-out eval family, with ``sigma`` /
``cond_mean`` / ``cond_std`` taken from its ``results.json``.

The figure's shared x-axis is reference samples consumed per observation,
recorded per method per cell in ``draws_consumed``, not the node budget.

Each peer must reproduce its own source's behaviour on a case where that
behaviour is known; a peer that fails its control is withheld and the
omission is stated. Thinning additionally carries a log-log slope band (see
``THINNING_SLOPE_BAND``), scored on a closed-form ruler so no bank noise
floor enters the levels the slope is fitted to.

At ``M_EXT = [256, 512]`` the thinning arm runs the peer's own published
tooling, ``goodpoints`` 0.6.3, which never builds the input-sized ``(n, n)``
Gram: ``kt.thin`` with ``store_K=False`` streams kernel rows in ``O(nd)``
memory, and ``compresspp_kt``'s largest allocation is the dense Gram of its
concatenated per-bin coresets. ``--phase extend`` appends those cells to the
saved record, per backend:

  * ``M = 256`` (n = 65536): ``goodpoints.kt.thin`` -- the full quadratic-
    time construction (KT-SPLIT + KT-SWAP), target kernel for both stages
    (matching the dense backend's own Gram convention), ``delta = 0.5``, the
    row-mean statistic its swap stage reads handed in precomputed (blocked,
    O(n) memory) outside the timed window.
  * ``M = 512`` (n = 262144): ``goodpoints.compress.compresspp_kt`` --
    Compress++(g = 4), the peer's own near-linear accelerator (Shetty,
    Dwivedi and Mackey, ICLR 2022), because the quadratic ``kt.thin`` costs
    about 80 minutes per repeat at this n. Its guarantee concedes at most a
    factor of four in error. At M = 256, where both backends run, the record
    carries ``thinning_compresspp`` beside the plotted plain-KT cell.

  The extension cells carry the arms that need no candidate pool: floor,
  reweight, ours, ours_move_raw, ours_samples_only, oracle_move, thinning
  (+ _rw, + input floor). The pool peers (herding / sbq / recombine /
  msip_dd) stay on the [32, 64, 128] grid: at M = 256 / 512 their shared
  512-atom pool would sit within a factor of two of the node budget, which
  the pool rule exists to prevent. The protocol at the new cells is the same
  one: the same ``BASE_SEED + CONFIRM_EVAL_OFFSET + 1000*M + r`` eval seeds,
  the same ``N_REPEATS``, the same per-arm generator streams, the same
  exact-reference scoring. Both goodpoints backends pass the same controls
  as the dense backend before any cell is written; a failing backend
  withholds its cells.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/dq_vs_thinning.py
    CUDA_VISIBLE_DEVICES="" python scripts/dq_vs_thinning.py --phase visualize
    CUDA_VISIBLE_DEVICES="" python scripts/dq_vs_thinning.py --phase extend
    # N workers own disjoint (case, budget) slices and touch only their own
    # sidecars; the final plain --phase extend merges once:
    ... --phase extend --ext-cases wedge --ext-budgets 256 --ext-no-merge

Reference:
  - Chen, Welling and Smola, "Super-samples from kernel herding", UAI 2010.
  - Huszar and Duvenaud, "Optimally-weighted herding is Bayesian quadrature",
    UAI 2012 (arXiv:1204.1664) -- the ``sbq`` peer.
  - Dwivedi and Mackey, "Kernel thinning", COLT 2021 / JMLR 25(152), 2024.
  - Shetty, Dwivedi and Mackey, ICLR 2022 (Compress++).
  - Hayakawa, Oberhauser and Lyons, NeurIPS 2022 (recombination).
  - Belhadji, Sharp and Marzouk, arXiv:2502.10600 (the data-driven MSIP
    quantizer, its Algorithm 2 and published settings) and arXiv:2605.14142
    (Table 3, the coordinate clamp its ``max_abs_coordinate`` is read
    against).
"""

from __future__ import annotations
from projorg import plotsdir  # noqa: E402

# Pin to CPU before importing torch.
import os  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402


from _ckpt import checkpoint_path  # noqa: E402
from _figstyle import apply_house_style, despine  # noqa: E402
from dq_safeguard_sweep import (  # noqa: E402
    find_run,
    load_net,
    load_state,
    target_for,
)
from projorg import configsdir, plotsdir
from projorg.config import read_config  # noqa: E402

from qfield.conditional import DataDrivenEmbeddings  # noqa: E402
from qfield.designed_quadrature import (  # noqa: E402
    DEFAULT_JITTER,
    CountingHook,
    admissible_budgets,
    constrained_value_direct,
    constrained_weights,
    cross_check_target,
    gaussian_target,
    gmm_target,
    gram,
    kernel_herding,
    kernel_thinning,
    mmd_sq,
    move_descend,
    nw_conditional_target,
    recombination_quadrature,
    sampled_bank_target,
    select_emission,
    sequential_bayesian_quadrature,
)
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)
from qfield.kernels import median_heuristic_sigma_sq  # noqa: E402
from qfield.msip import run_msip  # noqa: E402

FIG_DIR = plotsdir("paper")
DIAG_DIR = os.path.join(FIG_DIR, "diagnostics")

DTYPE = torch.float64
torch.set_num_threads(4)

# The node-budget grid for the full set of arms, capped at 128 by the
# thinning peer's own n = M^2 input schedule (see ``THINNING_INPUT``).
M_LIST = [32, 64, 128]
N_REPEATS = 64
BASE_SEED = 20260612
# Every seed stream is addressed as ``offset + 1000*M + r``, so one stream is
# 1000*max(M) + r wide. The offsets below keep the eval, tuning and per-arm
# bands at least 500 000 clear of one another.
TUNE_OFFSET = 2_000_000
N_TUNE_REPEATS = 16
JITTER = DEFAULT_JITTER

# Eval seeds are ``BASE_SEED + CONFIRM_EVAL_OFFSET + 1000*M + r``, below the
# tuning offset and below every per-arm stream offset.
CONFIRM_EVAL_OFFSET = 1_000_000

# The per-instance oracle descent, fixed and untuned.
ORACLE_LR = 0.05
ORACLE_ITERS = 150

# The data-driven quantizer at its published settings (arXiv:2502.10600: SE
# bandwidth 0.6, damped step 0.5, 1000 steps; its Algorithm 2 carries no box
# clamp, hence bounds=None). The 0.6 is a unit-scale number: their A.2 scales
# the bandwidth per problem (0.6 for the 2-D example, 2.25 for the Matern
# MNIST run, 0.025 elsewhere), so the scale-matched column runs the identical
# construction at the case's own trained ``sigma_eval``.
MSIP_DD_BANDWIDTH = 0.6
MSIP_DD_LR = 0.5
MSIP_DD_ITERS = 1000
MSIP_DD_CLAMP_REFERENCE = 1e3
# The diagonal inflation of the quantizer's Gram solve. Its source carries
# none; a tiny value keeps the solve defined.
MSIP_DD_INFL = 1e-8

# Seed-Gram conditioning threshold: past about 1e13 the constrained solve
# retains around two digits and rankings between solved-weight arms are solve
# noise. Cells past it are marked, never excluded and never ridged.
COND_K_THRESHOLD = 1e13

# CASE -> the trained amortizer the ``ours`` arm is one forward pass of.
# Two resolution kinds, neither of which is a sorted-glob or mtime pick:
#   * "find_run"        -- resolved through :func:`dq_safeguard_sweep.find_run`
#     which content-verifies against results.json and raises on ambiguity;
#   * "config_identity" -- resolved through :func:`_ckpt.checkpoint_path`,
#     which rebuilds the projorg experiment name from the config file byte for
#     byte (the only safe rule when several directories share a prefix).
# A case absent from this table has no trained network and is skipped.
CASE_RUNS = {
    # The sigma filter is the content check: the trained sigma 20.2856 read
    # from results.json, so no run at another bandwidth can resolve here. The
    # m_list infix does the same job for the node-budget grid.
    "gmm2": {
        "kind": "find_run",
        "spec": (
            "dq_gmm_d2_target-gmm_d-2_"
            "*m_list-32-64-128-256-512_*bandwidth-median_mmd_*",
            "gmm", 2, 3000, 20.2856, 1e-3,
        ),
    },
    # ``ignore_extra`` mirrors the script's own extended ignore_arg_list, so
    # the render-only knobs stay out of the rebuilt experiment name.
    "wedge": {
        "kind": "config_identity",
        "config": "dq_wedge_conditional.json",
        "ignore_extra": ("recon_M", "eval_repeats"),
        "overrides": {"bandwidth": "median_mmd"},
    },
}

# Per-arm generator offsets. Threading one generator through the arms in
# sequence would make every downstream arm's samples depend on which arms ran,
# so withholding a peer would silently change the peers that remain. The
# offsets clear the eval seeds and the tuning band.
ARM_SEED_OFFSET = {
    "seed": 0,
    "herding": 10_000_000,
    "thinning": 20_000_000,
    "recombine": 30_000_000,
    "sbq": 40_000_000,
    "msip_dd": 50_000_000,
}

# Thinning halves, so it is handed the n = M^2 input size its own analysis
# assumes. The in-repo ``kernel_thinning`` builds a dense (n, n) Gram over
# that schedule, which is why the full set of arms stops at M = 128:
#   M =  32 -> n =   1024 ->   8 MB
#   M =  64 -> n =   4096 -> 134 MB
#   M = 128 -> n =  16384 -> 2.15 GB
#   M = 256 -> n =  65536 ->  34 GB   (infeasible densely)
#   M = 512 -> n = 262144 -> 550 GB   (infeasible densely)
# The published tooling (``goodpoints``) avoids the dense Gram in O(nd)
# memory, so the thinning arm covers M_EXT via ``--phase extend``. The pool
# peers stay on M_LIST: their 512-atom pool would sit within a factor of two
# of the budget at M = 256 / 512.
M_EXT = [256, 512]
THINNING_INPUT = {M: M * M for M in M_LIST + M_EXT}
# Which published backend produces which extension cell: plain quadratic KT
# where it is computable in a day, Compress++(g=4) at the top budget.
EXT_BACKEND = {256: "goodpoints_kt", 512: "goodpoints_compresspp"}
COMPRESSPP_G = 4          # the accelerator paper's own experimental setting
COMPRESSPP_NUM_BINS = 4   # compresspp_kt default
KT_DELTA = 0.5            # kt.thin default failure-probability parameter
FLOOR_BLOCK = 4096        # blocked row-sum tile (O(n * block) memory)
# The pool sits above every node budget on the grid, so no search ever
# selects M atoms from a pool of M.
HERDING_POOL = 512          # candidate pool for the conditional gradient
SBQ_POOL = HERDING_POOL     # the same pool, so the two greedy searches differ
                            # only in whether the weights are re-solved inside
                            # the search
MSIP_DD_POOL = HERDING_POOL  # the same pool again: it is the msip_dd bank
RECOMBINE_POOL = 512        # pool the Caratheodory step reduces
RECOMBINE_NYSTROM = 256     # independent sample defining the truncation

# Each peer's own tuning grid, on the disjoint tuning-seed offset.
HERDING_MODES = ["herding", "fw", "line_search"]
THINNING_SWAP_PASSES = [0, 1, 2]
# The refinement budget of the weight-aware peer, tuned exactly as thinning's
# swap budget is. Zero keeps the construction nested in M: the prefix of the
# greedy selection is the selection at the smaller budget.
SBQ_SWAP_PASSES = [0, 1, 2]

# --- Peer-correctness thresholds -------------------------------------------
# SBQ at finite rank must integrate to machine precision; 1e-8 sits five
# orders below the best equal-weight value on the same pools, so the control
# separates the constructions by a margin no round-off can close while leaving
# room for the jitter-limited residual of a rank-deficient solve.
SBQ_FINITE_RANK_TOL = 1e-8
# SBQ must not lose to equal-weight herding on the same pool in the median;
# greedy search is not optimality, so the per-cell fraction is reported beside
# it rather than asserted.
SBQ_MEDIAN_GAP_MAX = 0.0
# Thinning's log-log acceptance band, in MMD^2-vs-n slope.
#
# Thinning from n inputs to sqrt(n) points guarantees root-MMD
# O(sqrt(log n / n)), i.e. MMD^2 = O(log n / n). That is not a power law, so
# its log-log slope depends on the grid: over n in {64, 256, 1024} the
# least-squares slope of log10(ln n / n) against log10(n) is (with three
# evenly log-spaced points, the secant)
#     (log10(6.9315/1024) - log10(4.1589/64)) / (log10(1024) - log10(64))
#   = (-2.16977 + 1.18718) / 1.20412 = -0.8160.
# An equal-sized i.i.d. sample is Omega(n^{-1/4}) in root-MMD, i.e. MMD^2
# slope -1/2. The upper edge is the midpoint separator between the two,
# (-0.8160 - 0.5)/2 = -0.658, rounded to -0.66.
#
# The lower edge is a one-sided guard only: the published result is an error
# upper bound, so a slope steeper than the guarantee is the peer doing better
# than its paper promises, while a slope past -2 means the levels, not the
# construction, are wrong.
#
# The ruler is closed form. A finite sample bank's own noise floor is additive
# in MMD^2, and an additive constant does not cancel from a log-log slope;
# :func:`_uniform01_target` removes it outright, since on U[0,1] the SE kernel
# mean and self-affinity are erf/exp forms.
THINNING_SLOPE_BAND = (-2.0, -0.66)
THINNING_SLOPE_GRID = (64, 256, 1024)   # n; M = sqrt(n) per the corollary
THINNING_SLOPE_REPEATS = 4
THINNING_SLOPE_SIGMA = 0.1  # the target's own scale on U[0,1]; the rate is
                            # bandwidth-independent, legibility is not

# The observation stream the conditional target is swept over, at most the
# wedge run's held-out eval observations, of which the first N_OBS are used.
N_OBS = 64

# --------------------------------------------------------------------------
# The design block, written into the record verbatim so the protocol travels
# with the numbers.
# --------------------------------------------------------------------------
DESIGN = {
    "status": (
        "CONFIRMATORY protocol. The 2026-08-05 measured tables in "
        "EXPERIMENT_DESIGN_v7.md sec. 3.9 are PILOT readings on the "
        "superseded old-rule gmm2 checkpoint (sigma = 4.9806) and are quoted "
        "nowhere as measured."
    ),
    "arms": [
        "floor", "reweight", "herding", "herding_rw", "sbq", "sbq_eq",
        "thinning", "thinning_rw", "thinning_input_floor", "recombine",
        "recombine_rw", "msip_dd", "msip_dd_rw", "msip_dd_raw",
        "msip_dd_scaled", "msip_dd_scaled_rw", "msip_dd_scaled_raw",
        "oracle_move", "ours_samples_only", "ours", "ours_move_raw",
    ],
    "pool": HERDING_POOL,
    "budgets": M_LIST,
    "budgets_extension": M_EXT,
    "budget_cap": {
        "full_arm_cap": 128,
        "uncapped_grid": [32, 64, 128, 256, 512],
        "thinning_dense_gram_gb": {
            "M=128": 2.15, "M=256": 34.0, "M=512": 550.0,
        },
        "ruling": (
            "2026-08-10 ruling 2 capped the grid at M = 128 because the "
            "in-repo thinning backend builds a dense (n, n) Gram over the "
            "peer's n = M^2 input schedule -- 2.15 GB at M = 128, 550 GB at "
            "M = 512. SUPERSEDED FOR THE THINNING ARM 2026-08-22: the dense "
            "input Gram was our implementation's shortcut, and the peer's "
            "published tooling (goodpoints 0.6.3) never builds it, so the "
            "thinning arm now covers the full deployed grid."
        ),
        "superseded": {
            "ruling_2026_08_10": (
                "2026-08-10 ruling 2: the budget grid is capped at M = 128 "
                "because kernel thinning's own n = M^2 input schedule "
                "builds a dense (n, n) Gram -- 2.15 GB at M = 128, 34 GB "
                "at M = 256, 550 GB at M = 512 -- a property of the peer, "
                "reported rather than engineered around."
            ),
            "note": (
                "kept verbatim for provenance; superseded 2026-08-22 for "
                "the thinning arm only (the pool peers' cap stands)"
            ),
        },
        "extension_2026_08_22": {
            "budgets": M_EXT,
            "backend_per_budget": {
                "256": (
                    "goodpoints.kt.thin -- full quadratic-time KT-SPLIT + "
                    "KT-SWAP, target kernel both stages, delta = 0.5, "
                    "store_K=False (O(nd) memory), row-mean statistic "
                    "precomputed outside the timed window"
                ),
                "512": (
                    "goodpoints.compress.compresspp_kt -- Compress++(g=4), "
                    "the peer's own near-linear accelerator (<= 4x error "
                    "concession, Shetty--Dwivedi--Mackey ICLR 2022); plain "
                    "kt.thin measures ~80 min per repeat at n = 262144 on "
                    "this box, infeasible at 64 repeats x 2 cases"
                ),
            },
            "concession_measurement": (
                "at M = 256 both backends run: the plotted thinning cell is "
                "plain KT and the record's thinning_compresspp column is "
                "Compress++ on the identical input sample and seed, so the "
                "accelerator's realized concession is measured at the one "
                "budget both can afford. UNITS (Gate-1 item 3): the record "
                "stores MMD^2, so the published <= 4x root-MMD concession "
                "is <= 16x on these columns; any quote beside the "
                "guarantee must be converted (sqrt of the MMD^2 ratio)"
            ),
            "extension_arms": [
                "floor", "reweight", "ours", "ours_move_raw",
                "ours_samples_only", "oracle_move", "thinning",
                "thinning_rw", "thinning_input_floor",
            ],
            "pool_peers_remain_capped": (
                "herding / sbq / recombine / msip_dd stay on the "
                "pre-registered [32, 64, 128] grid: their shared 512-atom "
                "pool (ruling 2) would sit within 2x of / equal to the node "
                "budget at M = 256 / 512 -- the select-M-from-a-pool-of-M "
                "degeneracy the pool rule exists to prevent. Extending them "
                "would re-register the pool, a design change rather than an "
                "extension."
            ),
        },
    },
    "repeats": N_REPEATS,
    "eval_seed_rule": (
        "BASE_SEED + CONFIRM_EVAL_OFFSET (= 1000000) + 1000*M + r, "
        "r < N_REPEATS (= 64) -- disjoint from the pilot's "
        "BASE_SEED + 1000*M + r (r < 256), from the tuning band at "
        "TUNE_OFFSET (= 2000000) and from every per-arm stream offset "
        "(>= 10000000)"
    ),
    "matched_quality_definition": (
        "A1 (verdict): per-new-observation construction cost at the SAME M, "
        "with quality (the median MMD^2 of 'rows') shown ADJACENT -- never a "
        "quality-equalized budget search"
    ),
    "mu_oracle_arms": ["reweight", "ours", "oracle_move", "herding", "sbq"],
    "thresholds": {
        "metric_cross_check": 1e-8,
        "sbq_finite_rank_mmd_sq": SBQ_FINITE_RANK_TOL,
        "sbq_median_gap_vs_herding_max": SBQ_MEDIAN_GAP_MAX,
        "thinning_slope_band": list(THINNING_SLOPE_BAND),
        "cond_K_threshold": COND_K_THRESHOLD,
        "herding_control_gap_max": 1e-9,
        "herding_inline_cross_check_gap_max": 1e-9,
    },
    "msip_dd_published_settings": {
        "bandwidth": MSIP_DD_BANDWIDTH, "lr": MSIP_DD_LR,
        "n_iters": MSIP_DD_ITERS, "bounds": None,
        "kernel_diag_infl": MSIP_DD_INFL,
        "clamp_reference": MSIP_DD_CLAMP_REFERENCE,
        "source": "arXiv:2502.10600 (settings); arXiv:2605.14142 Table 3 "
                  "(the R5 clamp level)",
        "infl_note": (
            "their listing carries NO diagonal inflation; 1e-8 keeps the "
            "solve defined and is UNIFIED with dq_currency_crossover.DD_INFL "
            "so the two sections' levels come off one solve"
        ),
    },
    "preregistered_readings": {
        "msip_dd": (
            "the LITERAL-PUBLISHED cell: bandwidth 0.6 transferred unchanged "
            "onto this case. On a d = 16 latent of trained scale 5.54 this "
            "column INCLUDES the scale transfer and is not, by itself, a "
            "reading about the predecessor's construction"
        ),
        "msip_dd_scaled": (
            "the SAME construction at the case's own trained sigma_eval -- "
            "their A.2's own per-problem bandwidth practice. This is the "
            "column a claim about the predecessor must be read against"
        ),
        "msip_dd_raw": (
            "the same nodes under the RAW K^{-1} v0 weights their Algorithm "
            "2's final line emits (not unit-sum). No dominance argument "
            "exists either way: the constrained fairness column upper-bounds "
            "only unit-sum weightings, so both are reported"
        ),
        "thinning_slope_lower_edge": (
            "a one-sided bug guard, NOT an acceptance gate: the published "
            "result is an error UPPER bound, so measuring steeper than the "
            "guarantee is the peer doing better than its paper promises"
        ),
    },
}

# One colour per arm.
PALETTE = {
    "floor": "#3a7ca5",
    "reweight": "#2a9d8f",
    "herding": "#9467bd",
    "sbq": "#c72e6d",
    "thinning": "#2ca02c",
    "recombine": "#8c564b",
    "msip_dd": "#17becf",
    "msip_dd_scaled": "#0f7d88",
    "ours": "#ff7f0e",
    "ours_samples_only": "#ffbb78",
    "oracle_move": "#7f7f7f",
}
LABELS = {
    "floor": "i.i.d. floor",
    "reweight": "i.i.d. nodes + reweight",
    "herding": "kernel herding",
    "sbq": "sequential Bayesian quadrature",
    "thinning": "kernel thinning",
    "thinning_input_floor": "thinning's input sample ($n = M^2$, equal w.)",
    "recombine": "recombination",
    "msip_dd": "data-driven MSIP (literal published bandwidth)",
    "msip_dd_scaled": "data-driven MSIP (scale-matched bandwidth)",
    "ours": "designed quadrature (ours, one pass)",
    "ours_samples_only": "ours, samples-only solve ($\\hat\\mu$, $n = M^2$)",
    "oracle_move": "per-instance oracle descent",
}
PLOT_KEYS = ("floor", "reweight", "herding", "sbq", "thinning", "recombine",
             "msip_dd", "msip_dd_scaled", "oracle_move", "ours_samples_only",
             "ours")
# Arms whose reported weights come from a solve against the node Gram, which
# is near-singular at large sigma: at a cond(K)-limited cell their between-arm
# rankings are solve noise, so the render marks them open there.
SOLVED_WEIGHT_KEYS = {"reweight", "sbq", "recombine", "msip_dd",
                      "msip_dd_scaled", "oracle_move", "ours_samples_only",
                      "ours"}


# --------------------------------------------------------------------------
# The reference budget: a pure function of the schedule.
# --------------------------------------------------------------------------
def draws_consumed_for(M: int) -> dict:
    """Reference samples each method consumes per observation at budget ``M``.

    This is the shared x-axis of the figure. The mirror arm consumes its ``M``
    seeds plus the ``n = M^2`` bank its solve reads; the pool peers consume
    the shared pool.
    """
    return {
        "floor": M,
        "reweight": M,
        "ours": M,
        "ours_move_raw": M,
        "ours_samples_only": M + THINNING_INPUT[M],
        "oracle_move": M,
        "herding": HERDING_POOL,
        "herding_rw": HERDING_POOL,
        "sbq": SBQ_POOL,
        "sbq_eq": SBQ_POOL,
        "thinning": THINNING_INPUT[M],
        "thinning_rw": THINNING_INPUT[M],
        "thinning_input_floor": THINNING_INPUT[M],
        "recombine": RECOMBINE_POOL + RECOMBINE_NYSTROM,
        "recombine_rw": RECOMBINE_POOL + RECOMBINE_NYSTROM,
        "msip_dd": MSIP_DD_POOL,
        "msip_dd_rw": MSIP_DD_POOL,
        "msip_dd_raw": MSIP_DD_POOL,
        # The scale-matched column is the same construction on the same pool:
        # only the bandwidth differs, so its budget is equal.
        "msip_dd_scaled": MSIP_DD_POOL,
        "msip_dd_scaled_rw": MSIP_DD_POOL,
        "msip_dd_scaled_raw": MSIP_DD_POOL,
    }


# --------------------------------------------------------------------------
# Targets.
# --------------------------------------------------------------------------
def wedge_fit_ledger(res: dict, n_obs: int = N_OBS) -> dict:
    """The wedge net's training cost as a per-family spend.

    Computed from the resolved run's own ``results.json`` (``train_secs``,
    ``n_steps``, ``batch_obs``, ``m_list``) and never measured anew. Per
    training step the trainer takes ``M`` seeds per observation in the batch
    and evaluates the kernel mean once at the ``M`` moved nodes per
    observation; the reference is closed form, so the density and score
    channel is zero. The count is the expectation under the trainer's own
    uniform ``M`` distribution (``mean(m_list)``).

    ``n_steps * batch_obs * mean(m_list)`` counts the training loop and
    nothing else, so it is a lower bound on what the run spent. The realized
    ``M`` sequence is not replayed: the trainer's generator is shared between
    the ``M`` choice, the observation permutation and every per-observation
    seed, so a replay that fell out of phase would produce a wrong exact count
    where the expectation is right on average. The envelope
    ``[envelope_lo, envelope_hi]`` bounds every realization instead.

    Args:
        res: The run's parsed ``results.json``.
        n_obs: Observations to amortize the fit over.

    Returns:
        The ledger, in points and seconds.
    """
    m_list = [int(m) for m in res["m_list"]]
    mean_m = sum(m_list) / len(m_list)
    steps_x_batch = float(res["n_steps"]) * float(res["batch_obs"])
    n_pts = steps_x_batch * mean_m
    lo = steps_x_batch * min(m_list)
    hi = steps_x_batch * max(m_list)
    train_secs = float(res["train_secs"])
    return {
        "role": (
            "A1/M8: the fit is a PER-FAMILY spend, amortized over the "
            "observation stream; no new measurement"
        ),
        "scope": "TRAIN-ONLY (the run's eval passes and every upstream build "
                 "are outside this ledger, so it is a LOWER bound)",
        "train_secs": train_secs,
        "n_steps": int(res["n_steps"]),
        "batch_obs": int(res["batch_obs"]),
        "mean_M": mean_m,
        "m_list": m_list,
        "n_target_train": 0,
        "n_mu_train": n_pts,
        "n_draw_train": n_pts,
        "envelope_lo": lo,
        "envelope_hi": hi,
        "envelope_applies_to": ["n_mu_train", "n_draw_train"],
        "realized_m_sequence": (
            "NOT REPLAYED: base_seed survives in the run's projorg name, but "
            "the trainer's generator is shared with the observation "
            "permutation and every per-observation seed draw, so a replay "
            "that fell out of phase would produce a wrong EXACT count. The "
            "envelope bounds every realization; the point value is the "
            "expectation under the trainer's own uniform M draw"
        ),
        "note": (
            "expectation under the trainer's uniform M draw; per step: M "
            "seeds drawn and one kernel-mean evaluation of M moved nodes, "
            "per observation in the batch; closed-form reference, so the "
            "density/score ledger is zero"
        ),
        "amortized_per_observation": {
            "n_obs": int(n_obs),
            "train_secs": train_secs / n_obs,
            "n_mu": n_pts / n_obs,
            "n_draw": n_pts / n_obs,
            "n_mu_lo": lo / n_obs,
            "n_mu_hi": hi / n_obs,
        },
        "provenance": (
            "results.json of the resolved dq_wedge_conditional run "
            "(train_secs, n_steps, batch_obs, m_list)"
        ),
    }


def _resolve_wedge(spec: dict) -> dict | None:
    """Resolve the wedge run by projorg config identity.

    :func:`_ckpt.checkpoint_path` rebuilds the experiment name from
    ``configs/dq_wedge_conditional.json`` plus the ``bandwidth=median_mmd``
    train-time override, with the script's own extended ignore list
    (``recon_M`` / ``eval_repeats``, the render-only knobs projorg would
    otherwise hash). Several directories share this run's prefix and differ
    only in the trained bandwidth, so any glob-sort rule would silently
    return a stale network.

    The network is rebuilt from the same config the identity was derived from,
    since the architecture fields are not in ``results.json``, then loaded
    from the resolved ``checkpoint.pth``. Family staging is deferred to
    :func:`_stage_wedge` so that resolution stays cheap.
    """
    ck = checkpoint_path(
        spec["config"], ignore_extra=spec["ignore_extra"], **spec["overrides"]
    )
    cdir = os.path.dirname(ck)
    rj = os.path.join(cdir, "results.json")
    if not (os.path.isfile(ck) and os.path.isfile(rj)):
        return None
    with open(rj) as fh:
        res = json.load(fh)
    cfg = read_config(os.path.join(configsdir(), spec["config"]))
    cfg.update(spec["overrides"])
    cfg["m_list"] = [int(v) for v in str(cfg["m_list"]).split(",")]
    net = ConditionalQuadratureAmortizer(
        d=int(cfg["r"]),
        hidden=int(cfg["hidden"]),
        n_blocks=int(cfg["n_blocks"]),
        n_heads=int(cfg["n_heads"]),
        cond_hidden=int(cfg["cond_hidden"]),
        delta_scale=float(cfg["delta_scale"]),
        spatial_size=int(cfg["n_img"]),
        cond_token_dim=int(cfg["cond_token_dim"]),
        cond_hidden_ch=int(cfg["cond_hidden_ch"]),
        cond_n_down=int(cfg["cond_n_down"]),
        cond_in_channels=int(cfg["cond_in_channels"]),
    ).to(DTYPE)
    state = torch.load(ck, map_location="cpu", weights_only=False)
    net.load_state_dict(state["state_dict"])
    net.eval()
    return {
        "dir": cdir, "res": res, "net": net, "cfg": cfg,
        "fit_ledger": wedge_fit_ledger(res),
    }


def _stage_wedge(run: dict) -> dict:
    """Stage the wedge run's held-out per-observation references (idempotent).

    Reuses ``dq_wedge_conditional``'s own machinery (the shared prior, the
    pooled informed basis with its quadrature-convergence assertion, and the
    streaming family build) to reconstruct that run's held-out eval family.
    ``sigma``, ``cond_mean`` and ``cond_std`` are taken from the run's
    ``results.json`` rather than recomputed, so every reference and every
    conditioner is on the ruler the network was trained under. The import is
    deferred because it pulls the subspace and dataset stacks.

    It inherits that script's whole-run RSS assertion: ``_build_family``
    asserts ``peak_rss_mb() < 2048``, read off ``RUSAGE_SELF``, i.e. the host
    process's lifetime peak. ``__main__`` therefore stages wedge first, before
    any 16384^2 thinning Gram exists in the process; the pre-check below
    covers callers that stage late, so an abort names its own cause.
    """
    if "recs" in run:
        return run
    import dq_wedge_conditional as wedge  # noqa: E402  (deferred, heavy)

    rss, bound = wedge.peak_rss_mb(), wedge._RSS_FAMILY_BOUND_MB
    if rss >= 0.8 * bound:
        print(
            f"[wedge] WARNING: this process's peak RSS is already "
            f"{rss:.0f} MB against dq_wedge_conditional's {bound:.0f} MB "
            "whole-run assertion, which the family build is about to check. "
            "That budget is the wedge SCRIPT's, not this harness's, and the "
            "earlier cases in this run are what put us here -- if the stage "
            "aborts, run the wedge case in its own process rather than "
            "raising the bound."
        )
    args = argparse.Namespace(**run["cfg"])
    exp = wedge.DQWedgeConditional(args)
    exp._build_prior()
    exp._build_basis()
    recs = exp._build_family(int(run["cfg"]["n_eval_obs"]), 100_000, "eval")
    exp.sigma = float(run["res"]["sigma"])
    exp.cond_mean = float(run["res"]["cond_mean"])
    exp.cond_std = float(run["res"]["cond_std"])
    run["exp"], run["recs"] = exp, recs

    def cond_fn(obs_index: int) -> torch.Tensor:
        """Standardized conditioner ``(1, 1, S, S)`` for one observation."""
        rec = recs[int(obs_index) % len(recs)]
        return (rec["cond"].unsqueeze(0) - exp.cond_mean) / exp.cond_std

    run["cond_fn"] = cond_fn
    return run


def resolve_case(case: str) -> dict | None:
    """Resolve the trained amortizer behind a case, or ``None`` if absent.

    Resolution is by projorg config identity, either through
    :func:`dq_safeguard_sweep.find_run`, which content-verifies against
    ``results.json`` and raises on ambiguity, or through
    :func:`_ckpt.checkpoint_path`, which rebuilds the experiment name from the
    config file. Neither rule ever takes a sorted-first or newest match;
    several runs of one target share a directory prefix and differ only in the
    trained bandwidth, so either shortcut would silently return a stale
    network.

    Args:
        case: A key of :data:`CASE_RUNS`, or any other case name (which has no
            trained network by construction).

    Returns:
        ``{"dir", "res", "net", ...}`` for a resolved run, or ``None`` when
        the case names no run or the run is not on disk.

    Raises:
        RuntimeError: If a ``find_run`` identity matches more than one run.
    """
    spec = CASE_RUNS.get(case)
    if spec is None:
        return None
    if spec["kind"] == "find_run":
        got = find_run(*spec["spec"])
        if got is None:
            return None
        cdir, res = got
        return {
            "dir": cdir, "res": res,
            "net": load_net(load_state(cdir), res["d"]),
        }
    return _resolve_wedge(spec)


def build_case(case: str, run: dict | None = None):
    """Return ``(target_or_factory, sigma_eval, label, conditional)``.

    ``conditional`` says whether the reference depends on an observation: a
    fixed reference cannot distinguish a per-target solve from an amortized
    one.

    Where a trained network exists, the target and the bandwidth are rebuilt
    from that run's own ``results.json`` rather than chosen here, so every arm
    is scored on the ruler the network was trained under. A case with no
    trained run keeps its own declared bandwidth and is skipped by
    :func:`run_case`.

    Args:
        case: ``"gmm2"`` (closed form, exact kernel mean, where every peer is
            exactly implementable), ``"wedge"`` (a family of exact latent
            Gaussians whose shape moves with the acquisition, staged through
            ``dq_wedge_conditional``'s own machinery), or ``"nw_cond"`` (the
            Nadaraya-Watson conditional, which has no checkpoint and is
            skipped).
        run: The record :func:`resolve_case` returned, or ``None``.

    Returns:
        ``(target_or_factory, sigma_eval, human_label, conditional)``.

    Raises:
        ValueError: On an unknown case, or on a case that names a trained run
            without one being supplied.
    """
    if case == "gmm2":
        if run is None:
            raise ValueError(
                "case 'gmm2' is scored on its trained run's own bandwidth; "
                "pass the record resolve_case('gmm2') returns"
            )
        sigma_eval = float(run["res"]["sigma"])
        return target_for(run["res"], whiten=False), sigma_eval, \
            "Gaussian-mixture posterior", False
    if case == "wedge":
        if run is None:
            raise ValueError(
                "case 'wedge' is scored on its trained run's own bandwidth; "
                "pass the record resolve_case('wedge') returns"
            )
        _stage_wedge(run)
        sigma_eval = float(run["res"]["sigma"])
        exp, recs = run["exp"], run["recs"]

        def factory(obs_index: int):
            """One held-out observation's EXACT latent Gaussian reference."""
            return exp._obs_target(recs[int(obs_index) % len(recs)])

        return factory, sigma_eval, "wedge-conditional tomography family", True
    if case == "nw_cond":
        sigma_eval = 1.0

        def factory(obs_index: int):
            """One observation's Nadaraya-Watson conditional reference.

            The atoms are a fixed bank; the observation moves the weights, so
            each index is a genuinely different reference and every peer must
            be rerun from scratch at each one.
            """
            gen = torch.Generator().manual_seed(BASE_SEED + 31 * obs_index)
            base = gaussian_target(sigma_eval, d=2, dtype=DTYPE)
            atoms = base.sampler(2048, gen)
            shift = torch.tensor(
                [0.35 * (obs_index % 7) - 1.0, 0.35 * (obs_index % 5) - 0.7],
                dtype=DTYPE,
            )
            logits = -((atoms - shift) ** 2).sum(-1) / 2.0
            alpha = torch.softmax(logits, dim=0)
            return nw_conditional_target(atoms, alpha, sigma_eval)

        return factory, sigma_eval, "Nadaraya--Watson conditional", True
    raise ValueError(f"unknown case {case!r}")


# --------------------------------------------------------------------------
# Arms.
# --------------------------------------------------------------------------
def _score(z, w, target, sigma_eval: float) -> float:
    return max(
        float(mmd_sq(z, w, target.mu_fn(z), target.c_rho, sigma_eval)), 0.0
    )


def _reweight(z, target, sigma_eval: float):
    """The constrained weights on another construction's nodes."""
    mu = target.mu_fn(z)
    return constrained_weights(z, mu, sigma_eval, JITTER)


def bank_mu_fn(bank: torch.Tensor, sigma: float):
    """``mu-hat_j = (1/n) sum_l k(x_l, z_j)`` over a bank: ``mu`` alone.

    Bit-identical to ``sampled_bank_target(bank, sigma).mu_fn`` in its
    ``n_mu = n`` case, and it skips that builder's ``O(n^2)`` V-statistic
    ``c_rho``, which the samples-only mirror never reads: its solve and its
    selection need only ``mu``, and its reported score is taken against the
    exact target's ``c_rho``. At ``n = M^2 = 4096`` the skipped build is
    16.7 M kernel evaluations per repeat.

    Args:
        bank: ``(n, d)`` reference samples.
        sigma: SE bandwidth.

    Returns:
        A callable ``z -> (M,)``, differentiable in ``z``.
    """
    inv_2sig2 = 1.0 / (2.0 * float(sigma) ** 2)

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        sq = torch.cdist(z.to(bank.dtype), bank) ** 2
        return torch.exp(-sq * inv_2sig2).mean(dim=-1)

    return mu_fn


def raw_stationary_weights(y, embeddings, infl: float, sigma: float):
    """Algorithm 2's own final line: ``w = K^{-1} v0``, not unit-sum.

    ``msip_weights`` returns ``solve(K, v0) / sum``, a normalization that
    deviates from arXiv:2502.10600's Algorithm 2, whose final line emits the
    raw solve. Both weightings are scored.

    The solve is done in the log-stable shifted variable and rescaled by
    ``exp(max log v0)``, which is safe here because ``v0`` is a KDE of a
    normalized SE kernel and so never exceeds 1 (the shift is <= 0).

    Args:
        y: Nodes ``(m, d)``.
        embeddings: The ``(log_v0, sigma^2 grad log v0)`` hook the map read.
        infl: The same diagonal inflation the map's solve used.
        sigma: The quantizer's own bandwidth (NOT the eval ruler).

    Returns:
        The raw weight vector ``(m,)``, or ``None`` when the nodes or the
        rescaling are not representable (reported as absent rather than as a
        number). The nodes are checked FIRST because the embeddings raise on
        a non-finite query batch by contract, and a diagnostic column may not
        be what takes a sweep down.
    """
    if not torch.isfinite(y).all():
        return None
    log_v0, _ = embeddings(y)
    m = int(y.shape[0])
    K = torch.exp(-torch.cdist(y, y) ** 2 / (2.0 * float(sigma) ** 2))
    K = K + float(infl) * torch.eye(m, dtype=y.dtype)
    shift = float(log_v0.max())
    if not math.isfinite(shift) or shift > 700.0:
        return None
    w = torch.linalg.solve(K, torch.exp(log_v0 - shift))
    return w * math.exp(shift)


def _net_forward(net, z0: torch.Tensor, cond: torch.Tensor | None):
    """One forward pass, conditioned when the case is (``(M, d) -> (M, d)``)."""
    with torch.no_grad():
        zin = z0.unsqueeze(0)
        out = net(zin) if cond is None else net(zin, cond)
    return out.squeeze(0).to(z0.dtype)


def our_arms(z0, net, target, sigma_eval: float, cond=None) -> dict:
    """The i.i.d. floor, the reweight lever, and one forward pass.

    ``ours`` is the safeguarded output: one call of the trained amortizer, the
    constrained solve at the displaced nodes, and then the selection over the
    three-arm menu. The moved arm before that selection is kept beside it as
    ``ours_move_raw``.

    The kernel-mean ledger is six evaluations of ``M`` points: the seed mean,
    the displaced-node mean, the selection's memoized pass over its two
    distinct node sets plus its determinism probe, and the rescoring of the
    selected quadrature. That is what ``hook.n_points`` returns here.

    Args:
        z0: i.i.d. seed nodes ``(M, d)``.
        net: The trained amortizer, in eval mode.
        target: The reference (supplies ``mu_fn`` / ``c_rho``).
        sigma_eval: The shared eval bandwidth.
        cond: Optional standardized conditioner ``(1, 1, S, S)`` for the
            conditional amortizer; ``None`` on an unconditional case.

    Returns:
        ``{floor, reweight, ours, ours_move_raw, selected, active, n_mu}``.
    """
    hook = CountingHook(target.mu_fn, "mu")
    M = z0.shape[0]
    mu0 = hook(z0)
    w_eq = torch.full((M,), 1.0 / M, dtype=z0.dtype)
    floor = max(float(mmd_sq(z0, w_eq, mu0, target.c_rho, sigma_eval)), 0.0)
    w_c = constrained_weights(z0, mu0, sigma_eval, JITTER)
    rw = max(float(mmd_sq(z0, w_c, mu0, target.c_rho, sigma_eval)), 0.0)
    z_net = _net_forward(net, z0, cond)
    mu_net = hook(z_net)
    w_net = constrained_weights(z_net, mu_net, sigma_eval, JITTER)
    ours = max(
        float(mmd_sq(z_net, w_net, mu_net, target.c_rho, sigma_eval)), 0.0
    )
    sel = select_emission(
        [(z0, w_eq), (z0, w_c), (z_net, w_net)],
        hook, sigma_eval, names=("seeds", "reweight", "move"),
    )
    z_s, w_s = sel["z"], sel["w"]
    mu_s = hook(z_s)
    shipped = max(
        float(mmd_sq(z_s, w_s, mu_s, target.c_rho, sigma_eval)), 0.0
    )
    return {"floor": floor, "reweight": rw, "ours": shipped,
            "ours_move_raw": ours, "selected": sel["name"],
            "active": bool(sel["active"]),
            "n_mu": hook.n_points}


def our_arms_samples_only(
    z0, net, target, sigma_eval: float, bank, cond=None
) -> dict:
    """The samples-only mirror of the amortized arm.

    The same forward pass on the same seeds, but every solve and the
    safeguarded selection read ``mu-hat``, the empirical kernel mean of the
    same ``n = M^2`` samples thinning consumes, instead of the exact oracle.
    The reported MMD^2 is still scored against the exact reference, so the
    mirror matches the information in the solve and not the ruler.

    ``mu-hat`` is ``sampled_bank_target``'s estimand without its ``c_rho``.
    That builder computes the V-statistic ``c_rho`` over the full bank at
    build time, ``O(n^2) = O(M^4)`` kernel evaluations, and this arm never
    reads it: the solve and the selection need only ``mu``, and the reported
    score uses the exact target's ``c_rho``. The estimand is therefore built
    here as the same closure (``mu_j = (1/n) sum_l k(x_l, z_j)``, the
    ``n_mu = n`` case of ``mu_fn``) and the quadratic build is skipped.

    Args:
        z0: i.i.d. seed nodes ``(M, d)``, the same seeds ``ours`` consumes.
        net: The trained amortizer, in eval mode.
        target: The exact reference, used only for the reported score.
        sigma_eval: The shared eval bandwidth.
        bank: The ``(n, d)`` bank, bit-identical to thinning's input sample
            (regenerated from thinning's own seed stream).
        cond: Optional conditioner, as in :func:`our_arms`.

    Returns:
        ``{ours_samples_only, selected, n_draw}``: the exact-reference score
        of the samples-only quadrature, the arm the selection picked, and the
        reference samples consumed (``M`` seeds plus the bank).
    """
    mu_hat = bank_mu_fn(bank, sigma_eval)
    M = z0.shape[0]
    w_eq = torch.full((M,), 1.0 / M, dtype=z0.dtype)
    mu0_hat = mu_hat(z0)
    w_c = constrained_weights(z0, mu0_hat, sigma_eval, JITTER)
    z_net = _net_forward(net, z0, cond)
    mu_net_hat = mu_hat(z_net)
    w_net = constrained_weights(z_net, mu_net_hat, sigma_eval, JITTER)
    sel = select_emission(
        [(z0, w_eq), (z0, w_c), (z_net, w_net)],
        mu_hat, sigma_eval, names=("seeds", "reweight", "move"),
    )
    val = _score(sel["z"], sel["w"], target, sigma_eval)
    return {
        "ours_samples_only": val, "selected": sel["name"],
        "n_draw": int(M + bank.shape[0]),
    }


def oracle_arm(z0, target, sigma_eval: float) -> dict:
    """The per-instance ``K``-step descent the map amortizes.

    Its ledger is kept separate: at ``ORACLE_ITERS`` steps it reads the kernel
    mean ``2 K + 1`` times per observation, where one forward pass reads it
    twice.

    Returns:
        ``{oracle_move, n_mu}``.
    """
    hook = CountingHook(target.mu_fn, "mu")
    _, mv = move_descend(
        z0, hook, target.c_rho, sigma_eval, ORACLE_LR, ORACLE_ITERS, JITTER
    )
    return {"oracle_move": mv, "n_mu": hook.n_points}


def herding_arm(target, M: int, sigma_eval: float, gen, mode: str) -> dict:
    """Kernel herding on a pool drawn from the reference."""
    hook = CountingHook(target.mu_fn, "mu")
    z_pool = target.sampler(HERDING_POOL, gen)
    out = kernel_herding(z_pool, hook(z_pool), sigma_eval, M, mode=mode)
    val = _score(out["z"], out["w"], target, sigma_eval)
    val_rw = _score(
        out["z"], _reweight(out["z"], target, sigma_eval), target, sigma_eval
    )
    return {
        "z": out["z"], "w": out["w"],
        "mmd_sq": val, "mmd_sq_rw": val_rw, "n_draw": HERDING_POOL,
        "n_mu": hook.n_points, "n_distinct": out["n_distinct"], "mode": mode,
    }


def sbq_arm(target, M: int, sigma_eval: float, gen, swaps: int) -> dict:
    """Weight-aware greedy selection on a pool drawn from the reference.

    Every candidate is scored after the unit-sum weights are re-solved at the
    enlarged node set, and the rule it emits carries those same weights. It
    reads exactly the hook herding reads, the kernel mean at the pool atoms in
    one evaluation of ``P`` points, and consumes exactly the same ``P``
    reference samples, so the two differ only in whether the search knows
    about the weights.

    ``mmd_sq_eq`` is this arm's own nodes under equal weights. It mirrors the
    peers' ``_rw`` variant: a weight-aware search chooses configurations that
    are only good once the signed weights are applied, and the gap between the
    two columns is how much of the arm's margin is the weighting rather than
    the placement.

    ``value_gap`` is the agreement between the incremental criterion the
    selection ran on and an independent dense re-solve at the selected nodes.
    It pins the rank-one recursion at run time.
    """
    hook = CountingHook(target.mu_fn, "mu")
    z_pool = target.sampler(SBQ_POOL, gen)
    mu_pool = hook(z_pool)
    K_pool = gram(z_pool, sigma_eval)
    out = sequential_bayesian_quadrature(
        z_pool, mu_pool, sigma_eval, M, c_rho=target.c_rho, K_pool=K_pool,
        swap_passes=swaps, jitter=JITTER,
    )
    val = _score(out["z"], out["w"], target, sigma_eval)
    m = int(out["z"].shape[0])
    w_eq = torch.full((m,), 1.0 / m, dtype=out["z"].dtype)
    val_eq = _score(out["z"], w_eq, target, sigma_eval)
    sel = out["indices"]
    eye = torch.eye(m, dtype=K_pool.dtype)
    direct = constrained_value_direct(
        K_pool[sel][:, sel] + JITTER * eye, mu_pool[sel], target.c_rho
    )
    return {
        # z / w are exported so a caller can re-score this arm on a bank
        # other than the one it was fitted against.
        "z": out["z"], "w": out["w"],
        "mmd_sq": val, "mmd_sq_eq": val_eq, "n_draw": SBQ_POOL,
        "n_mu": hook.n_points, "n_selected": m, "swap_passes": swaps,
        "n_swaps": out["n_swaps"], "rank_limited": out["rank_limited"],
        "unit_sum_err": abs(out["unit_sum"] - 1.0),
        "value_gap": abs(out["value"] - direct),
    }


def thinning_arm(
    target, M: int, sigma_eval: float, gen, swaps: int,
    input_floor: bool = False,
) -> dict:
    """Kernel thinning from ``M^2`` samples down to ``M`` equal-weight nodes.

    With ``input_floor=True`` (the eval loop only; tuning skips the extra
    Gram) the arm also scores its own input sample: the ``n = M^2`` samples it
    consumed, under equal weights, against the exact reference. That is the
    accuracy already present in the sample it thins, and the figure draws it
    as a band beside the arm.

    ``wall_secs`` is measured here, around the construction only: the
    input-floor score is a dense ``(M^2, M^2)`` kernel mean the peer's own
    construction never computes, and timing it would charge the peer for work
    this harness asked for. The caller accumulates this number rather than
    wrapping the call.
    """
    n_in = THINNING_INPUT[M]
    z = target.sampler(n_in, gen)
    t0 = time.time()
    out = kernel_thinning(z, sigma_eval, M, gen, swap_passes=swaps)
    val = _score(out["z"], out["w"], target, sigma_eval)
    val_rw = _score(
        out["z"], _reweight(out["z"], target, sigma_eval), target, sigma_eval
    )
    secs = time.time() - t0
    rec = {
        "z": out["z"], "w": out["w"],
        "mmd_sq": val, "mmd_sq_rw": val_rw, "n_draw": n_in,
        "n_mu": 0, "n_rounds": out["n_rounds"], "swap_passes": swaps,
        "wall_secs": secs,
    }
    if input_floor:
        w_in = torch.full((n_in,), 1.0 / n_in, dtype=z.dtype)
        rec["input_floor"] = _score(z, w_in, target, sigma_eval)
    return rec


# --------------------------------------------------------------------------
# The thinning peer at M = 256 / 512 via its authors' published tooling
# (goodpoints), which never materializes the dense (n, n) Gram the in-repo
# backend builds.
# --------------------------------------------------------------------------
def _se_kernel_np(sigma: float):
    """The SE kernel in goodpoints' calling convention.

    goodpoints passes ``kernel(y, X)`` with 2-D arrays and expects numpy row
    broadcasting: for ``y`` of shape ``(1, d)`` the return is the ``(t,)``
    row of evaluations against ``X``'s ``t`` rows, and for ``y`` of shape
    ``(n, d)`` against ``X`` of shape ``(n, d)`` it is the ``(n,)`` diagonal
    ``k(y_i, X_i)`` (its ``refine_X`` calls ``kernel(X, X)`` for exactly
    that). Both fall out of one broadcasted expression.

    Args:
        sigma: SE bandwidth (the case's own eval ruler).

    Returns:
        ``k(y, X) -> np.ndarray`` in the convention above, float64.
    """
    inv = 1.0 / (2.0 * float(sigma) ** 2)

    def k(y: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.exp(-np.sum((y - X) ** 2, axis=-1) * inv)

    return k


def _pick_floor_device() -> str:
    """Device for the untimed row-sum diagnostic, re-checked per call.

    CPU unless all of: ``DQ_THIN_FLOOR_DEVICE=auto`` is exported, the caller
    exported a non-empty ``CUDA_VISIBLE_DEVICES`` (this module's own default
    leaves it empty), torch sees a device, and the GPU is idle, meaning at
    least 4 GiB free and utilization at most 20 percent, read via NVML
    without allocating. The statistic sits outside every timed window, so no
    wall-clock number depends on the device. The record stores the device
    per repeat.
    """
    if os.environ.get("DQ_THIN_FLOOR_DEVICE", "cpu") != "auto":
        return "cpu"
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cpu"
    try:
        import pynvml  # noqa: E402  (ships with torch's cuda extras)

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            free_b = pynvml.nvmlDeviceGetMemoryInfo(handle).free
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        finally:
            pynvml.nvmlShutdown()
        if (free_b >= 4 * 1024**3 and util <= 20
                and torch.cuda.is_available()):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _gram_row_sums(
    z: torch.Tensor, sigma: float, block: int = FLOOR_BLOCK,
    device: str = "cpu",
) -> torch.Tensor:
    """Row sums of the SE Gram over ``z`` in ``O(n * block)`` memory.

    Symmetric-block accumulation: each off-diagonal tile is computed once
    and credited to both its row band and its column band, so the pass costs
    ``n^2 / 2`` kernel evaluations and never holds more than one
    ``(block, block)`` tile. Two consumers, one pass:

      * ``meanK = row_sums / n`` -- the row-mean statistic goodpoints'
        KT-SWAP stage reads (handed in so the tooling skips its own
        Python-loop recomputation of the same numbers);
      * ``row_sums.sum() / n^2`` -- the V-statistic term of the input-floor
        diagnostic, the same ``w^T K w`` at equal weights that
        :func:`_score` computes densely at the small budgets.

    Args:
        z: Input points ``(n, d)``, float64.
        sigma: SE bandwidth.
        block: Tile edge.
        device: ``"cpu"`` or ``"cuda"`` (the caller picks politely).

    Returns:
        ``(n,)`` float64 row sums, always on CPU.
    """
    n = int(z.shape[0])
    inv = 1.0 / (2.0 * float(sigma) ** 2)
    zd = z.to(device)
    sums = torch.zeros(n, dtype=z.dtype, device=device)
    n_blocks = (n + block - 1) // block
    for i in range(n_blocks):
        i0, i1 = i * block, min((i + 1) * block, n)
        zi = zd[i0:i1]
        for j in range(i, n_blocks):
            j0, j1 = j * block, min((j + 1) * block, n)
            tile = torch.exp(-(torch.cdist(zi, zd[j0:j1]) ** 2) * inv)
            sums[i0:i1] += tile.sum(dim=1)
            if j > i:
                sums[j0:j1] += tile.sum(dim=0)
    out = sums.cpu()
    if device == "cuda":
        del sums, zd
        torch.cuda.empty_cache()  # leave the shared card as found
    return out


def thinning_arm_published(
    target, M: int, sigma_eval: float, gen, backend: str,
    cpp_diag: bool = False, skip_floor: bool = False,
) -> dict:
    """Kernel thinning at ``M in M_EXT`` via the peer's published tooling.

    The same schedule, stream and scoring as :func:`thinning_arm`: the
    ``n = M^2`` input sample is taken from the identical generator stream, so
    it stays bit-identical to the mirror arm's bank, the coreset is scored
    equal-weight against the exact reference, the ``_rw`` variant gets the
    constrained weights, and the input sample's own floor is recorded. Only
    the construction is swapped for goodpoints:

      * ``backend="goodpoints_kt"``: ``kt.thin`` (KT-SPLIT + KT-SWAP), target
        kernel for both stages, ``delta = KT_DELTA``, ``store_K=False`` so
        the dense Gram never exists. The row-mean statistic its swap stage
        reads is precomputed by :func:`_gram_row_sums` outside the timed
        window and handed in, since the tooling would otherwise spend minutes
        re-deriving the same numbers in a Python loop, and always on CPU, so
        the emitted coreset never depends on what the GPU was doing.
      * ``backend="goodpoints_compresspp"``: ``compress.compresspp_kt`` with
        ``g = COMPRESSPP_G``, Compress++, the peer's own near-linear
        accelerator, kernel built in (``b"gaussian"`` with
        ``k_params = [2 sigma^2]`` is the SE kernel). Used at M = 512, where
        plain ``kt.thin`` measures about 80 minutes per repeat.

    ``wall_secs`` times the published tooling's construction call alone. The
    goodpoints seed is taken from the arm's own generator after the input
    sample, so cells are exactly replayable.

    With ``cpp_diag=True`` (the M = 256 cells) the accelerator also runs on
    the identical input sample and seed and its score is returned as
    ``mmd_sq_compresspp``, beside the plotted plain-KT cell.

    ``skip_floor=True`` omits the input-floor diagnostic only, and exists
    for the quiet wall-clock pass (:func:`run_wall_pass`), which re-times a
    construction whose scores are already recorded. It cannot change what is
    timed, since the floor sits outside the timed window by construction. The
    kt backend still computes the row sums even under this flag, because that
    statistic is handed into ``kt.thin``: dropping it would push the tooling's
    own recomputation inside the timed window and measure a different thing.

    Returns:
        ``{mmd_sq, mmd_sq_rw, input_floor, n_draw, n_mu, n_rounds,
        wall_secs, backend, floor_device[, mmd_sq_compresspp,
        wall_secs_compresspp]}``.
    """
    n_in = THINNING_INPUT[M]
    z = target.sampler(n_in, gen)
    gp_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
    X = np.ascontiguousarray(z.numpy())
    n_rounds = int(round(math.log2(n_in / M)))
    assert 2**n_rounds * M == n_in, (M, n_in)

    def _cpp_indices():
        from goodpoints import compress as gp_compress  # noqa: E402

        t0 = time.time()
        idx = gp_compress.compresspp_kt(
            X, b"gaussian", k_params=np.array([2.0 * float(sigma_eval) ** 2]),
            g=COMPRESSPP_G, num_bins=COMPRESSPP_NUM_BINS, delta=KT_DELTA,
            seed=gp_seed,
        )
        idx = np.ascontiguousarray(np.asarray(idx, dtype=np.int64))
        return idx, time.time() - t0

    rec: dict = {"backend": backend, "n_draw": n_in, "n_mu": 0,
                 "n_rounds": n_rounds}
    if backend == "goodpoints_kt":
        from goodpoints import kt as gp_kt  # noqa: E402

        # CPU always: this statistic feeds the peer's swap stage, and the
        # emitted coreset must not depend on the floor helper's device.
        row_sums = _gram_row_sums(z, sigma_eval, device="cpu")
        rec["floor_device"] = "cpu"
        k_np = _se_kernel_np(sigma_eval)
        t0 = time.time()
        idx_np = gp_kt.thin(
            X, n_rounds, k_np, k_np, delta=KT_DELTA, seed=gp_seed,
            store_K=False, meanK=(row_sums / n_in).numpy(),
        )
        rec["wall_secs"] = time.time() - t0
        idx_np = np.ascontiguousarray(np.asarray(idx_np, dtype=np.int64))
        if cpp_diag:
            cpp_idx, cpp_secs = _cpp_indices()
            assert cpp_idx.shape[0] == M, cpp_idx.shape
            zc = z[torch.from_numpy(cpp_idx)]
            w_eq = torch.full((M,), 1.0 / M, dtype=z.dtype)
            rec["mmd_sq_compresspp"] = _score(zc, w_eq, target, sigma_eval)
            rec["wall_secs_compresspp"] = cpp_secs
    elif backend == "goodpoints_compresspp":
        idx_np, rec["wall_secs"] = _cpp_indices()
        if skip_floor:
            row_sums, rec["floor_device"] = None, "skipped"
        else:
            device = _pick_floor_device()
            row_sums = _gram_row_sums(z, sigma_eval, device=device)
            rec["floor_device"] = device
    else:
        raise ValueError(f"unknown published-tooling backend {backend!r}")

    assert idx_np.shape[0] == M, (
        f"goodpoints returned {idx_np.shape[0]} indices for M={M}: the "
        f"n = M^2 schedule maps exactly onto its dyadic geometry, so this "
        "is a wiring bug, not a rounding question"
    )
    z_thin = z[torch.from_numpy(idx_np)]
    w_eq = torch.full((M,), 1.0 / M, dtype=z.dtype)
    rec["mmd_sq"] = _score(z_thin, w_eq, target, sigma_eval)
    rec["mmd_sq_rw"] = _score(
        z_thin, _reweight(z_thin, target, sigma_eval), target, sigma_eval
    )
    # The input sample's own floor: (1/n^2) sum K - (2/n) sum mu + c_rho,
    # the same equal-weight mmd_sq the dense path scores, assembled from the
    # blocked row sums.
    if not skip_floor:
        quad = float(row_sums.sum()) / (float(n_in) ** 2)
        mu_mean = float(target.mu_fn(z).mean())
        rec["input_floor"] = max(
            quad - 2.0 * mu_mean + float(target.c_rho), 0.0
        )
    return rec


def recombine_arm(target, M: int, sigma_eval: float, gen) -> dict:
    """Recombination onto ``<= M`` pool atoms with convex weights.

    The Nystrom sample is drawn from a SEPARATE generator stream, because the
    truncation must be independent of the pool it recombines for the imported
    bound to apply at all.
    """
    z_pool = target.sampler(RECOMBINE_POOL, gen)
    gen_nys = torch.Generator().manual_seed(
        int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
    )
    z_nys = target.sampler(RECOMBINE_NYSTROM, gen_nys)
    out = recombination_quadrature(z_pool, z_nys, sigma_eval, M)
    val = _score(out["z"], out["w"], target, sigma_eval)
    val_rw = _score(
        out["z"], _reweight(out["z"], target, sigma_eval), target, sigma_eval
    )
    return {
        "z": out["z"], "w": out["w"],
        "mmd_sq": val, "mmd_sq_rw": val_rw,
        "n_draw": RECOMBINE_POOL + RECOMBINE_NYSTROM,
        "n_mu": 0, "rank": out["rank"], "n_atoms": int(out["z"].shape[0]),
    }


def msip_dd_arm(
    target, M: int, sigma_eval: float, gen, bandwidth: float | None = None
) -> dict:
    """The data-driven quantizer of arXiv:2502.10600 on the shared pool.

    :class:`DataDrivenEmbeddings` over the same ``P``-atom pool the other
    quantizer peers consume (the pool is its bank), driven through
    :func:`run_msip` at that paper's published settings (damped step 0.5,
    1000 steps, ``bounds=None``, since its Algorithm 2 carries no box) from
    ``M`` initial nodes resampled off the pool, so no extra reference samples
    are taken. No density and no score is read, and its embedding evaluations
    are recorded on their own axis, ``n_emb_points``, pinned to the
    ``(T + 1) * M`` count.

    ``bandwidth=None`` runs the literal published 0.6; passing the case's own
    ``sigma_eval`` runs the scale-matched column.

    ``mmd_sq`` uses ``run_msip``'s unit-sum-normalized stationary weights;
    ``mmd_sq_raw`` uses the raw ``K^{-1} v0`` its Algorithm 2's final line
    emits. Both are reported.

    The trajectory-wide max absolute coordinate is read destructively via
    ``pop_query_stats`` immediately after the run, before any scoring and
    before the raw-weight solve re-enters the embeddings, and is reported
    against the clamp of arXiv:2605.14142 Table 3.
    """
    bw = MSIP_DD_BANDWIDTH if bandwidth is None else float(bandwidth)
    z_pool = target.sampler(MSIP_DD_POOL, gen)
    emb = DataDrivenEmbeddings(z_pool, bw, dtype=DTYPE)
    idx = torch.multinomial(
        torch.ones(MSIP_DD_POOL, dtype=DTYPE), M, replacement=False,
        generator=gen,
    )
    y, w, _ = run_msip(
        z_pool[idx], emb, MSIP_DD_LR, MSIP_DD_INFL, bw,
        MSIP_DD_ITERS, bounds=None, progress=False,
    )
    max_abs, n_queries = emb.pop_query_stats()
    expected = (MSIP_DD_ITERS + 1) * M
    assert n_queries == expected, (
        f"msip_dd embedding-evaluation count {n_queries} != the pinned "
        f"(T + 1) * M = {expected}; the R5 read no longer covers the "
        "trajectory"
    )
    val = _score(y, w, target, sigma_eval)
    val_rw = _score(y, _reweight(y, target, sigma_eval), target, sigma_eval)
    # The source's own weights, raw K^{-1} v0 rather than unit-sum. Scored
    # after the destructive pop above, so it cannot contaminate the
    # trajectory statistic.
    w_raw = raw_stationary_weights(y, emb, MSIP_DD_INFL, bw)
    val_raw = None if w_raw is None else _score(y, w_raw, target, sigma_eval)
    return {
        "mmd_sq": val, "mmd_sq_rw": val_rw, "mmd_sq_raw": val_raw,
        "raw_weight_sum": None if w_raw is None else float(w_raw.sum()),
        "bandwidth": bw, "n_draw": MSIP_DD_POOL,
        "n_mu": 0, "n_emb_points": n_queries,
        "max_abs_coordinate": float(max_abs),
    }


# --------------------------------------------------------------------------
# Peer-correctness controls. A peer that fails its own control is withheld.
# --------------------------------------------------------------------------
def control_herding_finite_rank(sigma_eval: float) -> dict:
    """Herding must beat the i.i.d. floor where the fast rate is guaranteed.

    Bach, Lacoste-Julien and Obozinski, Proposition 1: on a finite-dimensional
    RKHS with a compact domain and a full-support measure, the kernel mean lies
    in the relative interior of the marginal polytope and herding attains the
    fast rate. A finite feature map is built explicitly here so the condition
    holds by construction; if herding does not beat i.i.d. there, the
    implementation is wrong.

    The value that passes or fails is read off :func:`kernel_herding` itself
    (classical ``1/(t+1)`` schedule on the rank-``q`` linear Gram). The inline
    Frank-Wolfe recursion is retained only as a cross-check: it optimizes the
    identical objective on the identical schedule, so the two values must
    agree to round-off, and a disagreement withholds the arm exactly as a
    failed gap would.
    """
    gen = torch.Generator().manual_seed(BASE_SEED + 991)
    n, q = 256, 4
    z = torch.rand(n, q, generator=gen, dtype=DTYPE)
    K = z @ z.T  # a genuinely rank-q linear kernel
    mu = K.mean(dim=1)
    gaps, inline_gaps = [], []
    for M in (4, 8, 16):
        out = kernel_herding(z, mu, sigma_eval, M, mode="herding", K_pool=K)
        w_full = torch.zeros(n, dtype=DTYPE)
        for i in out["indices"]:
            w_full[int(i)] += 1.0 / M
        herd = float(w_full @ K @ w_full - 2.0 * w_full @ mu)
        # Inline Frank-Wolfe on the same objective: the cross-check only.
        w = torch.zeros(n, dtype=DTYPE)
        Kw = torch.zeros(n, dtype=DTYPE)
        for t in range(M):
            i = int(torch.argmin(Kw - mu))
            g = 1.0 / (t + 1.0)
            w = (1.0 - g) * w
            w[i] += g
            Kw = (1.0 - g) * Kw + g * K[:, i]
        herd_inline = float(w @ K @ w - 2.0 * w @ mu)
        inline_gaps.append(abs(herd - herd_inline))
        idx = torch.randperm(n, generator=gen)[:M]
        u = torch.zeros(n, dtype=DTYPE)
        u[idx] = 1.0 / M
        iid = float(u @ K @ u - 2.0 * u @ mu)
        gaps.append(herd - iid)
    return {
        "gaps": gaps,
        "inline_cross_check_gaps": inline_gaps,
        "passes": (
            all(g <= 1e-9 for g in gaps)
            and max(inline_gaps) <= 1e-9
        ),
    }


def control_thinning_beats_iid(sigma_eval: float, thin_fn=None) -> dict:
    """Thinning must beat an equal-sized i.i.d. sample on a compact target.

    The published claim in the paper's own units: thinning from ``n`` inputs
    delivers ``sqrt(n)`` points whose integration error is
    ``O(sqrt(log n / n))`` where an equal-sized i.i.d. sample suffers
    ``Omega(n^{-1/4})``. This control checks the direction of that contrast at
    a fixed budget; the rate itself is checked by
    :func:`control_thinning_rate_slope` against the registered band. Both read
    :func:`kernel_thinning` by default; ``thin_fn``
    (``(z, sigma, M, gen) -> index tensor``) points the same control at a
    goodpoints backend, which must clear it before any extension cell is
    written.
    """
    gen = torch.Generator().manual_seed(BASE_SEED + 992)
    n, M = 1024, 32
    z = torch.rand(n, 1, generator=gen, dtype=DTYPE)
    if thin_fn is None:
        idx_thin = kernel_thinning(z, sigma_eval, M, gen,
                                   swap_passes=1)["indices"]
    else:
        idx_thin = thin_fn(z, sigma_eval, M, gen)
    K = gram(z, sigma_eval)
    row_mean = K.mean(dim=1)

    def obj(idx):
        m = int(idx.numel())
        return float(
            K[idx][:, idx].sum() / (m * m) - 2.0 * row_mean[idx].sum() / m
        )

    thin = obj(idx_thin)
    iid_vals = []
    for r in range(64):
        g = torch.Generator().manual_seed(BASE_SEED + 992 + 7919 * r)
        iid_vals.append(obj(torch.randperm(n, generator=g)[:M]))
    iid = float(torch.tensor(iid_vals, dtype=DTYPE).median())
    return {"thinning": thin, "iid_median": iid, "passes": thin < iid}


def _uniform01_target(sigma: float):
    """Exact SE kernel mean and self-affinity of ``U[0, 1]``.

    The slope control's ruler. For ``k(x, y) = exp(-(x - y)^2 / (2 s^2))`` on
    the unit interval both moments are elementary:

        mu(x)  = int_0^1 k(x, y) dy
               = s sqrt(pi/2) [erf(x / (s sqrt2)) + erf((1 - x) / (s sqrt2))],
        c_rho  = int_0^1 int_0^1 k
               = 2 s sqrt(pi/2) erf(1 / (s sqrt2))
                 - 2 s^2 [1 - exp(-1 / (2 s^2))],

    the second from the triangular density ``1 - |t|`` of the difference.

    A finite sample bank's error is additive in ``MMD^2``, and an additive
    constant does not cancel from a log-log slope, so a closed-form ruler
    removes it outright instead of correcting for it.

    Args:
        sigma: SE bandwidth.

    Returns:
        ``(mu_fn, c_rho)`` -- ``mu_fn`` maps ``(n, 1)`` nodes to ``(n,)``.
    """
    s = float(sigma)
    root2 = math.sqrt(2.0)
    coef = s * math.sqrt(math.pi / 2.0)

    def mu_fn(z: torch.Tensor) -> torch.Tensor:
        x = z.squeeze(-1)
        return coef * (
            torch.erf(x / (s * root2)) + torch.erf((1.0 - x) / (s * root2))
        )

    c_rho = torch.tensor(
        2.0 * coef * math.erf(1.0 / (s * root2))
        - 2.0 * s * s * (1.0 - math.exp(-1.0 / (2.0 * s * s))),
        dtype=DTYPE,
    )
    return mu_fn, c_rho


def control_thinning_rate_slope(thin_fn=None) -> dict:
    """Thinning's log-log MMD^2-vs-n slope must land in the registered band.

    ``thin_fn`` (``(z, sigma, M, gen) -> index tensor``) points the control
    at a goodpoints backend instead of :func:`kernel_thinning`. The band is
    the same one: a constant-factor error concession is an additive log-log
    offset and moves no slope.

    The acceptance band is :data:`THINNING_SLOPE_BAND`, whose derivation sits
    at its definition: the guarantee ``MMD^2 = O(log n / n)`` fits ``-0.816``
    over this grid, the i.i.d. alternative sits at ``-0.5``, the upper edge is
    the midpoint separator ``-0.66``, and the lower edge is a one-sided guard.

    Grid: ``n`` in :data:`THINNING_SLOPE_GRID`, ``M = sqrt(n)`` per the
    corollary's own coupling, median over :data:`THINNING_SLOPE_REPEATS` fresh
    input samples, scored at bandwidth :data:`THINNING_SLOPE_SIGMA` against
    the closed-form ``U[0, 1]`` ruler of :func:`_uniform01_target`, the
    measure the inputs come from, so the levels the slope is fitted to carry
    no additive bank noise floor.
    """
    sig = THINNING_SLOPE_SIGMA
    mu_fn, c_rho = _uniform01_target(sig)

    def mmd_sq_exact(nodes):
        m = int(nodes.shape[0])
        w = torch.full((m,), 1.0 / m, dtype=DTYPE)
        return max(float(mmd_sq(nodes, w, mu_fn(nodes), c_rho, sig)), 0.0)

    medians = []
    for n in THINNING_SLOPE_GRID:
        M = int(round(n ** 0.5))
        vals = []
        for r in range(THINNING_SLOPE_REPEATS):
            g = torch.Generator().manual_seed(
                BASE_SEED + 996 + 1000 * n + r
            )
            z = torch.rand(n, 1, generator=g, dtype=DTYPE)
            if thin_fn is None:
                nodes = kernel_thinning(z, sig, M, g, swap_passes=1)["z"]
            else:
                nodes = z[thin_fn(z, sig, M, g)]
            vals.append(mmd_sq_exact(nodes))
        medians.append(float(torch.tensor(vals, dtype=DTYPE).median()))
    # Least-squares slope of log10(MMD^2) against log10(n), on levels that are
    # exact expectations of the estimand -- no additive constant to remove.
    xs = torch.log10(torch.tensor(
        [float(n) for n in THINNING_SLOPE_GRID], dtype=DTYPE
    ))
    ys = torch.log10(torch.tensor(medians, dtype=DTYPE).clamp_min(1e-300))
    xc = xs - xs.mean()
    slope = float((xc * (ys - ys.mean())).sum() / (xc * xc).sum())
    lo, hi = THINNING_SLOPE_BAND
    return {
        "grid_n": list(THINNING_SLOPE_GRID),
        "median_mmd_sq": medians,
        "slope": slope,
        "band": [lo, hi],
        "ruler": "closed-form U[0,1] (erf kernel mean, exact c_rho)",
        "c_rho": float(c_rho),
        "guarantee_slope_on_this_grid": -0.816,
        "iid_slope": -0.5,
        "lower_edge_role": (
            "one-sided bug guard: the published result is an error UPPER "
            "bound, so a steeper slope is the peer beating its own guarantee"
        ),
        "passes": lo <= slope <= hi,
    }


def control_recombination_is_exact_at_finite_rank(sigma_eval: float) -> dict:
    """Recombination must integrate the truncation EXACTLY once ``M > rank``.

    The property the imported bound rests on: with a rank-``q`` feature map and
    ``M >= q + 1`` atoms, the reduced measure reproduces every feature mean to
    machine precision, so the only error left is the kernel tail.
    """
    gen = torch.Generator().manual_seed(BASE_SEED + 993)
    target = gaussian_target(sigma_eval, d=2, dtype=DTYPE)
    z_pool = target.sampler(256, gen)
    gen_nys = torch.Generator().manual_seed(BASE_SEED + 993 + 90000)
    z_nys = target.sampler(64, gen_nys)
    M = 12
    out = recombination_quadrature(z_pool, z_nys, sigma_eval, M)
    from qfield.designed_quadrature import nystrom_features
    feats = nystrom_features(z_pool, z_nys, sigma_eval, q=M - 1)
    before = feats.mean(dim=0)
    after = out["w"] @ feats[out["indices"]]
    err = float((before - after).abs().max())
    return {"max_feature_mean_error": err, "passes": err < 1e-8}


def control_sbq_is_exact_at_finite_rank_and_dominates(sigma_eval: float) -> dict:
    """Three properties of :func:`sequential_bayesian_quadrature`.

    (i) Exact integration at finite rank. On an explicitly rank-``q`` feature
    map the reference kernel mean lies in the span of the features, and once
    the budget exceeds ``q`` a signed unit-sum rule reproduces it exactly, so
    the achieved ``MMD^2`` is zero to machine precision. The unit-sum
    constraint is what makes ``q + 1`` rather than ``q`` the interesting
    budget: at ``q`` nodes the unconstrained solve is already exact while the
    constrained one still pays ``(B - 1)^2 / C``, and the extra node is what
    buys the constraint back. Equal-weight herding, on the same pool and the
    same budget, cannot do this at all. Threshold:
    :data:`SBQ_FINITE_RANK_TOL`.

    (ii) Never worse than equal-weight herding on the same pool. The source
    paper's claim is that reweighting the herded set dominates the fixed
    schedule, so a construction that searches under the reweighting must not
    lose to one that does not. Measured over a grid of targets, budgets and
    seeds sharing the pool and the Gram, and reported as a fraction rather
    than asserted, since greedy search is not optimality and a cell can go the
    other way without the implementation being wrong. The pass condition is on
    the median gap (:data:`SBQ_MEDIAN_GAP_MAX`).

    (iii) The incremental recursion is the optimum it claims to be. The
    selection runs on a rank-one update; the reported criterion must equal an
    independent dense re-solve at the selected nodes, and the criterion along
    the greedy prefix must be non-increasing, since extending an optimal
    weight vector by a zero is feasible and preserves the unit sum.
    """
    gen = torch.Generator().manual_seed(BASE_SEED + 994)

    # (i) rank-q linear kernel, so dim H = q exactly and by construction.
    rank_rows = []
    for q in (3, 4, 6):
        feats = torch.rand(256, q, generator=gen, dtype=DTYPE)
        K = feats @ feats.T
        mu = K.mean(dim=1)
        c_rho = torch.tensor(float(K.mean()), dtype=DTYPE)
        for M in (q + 1, q + 3):
            out = sequential_bayesian_quadrature(
                feats, mu, 1.0, M, c_rho=c_rho, K_pool=K, jitter=JITTER
            )
            herd = kernel_herding(feats, mu, 1.0, M, K_pool=K)
            hw = torch.zeros(256, dtype=DTYPE)
            hw[herd["indices"]] += 1.0 / M
            rank_rows.append({
                "q": q, "M": M,
                "sbq": max(float(out["value"]), 0.0),
                "herding": float(
                    hw @ K @ hw - 2.0 * (hw @ mu) + c_rho
                ),
            })
    exact = max(r["sbq"] for r in rank_rows)
    herd_at_rank = min(r["herding"] for r in rank_rows)

    # (ii)/(iii) on the two closed-form references, same pool, same Gram.
    gaps, prefix_rise, value_gaps = [], [], []
    for target in (gaussian_target(sigma_eval, d=2, dtype=DTYPE),
                   gmm_target(sigma_eval, dtype=DTYPE)):
        for M in (4, 8, 16, 32):
            for r in range(8):
                g = torch.Generator().manual_seed(
                    BASE_SEED + 994 + 1000 * M + r
                )
                z_pool = target.sampler(256, g)
                mu_pool = target.mu_fn(z_pool)
                K = gram(z_pool, sigma_eval)
                out = sequential_bayesian_quadrature(
                    z_pool, mu_pool, sigma_eval, M, c_rho=target.c_rho,
                    K_pool=K, jitter=JITTER,
                )
                v_sbq = _score(out["z"], out["w"], target, sigma_eval)
                best_h = None
                for mode in HERDING_MODES:
                    h = kernel_herding(z_pool, mu_pool, sigma_eval, M,
                                       mode=mode, K_pool=K)
                    v = _score(h["z"], h["w"], target, sigma_eval)
                    best_h = v if best_h is None else min(best_h, v)
                gaps.append(v_sbq - best_h)
                pref = out["value_prefix"]
                prefix_rise.append(
                    max([pref[i + 1] - pref[i]
                         for i in range(len(pref) - 1)] + [0.0])
                )
                sel = out["indices"]
                eye = torch.eye(int(sel.numel()), dtype=DTYPE)
                value_gaps.append(abs(out["value"] - constrained_value_direct(
                    K[sel][:, sel] + JITTER * eye, mu_pool[sel], target.c_rho
                )))
    med_gap = float(torch.tensor(gaps, dtype=DTYPE).median())
    frac_no_worse = float(
        (torch.tensor(gaps, dtype=DTYPE) <= 1e-12).to(DTYPE).mean()
    )
    return {
        "finite_rank": rank_rows,
        "max_finite_rank_mmd_sq": exact,
        "min_herding_mmd_sq_at_finite_rank": herd_at_rank,
        "median_gap_vs_herding": med_gap,
        "frac_no_worse_than_herding": frac_no_worse,
        "max_prefix_increase": max(prefix_rise),
        "max_incremental_vs_direct_gap": max(value_gaps),
        "passes": (
            exact < SBQ_FINITE_RANK_TOL
            and herd_at_rank > 1e-5
            and med_gap <= SBQ_MEDIAN_GAP_MAX
            and max(prefix_rise) < 1e-10
            and max(value_gaps) < 1e-9
        ),
    }


def _msip_dd_control_leg(
    d: int, bandwidth: float | None, sigma_eval: float, seed: int, M: int = 16
) -> dict:
    """One (dimension, bandwidth) cell of the msip_dd correctness control.

    The source's own claim (arXiv:2502.10600), on one bank: the weighted
    quantization of an empirical measure integrates it better than ``M`` of
    its own atoms under equal weights. Measured end to end,
    :class:`DataDrivenEmbeddings` through :func:`run_msip` at the published
    step, iteration count and no-box settings, against the empirical-bank
    estimand of :func:`sampled_bank_target` at the same bandwidth the dynamics
    use, versus the median of 32 equal-weight ``M``-row subsamples of the same
    bank.

    ``bandwidth=None`` means scale-matched: the bank's own median pairwise
    distance (:func:`median_heuristic_sigma_sq`, the median heuristic itself
    and not the SVGD ``h^2``, which divides by ``2 log(M + 1)`` and lands a
    factor of three low). At ``d = 2`` that heuristic returns about 0.66, so
    the published 0.6 is scale-matched at unit scale, which is why the literal
    number transfers there and not at ``d = 16``.
    """
    gen = torch.Generator().manual_seed(seed)
    target = gaussian_target(sigma_eval, d=d, dtype=DTYPE)
    bank = target.sampler(512, gen)
    bw = (math.sqrt(median_heuristic_sigma_sq(bank)) if bandwidth is None
          else float(bandwidth))
    hat = sampled_bank_target(bank, bw, name=f"msip_dd_control_d{d}")
    emb = DataDrivenEmbeddings(bank, bw, dtype=DTYPE)
    idx = torch.multinomial(
        torch.ones(512, dtype=DTYPE), M, replacement=False, generator=gen
    )
    y, w, _ = run_msip(
        bank[idx], emb, MSIP_DD_LR, MSIP_DD_INFL, bw,
        MSIP_DD_ITERS, bounds=None, progress=False,
    )
    max_abs, n_queries = emb.pop_query_stats()
    msip_val = max(float(mmd_sq(y, w, hat.mu_fn(y), hat.c_rho, bw)), 0.0)
    iid_vals = []
    for r in range(32):
        g = torch.Generator().manual_seed(seed + 7919 * r)
        sub = torch.multinomial(
            torch.ones(512, dtype=DTYPE), M, replacement=False, generator=g
        )
        w_eq = torch.full((M,), 1.0 / M, dtype=DTYPE)
        iid_vals.append(max(float(mmd_sq(
            bank[sub], w_eq, hat.mu_fn(bank[sub]), hat.c_rho, bw,
        )), 0.0))
    iid_med = float(torch.tensor(iid_vals, dtype=DTYPE).median())
    return {
        "d": int(d),
        "bandwidth": bw,
        "scale_matched": bandwidth is None,
        "msip_dd": msip_val,
        "iid_subsample_median": iid_med,
        "max_abs_coordinate": float(max_abs),
        "n_emb_points": int(n_queries),
        "count_pinned": n_queries == (MSIP_DD_ITERS + 1) * M,
        "beats_subsampling": msip_val < iid_med,
    }


def control_msip_dd_quantizes(sigma_eval: float) -> dict:
    """The data-driven MSIP peer must quantize at both scales it is run at.

    Three cells, each with the trajectory count asserted equal to the pinned
    ``(T + 1) * M`` so the coordinate read covers the whole run:

      ``d2_published``  -- the unit-scale ``d = 2`` Gaussian bank at the
        literal published bandwidth 0.6.
      ``d16_scale_matched`` -- a ``d = 16`` bank at that bank's own median
        pairwise distance. This is the pass condition, and it certifies that
        the construction transfers to the dimension the wedge leg runs at.
      ``d16_published`` -- the same ``d = 16`` bank at the literal 0.6. A
        reading, never a pass condition: the 0.6 lands a factor of about 2.6
        below that bank's own scale and the quantizer then sits at or above
        equal-weight bank subsampling, which is why the sweep carries a
        scale-matched ``msip_dd_scaled`` column.
    """
    d2 = _msip_dd_control_leg(2, MSIP_DD_BANDWIDTH, sigma_eval, BASE_SEED + 995)
    d16_matched = _msip_dd_control_leg(
        16, None, sigma_eval, BASE_SEED + 995 + 61
    )
    d16_literal = _msip_dd_control_leg(
        16, MSIP_DD_BANDWIDTH, sigma_eval, BASE_SEED + 995 + 61
    )
    return {
        "msip_dd": d2["msip_dd"],
        "iid_subsample_median": d2["iid_subsample_median"],
        "max_abs_coordinate": d2["max_abs_coordinate"],
        "n_emb_points": d2["n_emb_points"],
        "d2_published": d2,
        "d16_scale_matched": d16_matched,
        "d16_published": d16_literal,
        "scale_transfer_reading": (
            "PRE-REGISTERED: the literal 0.6 is a unit-scale number. At "
            "d = 16 it sits well below the bank's own median pairwise "
            "distance and the quantizer no longer beats equal-weight "
            "subsampling, while the SAME construction at the scale-matched "
            "bandwidth does. The sweep's msip_dd column therefore includes "
            "the transfer; msip_dd_scaled is the construction's own column"
        ),
        "passes": (
            d2["beats_subsampling"] and d2["count_pinned"]
            and d16_matched["beats_subsampling"] and d16_matched["count_pinned"]
            and d16_literal["count_pinned"]
        ),
    }


def run_controls(sigma_eval: float) -> dict:
    """All peer-correctness controls; a peer that fails is withheld."""
    thin_dir = control_thinning_beats_iid(sigma_eval)
    thin_slope = control_thinning_rate_slope()
    return {
        "herding": control_herding_finite_rank(sigma_eval),
        "sbq": control_sbq_is_exact_at_finite_rank_and_dominates(sigma_eval),
        "thinning": {
            "direction": thin_dir,
            "slope": thin_slope,
            "passes": thin_dir["passes"] and thin_slope["passes"],
        },
        "recombine": control_recombination_is_exact_at_finite_rank(sigma_eval),
        "msip_dd": control_msip_dd_quantizes(sigma_eval),
    }


def _goodpoints_thin_fn(backend: str):
    """A ``(z, sigma, M, gen) -> index tensor`` view of a goodpoints backend.

    Lets the controls run unchanged against the published tooling: the
    goodpoints seed is taken from the control's own generator, and the
    geometry is the controls' own, whose grids are powers of four, so both
    ``kt.thin``'s dyadic halving and ``compresspp_kt``'s sqrt(n') land exactly
    on the requested ``M``.
    """
    def thin_fn(z, sigma, M, gen):
        n = int(z.shape[0])
        gp_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
        X = np.ascontiguousarray(z.numpy())
        if backend == "goodpoints_kt":
            from goodpoints import kt as gp_kt  # noqa: E402

            k_np = _se_kernel_np(sigma)
            idx = gp_kt.thin(
                X, int(round(math.log2(n / M))), k_np, k_np, delta=KT_DELTA,
                seed=gp_seed, store_K=False,
            )
        elif backend == "goodpoints_compresspp":
            from goodpoints import compress as gp_compress  # noqa: E402

            idx = gp_compress.compresspp_kt(
                X, b"gaussian",
                k_params=np.array([2.0 * float(sigma) ** 2]),
                g=COMPRESSPP_G, num_bins=COMPRESSPP_NUM_BINS, delta=KT_DELTA,
                seed=gp_seed,
            )
        else:
            raise ValueError(f"unknown backend {backend!r}")
        # goodpoints may hand back a reversed VIEW (negative stride), which
        # torch.from_numpy refuses; ascontiguousarray copies it flat.
        idx = np.ascontiguousarray(np.asarray(idx, dtype=np.int64))
        assert idx.shape[0] == M, (backend, n, M, idx.shape)
        return torch.from_numpy(idx)

    return thin_fn


def control_compresspp_cascade() -> dict:
    """The deployed accelerator path must clear the direction control too.

    ``compresspp_kt`` short-circuits to a direct dense thin whenever
    ``log2(sqrt(n)) <= g + log2(sqrt(num_bins))``: at g = 4, num_bins = 4
    that is every ``n <= 1024``, which is every grid the other controls run
    on. The M = 512 cells run the other branch, the Compress(g) cascade over
    per-bin coresets, so this control runs the cascade at its smallest
    engaging size, ``n = 4096 -> M = 64``, and asks the same direction
    question: the coreset must beat the median equal-sized i.i.d. subsample of
    the same input, in MMD^2 to the input measure.
    """
    n, M = 4096, 64
    assert (n.bit_length() - 1) // 2 > COMPRESSPP_G + (
        (COMPRESSPP_NUM_BINS.bit_length() - 1) // 2
    ), "n too small: the Compress cascade would short-circuit to dense KT"
    gen = torch.Generator().manual_seed(BASE_SEED + 997)
    z = torch.rand(n, 1, generator=gen, dtype=DTYPE)
    idx = _goodpoints_thin_fn("goodpoints_compresspp")(z, 0.1, M, gen)
    K = gram(z, 0.1)
    row_mean = K.mean(dim=1)

    def obj(sel):
        m = int(sel.numel())
        return float(
            K[sel][:, sel].sum() / (m * m) - 2.0 * row_mean[sel].sum() / m
        )

    cpp = obj(idx)
    iid_vals = []
    for r in range(64):
        g = torch.Generator().manual_seed(BASE_SEED + 997 + 7919 * r)
        iid_vals.append(obj(torch.randperm(n, generator=g)[:M]))
    iid = float(torch.tensor(iid_vals, dtype=DTYPE).median())
    return {"n": n, "M": M, "compresspp": cpp, "iid_median": iid,
            "passes": cpp < iid}


def run_backend_controls() -> dict:
    """The same two thinning controls, for the goodpoints backends.

    Direction (beats an equal-sized i.i.d. sample) and the log-log slope band,
    each run through :func:`_goodpoints_thin_fn`; the compresspp backend
    additionally clears :func:`control_compresspp_cascade`, because the
    small-n controls short-circuit past the cascade branch the M = 512 cells
    deploy. A backend that fails is withheld exactly as a dense-path peer
    would be.
    """
    out = {}
    for backend in sorted(set(EXT_BACKEND.values())):
        thin_fn = _goodpoints_thin_fn(backend)
        direction = control_thinning_beats_iid(1.0, thin_fn=thin_fn)
        slope = control_thinning_rate_slope(thin_fn=thin_fn)
        out[backend] = {
            "direction": direction,
            "slope": slope,
            "passes": direction["passes"] and slope["passes"],
        }
        if backend == "goodpoints_compresspp":
            cascade = control_compresspp_cascade()
            out[backend]["cascade"] = cascade
            out[backend]["passes"] = (
                out[backend]["passes"] and cascade["passes"]
            )
    return out


# --------------------------------------------------------------------------
# Tuning.
# --------------------------------------------------------------------------
def tune_peers(target, M: int, sigma_eval: float) -> dict:
    """Pick each peer's own knobs on seeds disjoint from the eval seeds.

    Every tunable peer is tuned on the same grid shape: its own knob, the same
    ``N_TUNE_REPEATS`` tuning repeats, the same disjoint offset. The
    weight-aware peer gets its refinement budget tuned exactly as thinning
    gets its swap budget. ``msip_dd`` is exempt, since it runs at its authors'
    published operating point. On a conditional family the knobs are tuned on
    the first held-out observation and held fixed across the stream, so the
    per-observation cost axis hides no per-observation tuning pass.
    """
    best_h, best_t, best_s = None, None, None
    for mode in HERDING_MODES:
        vals = []
        for r in range(N_TUNE_REPEATS):
            g = torch.Generator().manual_seed(
                BASE_SEED + TUNE_OFFSET + 1000 * M + r
            )
            vals.append(herding_arm(target, M, sigma_eval, g, mode)["mmd_sq"])
        med = float(torch.tensor(vals, dtype=DTYPE).median())
        if best_h is None or med < best_h["median"]:
            best_h = {"mode": mode, "median": med}
    for swaps in THINNING_SWAP_PASSES:
        vals = []
        for r in range(N_TUNE_REPEATS):
            g = torch.Generator().manual_seed(
                BASE_SEED + TUNE_OFFSET + 1000 * M + r
            )
            vals.append(
                thinning_arm(target, M, sigma_eval, g, swaps)["mmd_sq"]
            )
        med = float(torch.tensor(vals, dtype=DTYPE).median())
        if best_t is None or med < best_t["median"]:
            best_t = {"swap_passes": swaps, "median": med}
    for swaps in SBQ_SWAP_PASSES:
        vals = []
        for r in range(N_TUNE_REPEATS):
            g = torch.Generator().manual_seed(
                BASE_SEED + TUNE_OFFSET + 1000 * M + r
            )
            vals.append(sbq_arm(target, M, sigma_eval, g, swaps)["mmd_sq"])
        med = float(torch.tensor(vals, dtype=DTYPE).median())
        if best_s is None or med < best_s["median"]:
            best_s = {"swap_passes": swaps, "median": med}
    return {"herding": best_h, "thinning": best_t, "sbq": best_s}


# --------------------------------------------------------------------------
# The sweep.
# --------------------------------------------------------------------------
def _arm_gen(seed: int, arm: str) -> torch.Generator:
    """A generator for ONE arm, derived from the repeat seed by a fixed offset.

    Every arm samples from its own stream, so withholding a peer changes that
    peer's column and nothing else. Sharing one sequential generator would tie
    the peers' samples to the set of arms that ran. The samples-only mirror
    deliberately re-seeds thinning's stream, because its bank must be
    bit-identical to the input sample thinning consumes.
    """
    return torch.Generator().manual_seed(int(seed + ARM_SEED_OFFSET[arm]))


def run_case(case: str, withheld: set[str]) -> dict:
    """Sweep one target over the budget grid and, where conditional, a stream.

    A case whose trained amortizer is not on disk is skipped and says so: the
    per-instance oracle is a different object and may not stand in for the
    one-pass arm.
    """
    run = resolve_case(case)
    if run is None:
        reason = (
            "no trained amortizer is registered for this case"
            if case not in CASE_RUNS
            else "the registered checkpoint is not on disk"
        )
        print(
            f"[skip] case {case!r}: {reason}. The per-instance oracle is NOT "
            "substituted -- it is a different construction, and labelling it "
            "'ours' is exactly the defect this skip exists to prevent."
        )
        return {"case": case, "skipped": True, "reason": reason,
                "registered": case in CASE_RUNS}
    net = run["net"]
    built, sigma_eval, label, conditional = build_case(case, run)
    t_start = time.time()

    def _target(obs: int):
        return built(obs) if conditional else built

    gap = cross_check_target(_target(0), sigma_eval, seed=BASE_SEED)
    if gap > 1e-8:
        raise AssertionError(
            f"[{case}] metric cross-check {gap:.3e} exceeds 1e-8; no arm runs"
        )

    obs_stream = [r % N_OBS for r in range(N_REPEATS)] if conditional else None
    rows: list[dict] = []
    for M in M_LIST:
        tuned = tune_peers(_target(0), M, sigma_eval)
        acc: dict[str, list[float]] = {}
        cost: dict[str, int] = {}
        wall: dict[str, float] = {}
        conds: list[float] = []  # cond(K) of the seed Gram per repeat
        # Run-time diagnostics that pin an arm's internal algebra rather than
        # its score: whether the selection criterion agreed with an
        # independent re-solve, and the max absolute coordinate against the
        # clamp reference.
        diag: dict[str, float] = {}

        def _push(key: str, val: float) -> None:
            acc.setdefault(key, []).append(val)

        for r in tqdm(range(N_REPEATS), desc=f"{case} M={M}", leave=False):
            obs = r % N_OBS if conditional else 0
            target = _target(obs)
            cond = run["cond_fn"](obs) if "cond_fn" in run else None
            seed = BASE_SEED + CONFIRM_EVAL_OFFSET + 1000 * M + r
            z0 = target.sampler(M, _arm_gen(seed, "seed"))

            # Seed-Gram conditioning at the eval bandwidth: recorded and
            # thresholded, never excluded.
            K0 = torch.exp(
                -torch.cdist(z0, z0) ** 2 / (2.0 * sigma_eval * sigma_eval)
            )
            conds.append(float(torch.linalg.cond(K0)))

            t0 = time.time()
            ours = our_arms(z0, net, target, sigma_eval, cond=cond)
            wall["ours"] = wall.get("ours", 0.0) + (time.time() - t0)
            for k in ("floor", "reweight", "ours", "ours_move_raw"):
                _push(k, ours[k])
            cost["ours_n_mu"] = ours["n_mu"]
            cost["ours_n_draw"] = M

            # The samples-only mirror: mu-hat from the same n = M^2 samples
            # thinning consumes, from a fresh generator on an identical
            # stream, so the bank is bit-identical to thinning's input sample.
            # Taking the bank sits outside the timed window: it is already
            # charged on the reference-samples axis, and timing it here would
            # double-count it into the per-observation wall clock.
            bank = target.sampler(THINNING_INPUT[M], _arm_gen(seed, "thinning"))
            t0 = time.time()
            mirror = our_arms_samples_only(
                z0, net, target, sigma_eval, bank, cond=cond
            )
            wall["ours_samples_only"] = wall.get("ours_samples_only", 0.0) + (
                time.time() - t0
            )
            _push("ours_samples_only", mirror["ours_samples_only"])
            cost["ours_samples_only_n_draw"] = mirror["n_draw"]

            t0 = time.time()
            orc = oracle_arm(z0, target, sigma_eval)
            wall["oracle_move"] = wall.get("oracle_move", 0.0) + (
                time.time() - t0
            )
            _push("oracle_move", orc["oracle_move"])
            cost["oracle_move_n_mu"] = orc["n_mu"]
            cost["oracle_move_n_draw"] = M

            if "herding" not in withheld:
                t0 = time.time()
                out = herding_arm(target, M, sigma_eval,
                                  _arm_gen(seed, "herding"),
                                  tuned["herding"]["mode"])
                wall["herding"] = wall.get("herding", 0.0) + (time.time() - t0)
                _push("herding", out["mmd_sq"])
                _push("herding_rw", out["mmd_sq_rw"])
                cost["herding_n_mu"] = out["n_mu"]
                cost["herding_n_draw"] = out["n_draw"]

            if "sbq" not in withheld:
                t0 = time.time()
                out = sbq_arm(target, M, sigma_eval,
                              _arm_gen(seed, "sbq"),
                              tuned["sbq"]["swap_passes"])
                wall["sbq"] = wall.get("sbq", 0.0) + (time.time() - t0)
                _push("sbq", out["mmd_sq"])
                _push("sbq_eq", out["mmd_sq_eq"])
                cost["sbq_n_mu"] = out["n_mu"]
                cost["sbq_n_draw"] = out["n_draw"]
                diag["sbq_value_gap"] = max(
                    diag.get("sbq_value_gap", 0.0), out["value_gap"]
                )
                diag["sbq_unit_sum_err"] = max(
                    diag.get("sbq_unit_sum_err", 0.0), out["unit_sum_err"]
                )
                diag["sbq_n_swaps"] = (
                    diag.get("sbq_n_swaps", 0) + out["n_swaps"]
                )

            if "thinning" not in withheld:
                out = thinning_arm(target, M, sigma_eval,
                                   _arm_gen(seed, "thinning"),
                                   tuned["thinning"]["swap_passes"],
                                   input_floor=True)
                # The arm times itself around the construction only, so the
                # input-floor diagnostic Gram, which is work this harness
                # asked for rather than work the peer does, stays out.
                wall["thinning"] = wall.get("thinning", 0.0) + out["wall_secs"]
                _push("thinning", out["mmd_sq"])
                _push("thinning_rw", out["mmd_sq_rw"])
                _push("thinning_input_floor", out["input_floor"])
                cost["thinning_n_draw"] = out["n_draw"]

            if "recombine" not in withheld:
                t0 = time.time()
                out = recombine_arm(target, M, sigma_eval,
                                    _arm_gen(seed, "recombine"))
                wall["recombine"] = wall.get("recombine", 0.0) + (
                    time.time() - t0
                )
                _push("recombine", out["mmd_sq"])
                _push("recombine_rw", out["mmd_sq_rw"])
                cost["recombine_n_draw"] = out["n_draw"]

            if "msip_dd" not in withheld:
                # The literal published bandwidth 0.6, transferred unchanged,
                # and beside it the same construction at this case's own
                # trained scale. Both read the same pool from the same
                # stream, so the pair differs in the bandwidth alone.
                t0 = time.time()
                out = msip_dd_arm(target, M, sigma_eval,
                                  _arm_gen(seed, "msip_dd"))
                wall["msip_dd"] = wall.get("msip_dd", 0.0) + (
                    time.time() - t0
                )
                _push("msip_dd", out["mmd_sq"])
                _push("msip_dd_rw", out["mmd_sq_rw"])
                if out["mmd_sq_raw"] is not None:
                    _push("msip_dd_raw", out["mmd_sq_raw"])
                cost["msip_dd_n_draw"] = out["n_draw"]
                cost["msip_dd_n_emb_points"] = out["n_emb_points"]
                diag["msip_dd_max_abs_coordinate"] = max(
                    diag.get("msip_dd_max_abs_coordinate", 0.0),
                    out["max_abs_coordinate"],
                )

                t0 = time.time()
                sc = msip_dd_arm(target, M, sigma_eval,
                                 _arm_gen(seed, "msip_dd"),
                                 bandwidth=sigma_eval)
                wall["msip_dd_scaled"] = wall.get("msip_dd_scaled", 0.0) + (
                    time.time() - t0
                )
                _push("msip_dd_scaled", sc["mmd_sq"])
                _push("msip_dd_scaled_rw", sc["mmd_sq_rw"])
                if sc["mmd_sq_raw"] is not None:
                    _push("msip_dd_scaled_raw", sc["mmd_sq_raw"])
                cost["msip_dd_scaled_n_draw"] = sc["n_draw"]
                cost["msip_dd_scaled_n_emb_points"] = sc["n_emb_points"]
                diag["msip_dd_scaled_bandwidth"] = sc["bandwidth"]
                diag["msip_dd_scaled_max_abs_coordinate"] = max(
                    diag.get("msip_dd_scaled_max_abs_coordinate", 0.0),
                    sc["max_abs_coordinate"],
                )

        cond_med = float(torch.tensor(conds, dtype=DTYPE).median())
        rows.append({
            "M": M,
            "tuned": tuned,
            "median": {k: float(torch.tensor(v, dtype=DTYPE).median())
                       for k, v in acc.items()},
            "iqr": {k: [float(torch.tensor(v, dtype=DTYPE).quantile(0.25)),
                        float(torch.tensor(v, dtype=DTYPE).quantile(0.75))]
                    for k, v in acc.items()},
            # Raw per-repeat values: the per-observation readings and the
            # paired tests read these, and neither is derivable from a median.
            "values": acc,
            "cost": cost,
            # The reference budget, per method, for this cell.
            "draws_consumed": draws_consumed_for(M),
            # Seed-Gram conditioning: mark, never exclude, never ridge.
            "cond_K_seed": [
                cond_med,
                float(torch.tensor(conds, dtype=DTYPE).quantile(0.25)),
                float(torch.tensor(conds, dtype=DTYPE).quantile(0.75)),
            ],
            "cond_limited": cond_med > COND_K_THRESHOLD,
            "wall_secs": wall,
            "diagnostics": diag,
        })

    # Per-new-observation construction cost at the same M, with quality shown
    # adjacent.
    amortization = {}
    for row in rows:
        per = {}
        for arm in ("ours", "ours_samples_only", "oracle_move", "herding",
                    "sbq", "thinning", "recombine", "msip_dd",
                    "msip_dd_scaled"):
            if arm not in row["median"]:
                continue
            per[arm] = {
                "wall_secs_per_obs":
                    row["wall_secs"].get(arm, 0.0) / N_REPEATS,
                "n_mu_per_obs": row["cost"].get(f"{arm}_n_mu", 0),
                "n_draw_per_obs": row["draws_consumed"][arm],
                "quality_median_mmd_sq": row["median"][arm],
            }
        amortization[str(row["M"])] = per

    record = {
        "case": case,
        "skipped": False,
        "label": label,
        "conditional": conditional,
        "sigma_eval": sigma_eval,
        "cross_check_rel": gap,
        "design": DESIGN,
        # The network is named in the record, so a reader can tell which
        # trained run produced the ``ours`` column and at which bandwidth.
        "trained_run": {
            "dir": os.path.basename(run["dir"]),
            "target": run["res"].get("target", "wedge_conditional_family"),
            "d": int(run["res"].get("d") or run["res"].get("r", 0)),
            "sigma": float(run["res"]["sigma"]),
            "n_steps": int(run["res"].get("n_steps", 0)),
        },
        "oracle": {"lr": ORACLE_LR, "iters": ORACLE_ITERS,
                   "role": "the K-step competitor the map amortizes; NOT ours"},
        "rows": rows,
        "axis_amortization": {
            "definition": DESIGN["matched_quality_definition"],
            # What the wall clock does and does not contain.
            "wall_secs_window": (
                "CONSTRUCTION ONLY for our arm, the samples-only mirror and "
                "thinning: their reference draws sit outside the timed window "
                "(they are charged on the disclosed-draw axis, and timing "
                "them would double-count), and thinning's input-floor "
                "diagnostic Gram -- work this harness asked for, not work the "
                "peer does -- is excluded by the arm's own timer. The pool "
                "peers (herding / sbq / recombine / msip_dd) DO carry their "
                "own pool draw inside their window; at pool 512 that is a "
                "sub-millisecond charge against their own construction, i.e. "
                "the residual asymmetry runs AGAINST them, and it is "
                "disclosed here rather than removed"
            ),
            "per_M": amortization,
        },
        # The admissible-budget set of each method.
        "axis_budget": {
            "ours": {"admissible": M_LIST, "constructions_to_cover": 1},
            "ours_samples_only": {
                "admissible": M_LIST, "constructions_to_cover": 1,
                "note": "shares the trained map; only the solve's estimand "
                        "differs",
            },
            "oracle_move": {"admissible": M_LIST,
                            "constructions_to_cover": len(M_LIST)},
            "herding": {"admissible": M_LIST, "constructions_to_cover":
                        len(M_LIST)},
            # Greedy selection is nested when it is not refined: the length-M'
            # prefix of a length-M selection is the selection at M', so one
            # run of the search covers the whole grid. A refinement pass
            # destroys that, because a swap at one budget is not a swap at
            # another, so the count is reported per tuned configuration rather
            # than assumed.
            "sbq": {
                "admissible": M_LIST,
                "constructions_to_cover": {
                    str(row["M"]): (
                        1 if row["tuned"]["sbq"]["swap_passes"] == 0
                        else len(M_LIST)
                    )
                    for row in rows
                },
                "nested_when_unrefined": True,
            },
            "thinning": {
                "admissible": {str(M): admissible_budgets(THINNING_INPUT[M])
                               for M in M_LIST},
                "constructions_to_cover": len(M_LIST),
            },
            "recombine": {"admissible": M_LIST,
                          "constructions_to_cover": len(M_LIST)},
            "msip_dd": {
                "admissible": M_LIST,
                "constructions_to_cover": len(M_LIST),
                "note": "per-observation descent at its authors' published "
                        "settings; one run per (observation, budget)",
            },
            "msip_dd_scaled": {
                "admissible": M_LIST,
                "constructions_to_cover": len(M_LIST),
                "note": "the same construction at the case's own trained "
                        "sigma_eval (Gate-1 M3); identical budget axis, "
                        "identical pool, bandwidth alone differs",
            },
        },
        "withheld": sorted(withheld),
        "runtime_secs": time.time() - t_start,
    }
    if obs_stream is not None:
        record["obs_stream"] = obs_stream
    if "fit_ledger" in run:
        record["fit_ledger"] = run["fit_ledger"]
    return record


# --------------------------------------------------------------------------
# The extension driver (--phase extend): appends the M_EXT thinning cells,
# and the pool-free arms beside them, to the saved record. Every repeat is
# independently seeded, so a killed sweep restarts where it stopped.
# --------------------------------------------------------------------------
def _ext_progress_path(case: str, M: int) -> str:
    """One sidecar per (case, budget) cell, so parallel workers, each owning
    disjoint cells via ``--ext-cases`` / ``--ext-budgets``, never
    read-modify-write each other's repeats. kt.thin is single-threaded pure
    Python, so cell-level processes are the only real parallelism here."""
    return os.path.join(
        DIAG_DIR, f"dq_vs_thinning_ext_progress_{case}_{M}.json"
    )


def _load_ext_cell(case: str, M: int, backend: str) -> dict:
    path = _ext_progress_path(case, M)
    if os.path.isfile(path):
        with open(path) as fh:
            cell = json.load(fh)
        assert cell["backend"] == backend, (
            f"sidecar {path} was written by backend {cell['backend']!r}; "
            f"now asked for {backend!r} -- delete it to change backends"
        )
        return cell
    return {"backend": backend, "repeats": {}}


def _atomic_json_dump(obj, path: str, **kw) -> None:
    """Write JSON via a temporary file and ``os.replace``, so a crash
    mid-dump cannot truncate a record that took a day of repeats to earn."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, **kw)
    os.replace(tmp, path)


def _save_ext_cell(case: str, M: int, cell: dict) -> None:
    _atomic_json_dump(cell, _ext_progress_path(case, M))


def run_extension_cell(
    case: str, run: dict, built, sigma_eval: float, conditional: bool,
    M: int, backend: str, n_repeats: int,
) -> dict:
    """One (case, M) extension cell, in run_case's row schema exactly.

    The per-repeat protocol is run_case's verbatim (same eval seeds, same
    per-arm streams, same scoring), restricted to the arms that need no
    candidate pool, with the thinning construction produced by the published
    tooling. Completed repeats are read back from the cell's own sidecar file,
    so the cell is resumable, idempotent, and safe to run beside workers
    owning other cells.
    """
    net = run["net"]

    def _target(obs: int):
        return built(obs) if conditional else built

    cell = _load_ext_cell(case, M, backend)
    for r in tqdm(range(n_repeats), desc=f"{case} M={M} [{backend}]",
                  leave=False):
        if str(r) in cell["repeats"]:
            continue
        obs = r % N_OBS if conditional else 0
        target = _target(obs)
        cond = run["cond_fn"](obs) if "cond_fn" in run else None
        seed = BASE_SEED + CONFIRM_EVAL_OFFSET + 1000 * M + r
        z0 = target.sampler(M, _arm_gen(seed, "seed"))
        K0 = torch.exp(
            -torch.cdist(z0, z0) ** 2 / (2.0 * sigma_eval * sigma_eval)
        )
        cond_val = float(torch.linalg.cond(K0))

        acc: dict[str, float] = {}
        wall: dict[str, float] = {}
        cost: dict[str, int] = {}

        t0 = time.time()
        ours = our_arms(z0, net, target, sigma_eval, cond=cond)
        wall["ours"] = time.time() - t0
        for k in ("floor", "reweight", "ours", "ours_move_raw"):
            acc[k] = ours[k]
        cost["ours_n_mu"] = ours["n_mu"]
        cost["ours_n_draw"] = M

        # The samples-only mirror: bank bit-identical to thinning's input
        # (fresh generator, identical stream), taken outside the timed
        # window as in run_case.
        bank = target.sampler(THINNING_INPUT[M], _arm_gen(seed, "thinning"))
        t0 = time.time()
        mirror = our_arms_samples_only(
            z0, net, target, sigma_eval, bank, cond=cond
        )
        wall["ours_samples_only"] = time.time() - t0
        acc["ours_samples_only"] = mirror["ours_samples_only"]
        cost["ours_samples_only_n_draw"] = mirror["n_draw"]
        del bank

        t0 = time.time()
        orc = oracle_arm(z0, target, sigma_eval)
        wall["oracle_move"] = time.time() - t0
        acc["oracle_move"] = orc["oracle_move"]
        cost["oracle_move_n_mu"] = orc["n_mu"]
        cost["oracle_move_n_draw"] = M

        thin_out = thinning_arm_published(
            target, M, sigma_eval, _arm_gen(seed, "thinning"), backend,
            cpp_diag=(backend == "goodpoints_kt"),
        )
        wall["thinning"] = thin_out["wall_secs"]
        acc["thinning"] = thin_out["mmd_sq"]
        acc["thinning_rw"] = thin_out["mmd_sq_rw"]
        acc["thinning_input_floor"] = thin_out["input_floor"]
        if "mmd_sq_compresspp" in thin_out:
            acc["thinning_compresspp"] = thin_out["mmd_sq_compresspp"]
            wall["thinning_compresspp"] = thin_out["wall_secs_compresspp"]
        cost["thinning_n_draw"] = thin_out["n_draw"]

        cell["repeats"][str(r)] = {
            "acc": acc, "wall": wall, "cond": cond_val, "cost": cost,
            "floor_device": thin_out["floor_device"],
        }
        _save_ext_cell(case, M, cell)

    reps = [cell["repeats"][str(r)] for r in range(n_repeats)]
    acc_lists: dict[str, list[float]] = {}
    wall_sums: dict[str, float] = {}
    for rep in reps:
        for k, v in rep["acc"].items():
            acc_lists.setdefault(k, []).append(v)
        for k, v in rep["wall"].items():
            wall_sums[k] = wall_sums.get(k, 0.0) + v
    conds = torch.tensor([rep["cond"] for rep in reps], dtype=DTYPE)
    cond_med = float(conds.median())
    return {
        "M": M,
        # No tuning pass at the extension budgets: the published tooling
        # runs at its own operating point (kt.thin's built-in KT-SWAP,
        # Compress++ at g=4), as msip_dd does.
        "tuned": {
            "herding": None, "sbq": None,
            "thinning": {
                "backend": backend, "swap_passes": None,
                "note": "published tooling at its authors' operating "
                        "point; no knob tuned",
            },
        },
        "median": {k: float(torch.tensor(v, dtype=DTYPE).median())
                   for k, v in acc_lists.items()},
        "iqr": {k: [float(torch.tensor(v, dtype=DTYPE).quantile(0.25)),
                    float(torch.tensor(v, dtype=DTYPE).quantile(0.75))]
                for k, v in acc_lists.items()},
        "values": acc_lists,
        "cost": reps[-1]["cost"],
        "draws_consumed": draws_consumed_for(M),
        "cond_K_seed": [
            cond_med,
            float(conds.quantile(0.25)),
            float(conds.quantile(0.75)),
        ],
        "cond_limited": cond_med > COND_K_THRESHOLD,
        "wall_secs": wall_sums,
        "diagnostics": {
            "thinning_backend": backend,
            "floor_devices": sorted({rep["floor_device"] for rep in reps}),
            "goodpoints_seed_rule": (
                "drawn from the thinning arm's own generator stream, "
                "immediately after the input sample"
            ),
        },
        "arm_scope": (
            "extension cell (2026-08-22): pool-free arms only; the pool "
            "peers stay on the pre-registered [32, 64, 128] grid (see "
            "design.budget_cap.extension_2026_08_22)"
        ),
    }


def run_extension(
    json_path: str, n_repeats: int = N_REPEATS,
    cases: tuple[str, ...] = ("wedge", "gmm2"),
    budgets: tuple[int, ...] | None = None, merge: bool = True,
) -> dict:
    """Extend the saved record's thinning arm to the full deployed grid.

    Loads the record, gates the goodpoints backends on the same controls,
    runs the requested cells per case (wedge first, since the RSS-lifetime
    assertion of ``_stage_wedge`` must see a small process), and merges the
    new rows into the record in place. A smoke run
    (``n_repeats != N_REPEATS``) writes a sidecar instead of the record.

    ``cases`` / ``budgets`` / ``merge`` exist for parallel work: N workers
    each own a disjoint (case, budget) slice with ``merge=False``, so repeats
    land in per-cell sidecars and the record is not touched, and one final
    full invocation finds every repeat already on disk, assembles the rows,
    and merges once, so no two processes ever write one file.
    """
    with open(json_path) as fh:
        results = json.load(fh)
    backend_controls = run_backend_controls()
    results.setdefault("controls", {})["thinning_goodpoints"] = (
        backend_controls
    )
    failed = {b for b, c in backend_controls.items() if not c["passes"]}
    for b in sorted(failed):
        print(
            f"[C1] goodpoints backend {b!r} FAILED the pre-registered "
            "thinning controls and its cells are WITHHELD; the omission is "
            "reported rather than the number."
        )
    budgets = tuple(M_EXT) if budgets is None else tuple(budgets)
    assert all(M in M_EXT for M in budgets), budgets
    # Wedge first: _stage_wedge's whole-process RSS assertion must run before
    # any multi-GB backend allocation, so the order is not left to the
    # caller's argument order.
    cases = tuple(sorted(cases, key=lambda c: (c != "wedge", c)))
    for case in cases:
        res = results["cases"].get(case)
        if res is None or res.get("skipped", False):
            print(f"[extend] case {case!r} absent or skipped; not extended")
            continue
        run = resolve_case(case)
        if run is None:
            print(f"[extend] case {case!r}: checkpoint gone; not extended")
            continue
        built, sigma_eval, _, conditional = build_case(case, run)
        rec_sigma = float(res["sigma_eval"])
        assert abs(sigma_eval - rec_sigma) < 1e-12, (
            f"resolved run's sigma {sigma_eval} != record's {rec_sigma}; "
            "the checkpoint under this identity changed since the "
            "confirmatory sweep -- refusing to mix rulers in one figure"
        )

        def _target(obs: int):
            return built(obs) if conditional else built

        gap = cross_check_target(_target(0), sigma_eval, seed=BASE_SEED)
        if gap > 1e-8:  # the same cross-check as run_case
            raise AssertionError(
                f"[{case}] metric cross-check {gap:.3e} exceeds 1e-8"
            )
        new_rows = []
        for M in budgets:
            backend = EXT_BACKEND[M]
            if backend in failed:
                continue
            new_rows.append(run_extension_cell(
                case, run, built, sigma_eval, conditional, M, backend,
                n_repeats,
            ))
        if not merge:
            print(f"[extend] case {case!r}: worker slice done "
                  f"(budgets {list(budgets)}); record untouched.")
            continue
        replaced = {int(row["M"]) for row in new_rows}
        res["rows"] = sorted(
            [row for row in res["rows"] if int(row["M"]) not in replaced]
            + new_rows,
            key=lambda row: int(row["M"]),
        )
        # The amortization axis at the new budgets, run_case's own formula.
        for row in new_rows:
            per = {}
            for arm in ("ours", "ours_samples_only", "oracle_move",
                        "thinning"):
                if arm not in row["median"]:
                    continue
                per[arm] = {
                    "wall_secs_per_obs":
                        row["wall_secs"].get(arm, 0.0) / n_repeats,
                    "n_mu_per_obs": row["cost"].get(f"{arm}_n_mu", 0),
                    "n_draw_per_obs": row["draws_consumed"][arm],
                    "quality_median_mmd_sq": row["median"][arm],
                }
            res["axis_amortization"]["per_M"][str(row["M"])] = per
        # The extension cells' walls were taken under worker co-scheduling
        # and the other cells' single-process, so the two regimes may not be
        # read against each other. The MMD^2 columns are timing-independent.
        res["axis_amortization"]["extension_wall_disclosure"] = {
            "status": (
                "DISCLOSED, NOT RE-MEASURED (Gate-1 item 2, resolved "
                "2026-08-24). The wall_secs of the M = 256 / 512 cells were "
                "measured under FOUR-WAY worker co-scheduling (one process "
                "per (case, budget) cell, nice 10) on a shared box, while "
                "the M <= 128 cells ran single-process. The A1 wall column "
                "is therefore contention-inflated at the extension budgets "
                "and is NOT comparable across the 128/256 boundary."
            ),
            "why_not_re_measured": (
                "a quiet single-process re-timing was attempted and "
                "ABANDONED for a measured reason, not a scheduling one: at "
                "the calibration budget M = 128 -- where the confirmatory "
                "sweep and the re-timing pass BOTH run single-process, so "
                "they must agree -- the re-timing read one forward pass at "
                "12.2 ms against the swept column's 5.6 ms, a factor of "
                "2.2 apart. Wall clocks on this box are not reproducible "
                "across passes to better than that, so a second pass "
                "cannot license cross-pass comparison either; the box also "
                "re-filled with other jobs mid-pass. Reporting the "
                "contended numbers WITH this disclosure is the honest "
                "option, and it is the one taken."
            ),
            "consequence_for_the_prose": (
                "no wall-clock reading at an extension budget is quoted "
                "anywhere. The only wall numbers in sec. 6.4 (the SBQ "
                "per-observation cost against one forward pass) are both "
                "from the confirmatory sweep at the SAME budget and the "
                "SAME process, which is the only comparison this "
                "instrument supports."
            ),
            "mmd_unaffected": (
                "the MMD^2 columns are timing-independent and are the "
                "swept 64-repeat values throughout"
            ),
        }
        # One map covers the grid the rows now realize; the peers' sets
        # extend where their construction does.
        grid_present = sorted({int(row["M"]) for row in res["rows"]})
        res["axis_budget"]["ours"]["admissible"] = grid_present
        res["axis_budget"]["ours_samples_only"]["admissible"] = grid_present
        res["axis_budget"]["oracle_move"] = {
            "admissible": grid_present,
            "constructions_to_cover": len(grid_present),
        }
        res["axis_budget"]["thinning"]["admissible"] = {
            str(M): admissible_budgets(THINNING_INPUT[M])
            for M in grid_present
        }
        res["axis_budget"]["thinning"]["constructions_to_cover"] = (
            len(grid_present)
        )
        res["axis_budget"]["thinning"]["backend_per_budget"] = {
            "32-128": "in-repo dense-Gram kernel_thinning (tuned swaps)",
            "256": EXT_BACKEND[256], "512": EXT_BACKEND[512],
        }
        res["design"] = DESIGN
    if not merge:
        print("[extend] worker mode (--ext-no-merge): no record written.")
        return results
    results["design"] = DESIGN
    out_path = json_path if n_repeats == N_REPEATS else (
        json_path.replace(".json", "_smoke.json")
    )
    _atomic_json_dump(results, out_path, indent=2)
    print(f"Saved to {out_path}")
    if n_repeats != N_REPEATS:
        print(
            f"[extend] SMOKE RUN ({n_repeats} repeats): the shipped record "
            "was NOT touched; the per-cell sidecar repeats are reusable by "
            "the real sweep (identical seeds)."
        )
    return results


# --------------------------------------------------------------------------
# The quiet wall-clock pass. The extension cells' MMD^2 was swept by four
# co-scheduled workers, which inflates their wall column by up to two orders
# through contention rather than cost. The scores stay as swept; only the
# timing is retaken, single-process on an idle machine.
# --------------------------------------------------------------------------
WALL_QUIET_REPEATS = 8
# The calibration budget: the largest budget the sweep already timed
# single-process. Re-timing the cheap arms there says whether this pass
# reproduces the recorded column, which is what licenses reading the
# extension walls against it. Its thinning arm is not retimed, since the
# dense backend at M = 128 costs about 13 minutes per repeat.
WALL_CALIBRATION_M = 128


def run_wall_pass(
    json_path: str, n_repeats: int = WALL_QUIET_REPEATS,
    cases: tuple[str, ...] = ("wedge", "gmm2"),
) -> dict:
    """Re-time the timed windows single-process; write ``wall_secs_quiet``.

    Runs the same arms on the same eval seeds as the sweep, so the timing is
    of identical work, but scores nothing and writes no MMD: the record's
    values are the swept ones and this pass may not perturb them. The
    input-floor diagnostic is skipped, since it is outside every timed window
    by construction, which is what makes the pass affordable at
    ``n = 262144``.

    Two numbers travel with the timings: the repeat count, fewer than the
    sweep's because a wall-clock mean does not need sixty-four repeats, and
    the calibration reading at ``WALL_CALIBRATION_M``, where this pass and
    the sweep timed the same arms under the same single-process conditions
    and must therefore agree.
    """
    with open(json_path) as fh:
        results = json.load(fh)
    for case in tuple(sorted(cases, key=lambda c: (c != "wedge", c))):
        res = results["cases"].get(case)
        if res is None or res.get("skipped", False):
            continue
        run = resolve_case(case)
        if run is None:
            print(f"[wall] case {case!r}: checkpoint gone; not retimed")
            continue
        built, sigma_eval, _, conditional = build_case(case, run)
        net = run["net"]
        rows = {int(row["M"]): row for row in res["rows"]}
        for M in [WALL_CALIBRATION_M] + list(M_EXT):
            row = rows.get(M)
            if row is None:
                continue
            do_thin = M in M_EXT
            acc: dict[str, list[float]] = {}
            for r in tqdm(range(n_repeats), desc=f"wall {case} M={M}",
                          leave=False):
                obs = r % N_OBS if conditional else 0
                target = built(obs) if conditional else built
                cond = run["cond_fn"](obs) if "cond_fn" in run else None
                seed = BASE_SEED + CONFIRM_EVAL_OFFSET + 1000 * M + r
                z0 = target.sampler(M, _arm_gen(seed, "seed"))

                t0 = time.time()
                our_arms(z0, net, target, sigma_eval, cond=cond)
                acc.setdefault("ours", []).append(time.time() - t0)

                bank = target.sampler(
                    THINNING_INPUT[M], _arm_gen(seed, "thinning")
                )
                t0 = time.time()
                our_arms_samples_only(
                    z0, net, target, sigma_eval, bank, cond=cond
                )
                acc.setdefault("ours_samples_only", []).append(
                    time.time() - t0
                )
                del bank

                t0 = time.time()
                oracle_arm(z0, target, sigma_eval)
                acc.setdefault("oracle_move", []).append(time.time() - t0)

                if do_thin:
                    out = thinning_arm_published(
                        target, M, sigma_eval, _arm_gen(seed, "thinning"),
                        EXT_BACKEND[M],
                        cpp_diag=(EXT_BACKEND[M] == "goodpoints_kt"),
                        skip_floor=True,
                    )
                    acc.setdefault("thinning", []).append(out["wall_secs"])
                    if "wall_secs_compresspp" in out:
                        acc.setdefault("thinning_compresspp", []).append(
                            out["wall_secs_compresspp"]
                        )
            row["wall_secs_quiet"] = {
                k: float(torch.tensor(v, dtype=DTYPE).mean())
                for k, v in acc.items()
            }
            row["wall_secs_quiet_median"] = {
                k: float(torch.tensor(v, dtype=DTYPE).median())
                for k, v in acc.items()
            }
            row["wall_quiet_repeats"] = int(n_repeats)
            row["wall_quiet_note"] = (
                "single-process on an idle box, per observation, timing the "
                "SAME windows the sweep timed; scores untouched"
                + ("" if do_thin else
                   "; CALIBRATION budget -- the thinning arm is not retimed "
                   "here, only the arms whose shipped walls this pass is "
                   "checked against")
            )
            print(f"[wall] {case} M={M}: " + " ".join(
                f"{k}={1000 * v:.1f}ms"
                for k, v in row["wall_secs_quiet"].items()
            ))
        cal = rows.get(WALL_CALIBRATION_M, {})
        if "wall_secs_quiet" in cal:
            swept = {k: cal["wall_secs"].get(k, 0.0) / N_REPEATS
                     for k in cal["wall_secs_quiet"]}
            res["axis_amortization"]["wall_quiet_calibration"] = {
                "budget": WALL_CALIBRATION_M,
                "swept_secs_per_obs": swept,
                "quiet_secs_per_obs": cal["wall_secs_quiet"],
                "ratio_quiet_over_swept": {
                    k: (cal["wall_secs_quiet"][k] / swept[k])
                    if swept[k] > 0 else None
                    for k in cal["wall_secs_quiet"]
                },
                "role": (
                    "at this budget BOTH the confirmatory sweep and the "
                    "quiet pass ran single-process, so the ratio is a "
                    "measurement-noise reading: it is what licenses "
                    "comparing the extension budgets' quiet walls against "
                    "the shipped column"
                ),
            }
        res["axis_amortization"]["extension_wall_disclosure"] = {
            "status": (
                "the wall_secs of the M = 256 / 512 cells were swept under "
                "FOUR-WAY worker co-scheduling and are contention-inflated; "
                "wall_secs_quiet on those rows is a single-process "
                "re-measurement of the SAME timed windows over "
                "wall_quiet_repeats repeats."
            ),
            "READ_THE_CALIBRATION_FIRST": (
                "wall_secs_quiet may be quoted against the shipped column "
                "ONLY IF wall_quiet_calibration's ratio_quiet_over_swept "
                "sits near one at the calibration budget, where both "
                "passes ran single-process and must therefore agree. It "
                "did NOT on the 2026-08-24 attempt (2.2x on one forward "
                "pass), which is why the shipped record carries the "
                "contended numbers with a disclosure instead. A ratio far "
                "from one means this box's wall clocks are not "
                "reproducible across passes, and no cross-pass wall "
                "comparison is licensed -- not that the quiet pass is the "
                "better number."
            ),
            "mmd_unaffected": (
                "the MMD^2 columns are timing-independent and are the "
                "swept 64-repeat values throughout"
            ),
        }
    _atomic_json_dump(results, json_path, indent=2)
    print(f"Saved to {json_path}")
    return results


# --------------------------------------------------------------------------
# Render.
# --------------------------------------------------------------------------
def render(results: dict, out_path: str) -> None:
    """Discrepancy against the reference budget, per case.

    The shared x-axis is reference samples consumed per observation, never the
    node budget alone, so a pool peer's flat spend and thinning's ``M^2``
    schedule are visible as geometry. Thinning's own input-sample floor is
    drawn as a band beside its arm, and cond(K)-limited cells are drawn open
    for solved-weight arms. A skipped case is not drawn: an empty panel is
    preferable to a panel whose ``ours`` curve came from something other than
    the trained map.
    """
    apply_house_style()
    cases = [c for c, res in results["cases"].items()
             if not res.get("skipped", False)]
    skipped = [c for c in results["cases"] if c not in cases]
    if skipped:
        print(f"[render] cases skipped for want of a checkpoint: {skipped}")
    if not cases:
        raise RuntimeError(
            "every case was skipped; there is no ours-arm to plot and the "
            "figure would misrepresent the oracle as the shipped construction"
        )
    fig, axes = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 4.1))
    axes = [axes] if len(cases) == 1 else list(axes)

    for ax, case in zip(axes, cases):
        res = results["cases"][case]
        rows = res["rows"]
        limited = [row.get("cond_limited", False) for row in rows]
        # Thinning's own input-sample floor: the band beside the arm.
        band_ok = all("thinning_input_floor" in row["median"] for row in rows)
        if band_ok:
            xs = [row["draws_consumed"]["thinning_input_floor"]
                  for row in rows]
            med = [row["median"]["thinning_input_floor"] for row in rows]
            lo = [row["iqr"]["thinning_input_floor"][0] for row in rows]
            hi = [row["iqr"]["thinning_input_floor"][1] for row in rows]
            ax.fill_between(xs, lo, hi, color=PALETTE["thinning"],
                            alpha=0.14, lw=0.0)
            ax.plot(xs, med, color=PALETTE["thinning"], lw=1.2, ls=":",
                    label=LABELS["thinning_input_floor"])
        for key in PLOT_KEYS:
            # The arms live on different sub-grids of the budget axis: the
            # pool peers stop at M = 128, while the pool-free arms and
            # thinning run the full grid. An arm is therefore drawn over the
            # cells it has rather than dropped for the cells it lacks.
            pts = [
                (row["draws_consumed"][key], row["median"][key], lim)
                for row, lim in zip(rows, limited)
                if row["median"].get(key) is not None
            ]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lims = [p[2] for p in pts]
            # The oracle is a competitor, not the amortized arm: dashed and
            # grey, so no reader can take it for the one-pass output.
            dashed = key == "oracle_move"
            ax.plot(xs, ys, color=PALETTE[key], lw=2.0,
                    ls="--" if dashed else "-", label=LABELS[key])
            # Solved-weight arms at cond(K)-limited cells are drawn open:
            # their values stand, their between-arm rankings do not.
            for x, y, lim in zip(xs, ys, lims):
                open_marker = lim and key in SOLVED_WEIGHT_KEYS
                ax.plot([x], [y], marker="o", ms=3.6, color=PALETTE[key],
                        mfc="none" if open_marker else PALETTE[key],
                        mew=1.0)
        # Annotate the node budget along the ``ours`` arm, so the M grid
        # stays legible on the samples-consumed axis.
        for row in rows:
            if "ours" in row["median"]:
                ax.annotate(
                    f"M={row['M']}",
                    (row["draws_consumed"]["ours"], row["median"]["ours"]),
                    textcoords="offset points", xytext=(0, -11),
                    fontsize=6, color="0.35", ha="center",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("reference draws consumed per observation "
                      "(disclosed budget)")
        ax.set_ylabel(r"$\mathrm{MMD}^2$ to the reference")
        ax.set_title(res["label"])
        despine(ax)
    # The legend sits below the panels: over the full grid the amortized arm
    # descends through the left panel's lower-left corner, the oracle crosses
    # it, and the M = 256 / 512 annotations land on the entry text. Eleven
    # entries do not fit inside the data area without covering an arm.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.4, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               handlelength=1.8, columnspacing=1.4)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for ext in ("pdf", "png"):
        p = f"{out_path}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved to {p}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "visualize", "extend", "walltime"),
        default="all",
        help="'all' runs the controls then the sweep then renders; "
        "'visualize' re-renders from the saved record; 'extend' appends the "
        "M = 256 / 512 thinning cells to the saved record via the "
        "goodpoints backends (resumable) and re-renders; 'walltime' "
        "re-times the timed windows single-process on a quiet box and "
        "writes wall_secs_quiet (scores untouched).",
    )
    parser.add_argument(
        "--ext-repeats", type=int, default=N_REPEATS,
        help="SMOKE-TESTING ONLY: cap the extension's repeat count. Any "
        "value other than N_REPEATS writes a sidecar record instead of the "
        "shipped one.",
    )
    parser.add_argument(
        "--ext-cases", default="wedge,gmm2",
        help="extend only these cases (comma list) -- the parallel-worker "
        "slice knob; each worker must own a DISJOINT (case, budget) set.",
    )
    parser.add_argument(
        "--ext-budgets", default=None,
        help="extend only these budgets (comma list from M_EXT) -- the "
        "other worker slice knob.",
    )
    parser.add_argument(
        "--ext-no-merge", action="store_true",
        help="worker mode: compute repeats into the per-cell sidecars and "
        "leave the record and the figure untouched; one final plain "
        "--phase extend merges everything once.",
    )
    args = parser.parse_args()

    torch.set_default_dtype(DTYPE)
    os.makedirs(DIAG_DIR, exist_ok=True)
    json_path = os.path.join(DIAG_DIR, "dq_vs_thinning.json")

    if args.phase == "visualize":
        with open(json_path) as fh:
            results = json.load(fh)
        print(f"Loaded {json_path} (visualize-only; no recompute).")
    elif args.phase == "walltime":
        results = run_wall_pass(json_path)
        print("[wall] scores untouched; re-rendering is unnecessary but "
              "harmless (the figure reads medians, not walls).")
        sys.exit(0)
    elif args.phase == "extend":
        results = run_extension(
            json_path, n_repeats=args.ext_repeats,
            cases=tuple(c for c in args.ext_cases.split(",") if c),
            budgets=(None if args.ext_budgets is None else tuple(
                int(b) for b in args.ext_budgets.split(",") if b
            )),
            merge=not args.ext_no_merge,
        )
        if args.ext_no_merge:
            sys.exit(0)
        if args.ext_repeats != N_REPEATS:
            print("[extend] smoke run: not rendering into paper/figures.")
            sys.exit(0)
    else:
        controls = run_controls(1.0)
        withheld = {k for k, v in controls.items() if not v["passes"]}
        for k in sorted(withheld):
            print(
                f"[C1] peer {k!r} FAILED its own correctness control and is "
                "WITHHELD. An under-implemented competitor is a strawman; the "
                "omission is reported rather than the number."
            )
        results = {"design": DESIGN, "controls": controls, "cases": {}}
        # Wedge must stage before any 16384^2 Gram exists in this process.
        for case in ("wedge", "gmm2"):
            results["cases"][case] = run_case(case, withheld)
        _atomic_json_dump(results, json_path, indent=2)
        print(f"Saved to {json_path}")

    render(results, os.path.join(FIG_DIR, "dq_vs_thinning"))
