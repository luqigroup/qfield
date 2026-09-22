"""SVGD versus the designed quadrature, on closed-form targets.

SVGD returns an equally weighted particle cloud read off the posterior score
and re-run per query; the designed quadrature returns a signed-weight,
score-free, amortized object built from the kernel mean of ``rho``. The
commensurable axis is integration error of the same ``rho``, so every arm is
scored by the same exact SE-MMD^2 to the true posterior at one fixed
evaluation bandwidth ``sigma_eval``.

Every case is scored at the bandwidth its network was fitted under, read from
the resolved run's own ``results.json``, and on the deployed node-budget grid.

Targets (closed-form ``rho = pi``, exact ``mu`` and ``c_rho``, so the estimand
the weights are solved from and the estimand the arms are scored on coincide):
  * ``lg2``  -- a 2-D Gaussian posterior at its trained median-MMD bandwidth;
  * ``gmm2`` -- a 5-mode 2-D GMM at its trained median-MMD bandwidth (the
    SVGD-rule ``h_sq`` is used only inside the sampler's own dynamics);
  * ``g8``   -- an 8-D Gaussian posterior at its trained median-MMD bandwidth.

Seven arms, all scored by the same ``mmd_sq(z, w, mu_fn(z), c_rho,
sigma_eval)`` on the same ``M`` i.i.d. seed nodes ``z0`` per repeat, which
separates node motion from weighting:
  1. ``floor``          -- ``(z0, 1/M)``: the i.i.d.-MC integration error.
  2. ``reweight``       -- ``(z0, w_c)``: i.i.d. nodes, constrained signed
     weights, with no network and no descent.
  3. ``svgd``           -- tuned-SVGD particles, uniform ``1/M``.
  4. ``svgd_rw``        -- tuned-SVGD particles with the constrained weights,
     which isolates the reweighting gain on SVGD's own cloud.
  5. ``amortized``      -- one forward pass of the trained amortizer at the
     seeds, the closed-form constrained solve at its displaced nodes, and the
     safeguarded selection over {seeds, reweight, move}.
  6. ``amortized_raw``  -- the same forward pass without the safeguarded
     selection, recorded but not drawn.
  7. ``oracle_move``    -- per-instance MMD^2 descent from ``z0`` plus
     constrained weights: the oracle the amortizer approximates.

SVGD is tuned over a grid of internal bandwidth h, step lr and iteration
count, the configuration picked on a disjoint tuning-seed offset to minimize
the evaluation MMD, then frozen and evaluated on the held-out evaluation
seeds. The oracle arm uses a fixed, untuned ``oracle_lr`` and ``oracle_iters``,
and the amortized arm is a frozen checkpoint with no per-case knob. Per-cell
paired win fractions (amortized against SVGD on the same repeat) are recorded
beside the medians. A functional spot check (``x_1^2`` and a bounded bump) is
reported for every arm, since signed weights can integrate a bounded ``h``
worse than uniform weights despite a lower MMD.

CPU only; float64; per-repeat ``torch.Generator`` determinism (seed
``base_seed + 1000 * M + r``, disjoint tuning offset).

Run:
    CUDA_VISIBLE_DEVICES="" python scripts/dq_vs_svgd.py

Reference:
  - Liu & Wang, "Stein Variational Gradient Descent", NeurIPS 2016 (SVGD).
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (the MMD-optimal designed
    quadrature; SE closed forms).
"""

from __future__ import annotations
from projorg import plotsdir  # noqa: E402

# Pin CPU before importing torch.
import os  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from qfield.designed_quadrature import (  # noqa: E402
    constrained_weights,
    cross_check_target,
    gaussian_target,
    gmm_target,
    mmd_sq,
    move_descend,
    select_emission,
)
from qfield.designed_quadrature.arms import (  # noqa: E402
    COND_LIMIT,
    seed_gram_cond,
)
from qfield.designed_quadrature.target import (  # noqa: E402
    _build_gmm,
    _cncv_gaussian_posterior,
)
from qfield.kernels import (  # noqa: E402
    median_heuristic_h_sq,      # SVGD dynamics only (Liu and Wang)
    median_heuristic_sigma_sq,  # the MMD evaluation bandwidth
)
from qfield.samplers.svgd import run_svgd  # noqa: E402

# Sibling script modules (not a package): the checkpoint resolver, which
# raises on an ambiguous match, and the amortizer rebuild.
from dq_safeguard_sweep import (  # noqa: E402
    find_run,
    load_net,
    load_state,
    target_for,
)

FIG_DIR = plotsdir("paper")
DIAG_DIR = os.path.join(FIG_DIR, "diagnostics")

DTYPE = torch.float64

# Tiny intra-op pool: the workload is many small solves, where more threads
# only thrash.
torch.set_num_threads(4)

# Sweep and repeats: the closed forms make 64 repeats cheap and the
# intervals tight.
M_LIST = [32, 64, 128, 256, 512]
N_REPEATS = 64
BASE_SEED = 20260613
# Tuning seeds disjoint from the evaluation seeds base_seed + 1000*M + r,
# which span base_seed + [32000, 512063] for M <= 512 and r < 64; the offset
# below clears that range.
TUNE_OFFSET = 2_000_000
N_TUNE_REPEATS = 16

JITTER = 1e-10
# The per-instance oracle arm, fixed and untuned.
ORACLE_LR = 0.05
ORACLE_ITERS = 150

# The projorg identity token selecting a run trained on this node-budget
# grid.
GRID_TOKEN = "m_list-32-64-128-256-512"

# Case -> the trained amortizer the ``amortized`` arm is one forward pass of:
# ``(glob, target family, dimension, minimum steps, trained sigma)``, resolved
# by :func:`dq_safeguard_sweep.find_run`, which verifies the match against
# ``results.json`` and raises when it is ambiguous. Several runs of one target
# share a directory prefix and differ only in the bandwidth rule or the budget
# grid, so the glob must name both. ``sigma`` is ``None`` because all three
# bandwidths are trained median heuristics: the resolved run's own
# ``results.json`` supplies ``sigma_eval``.
CASE_RUNS = {
    "lg2": (
        "designed_quadrature_gaussian_amortized_target-gaussian_d-2_"
        f"*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*",
        "gaussian", 2, 3000, None,
    ),
    "gmm2": (
        f"dq_gmm_d2_target-gmm_d-2_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*",
        "gmm", 2, 3000, None,
    ),
    # The d = 8 network is isotropic (whiten-0) at the median-MMD bandwidth.
    "g8": (
        f"dq_gaussian_d8_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*",
        "gaussian", 8, 3000, None,
    ),
}

# SVGD's own hyperparameters: internal kernel bandwidth h, step lr and
# iterations. The evaluation kernel is fixed at sigma_eval and shared by all
# arms, while SVGD's internal h is free. "median" means
# h = sqrt(median_heuristic_h_sq(z_init)), the Liu and Wang default; the
# floats are explicit fixed bandwidths, in units of sigma rather than h^2.
SVGD_BANDWIDTHS = ["median", 0.5, 1.0, 2.0]
SVGD_LRS = [0.05, 0.1, 0.25, 0.5, 1.0]
SVGD_ITERS = [200, 500, 1000]

# Semantic palette, one concept to one colour.
PALETTE = {
    "floor": "#3a7ca5",     # the independent-sample floor
    "reweight": "#2a9d8f",  # the reweight-only lever
    "svgd": "#1f77b4",      # tuned SVGD
    "svgd_rw": "#9467bd",   # the SVGD cloud under the reweighting
    "amortized": "#ff7f0e",     # the safeguarded one-pass rule
    "amortized_raw": "#c98a4b",  # the moved arm without the safeguard
    "oracle_move": "#7f7f7f",   # the per-instance descent
}
LABELS = {
    "floor": "i.i.d. floor",
    "reweight": "i.i.d. nodes + reweight",
    "svgd": "SVGD (uniform, per query)",
    "svgd_rw": "SVGD nodes + reweight",
    "amortized": "designed quadrature (ours, safeguarded one pass)",
    "amortized_raw": "moved arm without the safeguard",
    "oracle_move": "per-instance descent (fixed horizon)",
}
# The arms the figure draws, in draw order. ``amortized_raw`` is recorded but
# not drawn: it is a diagnostic on the selection, not a competitor.
PLOT_ARMS = ("floor", "reweight", "svgd", "svgd_rw", "amortized", "oracle_move")
# Arms whose weights come out of the constrained solve. Past COND_LIMIT the
# solve is rank-deficient and differences between these arms are solve noise,
# so those cells are drawn open.
SOLVED_ARMS = ("reweight", "svgd_rw", "amortized", "amortized_raw", "oracle_move")


# --------------------------------------------------------------------------
# Targets and their closed-form scores (SVGD reads grad log pi).
# --------------------------------------------------------------------------
def _median_mmd_bandwidth(probe, n_ref: int = 4000) -> float:
    """The median-MMD bandwidth rule on one fixed reference sample.

    Returns ``sqrt(median_heuristic_sigma_sq(ref))``, not the Liu and Wang
    ``h_sq`` divisor, which belongs to a sampler's dynamics rather than to a
    metric. This is only a fallback: each case is scored at the bandwidth its
    own checkpoint was trained at.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(BASE_SEED + 99)
    return math.sqrt(median_heuristic_sigma_sq(probe.sampler(int(n_ref), gen)))


def resolve_case(case: str) -> dict:
    """Resolve the trained amortizer behind a case, by projorg config identity.

    Resolution is :func:`dq_safeguard_sweep.find_run`, which verifies the run
    against its ``results.json`` (target family, dimension, training steps,
    trained bandwidth) and raises when several runs match. A missing run is
    fatal rather than a skip.

    Returns:
        ``{"dir", "res", "net"}``.

    Raises:
        FileNotFoundError: No run matches the case's identity.
    """
    glob_pat, target, d, n_steps_min, sigma = CASE_RUNS[case]
    got = find_run(glob_pat, target, d, n_steps_min, sigma=sigma)
    if got is None:
        raise FileNotFoundError(
            f"[{case}] no trained amortizer matches the pinned identity: "
            f"glob {glob_pat!r}, filter target={target}, d={d}, "
            f"n_steps>={n_steps_min}, sigma={sigma}. The amortized arm cannot "
            "be drawn without it, and the per-instance oracle must never "
            "stand in for it."
        )
    cdir, res = got
    net = load_net(load_state(cdir), int(res["d"]))
    return {"dir": cdir, "res": res, "net": net}


def case_setup(case: str, run: dict):
    """``(target, score_fn, sigma_eval, label, xform)`` for a trained run.

    The score, the label and the frame reconciliation come from
    :func:`build_case`; the bandwidth comes from the run's own
    ``results.json``, so every arm is scored at the bandwidth the network was
    fitted under. The rebuilt reference is cross-checked against the run's own
    rebuild (:func:`dq_safeguard_sweep.target_for`), since a mismatch would put
    the network and the metric on two different references.

    Raises:
        RuntimeError: The case's reference and the run's disagree.
    """
    target, score_fn, sigma_eval, label, xform = build_case(
        case, float(run["res"]["sigma"])
    )
    theirs = target_for(run["res"], whiten=False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(BASE_SEED + 31337)
    probe = target.sampler(11, gen)
    gap = float((target.mu_fn(probe) - theirs.mu_fn(probe)).abs().max())
    c_gap = abs(float(target.c_rho) - float(theirs.c_rho))
    if gap > 1e-12 or c_gap > 1e-12:
        raise RuntimeError(
            f"[{case}] the case's reference is not the trained run's: "
            f"max |mu - mu'| = {gap:.3e}, |c_rho - c_rho'| = {c_gap:.3e}"
        )
    return target, score_fn, sigma_eval, label, xform


def build_case(case: str, sigma_eval: float | None = None):
    """Return ``(target, score_fn, sigma_eval, label, svgd_xform)`` for a case.

    ``target`` is the designed-quadrature :class:`_Target` (exact ``mu_fn``,
    ``c_rho`` and sampler, the common metric all arms are scored under).
    ``score_fn(z) -> grad log pi(z)`` is the exact posterior score SVGD reads
    to move its particles, built from the same posterior the metric uses.

    ``sigma_eval`` is supplied by the caller and is the trained bandwidth read
    off the resolved run's ``results.json`` (:func:`resolve_case`). The
    per-case default used when ``sigma_eval`` is ``None`` is the same
    median-MMD rule the trainers apply, a reference sample's
    ``sqrt(median_heuristic_sigma_sq)``.

    ``svgd_xform`` reconciles the SVGD coordinate frame with the metric's:

      * ``None`` -- the metric and the score live in the same ambient frame, so
        SVGD seeds, moves and is scored all in that frame. Every current case
        is of this kind; the whitened-frame branch below is the generic
        reconciliation a whitened case needs.
      * a dict ``{"to_score": fn, "to_kernel": fn}`` -- the metric lives in a
        whitened frame ``u`` while the score lives in the ambient frame ``x``.
        The target's ``sampler`` returns whitened seed nodes ``u``; ``to_score``
        lifts them to ambient ``x = L u`` so SVGD moves with the true ambient
        score ``grad log pi(x)``, and ``to_kernel`` whitens the moved particles
        back ``u = R x`` so they are scored by the same whitened MMD^2 as every
        other arm. Only the scoring frame changes; the score itself is the
        unmodified true ambient posterior score.

    Args:
        case: One of ``"lg2"`` (2-D Gaussian), ``"gmm2"`` (2-D 5-mode GMM),
            ``"g8"`` (8-D Gaussian).
        sigma_eval: The shared evaluation bandwidth. ``None`` falls back to the
            case's own median-MMD heuristic.

    Returns:
        ``(target, score_fn, sigma_eval, human_label, svgd_xform)``.
    """
    if case == "lg2":
        if sigma_eval is None:
            sigma_eval = _median_mmd_bandwidth(
                gaussian_target(1.0, d=2, dtype=DTYPE)
            )
        target = gaussian_target(sigma_eval, d=2, dtype=DTYPE)
        mu_post = torch.tensor(target.meta["mu_post"], dtype=DTYPE)
        sigma_post = torch.tensor(target.meta["Sigma_post"], dtype=DTYPE)
        prec = torch.linalg.inv(sigma_post)

        def score_fn(z: torch.Tensor) -> torch.Tensor:
            # grad log N(mu, Sigma) = -Sigma^{-1} (z - mu).
            return -(z.to(DTYPE) - mu_post) @ prec.T

        return target, score_fn, sigma_eval, "2-D linear-Gaussian posterior", None

    if case == "gmm2":
        # Evaluation bandwidth: the median heuristic on a reference sample of
        # rho, shared by every arm. SVGD's own tuning grid still includes the
        # Liu and Wang "median".
        if sigma_eval is None:
            sigma_eval = _median_mmd_bandwidth(gmm_target(1.0, d=2, dtype=DTYPE))
        target = gmm_target(sigma_eval, d=2, dtype=DTYPE)
        # meta lacks covs and weights, so rebuild the same GMM for its .score;
        # the cross-check below asserts it is the same mixture.
        gmm = _build_gmm(2, DTYPE)

        def score_fn(z: torch.Tensor) -> torch.Tensor:
            return gmm.score(z.to(DTYPE))  # grad log pi (normalized mixture)

        # The rebuilt GMM's MMD^2 must equal the target's on a random
        # weighting, so the score and the metric describe the same pi.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(BASE_SEED + 4242)
        zc = target.sampler(7, gen)
        wc = torch.randn(7, generator=gen, dtype=DTYPE)
        wc = wc / wc.sum()
        mine = float(gmm.mmd_sq(zc, wc, sigma_eval))
        theirs = float(target.cross_check_reference(zc, wc))
        rel = abs(mine - theirs) / (abs(theirs) + 1e-30)
        assert rel < 1e-8, f"GMM score-vs-ruler mismatch: rel={rel:.2e}"

        return target, score_fn, sigma_eval, "2-D 5-mode anisotropic GMM", None

    if case == "g8":
        d = 8
        if sigma_eval is None:
            sigma_eval = _median_mmd_bandwidth(
                gaussian_target(1.0, d=d, dtype=DTYPE)
            )
        target = gaussian_target(sigma_eval, d=d, dtype=DTYPE)
        # The score must be the target's own posterior. ``gaussian_target``
        # builds the posterior at its default y* seed and records only the
        # prior metadata, so the moments are rebuilt through the same library
        # call rather than read off ``meta``. The identity
        # ``E_rho[x_1^2] = mu_1^2 + Sigma_11`` is asserted against the target's
        # own ``e_x1sq`` before the score is handed to SVGD: a different seed
        # or observation-noise default would otherwise put the sampler's score
        # and the metric on two different posteriors.
        mu_post, sigma_post, _ = _cncv_gaussian_posterior(
            d, sigma_obs=0.3, seed=20260612, dtype=DTYPE
        )
        got = float(mu_post[0] ** 2 + sigma_post[0, 0])
        assert abs(got - float(target.e_x1sq)) <= 1e-10 * max(1.0, abs(got)), (
            f"[g8] the rebuilt posterior is not the target's: "
            f"E[x_1^2] {got} vs {target.e_x1sq}"
        )
        prec = torch.linalg.inv(sigma_post)

        def score_fn(x: torch.Tensor) -> torch.Tensor:
            # grad log N(mu, Sigma) = -Sigma^{-1} (x - mu).
            return -(x.to(DTYPE) - mu_post) @ prec.T

        return (
            target, score_fn, sigma_eval, "8-D Gaussian posterior", None,
        )

    raise ValueError(f"unknown case {case!r}")


# --------------------------------------------------------------------------
# The arms, all scored by the same mmd_sq at sigma_eval.
# --------------------------------------------------------------------------
def _score(z: torch.Tensor, w: torch.Tensor, target, sigma_eval: float) -> float:
    """Exact SE-MMD^2 of ``(z, w)`` to the true ``rho`` (clamped at 0)."""
    val = float(mmd_sq(z.to(DTYPE), w.to(DTYPE), target.mu_fn(z), target.c_rho, sigma_eval))
    return max(val, 0.0)


def amortized_arm(z0: torch.Tensor, net, target, sigma_eval: float) -> dict:
    """The safeguarded one-pass rule on one seed set.

    One forward pass of the trained amortizer at the i.i.d. seeds, the
    closed-form unit-sum-constrained solve at its displaced nodes, and the
    selection over the three-arm menu {seeds, reweight, move}. There is no
    per-instance descent, no score and no per-observation tuning: the only
    per-repeat work is one network evaluation and two linear solves.

    The returned value is the selected one, and the bare moved arm is returned
    beside it as ``amortized_raw`` so the selection's effect stays visible.

    ``fitting_criterion=True`` is passed to the selection because on these
    closed-form references the criterion's ``mu_fn`` is the estimand the
    weights were solved against, so the seed arm can never strictly beat the
    same-node reweighting and the selection carries a live check of the solve.

    Args:
        z0: i.i.d. seed nodes ``(M, d)``, the same seeds every other arm
            consumes on this repeat.
        net: The trained amortizer, in eval mode.
        target: The reference (supplies ``mu_fn`` and ``c_rho``).
        sigma_eval: The shared evaluation bandwidth.

    Returns:
        ``{amortized, amortized_raw, selected, active, z, w}`` -- the two
        scores, the menu member chosen, whether the safeguard moved the answer
        off the network's arm, and the rule itself.
    """
    M = int(z0.shape[0])
    mu_fn = getattr(target, "mu_fn_eval", None) or target.mu_fn
    with torch.no_grad():
        mu0 = mu_fn(z0)
        w_eq = torch.full((M,), 1.0 / M, dtype=DTYPE)
        w_rw = constrained_weights(z0, mu0, sigma_eval, JITTER)
        z_net = net(z0.unsqueeze(0).to(DTYPE)).squeeze(0).to(DTYPE)
        mu_net = mu_fn(z_net)
        w_net = constrained_weights(z_net, mu_net, sigma_eval, JITTER)
        raw = max(
            float(mmd_sq(z_net, w_net, mu_net, target.c_rho, sigma_eval)), 0.0
        )
        sel = select_emission(
            [(z0, w_eq), (z0, w_rw), (z_net, w_net)],
            mu_fn, sigma_eval, names=("seeds", "reweight", "move"),
            fitting_criterion=True,
        )
        z_s, w_s = sel["z"], sel["w"]
        shipped = max(
            float(mmd_sq(z_s, w_s, mu_fn(z_s), target.c_rho, sigma_eval)), 0.0
        )
    return {
        "amortized": shipped,
        "amortized_raw": raw,
        "selected": sel["name"],
        "active": bool(sel["active"]),
        "z": z_s,
        "w": w_s,
    }


def _paired_frac(a: list[float], b: list[float]) -> float:
    """Fraction of paired repeats on which ``a`` is strictly below ``b``.

    The medians say which arm is ahead in the middle of the distribution, not
    how often the ordering holds on one repeat. Both arms are scored on a
    shared seed set per repeat, so the pairing is exact. Ties count for neither
    side.
    """
    if len(a) != len(b):
        raise ValueError("paired win fraction needs equal-length arms")
    return sum(1 for x, y in zip(a, b) if x < y) / max(len(a), 1)


def run_svgd_arm(
    z0: torch.Tensor,
    score_fn,
    bandwidth,
    lr: float,
    n_iters: int,
    sigma_eval: float,
    svgd_xform=None,
) -> torch.Tensor:
    """Run SVGD from ``z0`` and return its final particles in the metric frame.

    ``z0`` is a seed node set in the frame ``mmd_sq`` scores in. When
    ``svgd_xform`` is ``None`` the metric and the score share that frame, so
    SVGD runs directly on ``z0`` and the result is returned unchanged. When
    ``svgd_xform`` is given, the seed is first lifted to the ambient score
    frame (``to_score``), SVGD moves there with the true ambient posterior
    score, and the moved particles are whitened back to the metric frame
    (``to_kernel``) so they are scored by the same whitened MMD^2 as every
    other arm. Only the coordinate frame is mapped; the score SVGD descends is
    never altered.

    ``bandwidth`` is SVGD's internal kernel bandwidth h: ``"median"`` gives
    ``sqrt(median_heuristic_h_sq(.))`` (the helper returns h^2, so the square
    root converts it to a bandwidth) computed in the score frame SVGD runs in,
    else the explicit float. ``sigma_eval`` is not used inside SVGD; it is the
    shared metric applied afterwards.
    """
    # Lift the seed into the frame SVGD moves in.
    x0 = svgd_xform["to_score"](z0) if svgd_xform is not None else z0.to(DTYPE)
    if bandwidth == "median":
        h = math.sqrt(median_heuristic_h_sq(x0))  # h^2 -> bandwidth (score frame)
    else:
        h = float(bandwidth)
    x_svgd, _ = run_svgd(
        x0, score_fn, lr=lr, sigma=h, n_iters=int(n_iters),
        bounds=None, progress=False,
    )
    # Map the moved particles back into the metric frame for scoring.
    if svgd_xform is not None:
        return svgd_xform["to_kernel"](x_svgd).to(DTYPE)
    return x_svgd.to(DTYPE)


def tune_svgd(
    case: str, target, score_fn, sigma_eval: float, M: int, svgd_xform=None
) -> dict:
    """Pick SVGD (bandwidth, lr, iters) minimizing the mean MMD on tuning seeds.

    Tunes on a tuning-seed offset disjoint from the evaluation seeds, under the
    same metric all arms are scored by. Returns the winning configuration and
    its tuning score.
    """
    # Tuning node sets on the disjoint offset.
    tune_draws = []
    for r in range(N_TUNE_REPEATS):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(BASE_SEED + TUNE_OFFSET + 1000 * M + r)
        tune_draws.append(target.sampler(M, gen))

    best = {"mmd": float("inf"), "bandwidth": None, "lr": None, "iters": None}
    for bw in SVGD_BANDWIDTHS:
        for lr in SVGD_LRS:
            for it in SVGD_ITERS:
                vals = []
                for z0 in tune_draws:
                    z_s = run_svgd_arm(
                        z0, score_fn, bw, lr, it, sigma_eval, svgd_xform
                    )
                    w_eq = torch.full((M,), 1.0 / M, dtype=DTYPE)
                    vals.append(_score(z_s, w_eq, target, sigma_eval))
                mean_mmd = float(torch.tensor(vals, dtype=DTYPE).mean())
                if mean_mmd < best["mmd"]:
                    best = {
                        "mmd": mean_mmd, "bandwidth": bw, "lr": lr, "iters": it,
                    }
    return best


# --------------------------------------------------------------------------
# Functional spot check (x_1^2 and a bounded bump), one M, every arm.
# --------------------------------------------------------------------------
def _bump(z: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """A bounded test function h(x) = exp(-||x - center||^2 / 2), in [0, 1]."""
    return torch.exp(-0.5 * ((z - center) ** 2).sum(dim=-1))


def functional_spot_check(
    case: str, target, score_fn, sigma_eval: float, svgd_cfg: dict, M: int,
    net=None, svgd_xform=None,
) -> dict:
    """``|E_Q[h] - E_rho[h]|`` for ``h in {x_1^2, bump}``, every arm.

    The constants-exact unit-sum weights integrate ``x_1^2`` and the bump only
    as well as the kernel resolves them, so this surfaces whether signed
    weights hurt a bounded ``h`` despite a lower MMD. ``E_rho[h]`` is estimated
    with a large i.i.d. reference sample, the same for every arm, so the
    comparison is unbiased relative to that reference. All quantities are in
    the metric frame: nodes, the reference and the test functions all live
    there.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(BASE_SEED + 7)
    z0 = target.sampler(M, gen)

    # Large reference for E_rho[h], shared across arms; the bump is centered
    # on the reference mean.
    gref = torch.Generator(device="cpu")
    gref.manual_seed(BASE_SEED + 8)
    ref = target.sampler(200_000, gref)
    center = ref.mean(dim=0)
    e_x1sq = float((ref[:, 0] ** 2).mean())
    e_bump = float(_bump(ref, center).mean())

    def errs(z, w):
        w = w.to(DTYPE)
        ex1 = abs(float((w * z[:, 0] ** 2).sum()) - e_x1sq)
        ebp = abs(float((w * _bump(z, center)).sum()) - e_bump)
        return ex1, ebp

    w_eq = torch.full((M,), 1.0 / M, dtype=DTYPE)
    w_rw = constrained_weights(z0, target.mu_fn(z0), sigma_eval, JITTER)
    z_s = run_svgd_arm(
        z0, score_fn, svgd_cfg["bandwidth"], svgd_cfg["lr"],
        svgd_cfg["iters"], sigma_eval, svgd_xform,
    )
    w_srw = constrained_weights(z_s, target.mu_fn(z_s), sigma_eval, JITTER)
    z_mv, _ = move_descend(
        z0, target.mu_fn, target.c_rho, sigma_eval, ORACLE_LR, ORACLE_ITERS,
        JITTER,
    )
    w_mv = constrained_weights(z_mv, target.mu_fn(z_mv), sigma_eval, JITTER)
    ours = amortized_arm(z0, net, target, sigma_eval)

    per_arm = {
        "floor": errs(z0, w_eq),
        "reweight": errs(z0, w_rw),
        "svgd": errs(z_s, w_eq),
        "svgd_rw": errs(z_s, w_srw),
        "amortized": errs(ours["z"], ours["w"]),
        "oracle_move": errs(z_mv, w_mv),
    }
    return {
        "M": M,
        "e_x1sq": e_x1sq,
        "e_bump": e_bump,
        "selected": ours["selected"],
        "x1sq": {k: v[0] for k, v in per_arm.items()},
        "bump": {k: v[1] for k, v in per_arm.items()},
    }


# --------------------------------------------------------------------------
# The sweep.
# --------------------------------------------------------------------------
def _median_iqr(vals) -> tuple[float, float, float]:
    t = torch.tensor(vals, dtype=DTYPE)
    t = t[torch.isfinite(t)]
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


def run_case(case: str) -> dict:
    """Resolve the network, tune SVGD, then sweep ``M`` over the seven arms.

    The trained checkpoint is resolved first because it fixes the bandwidth:
    every arm in the case is scored at the bandwidth the network was fitted
    under (:func:`case_setup`). The sweep then pairs all seven arms on one
    i.i.d. seed set per repeat, so the medians, the interquartile ranges and
    the paired win fractions all describe the same repeats.
    """
    run = resolve_case(case)
    target, score_fn, sigma_eval, label, svgd_xform = case_setup(case, run)
    net = run["net"]

    # Metric cross-check: explicit MMD^2 against the target's independent
    # closed form, so the score and the metric describe the same posterior.
    rel = cross_check_target(target, sigma_eval, seed=BASE_SEED + 777)
    assert rel < 1e-8, f"[{case}] metric cross-check failed: rel={rel:.2e}"
    print(
        f"[{case}] {label}  sigma_eval={sigma_eval:.6f} (trained)  "
        f"cross-check rel={rel:.2e}\n         net={os.path.basename(run['dir'])[:70]}"
    )

    arm_keys = tuple(PALETTE)
    rows = []
    svgd_cfgs = {}
    for M in M_LIST:
        cfg = tune_svgd(case, target, score_fn, sigma_eval, M, svgd_xform)
        svgd_cfgs[M] = cfg
        arms = {k: [] for k in arm_keys}
        draws = []
        selected = []
        active = 0
        for r in range(N_REPEATS):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(BASE_SEED + 1000 * M + r)
            z0 = target.sampler(M, gen)  # the same z0 for every arm
            draws.append(z0)

            w_eq = torch.full((M,), 1.0 / M, dtype=DTYPE)
            arms["floor"].append(_score(z0, w_eq, target, sigma_eval))

            w_rw = constrained_weights(z0, target.mu_fn(z0), sigma_eval, JITTER)
            arms["reweight"].append(_score(z0, w_rw, target, sigma_eval))

            z_s = run_svgd_arm(
                z0, score_fn, cfg["bandwidth"], cfg["lr"], cfg["iters"],
                sigma_eval, svgd_xform,
            )
            arms["svgd"].append(_score(z_s, w_eq, target, sigma_eval))
            w_srw = constrained_weights(z_s, target.mu_fn(z_s), sigma_eval, JITTER)
            arms["svgd_rw"].append(_score(z_s, w_srw, target, sigma_eval))

            ours = amortized_arm(z0, net, target, sigma_eval)
            arms["amortized"].append(ours["amortized"])
            arms["amortized_raw"].append(ours["amortized_raw"])
            selected.append(ours["selected"])
            active += int(ours["active"])

            z_mv, _ = move_descend(
                z0, target.mu_fn, target.c_rho, sigma_eval, ORACLE_LR,
                ORACLE_ITERS, JITTER,
            )
            w_mv = constrained_weights(z_mv, target.mu_fn(z_mv), sigma_eval, JITTER)
            arms["oracle_move"].append(_score(z_mv, w_mv, target, sigma_eval))

        row = {"M": M, "svgd_cfg": cfg}
        for k in arm_keys:
            row[k] = _median_iqr(arms[k])
        # Conditioning of the seed Gram at the evaluation bandwidth. Past
        # ``COND_LIMIT`` the constrained solve retains about two digits, many
        # weight vectors fit near-equally, and differences between
        # solved-weight arms are solve noise. Those cells are marked rather
        # than dropped, and the figure draws them open.
        cond = seed_gram_cond(torch.stack(draws, dim=0), sigma_eval)
        row["cond_K_seed"] = cond
        row["cond_limited"] = bool(cond["median"] > COND_LIMIT)
        # Paired win fractions on the same repeats: a median ordering says
        # nothing about how often the ordering holds.
        row["paired_win_frac"] = {
            "amortized_below_svgd": _paired_frac(
                arms["amortized"], arms["svgd"]
            ),
            "amortized_below_svgd_rw": _paired_frac(
                arms["amortized"], arms["svgd_rw"]
            ),
            "amortized_below_floor": _paired_frac(
                arms["amortized"], arms["floor"]
            ),
            "amortized_below_oracle": _paired_frac(
                arms["amortized"], arms["oracle_move"]
            ),
        }
        # Which menu member the safeguard emitted, and how often it moved the
        # answer off the network's own arm.
        row["selection"] = {
            "counts": {
                name: selected.count(name)
                for name in ("seeds", "reweight", "move")
            },
            "activation_rate": active / N_REPEATS,
        }
        # Ratios to the floor and to SVGD.
        fl = row["floor"][0]
        sv = row["svgd"][0]
        row["ratios"] = {
            k: (row[k][0] / fl if fl > 0 else float("inf")) for k in arm_keys
        }
        row["over_svgd"] = {
            k: (row[k][0] / sv if sv > 0 else float("inf")) for k in arm_keys
        }
        rows.append(row)
        print(
            f"  M={M:>3d} | floor={row['floor'][0]:.3e} "
            f"rw={row['reweight'][0]:.3e} "
            f"svgd={row['svgd'][0]:.3e} (h={cfg['bandwidth']},lr={cfg['lr']},"
            f"it={cfg['iters']}) svgd_rw={row['svgd_rw'][0]:.3e} "
            f"AMORT={row['amortized'][0]:.3e} "
            f"(raw={row['amortized_raw'][0]:.3e}, "
            f"amort/svgd={row['over_svgd']['amortized']:.3g}, "
            f"win={row['paired_win_frac']['amortized_below_svgd']:.2f}) "
            f"oracle={row['oracle_move'][0]:.3e}"
            f"{'  [cond-limited]' if row['cond_limited'] else ''}"
        )

    spot = functional_spot_check(
        case, target, score_fn, sigma_eval, svgd_cfgs[M_LIST[0]], M=M_LIST[0],
        net=net, svgd_xform=svgd_xform,
    )
    return {
        "case": case,
        "label": label,
        "sigma_eval": sigma_eval,
        "sigma_source": "trained (results.json of the resolved run)",
        "estimand": "closed form (exact mu_fn and c_rho; no sampled bank)",
        "trained_run": {
            "dir": os.path.basename(run["dir"]),
            "target": run["res"]["target"],
            "d": int(run["res"]["d"]),
            "sigma": float(run["res"]["sigma"]),
            "n_steps": int(run["res"]["n_steps"]),
            "trained_m_list": run["res"].get("m_list"),
        },
        "m_list": M_LIST,
        "n_repeats": N_REPEATS,
        "metric_cross_check_rel": rel,
        "rows": rows,
        "spot_check": spot,
        "target_meta": target.meta,
        "svgd_grid": {
            "bandwidths": SVGD_BANDWIDTHS, "lrs": SVGD_LRS, "iters": SVGD_ITERS,
        },
        "oracle_move_config": {
            "lr": ORACLE_LR, "iters": ORACLE_ITERS, "tuned": False,
            "role": "the K-step competitor the map amortizes; NOT our arm",
        },
        "cond_limit": COND_LIMIT,
    }


# --------------------------------------------------------------------------
# Figure: MMD against M, one panel per case.
# --------------------------------------------------------------------------
def render(results: dict, out_path: str) -> None:
    """One row of panels (one per case): MMD^2 to rho against M, log-log.

    Six curves per panel (``PLOT_ARMS``). The one-pass rule is the orange solid
    line and the per-instance oracle is grey. ``amortized_raw`` is in the
    record but off the figure.
    """
    cases = list(results.keys())
    with plt.rc_context(
        {"pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
         "font.size": 9, "legend.fontsize": 7.0}
    ):
        fig, axes = plt.subplots(
            1, len(cases), figsize=(4.6 * len(cases), 3.7), squeeze=False
        )
        axes = axes[0]
        for ax, case in zip(axes, cases):
            res = results[case]
            rows = res["rows"]
            missing = [a for a in PLOT_ARMS if a not in rows[0]]
            if missing:
                raise KeyError(
                    f"[{case}] the record predates the amortized arm (missing "
                    f"{missing}); re-run --phase all. Redrawing a pre-repair "
                    "record would put the per-instance oracle back on the "
                    "figure in place of the emission."
                )
            m_vals = [row["M"] for row in rows]
            # Cells whose seed-Gram cond(K) exceeds COND_LIMIT: the constrained
            # solve retains about two digits there, so differences between
            # solved-weight arms are solve noise, and those cells are drawn as
            # open, faded markers. The flag comes from the record itself, so
            # the figure and the JSON agree on which cells are marked.
            limited = [bool(row.get("cond_limited", False)) for row in rows]
            # A non-positive median is not a value. At the worst-conditioned
            # cells the constrained solve returns a discrepancy the clamp at
            # zero has already absorbed. Plotting such a cell at a token 1e-30
            # on a log axis draws a cliff that is pure floating point and
            # compresses every real curve in the panel into a band. The axis
            # floor is set from the smallest strictly positive median in the
            # panel, and the cell is marked there with a downward caret.
            pos = [
                row[arm][0] for arm in PLOT_ARMS for row in rows
                if row[arm][0] > 0.0
            ]
            y_floor = 0.25 * min(pos) if pos else 1e-30
            zeroed = False
            for arm in PLOT_ARMS:
                raw_med = [row[arm][0] for row in rows]
                med = [max(v, y_floor) for v in raw_med]
                lo = [max(row[arm][1], y_floor) for row in rows]
                hi = [max(row[arm][2], y_floor) for row in rows]
                yerr = [
                    [m - l for m, l in zip(med, lo)],
                    [h - m for m, h in zip(med, hi)],
                ]
                ls = "--" if arm in ("reweight", "svgd_rw") else "-"
                lw = 2.1 if arm == "amortized" else 1.6
                ax.errorbar(
                    m_vals, med, yerr=yerr, color=PALETTE[arm], marker="o",
                    markersize=4, linewidth=lw, capsize=2.5, linestyle=ls,
                    label=LABELS[arm],
                    zorder=4 if arm == "amortized" else 3,
                )
                # Overdraw the conditioning-limited cells of the solved-weight
                # arms as open marks; the floor and the equal-weight SVGD arm
                # involve no solve and stay solid.
                if arm in SOLVED_ARMS and any(limited):
                    mx = [m for m, L in zip(m_vals, limited) if L]
                    my = [v for v, L in zip(med, limited) if L]
                    ax.plot(
                        mx, my, linestyle="none", marker="o", markersize=5.5,
                        mfc="white", mec=PALETTE[arm], mew=1.4, alpha=0.9,
                        zorder=5,
                    )
                # Cells the solve took to zero: caret at the axis floor.
                zx = [m for m, v in zip(m_vals, raw_med) if v <= 0.0]
                if zx:
                    zeroed = True
                    ax.plot(
                        zx, [y_floor] * len(zx), linestyle="none", marker="v",
                        markersize=6, mfc="white", mec=PALETTE[arm], mew=1.4,
                        clip_on=False, zorder=6,
                    )
            ax.set_ylim(bottom=y_floor)
            note = []
            if any(limited):
                note.append(
                    "open marks: solve conditioning-limited "
                    "(values, not rankings)"
                )
            if zeroed:
                note.append(
                    "carets at the axis floor: solved to zero "
                    "(no resolvable value)"
                )
            if note:
                # Lower left: the curves fall to the right, so a note in the
                # lower-right corner would sit on the marks it describes.
                ax.annotate(
                    "\n".join(note),
                    xy=(0.02, 0.02), xycoords="axes fraction", ha="left",
                    va="bottom", fontsize=6.5, color="0.35",
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(m_vals)
            ax.set_xticklabels([str(m) for m in m_vals])
            ax.set_xlabel(r"node budget $M$")
            ax.set_title(res["label"], fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(r"$\mathrm{MMD}^2(Q,\pi)$  (lower is better)")
        # One legend under the row rather than inside a panel: with six arms
        # an in-axes legend covers the curves it names, and the entries are
        # shared by all three panels.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, frameon=False, ncol=3, loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
        )
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = f"{out_path}.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"Saved {p}")
        plt.close(fig)


if __name__ == "__main__":
    import argparse  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "visualize"), default="all",
        help="'all' recomputes (tunes SVGD + sweeps) then renders; "
        "'visualize' only re-renders the figure from the saved "
        "diagnostics/dq_vs_svgd.json (CPU, no SVGD sweep, no training).",
    )
    args = parser.parse_args()

    torch.set_default_dtype(DTYPE)
    os.makedirs(DIAG_DIR, exist_ok=True)
    json_path = os.path.join(DIAG_DIR, "dq_vs_svgd.json")

    if args.phase == "visualize":
        with open(json_path) as fh:
            results = json.load(fh)
        print(f"Loaded {json_path} (visualize-only; no recompute).")
    else:
        # The record is written after every case rather than once at the end,
        # so an interrupted run still leaves the finished cases on disk.
        results = {}
        for case in ("lg2", "gmm2", "g8"):
            results[case] = run_case(case)
            with open(json_path, "w") as fh:
                json.dump(results, fh, indent=2)
            print(f"Saved {json_path} ({len(results)}/3 cases)", flush=True)

    render(results, os.path.join(FIG_DIR, "dq_vs_svgd_mmd_vs_M"))
