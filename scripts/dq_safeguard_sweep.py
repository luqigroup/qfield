"""Safeguard-activation sweep.

Evaluation only, CPU only, no training. Measures how often the
deployment-time safeguard rejects the network's moved nodes, and verifies
the pathwise inequality on every sample.

The menu. At an observation and a budget ``M``, given seeds
``z0 ~ rho^{(x)M}``:

  * ``Q_0``  the seeds at equal weights (the i.i.d.-MC floor);
  * ``Q_rw`` the unit-sum-constrained weights at the same seeds;
  * ``Q_mv`` the constrained weights at the network's moved nodes.

Selection is by the computable criterion ``J_hat(Q) = w' K w - 2 w' mu_hat``,
which is evaluated on its own path (an explicit Gram plus two contractions,
no ``c_rho`` anywhere) exactly as a deployed implementation would compute it.
The arm the criterion selects is then scored by the independent
``mmd_sq_batched``. Keeping the two paths separate makes the reported
``MMD^2(Q_safe) <= MMD^2(Q_0)`` a check of the rule rather than an identity.

``Q_0`` can never win: the equal weights ``1/M`` are feasible for the unit-sum
QP solved at the same nodes, so ``J(Q_rw) <= J(Q_0)`` identically. The menu is
therefore formally ``{rw, mv}``; ``Q_0`` is kept in the code as a runtime
assertion (invariant I1). The safeguard is active when ``Q_rw`` beats
``Q_mv``.

Invariants checked on every sample:
  I1  ``MMD^2(Q_rw) <= MMD^2(Q_0)``, the constrained-QP floor lemma.
  I2  ``MMD^2(Q_safe) <= MMD^2(Q_0)``.

A violation aborts (:func:`invariant_failures`), covering every
exact-estimand cell, the primary ones and all four controls, plus the
headline/split reconciliation. On failure the record is written to a
quarantine name, the published one is left untouched, no figure is drawn,
and the run raises. The sampled-estimand cells (banana, ``gamma > 0``) are
reported and not checked: neither inequality is guaranteed pathwise there.

Exactness. All primary targets have an exact kernel mean (``gamma = 0``):
Gaussian, 2-component CNCV GMM, 5-mode GMM, their posterior-whitened
(Mahalanobis) counterparts, and the tomography latent Gaussian. The banana
target's ``mu`` is a finite reference-bank sum, so its selection is only
exact with respect to that bank's empirical measure, and it is reported in a
separate section.

Controls.
  C1  budgets outside the trained ``{32..512}`` range: ``M in
      {16, 1024, 2048}``.
  C2  distribution shift on a closed-form Gaussian target at the trained
      bandwidth: (a) posterior-mean translation by ``k`` posterior standard
      deviations, which leaves the target geometry and hence the floor law
      unchanged and moves only the network's input distribution,
      (b) covariance scaled by ``t^2``.
  C3  cross-target: a Gaussian-trained net scored on the GMM target (and
      vice versa) at the net's own trained bandwidth.
  C4  jitter sensitivity of the constrained solve at the extrapolated
      budgets.

Output: ``dq_safeguard_sweep.json`` (every per-cell record, the per-sample
invariant worsts, and the full design block) plus
``dq_safeguard_activation.{pdf,png}`` in the same directory.

Run (hours at the default --m_max 2048; the M >= 1024 columns are
single-threaded solves):
    CUDA_VISIBLE_DEVICES="" python scripts/dq_safeguard_sweep.py

Reference:
  Owen & Zhou, JASA 2000; Hesterberg 1995 (the two-arm control-variate
  safeguard this instantiates).
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

# Pin CPU before importing torch.
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import glob  # noqa: E402
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

from _figstyle import apply_house_style, despine  # noqa: E402

from qfield.designed_quadrature import (  # noqa: E402
    build_target,
    constrained_weights_batched,
    cross_check_target,
    gram_batched,
    latent_gaussian_target,
    mmd_sq_batched,
    whitened_gmm_cncv_target,
)
from qfield.designed_quadrature.net import (  # noqa: E402
    QuadratureAmortizer,
)

CKPT_ROOT = datadir("checkpoints")
OUT_DIR = datadir("records")

DTYPE = torch.float64
torch.set_num_threads(4)

JITTER = 1e-10
THEOREM_TOL = 1e-9

# A held-out seed stream, disjoint from the training and evaluation streams.
BASE_SEED = 20260612
SWEEP_SEED = BASE_SEED + 130000

# Budgets. C1 is the out-of-range control: one budget below the trained range
# {32,...,512} and two above it.
M_GRID = [16, 32, 64, 128, 256, 512, 1024, 2048]
# The trained range inside that grid.
M_IN_RANGE = [32, 64, 128, 256, 512]
# (repeats, block size) per budget. The (R, M, M) Gram is the memory driver,
# and the block is chosen to hold a roughly 17 MB per-Gram envelope. Runtime,
# not memory, is what this grid costs: the oneMKL workaround below forces
# every solve above M = 128 to run single-threaded, so the three largest
# budgets dominate the wall time.
R_PLAN = {
    16: (512, 128), 32: (512, 128), 64: (512, 128), 128: (128, 32),
    256: (128, 32), 512: (128, 8), 1024: (64, 2), 2048: (64, 1),
}


def r_plan(M: int) -> tuple[int, int]:
    """The replicate plan for a budget, rejecting off-grid budgets loudly.

    Direct ``R_PLAN[M]`` indexing would raise a bare ``KeyError`` deep inside
    a control loop.
    """
    if M not in R_PLAN:
        raise ValueError(
            f"budget M={M} has no replicate plan; the deployed plan covers "
            f"{sorted(R_PLAN)}. An old-grid budget list is probably being "
            "iterated."
        )
    return R_PLAN[M]

C_FLOOR = "#3a7ca5"   # the i.i.d.-MC floor.
C_MOVE = "#ff7f0e"    # the move arm / activation.
C_RW = "#2a9d8f"      # the reweight (safeguard fallback) arm.


# ==========================================================================
# Run discovery.
# ==========================================================================
# The identity token of a run trained on the deployed node-budget grid. Every
# glob carries it beside the bandwidth rule, since several runs of one target
# share a directory prefix.
GRID_TOKEN = "m_list-32-64-128-256-512"

# The twelve canonical single-seed runs:
# (label, glob, target-kind in results.json, d, min n_steps).
CANON = [
    ("gaussian d2",  f"designed_quadrature_gaussian_amortized_target-gaussian_d-2_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*",  "gaussian", 2, 3000),
    ("gaussian d4",  f"dq_gaussian_d4_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*",  "gaussian", 4, 3000),
    ("gaussian d8",  f"dq_gaussian_d8_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*",  "gaussian", 8, 3000),
    ("gaussian d16", f"dq_gaussian_d16_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*", "gaussian", 16, 3000),
    ("gaussian d32", f"dq_gaussian_d32_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*", "gaussian", 32, 3000),
    ("gmm2 d10", f"designed_quadrature_gmmcncv10_amortized*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*", "gmm_cncv", 10, 3000),
    ("gmm2 d32", f"designed_quadrature_gmmcncv32_amortized*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*", "gmm_cncv", 32, 3000),
    ("gmm2 d64", f"designed_quadrature_gmmcncv64_amortized*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*", "gmm_cncv", 64, 3000),
    ("gmm5 d10", f"dq_gmm_d10_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*", "gmm", 10, 3000),
    ("gmm5 d32", f"dq_gmm_d32_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*", "gmm", 32, 3000),
    ("gmm5 d64", f"dq_gmm_d64_*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*", "gmm", 64, 3000),
    ("banana d2",
     f"designed_quadrature_banana2_amortized*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*",
     "banana", 2, 3000),
]


def _recorded_d(res):
    """The latent dimension a run's ``results.json`` records, or ``None``.

    Runs written before the ``d`` field existed store ``d: null`` but still
    carry the target's own descriptors, so the dimension is recovered from
    ``target_meta`` rather than the run being declared unidentifiable. Nothing
    is guessed: a record that carries neither is skipped by the resolver.
    """
    d = res.get("d")
    if d is not None:
        return int(d)
    meta = res.get("target_meta") or {}
    for key in ("mu_post", "mu_u", "mean"):
        if key in meta:
            return int(len(meta[key]))
    return None


def find_run(glob_pat, target, d, n_steps_min, sigma=None, sigma_tol=1e-12):
    """Resolve ONE trained run dir; raise if the filter is not unique.

    Several directories share a name prefix, so the ``(target, d, n_steps)``
    filter, optionally sharpened by the trained bandwidth that distinguishes
    two otherwise identical runs of the same target, is the actual identity
    and must be unique. Ambiguity raises rather than resolving to a
    sorted-first or newest match, because either rule silently returns a
    stale run.

    Args:
        glob_pat: Directory glob under ``data/checkpoints``.
        target: The ``target`` field the run's ``results.json`` records.
        d: The latent dimension (see :func:`_recorded_d`).
        n_steps_min: Minimum recorded training steps, which excludes smoke runs.
        sigma: Optional exact trained bandwidth to match.
        sigma_tol: Absolute tolerance for that match.

    Returns:
        ``(dir, res)`` with ``res["d"]`` filled in, or ``None`` when no run
        matches.

    Raises:
        RuntimeError: If more than one run matches the filter.
    """
    hits = []
    for cdir in sorted(glob.glob(os.path.join(CKPT_ROOT, glob_pat))):
        rj = os.path.join(cdir, "results.json")
        cp = os.path.join(cdir, "checkpoint.pth")
        if not (os.path.exists(rj) and os.path.exists(cp)):
            continue
        with open(rj) as fh:
            r = json.load(fh)
        if r.get("target") != target:
            continue
        if _recorded_d(r) != d:
            continue
        if int(r.get("n_steps", 0)) < n_steps_min:
            continue
        if sigma is not None and abs(float(r["sigma"]) - float(sigma)) > sigma_tol:
            continue
        r = dict(r)
        r["d"] = d
        hits.append((cdir, r))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(
            f"ambiguous run {glob_pat!r} (target={target}, d={d}, "
            f"sigma={sigma}): "
            + ", ".join(os.path.basename(h[0]) for h in hits)
        )
    return hits[0]


def _find_canon(glob_pat, target, d, n_steps_min):
    """The canonical-run special case of :func:`find_run` (no sigma filter)."""
    return find_run(glob_pat, target, d, n_steps_min)


def _multiseed_runs(prefix):
    """All ``dq_ms_{iso,whiten}_*`` dirs, keyed by (config, seed).

    The multi-seed directories carry the seed in the name, so resolution is
    unambiguous by construction.
    """
    out = []
    for cdir in sorted(glob.glob(os.path.join(CKPT_ROOT, prefix + "*"))):
        rj = os.path.join(cdir, "results.json")
        cp = os.path.join(cdir, "checkpoint.pth")
        if not (os.path.exists(rj) and os.path.exists(cp)):
            continue
        with open(rj) as fh:
            r = json.load(fh)
        base = os.path.basename(cdir)
        tag = base[len(prefix):].split("_seed-")[0]
        seed = int(base.split("_seed-")[1].split("_")[0])
        out.append((cdir, r, tag, seed))
    return out


def load_net(state_dict, d) -> QuadratureAmortizer:
    """Rebuild the amortizer (fixed arch across every saved run) + weights."""
    net = QuadratureAmortizer(
        d=d, hidden=64, n_blocks=3, n_heads=4, cond_hidden=32, delta_scale=0.5
    ).to(DTYPE)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def load_state(cdir):
    ck = torch.load(
        os.path.join(cdir, "checkpoint.pth"), map_location="cpu",
        weights_only=False,
    )
    return ck["state_dict"]


# ==========================================================================
# Target rebuild, per family (all deterministic, all closed form).
# ==========================================================================
def target_for(res, whiten: bool):
    """Rebuild a run's target from its ``results.json``.

    Isotropic runs go through :func:`build_target`. Whitened runs are NOT
    covered by ``build_target``; they are rebuilt from the stored
    ``target_meta``:

      * ``whitened_gaussian``    -- the whitened Gaussian is ``N(R mu, I)``
        with ``R = meta['whiten']`` the recorded whitening map, so
        :func:`latent_gaussian_target` reproduces it exactly;
      * ``whitened_cncv_gmm_2comp`` -- rebuilt by the deterministic builder
        :func:`whitened_gmm_cncv_target` and cross-checked against the stored
        whitened means.
    """
    sigma = float(res["sigma"])
    d = int(res["d"])
    if not whiten:
        return build_target(res["target"], sigma, d=d, dtype=DTYPE)
    meta = res["target_meta"]
    kind = meta["kind"]
    if kind == "whitened_gaussian":
        r_mat = torch.tensor(meta["whiten"], dtype=DTYPE)
        mu_post = torch.tensor(meta["mu_post"], dtype=DTYPE)
        mu_u = r_mat @ mu_post
        return latent_gaussian_target(
            mu_u, torch.eye(d, dtype=DTYPE), sigma,
            name="gaussian_whitened", meta={"rebuilt_from": "results.json"},
            dtype=DTYPE,
        )
    if kind == "whitened_cncv_gmm_2comp":
        tgt = whitened_gmm_cncv_target(sigma, d=d, dtype=DTYPE)
        got = np.asarray(tgt.meta["means_whitened"], dtype=float)
        want = np.asarray(meta["means_whitened"], dtype=float)
        gap = float(np.abs(got - want).max())
        if gap > 1e-10:
            raise RuntimeError(
                f"whitened GMM rebuild mismatch at d={d}: {gap:.3e}"
            )
        return tgt
    raise ValueError(f"unhandled whitened target kind {kind!r}")


def _gen(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


# ==========================================================================
# The core measurement.
# ==========================================================================
def _wilson(k, n, z=1.959963984540054):
    """Wilson 95% interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - half), min(1.0, ctr + half))


class _threads:
    """Temporarily set the intra-op thread count.

    Environment workaround. ``torch.linalg.solve`` on a batched float64
    system with matrix size ``M = 256`` and more than one intra-op thread
    emits ``Intel oneMKL ERROR: Parameter 6 was incorrect on entry to
    DLASWP`` and then hangs indefinitely; with one thread the identical call
    returns in milliseconds. LU with partial pivoting is the same algorithm
    either way, and every arm inside a cell uses the same setting, so no
    comparison is affected.
    """

    def __init__(self, n):
        self.n = int(n)

    def __enter__(self):
        self.prev = torch.get_num_threads()
        torch.set_num_threads(self.n)

    def __exit__(self, *exc):
        torch.set_num_threads(self.prev)
        return False


# Above this budget the batched LAPACK solve must run single-threaded.
SOLVE_SERIAL_M = 128


def _criterion(z, w, mu, sigma):
    """The DEPLOYED selection criterion ``J_hat = w' K w - 2 w' mu_hat``.

    Computed on its own path, an explicit Gram plus the two contractions with
    no ``c_rho`` term, rather than as ``mmd_sq_batched`` minus a constant: a
    deployed safeguard never sees ``c_rho``, so the selection is made by the
    same quantity it would use, and the arm it picks is then scored by the
    independent :func:`mmd_sq_batched`.
    """
    K = gram_batched(z, sigma)
    return torch.einsum("rj,rjl,rl->r", w, K, w) - 2.0 * (w * mu).sum(dim=-1)


def sweep_cell(net, target, sigma, d, M, R, block, seed_off, jitter=JITTER):
    """One (run, budget) cell: the three arms on ``R`` held-out seed sets.

    Per sample: the three arms are built, the criterion ``J_hat`` is
    evaluated on its own path for each, the safeguard selects over the
    two-arm menu ``{rw, mv}`` by ``argmin J_hat``, and the selected arm is
    then scored by ``mmd_sq_batched``. Returns the per-sample MMD^2 of the
    floor / reweight / move / selected quadrature, the criterion values, and
    the per-block activation counts.
    """
    keys = ("a0", "arw", "amv", "asafe", "j0", "jrw", "jmv", "sel_mv",
            "w1_rw", "w1_mv")
    acc = {k: [] for k in keys}
    blk, bsz = [], []
    n_blocks = int(math.ceil(R / block))
    nthreads = 1 if M > SOLVE_SERIAL_M else 4
    for b in range(n_blocks):
        rb = min(block, R - b * block)
        bsz.append(rb)
        g = _gen(seed_off + 1000 * M + 7919 * b)
        z0 = target.sampler(rb * M, g).to(DTYPE).reshape(rb, M, d)
        with torch.no_grad(), _threads(nthreads):
            mu0 = target.mu_fn(z0.reshape(rb * M, d)).reshape(rb, M)
            w_eq = torch.full((rb, M), 1.0 / M, dtype=DTYPE)
            a0 = mmd_sq_batched(z0, w_eq, mu0, target.c_rho, sigma)
            w_rw = constrained_weights_batched(z0, mu0, sigma, jitter)
            arw = mmd_sq_batched(z0, w_rw, mu0, target.c_rho, sigma)
            zh = net(z0)
            muh = target.mu_fn(zh.reshape(rb * M, d)).reshape(rb, M)
            w_mv = constrained_weights_batched(zh, muh, sigma, jitter)
            amv = mmd_sq_batched(zh, w_mv, muh, target.c_rho, sigma)
            # The deployment-time rule, on its own numeric path.
            j0 = _criterion(z0, w_eq, mu0, sigma)
            jrw = _criterion(z0, w_rw, mu0, sigma)
            jmv = _criterion(zh, w_mv, muh, sigma)
            sel_mv = jmv <= jrw          # the move arm is the one deployed
            asafe = torch.where(sel_mv, amv, arw)
        for k, v in (("a0", a0), ("arw", arw), ("amv", amv), ("asafe", asafe),
                     ("j0", j0), ("jrw", jrw), ("jmv", jmv),
                     # Signed-weight mass: it scales the solve's coupling
                     # to any non-constant estimand error.
                     ("w1_rw", w_rw.abs().sum(-1)),
                     ("w1_mv", w_mv.abs().sum(-1))):
            acc[k].append(v.numpy())
        acc["sel_mv"].append(sel_mv.numpy())
        blk.append(int((~sel_mv).sum()))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["blocks"] = list(zip(blk, bsz))
    return out


def summarize(res, **tags):
    """Per-cell record: activation, regret, gain, and the two invariants."""
    a0, arw, amv = res["a0"], res["arw"], res["amv"]
    asafe = res["asafe"]
    blocks = res["blocks"]
    n = int(a0.size)
    active = ~res["sel_mv"]                  # the safeguard rejects the move
    k = int(active.sum())
    lo, hi = _wilson(k, n)

    # Invariants. The MMD^2 values are O(1e-3 .. 1e-1); round-off in the
    # quadratic form is ~1e-16 of the term scale, so absolute worsts are the
    # meaningful check. Relative worsts are also recorded.
    denom = np.maximum(np.abs(a0), 1e-30)
    i1_abs = float(np.max(arw - a0))
    i1_rel = float(np.max((arw - a0) / denom))
    i2_abs = float(np.max(asafe - a0))
    i2_rel = float(np.max((asafe - a0) / denom))

    # I1 in the CRITERION (what the rule actually compares): J(w_rw) <= J(1/M),
    # so the equal-weight arm can never be the argmin -- the menu is two-arm.
    j0, jrw, jmv = res["j0"], res["jrw"], res["jmv"]
    q0_wins = int(np.sum(j0 < np.minimum(jrw, jmv) - THEOREM_TOL))
    # Does the criterion pick the same arm the true MMD^2 would? Under an
    # exact estimand it must, up to round-off; a mismatch count > 0 is
    # numerical, not statistical.
    sel_mismatch = int(np.sum(res["sel_mv"] != (amv <= arw)))

    # The selection margin: |J(mv) - J(rw)| is the quantity the criterion
    # noise must be compared against. The 5th percentile is the noise-prone
    # tail.
    margin_j = np.abs(res["jmv"] - res["jrw"])

    gain_mv = 1.0 - amv / denom
    gain_rw = 1.0 - arw / denom
    gain_safe = 1.0 - asafe / denom
    regret = (amv - arw) / denom             # >0 exactly when active

    rec = dict(tags)
    rec.update({
        "M": int(tags["M"]), "n_draws": n, "n_active": k,
        "med_margin_j": float(np.median(margin_j)),
        "p05_margin_j": float(np.percentile(margin_j, 5)),
        "med_w1_rw": float(np.median(res["w1_rw"])),
        "max_w1_rw": float(np.max(res["w1_rw"])),
        "med_w1_mv": float(np.median(res["w1_mv"])),
        "max_w1_mv": float(np.max(res["w1_mv"])),
        "act_rate": k / n, "act_lo": lo, "act_hi": hi,
        "block_act": [c / s for c, s in blocks],
        "block_counts": [c for c, _ in blocks],
        "block_sizes": [s for _, s in blocks],
        "med_A0": float(np.median(a0)),
        "med_Arw": float(np.median(arw)),
        "med_Amv": float(np.median(amv)),
        "med_gain_mv": float(np.median(gain_mv)),
        "med_gain_rw": float(np.median(gain_rw)),
        "med_gain_safe": float(np.median(gain_safe)),
        "mean_gain_mv": float(np.mean(gain_mv)),
        "mean_gain_safe": float(np.mean(gain_safe)),
        "med_regret_active": (
            float(np.median(regret[active])) if k else float("nan")
        ),
        "p95_regret_active": (
            float(np.percentile(regret[active], 95)) if k else float("nan")
        ),
        "max_regret_active": (
            float(np.max(regret[active])) if k else float("nan")
        ),
        "med_gain_kept_inactive": (
            float(np.median(gain_mv[~active])) if k < n else float("nan")
        ),
        "i1_worst_abs": i1_abs, "i1_worst_rel": i1_rel,
        "i1_n_viol": int(np.sum(arw - a0 > THEOREM_TOL)),
        "i2_worst_abs": i2_abs, "i2_worst_rel": i2_rel,
        "i2_n_viol": int(np.sum(asafe - a0 > THEOREM_TOL)),
        "q0_wins": q0_wins,
        "sel_mismatch_vs_true_mmd": sel_mismatch,
        "j0_minus_jrw_min": float(np.min(j0 - jrw)),
        # When inactive the deployed output is the unsafeguarded one.
        "inactive_bit_identical": bool(
            np.all(asafe[~active] == amv[~active]) if k < n else True
        ),
        "n_A0_nonpos": int(np.sum(a0 <= 0.0)),
    })
    return rec


# ==========================================================================
# Run assembly.
# ==========================================================================
def build_run_list(quick=False):
    """(label, family, d, seed, kind, exactness, res, net) for every run."""
    runs = []
    missing = []
    for label, gp, tgt, d, nmin in CANON:
        got = _find_canon(gp, tgt, d, nmin)
        if got is None:
            # Collected rather than raised immediately, so one run reports
            # every missing reference; raised below rather than warned,
            # because a partial roster must not publish a record.
            print(f"  [MISSING canonical] {label}")
            missing.append(label)
            continue
        cdir, res = got
        runs.append({
            "label": label, "group": "canonical", "family": tgt,
            "d": d, "seed": -1, "whiten": False,
            "exact": tgt != "banana", "dir": cdir, "res": res,
        })
    if missing:
        raise RuntimeError(
            f"{len(missing)} canonical run(s) unresolved on the deployed "
            f"grid: {missing}. Retrain them (or fix the CANON glob) before "
            "sweeping; a partial roster must not publish."
        )
    for cdir, res, tag, seed in _multiseed_runs("dq_ms_iso_"):
        runs.append({
            "label": f"{res['target']} d{res['d']} s{seed}",
            "group": "ms_iso", "family": res["target"], "d": int(res["d"]),
            "seed": seed, "whiten": False,
            "exact": res["target"] != "banana", "dir": cdir, "res": res,
        })
    for cdir, res, tag, seed in _multiseed_runs("dq_ms_whiten_"):
        runs.append({
            "label": f"w-{res['target']} d{res['d']} s{seed}",
            "group": "ms_whiten", "family": "w_" + res["target"],
            "d": int(res["d"]), "seed": seed, "whiten": True,
            "exact": True, "dir": cdir, "res": res,
        })
    if quick:
        runs = [r for r in runs if r["group"] == "canonical"]
    return runs


def add_tomography(runs):
    """Append the trained tomography run (latent Gaussian, d = r = 7).

    Resolved by ``scripts/_ckpt.checkpoint_path``; the target is re-derived
    by re-running the run's own deterministic ``_build_problem`` /
    ``_build_target``.
    """
    try:
        from _ckpt import checkpoint_path, experiment_name  # noqa: E402
        from projorg.config import read_config  # noqa: E402
        from projorg import configsdir  # noqa: E402
        import dq_tomography as dqt  # noqa: E402
    except Exception as exc:  # pragma: no cover - optional leg
        print(f"  [tomography] import failed: {exc}")
        return None
    cfg_file = dqt.CONFIG_FILE
    # Resolve the median_mmd run and rebuild its target at the same rule.
    bw = "median_mmd"
    cp = checkpoint_path(cfg_file, bandwidth=bw)
    if not os.path.isfile(cp):
        print(f"  [tomography] checkpoint absent: {cp}")
        return None
    cfg = read_config(os.path.join(configsdir(), cfg_file))
    cfg["bandwidth"] = bw
    cfg["m_list"] = [int(v) for v in str(cfg["m_list"]).split(",")]
    cfg["experiment"] = experiment_name(cfg_file, bandwidth=bw)
    args = argparse.Namespace(**cfg)
    exp = dqt.DQTomography(args)
    sd = torch.load(cp, map_location="cpu", weights_only=False)["state_dict"]
    net = load_net(sd, exp.r)
    return {
        "label": f"tomography r{exp.r}", "group": "tomography",
        "family": "tomography", "d": int(exp.r), "seed": -1,
        "whiten": False, "exact": True, "dir": os.path.dirname(cp),
        "res": {"target": "tomography_latent", "d": int(exp.r),
                "sigma": float(exp.sigma)},
        "_target": exp.target, "_net": net,
    }


# ==========================================================================
# Controls.
# ==========================================================================
def abpde_conditional_leg(m_grid, n_y=60, r_per_y=256, r_big=64):
    """The observation axis: per-held-out-``y*`` exact NW conditionals (ABPDE).

    Every held-out ``y*`` has its own reference
    ``rho(.|y*) = sum_i alpha_i delta_{c_i}``, whose SE kernel mean and
    self-affinity are exact finite weighted sums, so ``gamma = 0`` holds. No
    PDE is solved: the setup reloads the cached joint bank and rebuilds the
    subspace deterministically, and the rebuilt bandwidth is asserted against
    the trained run's own ``results.json``.
    """
    try:
        from _ckpt import checkpoint_path, experiment_name  # noqa: E402
        from projorg import configsdir  # noqa: E402
        from projorg.config import read_config  # noqa: E402
        import dq_abpde as dqa  # noqa: E402
    except Exception as exc:  # pragma: no cover - optional leg
        print(f"  [abpde] import failed: {exc}")
        return []
    cfg_file = dqa.CONFIG_FILE
    cp = checkpoint_path(cfg_file)
    if not os.path.isfile(cp):
        print(f"  [abpde] checkpoint absent: {cp}")
        return []
    cfg = read_config(os.path.join(configsdir(), cfg_file))
    cfg["m_list"] = [int(v) for v in str(cfg["m_list"]).split(",")]
    cfg["experiment"] = experiment_name(cfg_file)
    args = argparse.Namespace(**cfg)
    exp = dqa.DQABPDE(args)
    exp._simulate_bank()
    exp._build_subspace()
    exp.sigma = exp._latent_sigma()
    exp.sigma_y, exp.ess_y = exp._select_sigma_y()

    ck = torch.load(cp, map_location="cpu", weights_only=False)
    res = ck["results"]
    gap = abs(float(res["sigma"]) - exp.sigma)
    if gap > 1e-10:
        raise RuntimeError(
            f"ABPDE rebuild mismatch: sigma {exp.sigma} vs trained "
            f"{res['sigma']} (gap {gap:.3e}) -- the bank / subspace rebuild "
            "is NOT the one the net was trained on"
        )
    net = load_net(ck["state_dict"], exp.r)

    idx = exp.idx_out[:int(n_y)]
    out = []
    bar = tqdm(total=len(idx) * len(m_grid), desc="abpde y*")
    for ti, i in enumerate(idx):
        tgt = exp._nw_target(exp.s_bank[i], name=f"y{int(i)}")
        chk = cross_check_target(tgt, exp.sigma, seed=SWEEP_SEED + 17)
        if chk > 1e-8:
            raise RuntimeError(f"ABPDE NW cross-check {chk:.3e} at y*={int(i)}")
        # Capped at the deployed top and blocked through r_plan: block = R
        # at M = 2048 builds (64, 2048, 2048) float64 Grams plus an attention
        # peak measured in GB.
        for M in [m for m in m_grid if m <= max(M_IN_RANGE)]:
            R = r_big if M > 64 else r_per_y
            blk = min(R, r_plan(M)[1])
            res = sweep_cell(
                net, tgt, exp.sigma, exp.r, M, R, blk,
                SWEEP_SEED + 4100 + 131 * ti,
            )
            out.append(summarize(
                res, M=M, y_index=int(i), label="abpde",
                r=int(exp.r), sigma=float(exp.sigma),
            ))
            bar.update(1)
    bar.close()
    return out


def gaussian_moments(d):
    """The ``(mu_post, Sigma_post)`` behind :func:`gaussian_target` at ``d``.

    ``gaussian_target`` records them in ``meta`` only at ``d = 2``; for the
    CNCV dimension-scaling posterior it keeps just the prior descriptors, so
    the moments are re-derived from the same deterministic builder the target
    itself calls.
    """
    tgt = build_target("gaussian", 1.0, d=d, dtype=DTYPE)
    meta = tgt.meta
    if "mu_post" in meta:
        return (torch.tensor(meta["mu_post"], dtype=DTYPE),
                torch.tensor(meta["Sigma_post"], dtype=DTYPE))
    from qfield.designed_quadrature.target import (  # noqa: E402
        _cncv_gaussian_posterior,
    )
    mu, cov, _ = _cncv_gaussian_posterior(d, 0.3, 20260612, DTYPE)
    return mu, cov


def control_shift(net, base_target, sigma, d, m_list, seed_off, mode, levels):
    """C2: re-score a trained net against a deliberately SHIFTED exact target.

    ``mode='mean'`` translates the Gaussian posterior mean by ``k`` posterior
    standard deviations along the leading coordinate. The SE-MMD and the
    floor law are translation invariant, so the target geometry, and hence
    the achievable gain, is unchanged and only the network's input
    distribution moves off-distribution. ``mode='cov'`` scales the covariance
    by ``t^2``. The bandwidth is held at the trained value and the target is
    rebuilt in closed form, so ``gamma`` stays exactly zero.
    """
    mu0, cov0 = gaussian_moments(d)
    # The level-0 / t-1 rebuild must reproduce the run's own target exactly.
    g_chk = _gen(seed_off + 5)
    z_chk = base_target.sampler(64, g_chk).to(DTYPE)
    ref = latent_gaussian_target(mu0, cov0, sigma, dtype=DTYPE)
    gap = float((ref.mu_fn(z_chk) - base_target.mu_fn(z_chk)).abs().max())
    gap_c = float(abs(ref.c_rho - base_target.c_rho))
    if max(gap, gap_c) > 1e-12:
        raise RuntimeError(
            f"C2 base rebuild mismatch at d={d}: mu {gap:.2e}, c_rho {gap_c:.2e}"
        )
    sd0 = float(torch.sqrt(cov0[0, 0]))
    out = []
    for lv in levels:
        if mode == "mean":
            shift = torch.zeros(d, dtype=DTYPE)
            shift[0] = float(lv) * sd0
            tgt = latent_gaussian_target(mu0 + shift, cov0, sigma, dtype=DTYPE)
        else:
            tgt = latent_gaussian_target(
                mu0, (float(lv) ** 2) * cov0, sigma, dtype=DTYPE
            )
        for M in m_list:
            R, blk = r_plan(M)
            res = sweep_cell(
                net, tgt, sigma, d, M, R, blk, seed_off
            )
            out.append(summarize(
                res, M=M, level=float(lv), mode=mode,
            ))
    return out


# ==========================================================================
# Figure.
# ==========================================================================
def make_figure(cells, ctrl_mean, abpde, path_stem):
    apply_house_style()
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(12.6, 3.5))

    # (a) activation rate vs M: per-family median line + full across-run range.
    fams = sorted({c["family"] for c in cells})
    cmap = plt.get_cmap("viridis")
    for i, fam in enumerate(fams):
        col = cmap(0.08 + 0.84 * i / max(1, len(fams) - 1))
        xs, ys, lo, hi = [], [], [], []
        for M in M_GRID:
            v = [c["act_rate"] for c in cells
                 if c["family"] == fam and c["M"] == M]
            if v:
                xs.append(M)
                ys.append(float(np.median(v)))
                lo.append(float(np.min(v)))
                hi.append(float(np.max(v)))
        axa.fill_between(xs, lo, hi, color=col, alpha=0.14, lw=0)
        axa.plot(xs, ys, marker="o", ms=3.5, lw=1.4, color=col, label=fam)
    axa.axvspan(min(M_IN_RANGE), max(M_IN_RANGE), color=C_FLOOR,
                alpha=0.10, zorder=0)
    axa.set_xscale("log", base=2)
    axa.set_xticks(M_GRID)
    axa.set_xticklabels([str(v) for v in M_GRID])
    axa.set_xlabel(r"node budget $M$")
    axa.set_ylabel("safeguard activation rate")
    axa.set_title("shaded band: trained budget range", fontsize=9.5)
    axa.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    despine(axa)

    # (b) activation rate vs distribution shift (control C2, mean translation).
    lv = sorted({c["level"] for c in ctrl_mean})
    ms = sorted({c["M"] for c in ctrl_mean})
    for j, M in enumerate(ms):
        col = plt.get_cmap("magma")(0.15 + 0.6 * j / max(1, len(ms) - 1))
        ys = [float(np.median([c["act_rate"] for c in ctrl_mean
                               if c["level"] == v and c["M"] == M]))
              for v in lv]
        axb.plot(lv, ys, marker="s", ms=3.5, lw=1.4, color=col,
                 label=rf"$M={M}$")
    axb.set_xlabel(r"posterior-mean translation (posterior std)")
    axb.set_ylabel("safeguard activation rate")
    axb.set_title("out-of-distribution probe", fontsize=9.5)
    axb.legend(frameon=False, fontsize=7, loc="best")
    despine(axb)

    # (c) the observation axis: per-y* activation, observations sorted by it.
    if abpde:
        ys_idx = sorted({c["y_index"] for c in abpde})
        mids = [m for m in M_IN_RANGE if any(c["M"] == m for c in abpde)]
        rank_by = {y: np.mean([c["act_rate"] for c in abpde
                               if c["y_index"] == y and c["M"] in mids])
                   for y in ys_idx}
        order = sorted(ys_idx, key=lambda y: -rank_by[y])
        # Four log-spread budgets of M_GRID, two in range and two outside.
        show = [m for m in (16, 64, 256, 1024) if any(c["M"] == m
                                                      for c in abpde)]
        for j, M in enumerate(show):
            col = plt.get_cmap("magma")(0.1 + 0.65 * j / max(1, len(show) - 1))
            lut = {c["y_index"]: c["act_rate"] for c in abpde if c["M"] == M}
            axc.plot(range(len(order)), [lut[y] for y in order], marker="o",
                     ms=2.6, lw=1.0, color=col, label=rf"$M={M}$")
        axc.set_xlabel("held-out observation, ranked by activation")
        axc.set_ylabel("safeguard activation rate")
        axc.set_title("observation axis (ABPDE, legacy)", fontsize=9.5)
        axc.legend(frameon=False, fontsize=7, loc="best")
    else:
        axc.text(0.5, 0.5, "observation axis not run", ha="center",
                 va="center", transform=axc.transAxes, fontsize=8,
                 color="0.45")
    despine(axc)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = f"{path_stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved to {p}")
    plt.close(fig)


# ==========================================================================
# Main.
# ==========================================================================
def invariant_failures(payload):
    """Every condition this sweep must satisfy before it writes a record.

    I1 and I2 are bug detectors on an inequality that holds by construction
    wherever the estimand is exact, so a nonzero count is an implementation
    defect and everything downstream is void. The offending cells are named,
    because a scalar count does not say where to look.

    The check covers every cell whose estimand is exact: the primary
    ``exact`` cells and all four controls, each of which rebuilds a
    closed-form Gaussian or GMM target (C2 asserts the rebuild to 1e-12
    itself). The sampled-estimand cells (banana, ``gamma > 0``) are reported
    and not checked: the constrained solve minimizes an estimated objective
    there, so neither inequality is guaranteed pathwise and a violation is a
    statement about the bank, not about the code.
    """
    fails = []
    cells = payload["cells"]
    exact_cells = [c for c in cells if c["exact"]]
    approx_cells = [c for c in cells if not c["exact"]]
    gated = list(exact_cells)
    for key in ("control_mean_shift", "control_cov_scale",
                "control_cross_target", "control_jitter"):
        gated += payload.get(key, [])

    def _name(c):
        bits = [str(c.get("label", "?")), f"M={c.get('M')}"]
        for k in ("level", "mode", "eval_target", "jitter"):
            if k in c:
                bits.append(f"{k}={c[k]}")
        return " ".join(bits)

    for field, what in (
        ("i1_n_viol", "I1 (MMD^2(Q_rw) <= MMD^2(Q_0), the constrained-QP "
                      "floor lemma)"),
        ("i2_n_viol", "I2 (MMD^2(Q_safe) <= MMD^2(Q_0), eq:t4-eq-safeguard "
                      "at gamma = 0)"),
        ("q0_wins", "Q0 wins (the equal weights beat the QP solved at the "
                    "same nodes)"),
        ("sel_mismatch_vs_true_mmd", "criterion-vs-true-MMD selection "
                                     "mismatch"),
    ):
        bad = [c for c in gated if int(c.get(field, 0))]
        if bad:
            fails.append(
                f"{sum(int(c[field]) for c in bad)} draws violate {what} "
                f"across {len(bad)} exact-estimand cells: "
                + ", ".join(f"{_name(c)} (n={c[field]})" for c in bad[:12])
            )

    bad = [c for c in gated if int(c.get("n_A0_nonpos", 0))]
    if bad:
        fails.append(
            "MMD^2(Q_0) <= 0 on some draws, so every relative gain divided by "
            "it is meaningless: "
            + ", ".join(f"{_name(c)} (n={c['n_A0_nonpos']})" for c in bad[:12])
        )
    bad = [c for c in gated if not c.get("inactive_bit_identical", True)]
    if bad:
        fails.append(
            "P6 fails -- on an INACTIVE draw the emitted arm is not bitwise "
            "the unsafeguarded one: " + ", ".join(_name(c) for c in bad[:12])
        )

    # The exact / sampled split, reconciled against the headline. A cell in
    # neither half, or a headline count that is not the sum of the half it
    # names, means the sections do not partition what was measured.
    if len(exact_cells) + len(approx_cells) != len(cells):
        fails.append(
            f"exact/sampled split does not reconcile: {len(exact_cells)} + "
            f"{len(approx_cells)} != {len(cells)} cells"
        )
    nd_exact = sum(int(c["n_draws"]) for c in exact_cells)
    if int(payload["headline"]["n_draws_exact"]) != nd_exact:
        fails.append(
            f"headline n_draws_exact {payload['headline']['n_draws_exact']} "
            f"!= {nd_exact} summed over the exact cells"
        )
    for field, cellfield in (("i1_violations", "i1_n_viol"),
                             ("i2_violations", "i2_n_viol"),
                             ("q0_wins", "q0_wins")):
        got = int(payload["headline"][field])
        want = sum(int(c[cellfield]) for c in exact_cells)
        if got != want:
            fails.append(
                f"headline {field} = {got} but the exact cells sum to {want}"
            )
    off = [c for c in cells
           if int(c["n_draws"]) != int(R_PLAN[int(c["M"])][0])]
    if off:
        fails.append(
            f"{len(off)} cells were not scored at their budget's planned "
            "repeat count: " + ", ".join(
                f"{_name(c)} R={c['n_draws']} (plan {R_PLAN[int(c['M'])][0]})"
                for c in off[:12]
            )
        )
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="canonical runs only (smoke test)")
    ap.add_argument("--no_tomography", action="store_true")
    ap.add_argument("--abpde", action="store_true")
    ap.add_argument("--m_max", type=int, default=2048,
                help="top swept budget; 2048 covers the deployed M_GRID incl. the out-of-range {16, 1024, 2048} cells")
    ap.add_argument("--replot", action="store_true",
                    help="re-render the figure from the saved JSON only")
    cli = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if cli.replot:
        with open(os.path.join(OUT_DIR, "dq_safeguard_sweep.json")) as fh:
            saved = json.load(fh)
        # The check is a pure function of the record, so the re-render is
        # checked too: a figure must not be drawn from a failed record.
        fails = invariant_failures(saved)
        if fails:
            raise AssertionError(
                "the saved record violates the invariants and must not be "
                "re-rendered:\n  - " + "\n  - ".join(fails)
            )
        make_figure(
            [c for c in saved["cells"] if c["exact"]],
            saved["control_mean_shift"],
            saved.get("abpde_observation_axis", []),
            os.path.join(OUT_DIR, "dq_safeguard_activation"),
        )
        return
    m_grid = [m for m in M_GRID if m <= cli.m_max]
    t_start = time.time()

    print("[load] runs")
    runs = build_run_list(quick=cli.quick)
    for r in runs:
        r["_net"] = load_net(load_state(r["dir"]), r["d"])
        r["_target"] = target_for(r["res"], r["whiten"])
    if not cli.no_tomography:
        tomo = add_tomography(runs)
        if tomo is not None:
            runs.append(tomo)
    print(f"  {len(runs)} runs")

    # Pre-flight: the explicit MMD must match every target's independent
    # closed form before anything is measured.
    print("[check] target cross-checks")
    worst_xchk = 0.0
    for r in runs:
        gap = cross_check_target(
            r["_target"], float(r["res"]["sigma"]), seed=SWEEP_SEED + 11
        )
        worst_xchk = max(worst_xchk, float(gap))
        if r["exact"] and gap > 1e-8:
            raise RuntimeError(f"cross-check failed for {r['label']}: {gap:.3e}")
    print(f"  worst relative cross-check gap = {worst_xchk:.3e}")

    # ---- main sweep ----------------------------------------------------
    cells = []
    bar = tqdm(total=len(runs) * len(m_grid), desc="sweep")
    for r in runs:
        sigma = float(r["res"]["sigma"])
        for M in m_grid:
            R, blk = r_plan(M)
            res = sweep_cell(
                r["_net"], r["_target"], sigma, r["d"], M, R, blk, SWEEP_SEED
            )
            cells.append(summarize(
                res, M=M, label=r["label"],
                group=r["group"], family=r["family"], d=r["d"],
                seed=r["seed"], sigma=sigma, exact=r["exact"],
            ))
            bar.update(1)
    bar.close()

    exact_cells = [c for c in cells if c["exact"]]
    approx_cells = [c for c in cells if not c["exact"]]

    # ---- controls ------------------------------------------------------
    print("[control C2] distribution shift (Gaussian, exact, trained sigma)")
    shift_runs = [r for r in runs
                  if r["family"] == "gaussian" and r["group"] == "canonical"]
    ctrl_mean, ctrl_cov = [], []
    for r in shift_runs:
        sigma = float(r["res"]["sigma"])
        base = r["_target"]
        for rec in control_shift(
            # Bottom / middle / top of the training grid.
            r["_net"], base, sigma, r["d"], [32, 128, 512],
            SWEEP_SEED + 3100, "mean", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
        ):
            rec["label"] = r["label"]
            ctrl_mean.append(rec)
        for rec in control_shift(
            r["_net"], base, sigma, r["d"], [32, 128, 512],
            SWEEP_SEED + 3200, "cov", [1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
        ):
            rec["label"] = r["label"]
            ctrl_cov.append(rec)

    print("[control C3] cross-target (net's own trained bandwidth)")
    ctrl_cross = []
    by_label = {r["label"]: r for r in runs}
    cross_pairs = [
        ("gmm2 d10", "gaussian", 10), ("gaussian d32", "gmm_cncv", 32),
        ("gmm2 d32", "gaussian", 32), ("gaussian d8", "gmm_cncv", 8),
    ]
    for lab, other, dd in cross_pairs:
        if lab not in by_label:
            continue
        r = by_label[lab]
        if int(r["d"]) != dd:
            continue
        sigma = float(r["res"]["sigma"])
        tgt = build_target(other, sigma, d=dd, dtype=DTYPE)
        for M in (32, 128, 512):  # grid bottom/mid/top
            R, blk = r_plan(M)
            res = sweep_cell(
                r["_net"], tgt, sigma, dd, M, R, blk, SWEEP_SEED + 3300
            )
            ctrl_cross.append(summarize(
                res, M=M, label=lab, eval_target=other, d=dd,
            ))

    print("[control C4] jitter sensitivity, in-range anchor + extrapolated")
    ctrl_jit = []
    # gaussian d4 is included because its M=512 cell is the one cell past
    # the conditioning limit.
    jit_labels = ["gaussian d2", "gaussian d4", "gaussian d32", "gmm2 d10"]
    for lab in jit_labels:
        if lab not in by_label:
            continue
        r = by_label[lab]
        sigma = float(r["res"]["sigma"])
        # 512 is the in-range anchor, the top trained budget; 1024 and 2048
        # are the extrapolated cells.
        for M in (512, 1024, 2048):
            if M > cli.m_max:
                continue
            R, blk = r_plan(M)
            # The three jitter cells share seed_off at fixed M, so z0 and zh
            # are identical across jitters and the contrast is paired. The
            # per-sample flip count against the baseline jitter also catches
            # +k/-k flips that cancel in the cell-level rate.
            base_sel = None
            for jit in (1e-10, 1e-8, 1e-6):
                res = sweep_cell(
                    r["_net"], r["_target"], sigma, r["d"], M, R, blk,
                    SWEEP_SEED, jitter=jit,
                )
                if base_sel is None:
                    base_sel = res["sel_mv"].copy()
                rec = summarize(res, M=M, label=lab, jitter=jit)
                rec["n_flips_vs_deployed_jitter"] = int(
                    np.sum(res["sel_mv"] != base_sel)
                )
                ctrl_jit.append(rec)

    # ---- the observation axis ------------------------------------------
    abpde = []
    if cli.abpde:
        print("[abpde] conditional leg (exact NW estimand per held-out y*)")
        try:
            abpde = abpde_conditional_leg(m_grid)
        except Exception as exc:
            print(f"  [abpde] FAILED: {exc}")

    # ---- verdicts ------------------------------------------------------
    print("\n=== INVARIANTS (exact-estimand cells) ===")
    i1v = sum(c["i1_n_viol"] for c in exact_cells)
    i2v = sum(c["i2_n_viol"] for c in exact_cells)
    q0w = sum(c["q0_wins"] for c in exact_cells)
    nd = sum(c["n_draws"] for c in exact_cells)
    print(f"draws={nd}  I1 violations={i1v}  I2 violations={i2v}  "
          f"Q0 wins={q0w}")
    print(f"worst I1 abs={max(c['i1_worst_abs'] for c in exact_cells):.3e}  "
          f"worst I2 abs={max(c['i2_worst_abs'] for c in exact_cells):.3e}")
    print(f"criterion-vs-true-MMD selection mismatches="
          f"{sum(c['sel_mismatch_vs_true_mmd'] for c in exact_cells)}  "
          f"min(J0 - Jrw)={min(c['j0_minus_jrw_min'] for c in exact_cells):.3e}")

    print("\n=== ACTIVATION (exact) by budget ===")
    for M in m_grid:
        v = [c["act_rate"] for c in exact_cells if c["M"] == M]
        g = [c["med_gain_mv"] for c in exact_cells if c["M"] == M]
        print(f"  M={M:>3}: median act={np.median(v):.4f} "
              f"max act={np.max(v):.4f} n_cells={len(v)} "
              f"median move gain={np.median(g):+.4f}")

    print("\n=== P4: activation vs move headroom (across exact cells) ===")
    xa = np.array([c["act_rate"] for c in exact_cells])
    xg = np.array([c["med_gain_mv"] for c in exact_cells])
    ok = np.isfinite(xa) & np.isfinite(xg)
    pear = float(np.corrcoef(xa[ok], xg[ok])[0, 1])
    rk = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    spear = float(np.corrcoef(rk(xa[ok]), rk(xg[ok]))[0, 1])
    print(f"  pearson={pear:+.3f}  spearman={spear:+.3f}  n={int(ok.sum())}")

    print("\n=== C2 mean-shift activation (median over Gaussian runs) ===")
    for lv in sorted({c["level"] for c in ctrl_mean}):
        for M in (32, 128, 512):  # grid bottom/mid/top
            v = [c["act_rate"] for c in ctrl_mean
                 if c["level"] == lv and c["M"] == M]
            if v:
                print(f"  shift={lv:>4} M={M:>2}: act={np.median(v):.4f}")

    print("\n=== C2 cov-scale activation ===")
    for lv in sorted({c["level"] for c in ctrl_cov}):
        for M in (32, 128, 512):  # grid bottom/mid/top
            v = [c["act_rate"] for c in ctrl_cov
                 if c["level"] == lv and c["M"] == M]
            if v:
                print(f"  t={lv:>4} M={M:>2}: act={np.median(v):.4f}")

    print("\n=== C3 cross-target activation ===")
    for c in ctrl_cross:
        print(f"  {c['label']:>12} -> {c['eval_target']:>9} M={c['M']:>2}: "
              f"act={c['act_rate']:.4f}  gain_mv={c['med_gain_mv']:+.4f}")

    print("\n=== C4 jitter at extrapolated budgets ===")
    for c in ctrl_jit:
        print(f"  {c['label']:>12} M={c['M']:>3} jitter={c['jitter']:.0e}: "
              f"act={c['act_rate']:.4f} I1abs={c['i1_worst_abs']:.2e} "
              f"I2abs={c['i2_worst_abs']:.2e} gain_mv={c['med_gain_mv']:+.4f}")

    if abpde:
        print("\n=== OBSERVATION AXIS (ABPDE NW conditionals; legacy, "
              "non-paper) ===")
        n_y = len({c["y_index"] for c in abpde})
        print(f"  {n_y} held-out y*, exact NW estimand (gamma = 0)")
        for M in m_grid:
            v = [c["act_rate"] for c in abpde if c["M"] == M]
            g = [c["med_gain_mv"] for c in abpde if c["M"] == M]
            if not v:
                continue
            pooled_k = sum(c["n_active"] for c in abpde if c["M"] == M)
            pooled_n = sum(c["n_draws"] for c in abpde if c["M"] == M)
            lo, hi = _wilson(pooled_k, pooled_n)
            n_y_any = sum(1 for c in abpde if c["M"] == M and c["n_active"] > 0)
            print(f"  M={M:>3}: pooled act={pooled_k / pooled_n:.4f} "
                  f"[{lo:.4f},{hi:.4f}]  per-y* max={np.max(v):.4f}  "
                  f"y* with any activation={n_y_any}/{len(v)}  "
                  f"median move gain={np.median(g):+.4f}")
        ai1 = sum(c["i1_n_viol"] for c in abpde)
        ai2 = sum(c["i2_n_viol"] for c in abpde)
        print(f"  I1 violations={ai1}  I2 violations={ai2}  "
              f"Q0 wins={sum(c['q0_wins'] for c in abpde)}  "
              f"draws={sum(c['n_draws'] for c in abpde)}")

    if approx_cells:
        print("\n=== SAMPLED-ESTIMAND (banana; gamma > 0, reported apart) ===")
        for M in m_grid:
            v = [c["act_rate"] for c in approx_cells if c["M"] == M]
            if v:
                print(f"  M={M:>3}: median act={np.median(v):.4f} "
                      f"I2viol={sum(c['i2_n_viol'] for c in approx_cells if c['M'] == M)}")

    # ---- outputs -------------------------------------------------------
    payload = {
        "design": {
            "menu": ["Q0 equal weights", "Q_rw constrained at seeds",
                     "Q_mv constrained at net nodes"],
            "criterion": "J_hat = w' K w - 2 w' mu (c_rho cancels)",
            "scorer": "mmd_sq_batched (= J_hat + c_rho; ranks identically)",
            "m_grid": m_grid,
            "trained_range": [min(M_IN_RANGE), max(M_IN_RANGE)],
            "r_plan": {str(k): v for k, v in R_PLAN.items()},
            "sweep_seed": SWEEP_SEED, "jitter": JITTER,
            "theorem_tol": THEOREM_TOL,
            "worst_cross_check_rel": worst_xchk,
            "n_runs": len(runs),
        },
        "cells": cells,
        "control_mean_shift": ctrl_mean,
        "control_cov_scale": ctrl_cov,
        "control_cross_target": ctrl_cross,
        "control_jitter": ctrl_jit,
        "abpde_observation_axis": abpde,
        "headline": {
            "n_draws_exact": nd, "i1_violations": i1v, "i2_violations": i2v,
            "q0_wins": q0w,
            "pearson_act_vs_gain": pear, "spearman_act_vs_gain": spear,
        },
        "runtime_secs": time.time() - t_start,
    }
    # On failure the record goes to a quarantine name, not the published
    # one: the sweep costs hours and its per-cell fields are the only way to
    # triage, but a failed run must not be consumed as a result.
    fails = invariant_failures(payload)
    out = os.path.join(OUT_DIR, "dq_safeguard_sweep.json")
    if fails:
        quarantine = out.replace(".json", ".FAILED.json")
        with open(quarantine, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nSaved the FAILED record to {quarantine} for triage; {out} "
              "was NOT overwritten and no figure was drawn.")
        raise AssertionError(
            "the sweep violates its own invariants and must not publish a "
            "record:\n  - " + "\n  - ".join(fails)
        )

    make_figure(exact_cells, ctrl_mean, abpde,
                os.path.join(OUT_DIR, "dq_safeguard_activation"))
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nSaved to {out}")
    print(f"total {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
