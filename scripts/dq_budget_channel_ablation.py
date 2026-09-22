"""Budget-channel ablation: the full-grid network retrained without the
explicit budget input.

The experiment measures whether the EXPLICIT budget input contributes anything
to the fitted map on the two closed-form toys. The primary endpoint is the
PAIRED per-instance contrast noexpl vs intact on shared held-out sample sets.

Arms per (target, replicate), paired by common random numbers (same net-init
seed, same training stream; the forward pass consumes no RNG, so both arms see
bit-identical (M, z0) training batches and differ only in the dead column, and
a runtime fingerprint asserts it):

  * ``intact``  -- ``budget_input="log_m"``.
  * ``noexpl``  -- ``budget_input=None``: the log-M column zeroed, the spread
    features kept, the capacity unchanged, and the column's weights frozen.

Run:
    CUDA_VISIBLE_DEVICES="" python scripts/dq_budget_channel_ablation.py

Re-render only, from the saved sidecar and without training:
    CUDA_VISIBLE_DEVICES="" python scripts/dq_budget_channel_ablation.py \
        --phase visualize

Stitch a killed run's partials into the sidecar and figure, without training:
    CUDA_VISIBLE_DEVICES="" python scripts/dq_budget_channel_ablation.py \
        --phase assemble

``--fallback`` runs the reduced schedule of 2 gaussian and 3 gmm replicates.

Per-arm state dicts and atomic partial JSONs are written under the log
directory, so a kill loses at most one arm and a relaunch reuses everything
already trained.

Reference:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (the per-instance MMD-optimal
    designed quadrature the network amortizes across M).
"""

from __future__ import annotations
from projorg import gitdir, logsdir, plotsdir  # noqa: E402

# CPU only: pin before importing torch.
import os  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sps  # noqa: E402

from qfield.designed_quadrature import (  # noqa: E402
    build_target,
    constrained_weights_batched,
    cross_check_target,
    mmd_sq_batched,
    move_descend_batched,
)
from qfield.designed_quadrature.move import RAY_KMAX  # noqa: E402
from qfield.designed_quadrature.net import QuadratureAmortizer  # noqa: E402
from qfield.kernels import median_heuristic_sigma_sq  # noqa: E402

DTYPE = torch.float64
torch.set_num_threads(4)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = plotsdir("paper")
_DIAG_DIR = os.path.join(_FIG_DIR, "diagnostics")
_LOG_DIR = logsdir("budget_channel")
_E6_JSON = os.path.join(_FIG_DIR, "dq_resolution_ablation.json")
_OUT_JSON = os.path.join(_DIAG_DIR, "dq_budget_channel_ablation.json")
_STATUS = os.path.join(_LOG_DIR, "status.txt")

# ---- experiment constants ----
TARGETS = ["gaussian", "gmm"]
TARGET_TITLE = {"gaussian": "Gaussian (unimodal)", "gmm": "GMM (5-mode)"}
D = 2
N_REF = 20000
N_MU = 4000
JITTER = 1e-10
SIGMA_N_SAMPLE = 4000
HIDDEN = 64
N_BLOCKS = 3
N_HEADS = 4
COND_HIDDEN = 32
DELTA_SCALE = 0.5
BATCH_SIZE = 64
N_STEPS = 4000
LR = 3e-4
EVAL_REPEATS = 256
ORACLE_LR = 0.05
ORACLE_ITERS = 150
BASE_SEED = 20260612
NET_SEED = 0
M_EVAL = [32, 64, 128, 256, 512]
M_TRAIN = [32, 64, 128, 256, 512]  # the full-grid schedule, both arms
_SALT = {"gaussian": 7000, "gmm": 14000}

# ---- arm and seed offsets ----
ARMS = ("intact", "noexpl")
BUDGET_INPUT = {"intact": "log_m", "noexpl": None}
# Common random numbers: BOTH arms of a replicate share the net seed and the
# stream offset. Replicate 0 is the primary.
REP_NET_OFF = [0, 41, 83]
REP_STREAM_OFF = [101, 606, 707]
VAL_OFF = 505          # the fixed val batches' generator
SUPERSET_OFF = 909     # the implicit-inference probe's fixed superset
FRESH_EVAL_OFF = 60000  # robustness re-score (standard eval band is 50000)
VAL_EVERY = 200
VAL_BATCH = 64
CLAMP_M = 128          # the sensitivity probe clamps the budget input here
BOOT_B = 10000
BOOT_SEED = 20260612
TOL_CONTROL = 1e-12    # relative tolerance for the control checks
PLATEAU_WINDOW = 500   # steps in the plateau criterion's window
PLATEAU_TOL = 0.01
PLATEAU_NOISE_MULT = 10.0  # a val pair counts only 10x above its noise floor
DISP_SHARE_MIN = 0.1   # verdict gate (d): displacement must carry >= 10%
USABLE_FRAC_MIN = 0.9  # verdict gate (c): >= 90% of pairs above the floor
ALPHA = 0.05
# The reduced schedule: fewer gaussian replicates.
FALLBACK_REPS = {"gaussian": 2, "gmm": 3}


# ------------------------------------------------------------------ helpers
def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=gitdir(),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _status_append(line: str) -> None:
    """Append a line to the driver's status file (if the log dir exists)."""
    try:
        with open(_STATUS, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _median_heuristic_sigma(target, n_sample: int, seed: int) -> float:
    """True median-heuristic SE bandwidth."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z = target.sampler(int(n_sample), gen)
    return math.sqrt(median_heuristic_sigma_sq(z))


def _draw_batch(target, M: int, B: int, gen: torch.Generator) -> torch.Tensor:
    """A fresh batch of ``B`` i.i.d. ``M``-sample sets, shape (B, M, d)."""
    return torch.stack(
        [target.sampler(int(M), gen) for _ in range(int(B))], dim=0
    )


def _emit_mmd_batched(net, z0, target, sigma, jitter):
    """Per-set exact SE-MMD^2 of the emitted quadrature."""
    B, M, d = z0.shape
    z = net(z0)
    mu = target.mu_fn(z.reshape(B * M, d)).reshape(B, M)
    w = constrained_weights_batched(z, mu, sigma, jitter)
    return mmd_sq_batched(z, w, mu, target.c_rho, sigma)


def _median_iqr(vals):
    """Median and quartiles of a list of values."""
    t = torch.tensor(vals, dtype=DTYPE)
    t = t[torch.isfinite(t)]
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


def _build_net(budget_input) -> QuadratureAmortizer:
    return QuadratureAmortizer(
        d=D, hidden=HIDDEN, n_blocks=N_BLOCKS, n_heads=N_HEADS,
        cond_hidden=COND_HIDDEN, delta_scale=DELTA_SCALE,
        budget_input=budget_input,
    ).to(DTYPE)


def _raw_stats(raw: torch.Tensor) -> dict:
    """Per-cell record of one rule's raw (pre-clamp where available) values."""
    med, q1, q3 = _median_iqr(torch.clamp(raw, min=0.0).tolist())
    neg = raw[raw < 0]
    return {
        "median": med, "q1": q1, "q3": q3,
        "raw_min": float(raw.min()),
        "n_negative": int((raw < 0).sum()),
        "neg_floor": float(neg.abs().max()) if neg.numel() else 0.0,
    }


def _rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _control(name: str, got: dict, want: dict, tol: float) -> None:
    """Assert per-key relative agreement with reference values.

    Prints the per-key relative differences before raising, so a uniform tiny
    drift is distinguishable from a localized defect.
    """
    diffs = {k: _rel_diff(got[k], want[k]) for k in want}
    worst = max(diffs.values())
    print(f"[control {name}] worst rel diff {worst:.3e} (tol {tol:.0e})")
    if worst > tol:
        for k in sorted(diffs, key=lambda k: -diffs[k]):
            print(f"  {k}: got {got[k]:.17e} want {want[k]:.17e} "
                  f"rel {diffs[k]:.3e}")
        raise AssertionError(
            f"positive control {name} FAILED (worst rel {worst:.3e} > "
            f"{tol:.0e}). Uniform ~1e-13..1e-9 drift across all keys => "
            "environment change (diagnose; never silently loosen); a "
            "localized miss => genuine harness defect. Do not proceed."
        )


def _fingerprint(M: int, z0: torch.Tensor) -> list:
    """A common-random-numbers tripwire for one training batch."""
    return [int(M), float(z0.sum()), float(z0.abs().sum())]


# ------------------------------------------------------------ eval machinery
def _eval_z0(target, seed_salt: int, band: int) -> dict[int, torch.Tensor]:
    """The shared held-out sample sets, at the standard or a fresh seed band."""
    z0_by_M = {}
    for M in M_EVAL:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(BASE_SEED) + band + seed_salt + 1000 * int(M))
        z0_by_M[M] = _draw_batch(target, M, EVAL_REPEATS, gen)
    return z0_by_M


def _score_nodes_raw(z: torch.Tensor, target, sigma) -> torch.Tensor:
    """UN-clamped deployed-style score of given nodes: solve + exact MMD^2."""
    R, M, d = z.shape
    mu = target.mu_fn(z.reshape(R * M, d)).reshape(R, M)
    w = constrained_weights_batched(z, mu, sigma, JITTER)
    return mmd_sq_batched(z, w, mu, target.c_rho, sigma)


def _eval_references(target, sigma, z0_by_M):
    """Floor, reweight-only, and oracle on the shared sample sets.

    ``oracle`` is ``move_descend_batched``'s own internally CLAMPED value.
    Because the clamp makes it structurally blind to negative excursions, the
    numerical-floor instrument instead reads ``oracle_rescored``: the returned
    nodes re-scored the deployed way with NO clamp (one extra solve per cell),
    so the deepest rule contributes its true cancellation amplitude.
    """
    out = {"floor": {}, "reweight": {}, "oracle": {}, "oracle_rescored": {},
           "oracle_disp": {}}
    for M in M_EVAL:
        z0 = z0_by_M[M]
        R = z0.shape[0]
        mu0 = target.mu_fn(z0.reshape(R * M, D)).reshape(R, M)
        w_eq = torch.full((R, M), 1.0 / M, dtype=DTYPE)
        out["floor"][M] = mmd_sq_batched(z0, w_eq, mu0, target.c_rho, sigma)
        w_rw = constrained_weights_batched(z0, mu0, sigma, JITTER)
        out["reweight"][M] = mmd_sq_batched(z0, w_rw, mu0, target.c_rho, sigma)
        z_mv, orc = move_descend_batched(
            z0, target.mu_fn, target.c_rho, sigma, ORACLE_LR, ORACLE_ITERS,
            JITTER,
        )
        out["oracle"][M] = orc
        out["oracle_rescored"][M] = _score_nodes_raw(z_mv, target, sigma)
        # Per-particle displacement (comparable to the implicit probe's
        # mean |dz_j|; a flattened Frobenius norm would grow ~sqrt(M)
        # mechanically and mislead the M-dependence read).
        disp = (z_mv - z0).norm(dim=-1).mean(dim=1)  # (R,)
        out["oracle_disp"][M] = float(disp.median())
    return out


def _eval_net(net, target, sigma, z0_by_M):
    """Per-M raw MMD^2 tensors for one net on the shared sample sets."""
    out = {}
    with torch.no_grad():
        for M in M_EVAL:
            out[M] = _emit_mmd_batched(net, z0_by_M[M], target, sigma, JITTER)
    return out


def _film_norms(net, z0: torch.Tensor) -> dict:
    """Per-block FiLM (gamma, beta) norms for one batch of seed sets."""
    from qfield.designed_quadrature.net import _set_features

    with torch.no_grad():
        feats = _set_features(z0)
        if net.budget_input is None:
            feats = torch.cat(
                [torch.zeros_like(feats[:, :1]), feats[:, 1:]], dim=-1
            )
        elif net.budget_input != "log_m":
            feats = torch.cat(
                [torch.full_like(feats[:, :1], float(net.budget_input)),
                 feats[:, 1:]], dim=-1,
            )
        cond = net.cond_mlp(feats)
        out = []
        for block in net.blocks:
            gamma, beta = block.film.lin(cond).chunk(2, dim=-1)
            out.append({"gamma": float(gamma.norm(dim=-1).mean()),
                        "beta": float(beta.norm(dim=-1).mean())})
    return out


# --------------------------------------------------------------- training
def _val_batches(target, seed_salt: int) -> dict[int, torch.Tensor]:
    """The fixed per-budget validation batches, built ONCE.

    Uses a dedicated generator (VAL_OFF), never the training stream: one
    accidental call on the training generator would shift every subsequent
    training batch.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(BASE_SEED) + seed_salt + VAL_OFF)
    return {M: _draw_batch(target, M, VAL_BATCH, gen) for M in M_EVAL}


def _train_arm(target, sigma, seed_salt, label, budget_input, rep,
               val_batches, n_steps: int = N_STEPS, lr: float = LR):
    """Train one arm of one replicate.

    Common random numbers: every arm of a replicate shares the net-init seed
    and the training stream; the forward pass consumes no RNG, so the
    sequences of (M_t, z0-batch_t) are bit-identical across arms, and the
    returned fingerprints assert it at run time. The contingency reuses this
    loop with ``n_steps`` and ``lr`` overridden; the first 4000 steps of an
    8000-step arm consume the identical generator sequence, so the extension
    is the same trajectory trained longer.
    """
    torch.manual_seed(NET_SEED + seed_salt + REP_NET_OFF[rep])
    net = _build_net(budget_input)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(BASE_SEED) + seed_salt + REP_STREAM_OFF[rep])
    m_choices = torch.tensor(list(M_TRAIN))

    if budget_input is None:
        col0 = net.cond_mlp[0].weight[:, 0].detach().clone()

    print(f"[train:{label} r{rep}] M~Uniform{list(M_TRAIN)} steps={n_steps} "
          f"batch={BATCH_SIZE} lr={lr} sigma={sigma:.4f} params={n_params}")
    log_every = max(1, n_steps // 8)
    curve, fps = [], {}
    ema = None
    t0 = time.time()
    net.train()
    for step in range(n_steps):
        M = int(m_choices[torch.randint(len(m_choices), (1,), generator=gen)])
        z0 = _draw_batch(target, M, BATCH_SIZE, gen)
        if step == 0 or step == n_steps - 1:
            fps[str(step)] = _fingerprint(M, z0)
        loss = _emit_mmd_batched(net, z0, target, sigma, JITTER).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        v = float(loss.detach())
        ema = v if ema is None else 0.98 * ema + 0.02 * v
        if (step % VAL_EVERY == 0 or step == n_steps - 1
                or step == n_steps - PLATEAU_WINDOW):
            net.eval()
            with torch.no_grad():
                val = {
                    str(M_): float(
                        _emit_mmd_batched(
                            net, val_batches[M_], target, sigma, JITTER
                        ).mean()
                    )
                    for M_ in M_EVAL
                }
            net.train()
            curve.append({"step": step, "train_ema": ema, "val": val})
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  [{label} r{rep}] step {step:>5d}/{n_steps}  EMA={ema:.4e}")
    wall = time.time() - t0
    print(f"  [{label} r{rep}] train wall-time {wall:.1f} s")
    net.eval()

    if budget_input is None:
        # The ablation must be provably dead: zero input to the first linear
        # layer => exactly zero gradient => the column never moves under Adam.
        assert torch.equal(net.cond_mlp[0].weight[:, 0].detach(), col0), (
            "ablated arm's budget column MOVED -- the ablation was not dead "
            "(weight decay added? flag broken?). Do not trust this run."
        )
        print(f"  [{label} r{rep}] budget column bitwise unchanged from init OK")

    return net, curve, wall, n_params, fps


def _plateau(curve) -> dict:
    """The plateau criterion, floored against evaluator noise.

    Relative val improvement over the last ~PLATEAU_WINDOW steps <
    PLATEAU_TOL, per budget; a budget is read only where BOTH val means sit
    PLATEAU_NOISE_MULT above that budget's own observed noise amplitude (the
    magnitude of the most negative val mean seen along the curve; a val mean
    at the cancellation floor makes the ratio O(1) noise). ``passed`` is None
    when no budget qualifies, that is, indeterminate at evaluator precision.
    """
    last = curve[-1]
    ref_step = last["step"] - PLATEAU_WINDOW
    ref = min(curve, key=lambda c: abs(c["step"] - ref_step))
    noise = {
        M: max(0.0, -min(c["val"][M] for c in curve)) for M in last["val"]
    }
    imps, skipped = {}, []
    for M in ref["val"]:
        a, b = ref["val"][M], last["val"][M]
        thresh = PLATEAU_NOISE_MULT * noise[M]
        if a <= thresh or b <= thresh or a <= 0.0 or b <= 0.0:
            skipped.append(M)
            continue
        imps[M] = (a - b) / a
    worst = max(imps.values()) if imps else None
    return {
        "per_M": imps, "worst": worst,
        "skipped_at_noise": skipped,
        "ref_step_realized": ref["step"],
        "passed": (bool(worst < PLATEAU_TOL) if worst is not None else None),
    }


# --------------------------------------------------------------- statistics
def _boot_seed(tag: str) -> int:
    """A deterministic per-(target, M, contrast) bootstrap seed."""
    return BOOT_SEED ^ zlib.crc32(tag.encode())


def _paired_cell(a: torch.Tensor, b: torch.Tensor, cell_floor: float,
                 tag: str) -> dict:
    """Paired per-draw statistics of a (numerator) vs b (denominator).

    Exact ties are dropped from the sign statistics: the clamp probe's home
    cell at M = CLAMP_M ties on every instance by construction, and counting
    ties as losses would assert an enormous difference between bit-identical
    arms. Censoring (verdict gate c): the log-ratio and the sign test run on
    USABLE pairs, both raw values above the cell's measured numerical floor,
    so sub-floor evaluator noise neither feeds the Holm family nor biases the
    median. Censoring is informative, since pairs drop exactly where a value
    went deepest. Confidence intervals cover sampling uncertainty only.
    """
    a_np, b_np = a.numpy(), b.numpy()
    n = len(a_np)
    tied = a_np == b_np
    usable = (a_np > cell_floor) & (b_np > cell_floor)
    use_eff = usable & ~tied
    n_usable, n_use_eff = int(usable.sum()), int(use_eff.sum())
    wins = int((a_np[use_eff] < b_np[use_eff]).sum())

    if n_use_eff > 0:
        if wins == 0:
            lo = 0.0
        else:
            lo = float(sps.beta.ppf(ALPHA / 2, wins, n_use_eff - wins + 1))
        if wins == n_use_eff:
            hi = 1.0
        else:
            hi = float(sps.beta.ppf(1 - ALPHA / 2, wins + 1, n_use_eff - wins))
        win_frac = wins / n_use_eff
        sign_p = float(sps.binomtest(wins, n_use_eff, 0.5).pvalue)
    else:
        lo, hi, win_frac, sign_p = 0.0, 1.0, None, 1.0

    out = {
        "n": n, "n_ties": int(tied.sum()), "n_usable": n_usable,
        "n_usable_effective": n_use_eff,
        "usable_frac": n_usable / n,
        "win_frac": win_frac, "win_ci": [lo, hi], "sign_p": sign_p,
    }
    if n_usable >= 32:
        lr = np.log(a_np[usable]) - np.log(b_np[usable])
        rng = np.random.default_rng(_boot_seed(tag))
        meds = np.median(
            rng.choice(lr, size=(BOOT_B, n_usable), replace=True), axis=1
        )
        out["median_log_ratio"] = float(np.median(lr))
        out["log_ratio_ci"] = [
            float(np.quantile(meds, ALPHA / 2)),
            float(np.quantile(meds, 1 - ALPHA / 2)),
        ]
    else:
        out["median_log_ratio"] = None
        out["log_ratio_ci"] = None
    if n_usable < n:
        out["at_evaluator_precision"] = (
            "full" if n_usable < 32 else "partial"
        )
    return out


def _holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down; returns per-key rejection at ALPHA."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    reject, still = {}, True
    for i, (k, p) in enumerate(items):
        still = still and (p <= ALPHA / (m - i))
        reject[k] = bool(still)
    return reject


# ------------------------------------------------------------- timing probe
def _timing_probe(target, sigma):
    """~20-step projection of the wall-time before committing to the run.

    An UNDERESTIMATE by construction: it prices training steps and the oracle
    iterations, not the val scoring, the clamp and fresh probes, or the
    reference reweight solves, all of which are much smaller. The projection
    is appended to the status file; past ~24 h the reduced ``--fallback``
    schedule is called out.
    """
    print("[timing probe] 5 training steps per M + 2 oracle iters per M ...")
    torch.manual_seed(999)
    net = _build_net("log_m")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(BASE_SEED) + 999)
    per_step = {}
    for M in M_EVAL:
        t0 = time.time()
        for _ in range(5):
            z0 = _draw_batch(target, M, BATCH_SIZE, gen)
            loss = _emit_mmd_batched(net, z0, target, sigma, JITTER).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        per_step[M] = (time.time() - t0) / 5
    train_per_net = N_STEPS * sum(per_step.values()) / len(per_step)
    oracle_s = 0.0
    for M in M_EVAL:
        z0 = _draw_batch(target, M, EVAL_REPEATS, gen)
        t0 = time.time()
        move_descend_batched(
            z0, target.mu_fn, target.c_rho, sigma, ORACLE_LR, 2, JITTER
        )
        oracle_s += (time.time() - t0) / 2 * ORACLE_ITERS
    return {"per_step_s": {str(M): s for M, s in per_step.items()},
            "train_per_net_s": train_per_net,
            "oracle_per_target_s": oracle_s}


def _project_and_report(timing: dict, reps: dict) -> float:
    n_nets = sum(len(ARMS) * reps[t] for t in TARGETS)
    total_h = (
        n_nets * timing["train_per_net_s"]
        + len(TARGETS) * timing["oracle_per_target_s"]
    ) / 3600
    line = (f"PROJECTED ~{total_h:.1f} h ({n_nets} nets x "
            f"{timing['train_per_net_s'] / 60:.1f} min + oracle "
            f"{timing['oracle_per_target_s'] / 60:.1f} min/target; "
            f"underestimate: excludes val/probe/reference evals)")
    print(f"[timing probe] {line}")
    _status_append(line)
    if total_h > 24:
        warn = ("FALLBACK REQUIRED: projection exceeds 24 h -- relaunch with "
                "--fallback (2 gaussian + 3 gmm replicates) per the design.")
        print(f"[timing probe] {warn}")
        _status_append(warn)
    return total_h


# ------------------------------------------------------------------ per-target
def run_target(tname: str, e6: dict, n_reps: int) -> dict:
    seed_salt = _SALT[tname]
    e6_t = e6[tname]
    print(f"\n########## arch-6 target = {tname} (d={D}, closed form) ##########")

    # Pre-training op order: probe target -> sigma -> rebuild -> cross-check
    # -> references -> training.
    probe = build_target(tname, 0.5, d=D, dtype=DTYPE, n_ref=N_REF, n_mu=N_MU)
    sigma = _median_heuristic_sigma(probe, SIGMA_N_SAMPLE,
                                    BASE_SEED + 1234 + seed_salt)
    print(f"[sigma] median heuristic ({tname}, d={D}): sigma={sigma:.4f}")
    _control("t0:sigma", {"sigma": sigma},
             {"sigma": e6_t["design"]["sigma"]}, TOL_CONTROL)
    target = build_target(tname, sigma, d=D, dtype=DTYPE, n_ref=N_REF,
                          n_mu=N_MU)

    rel = cross_check_target(target, sigma, seed=int(BASE_SEED) + 777 + seed_salt)
    print(f"[metric cross-check] rel={rel:.2e}")
    assert rel < 1e-8, f"t1: metric cross-check failed: rel={rel:.2e}"

    partial_path = os.path.join(_LOG_DIR, f"partial_{tname}.json")
    resumed = {}
    if os.path.exists(partial_path):
        with open(partial_path) as fh:
            resumed = json.load(fh)
        print(f"[resume] found {partial_path} with arms "
              f"{sorted(resumed.get('arms', {}))}")

    z0_by_M = _eval_z0(target, seed_salt, band=50000)

    ref_rules = ("floor", "reweight", "oracle", "oracle_rescored")
    if resumed and all(r in resumed.get("raw", {}) for r in ref_rules):
        # Reuse the expensive oracle from the partial; the floor is cheap and
        # is recomputed below to re-assert the control in this process.
        refs = {
            rule: {M: torch.tensor(resumed["raw"][rule][str(M)], dtype=DTYPE)
                   for M in M_EVAL}
            for rule in ref_rules
        }
        refs["oracle_disp"] = {
            M: resumed["oracle_disp_median"][str(M)] for M in M_EVAL
        }
        print("[resume] references reused from partial (oracle not re-run)")
        for M in M_EVAL:  # cheap floor recompute for a live control
            z0 = z0_by_M[M]
            R = z0.shape[0]
            mu0 = target.mu_fn(z0.reshape(R * M, D)).reshape(R, M)
            w_eq = torch.full((R, M), 1.0 / M, dtype=DTYPE)
            refs["floor"][M] = mmd_sq_batched(z0, w_eq, mu0, target.c_rho,
                                              sigma)
    else:
        t_ref0 = time.time()
        refs = _eval_references(target, sigma, z0_by_M)
        print(f"[references] wall {time.time() - t_ref0:.1f} s; oracle median "
              f"per-particle disp by M: "
              f"{ {M: round(refs['oracle_disp'][M], 4) for M in M_EVAL} }")

    # The floor path is deterministic; assert it exactly.
    _control(
        "t2:floor",
        {f"M{M}": _raw_stats(refs["floor"][M])["median"] for M in M_EVAL},
        {f"M{M}": e6_t["floor"][str(M)][0] for M in M_EVAL},
        TOL_CONTROL,
    )
    # The augmented candidate pool can only lower the minimum, so the oracle
    # is asserted one-sided.
    for M in M_EVAL:
        new = _raw_stats(refs["oracle"][M])["median"]
        old = e6_t["oracle"][str(M)][0]
        assert new <= old * (1 + 1e-9) + 1e-18, (
            f"oracle median ROSE vs record at M={M}: {new:.3e} > {old:.3e}"
        )
    print("[control oracle] one-sided (new <= tracked) OK at every M")

    val_batches = _val_batches(target, seed_salt)
    result = {
        "design": {
            "target": tname, "d": D, "sigma": sigma, "m_eval": M_EVAL,
            "m_train": M_TRAIN, "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
            "lr": LR, "eval_repeats": EVAL_REPEATS, "oracle_lr": ORACLE_LR,
            "oracle_iters": ORACLE_ITERS, "delta_scale": DELTA_SCALE,
            "metric_cross_check_rel": rel, "seed_salt": seed_salt,
            "n_replicates": n_reps,
            "rep_net_off": REP_NET_OFF[:n_reps],
            "rep_stream_off": REP_STREAM_OFF[:n_reps],
            "val_off": VAL_OFF, "val_every": VAL_EVERY, "val_batch": VAL_BATCH,
            "clamp_m": CLAMP_M, "boot_b": BOOT_B, "alpha": ALPHA,
            "plateau_window": PLATEAU_WINDOW, "plateau_tol": PLATEAU_TOL,
            "plateau_noise_mult": PLATEAU_NOISE_MULT,
            "disp_share_min": DISP_SHARE_MIN,
            "usable_frac_min": USABLE_FRAC_MIN,
        },
        "references": {
            rule: {str(M): _raw_stats(refs[rule][M]) for M in M_EVAL}
            for rule in ref_rules
        },
        "oracle_disp_median": {str(M): refs["oracle_disp"][M] for M in M_EVAL},
        "raw": {
            rule: {str(M): refs[rule][M].tolist() for M in M_EVAL}
            for rule in ref_rules
        },
        "arms": {}, "curves": {}, "plateau": {}, "wall_s": {},
        "train_fingerprints": {},
    }
    _atomic_json(partial_path, result)

    nets = {}
    for rep in range(n_reps):
        for arm in ARMS:
            key = f"{arm}_r{rep}"
            sd_path = os.path.join(_LOG_DIR, f"net_{tname}_{key}.pth")
            done = (
                key in resumed.get("arms", {})
                and key in resumed.get("raw", {})
                and os.path.exists(sd_path)
            )
            if done:
                print(f"[resume] {key}: reusing trained arm from partial")
                net = _build_net(BUDGET_INPUT[arm])
                net.load_state_dict(torch.load(sd_path, weights_only=True))
                net.eval()
                result["arms"][key] = resumed["arms"][key]
                result["raw"][key] = resumed["raw"][key]
                result["curves"][key] = resumed["curves"][key]
                result["plateau"][key] = resumed["plateau"][key]
                result["wall_s"][key] = resumed["wall_s"][key]
                result["train_fingerprints"][key] = (
                    resumed["train_fingerprints"][key]
                )
            else:
                net, curve, wall, n_params, fps = _train_arm(
                    target, sigma, seed_salt, arm, BUDGET_INPUT[arm], rep,
                    val_batches,
                )
                result["design"]["n_params"] = n_params
                ev = _eval_net(net, target, sigma, z0_by_M)
                result["arms"][key] = {
                    str(M): _raw_stats(ev[M]) for M in M_EVAL
                }
                result["raw"][key] = {str(M): ev[M].tolist() for M in M_EVAL}
                result["curves"][key] = curve
                result["plateau"][key] = _plateau(curve)
                result["wall_s"][key] = wall
                result["train_fingerprints"][key] = fps
                torch.save(net.state_dict(), sd_path)
            if rep == 0:
                nets[key] = net
            _atomic_json(partial_path, result)

            if arm == "intact" and rep == 0:
                # Control check, before any ablated training.
                _control(
                    "t3:intact_r0_warm",
                    {f"M{M}": result["arms"][key][str(M)]["median"]
                     for M in M_EVAL},
                    {f"M{M}": e6_t["warm"]["all"][str(M)][0] for M in M_EVAL},
                    TOL_CONTROL,
                )
                # Sensitivity probe: the SAME trained net, budget input
                # clamped to log CLAMP_M (net.py's own construction of the
                # constant), state_dict reloaded into a clamped-mode net.
                const = float(torch.log(torch.tensor(float(CLAMP_M))))
                clamp_net = _build_net(const)
                clamp_net.load_state_dict(net.state_dict())
                clamp_net.eval()
                ev_c = _eval_net(clamp_net, target, sigma, z0_by_M)
                result["arms"]["clamp_r0"] = {
                    str(M): _raw_stats(ev_c[M]) for M in M_EVAL
                }
                result["raw"]["clamp_r0"] = {
                    str(M): ev_c[M].tolist() for M in M_EVAL
                }
                _atomic_json(partial_path, result)

        # Common-random-numbers tripwire: both arms of the replicate must
        # have seen the SAME first and last training batches.
        fa = result["train_fingerprints"][f"intact_r{rep}"]
        fb = result["train_fingerprints"][f"noexpl_r{rep}"]
        assert fa == fb, (
            f"CRN VIOLATION at replicate {rep}: the two arms saw different "
            f"training batches ({fa} vs {fb}). The paired reading is void."
        )
        print(f"[CRN r{rep}] first/last training-batch fingerprints match OK")

    # Implicit-inference probe on the ablated net (rep 0): does displacement
    # scale track M with the column dead?  (i) fresh i.i.d. sets = the shared
    # eval draws; (ii) one fixed 512-superset subsampled to each M (spread
    # ~constant, density varying). Per-block FiLM (gamma, beta) norms ride
    # along for both modes and both rep-0 arms.
    gen_s = torch.Generator(device="cpu")
    gen_s.manual_seed(int(BASE_SEED) + seed_salt + SUPERSET_OFF)
    superset = target.sampler(max(M_EVAL), gen_s)  # (512, d)
    probe_out = {}
    for key in ("noexpl_r0", "intact_r0"):
        net_p = nets[key]
        entry = {"fresh": {}, "subsampled": {}, "film_fresh": {},
                 "film_subsampled": {}}
        with torch.no_grad():
            for M in M_EVAL:
                dz = net_p.displacement(z0_by_M[M])
                entry["fresh"][str(M)] = float(dz.norm(dim=-1).mean())
                entry["film_fresh"][str(M)] = _film_norms(net_p, z0_by_M[M])
                zs = superset[:M].unsqueeze(0)
                entry["subsampled"][str(M)] = float(
                    net_p.displacement(zs).norm(dim=-1).mean()
                )
                entry["film_subsampled"][str(M)] = _film_norms(net_p, zs)
        probe_out[key] = entry
    result["implicit_probe"] = probe_out
    print(f"[implicit probe] noexpl mean |dz| fresh: "
          f"{probe_out['noexpl_r0']['fresh']}")

    # Robustness: both rep-0 arms re-scored at a FRESH eval offset; the
    # verdict-unchanged check is computed below in the stats block.
    z0_fresh = _eval_z0(target, seed_salt, band=FRESH_EVAL_OFF)
    fresh_raw = {}
    for M in M_EVAL:
        z0 = z0_fresh[M]
        R = z0.shape[0]
        mu0 = target.mu_fn(z0.reshape(R * M, D)).reshape(R, M)
        w_eq = torch.full((R, M), 1.0 / M, dtype=DTYPE)
        fresh_raw.setdefault("floor", {})[M] = mmd_sq_batched(
            z0, w_eq, mu0, target.c_rho, sigma
        )
    for key in ("intact_r0", "noexpl_r0"):
        ev_f = _eval_net(nets[key], target, sigma, z0_fresh)
        fresh_raw[key] = ev_f
    result["fresh_eval"] = {
        rule: {str(M): _raw_stats(fresh_raw[rule][M]) for M in M_EVAL}
        for rule in fresh_raw
    }
    result["raw_fresh"] = {
        rule: {str(M): fresh_raw[rule][M].tolist() for M in M_EVAL}
        for rule in fresh_raw
    }

    # ------------------------------ paired statistics + the verdict gates
    # (gate a's Holm rejection is filled in later by main(), pooled across
    # BOTH targets' cells.)
    floor_rules = ["floor", "reweight", "oracle_rescored", "clamp_r0"] + [
        f"{arm}_r{r}" for arm in ARMS for r in range(n_reps)
    ]
    stats_block = {}
    for M in M_EVAL:
        raws = {
            r: torch.tensor(result["raw"][r][str(M)], dtype=DTYPE)
            for r in floor_rules
        }
        cell_floor = max(
            float(t[t < 0].abs().max()) if (t < 0).any() else 0.0
            for t in raws.values()
        )
        a, b = raws["noexpl_r0"], raws["intact_r0"]
        rw, fl = raws["reweight"], raws["floor"]
        pfx = f"{tname}:M{M}"
        cell = {
            "cell_floor": cell_floor,
            "noexpl_vs_intact": _paired_cell(a, b, cell_floor,
                                             f"{pfx}:ni"),
            "noexpl_vs_floor": _paired_cell(a, fl, cell_floor, f"{pfx}:nf"),
            "intact_vs_floor": _paired_cell(b, fl, cell_floor, f"{pfx}:if"),
            "noexpl_vs_reweight": _paired_cell(a, rw, cell_floor,
                                               f"{pfx}:nr"),
            "intact_vs_reweight": _paired_cell(b, rw, cell_floor,
                                               f"{pfx}:ir"),
            "clamp_vs_intact": _paired_cell(raws["clamp_r0"], b, cell_floor,
                                            f"{pfx}:ci"),
        }
        # Per-replicate paired contrast; a pair exists in every replicate.
        rep_mlr = []
        for r in range(n_reps):
            pr = _paired_cell(raws[f"noexpl_r{r}"], raws[f"intact_r{r}"],
                              cell_floor, f"{pfx}:ni_r{r}")
            rep_mlr.append(pr["median_log_ratio"])
        cell["replicate_paired_mlr"] = rep_mlr
        signs = {np.sign(v) for v in rep_mlr if v is not None}
        cell["replicate_sign_consistent"] = (
            len(signs) == 1 if len([v for v in rep_mlr if v is not None])
            == n_reps else None
        )
        # gate (d): does the intact arm's displacement carry margin here?
        usable = (b.numpy() > cell_floor) & (rw.numpy() > cell_floor)
        if usable.sum() >= 32:
            share = 1.0 - b.numpy()[usable] / rw.numpy()[usable]
            cell["disp_share_intact"] = float(np.median(share))
        else:
            cell["disp_share_intact"] = None
        cell["solve_dominated"] = (
            cell["disp_share_intact"] is None
            or cell["disp_share_intact"] < DISP_SHARE_MIN
        )
        # gate (b): replicate spread of each arm's per-cell median.
        for arm in ARMS:
            meds = [
                result["arms"][f"{arm}_r{r}"][str(M)]["median"]
                for r in range(n_reps)
            ]
            pos = [m for m in meds if m > cell_floor]
            cell[f"{arm}_replicate_log_spread"] = (
                float(np.log(max(pos)) - np.log(min(pos)))
                if len(pos) == len(meds) else None
            )
        # Fresh-offset verdict-unchanged check: the same-direction paired
        # contrast at the fresh offset.
        f_raws = {
            r: torch.tensor(result["raw_fresh"][r][str(M)], dtype=DTYPE)
            for r in ("floor", "intact_r0", "noexpl_r0")
        }
        f_floor = max(
            float(t[t < 0].abs().max()) if (t < 0).any() else 0.0
            for t in f_raws.values()
        )
        f_pair = _paired_cell(f_raws["noexpl_r0"], f_raws["intact_r0"],
                              f_floor, f"{pfx}:ni_fresh")
        cell["fresh_noexpl_vs_intact"] = f_pair
        m_std, m_fr = (cell["noexpl_vs_intact"]["median_log_ratio"],
                       f_pair["median_log_ratio"])
        cell["fresh_direction_agrees"] = (
            bool(np.sign(m_std) == np.sign(m_fr))
            if (m_std is not None and m_fr is not None) else None
        )
        stats_block[str(M)] = cell
    result["paired_stats"] = stats_block
    result["noexpl_plateau_passed"] = result["plateau"]["noexpl_r0"]["passed"]
    _atomic_json(partial_path, result)
    return result


def _finalize_gates(all_results: dict) -> None:
    """Pool the cells' sign p-values, run Holm once over that family, and
    write the verdict gates back into every cell."""
    pvals = {
        f"{tname}:M{M}":
            all_results["targets"][tname]["paired_stats"][str(M)]
            ["noexpl_vs_intact"]["sign_p"]
        for tname in all_results["targets"] for M in M_EVAL
    }
    holm = _holm(pvals)
    for tname, res in all_results["targets"].items():
        for M in M_EVAL:
            c = res["paired_stats"][str(M)]
            pc = c["noexpl_vs_intact"]
            spread = c["intact_replicate_log_spread"]
            ci = pc["log_ratio_ci"]
            gates = {
                # (a) Holm over the pooled cell family AND the paired CI
                # excludes 0. The sign test is the Holm input, and both run
                # on usable non-tied pairs.
                "a_holm_and_ci": bool(
                    holm[f"{tname}:M{M}"]
                    and ci is not None and (ci[0] > 0 or ci[1] < 0)
                ),
                "b_above_replicate_spread": bool(
                    pc["median_log_ratio"] is not None and spread is not None
                    and abs(pc["median_log_ratio"]) > spread
                ),
                # (c) near-full usability, not just >= 32.
                "c_above_numerical_floor": bool(
                    pc["median_log_ratio"] is not None
                    and pc["usable_frac"] >= USABLE_FRAC_MIN
                ),
                "d_displacement_carries_margin": not c["solve_dominated"],
            }
            c["verdict_gates"] = gates
            c["directional_verdict_allowed"] = all(gates.values())
    all_results["holm_family"] = {
        "cells": sorted(pvals), "alpha": ALPHA, "rejections": holm,
    }


# ------------------------------------------------------------ contingency
# The gaussian contingency: all 3 paired replicates extended to 8000 steps at
# the deployed learning rate, plus the ablated arm at the two sweep rates,
# judged at the same convergence budget. GMM gets no further compute, because
# a solve-dominated cell cannot change.
CONT_TARGET = "gaussian"
CONT_STEPS = 8000
CONT_ARMS = [
    # (label, budget_input, lr)
    ("intact8k", "log_m", LR),
    ("noexpl8k", None, LR),
    ("noexpl8k_lr1e4", None, 1e-4),
    ("noexpl8k_lr1e3", None, 1e-3),
]
_CONT_JSON = os.path.join(_DIAG_DIR, "dq_budget_channel_contingency.json")


def run_contingency() -> None:
    """The gaussian contingency: 8000-step paired replicates plus the
    ablated-arm learning-rate sweep. Reuses the main record's references,
    which are deterministic and training-independent, and its held-out sample
    sets; writes its own record and partials, never touching the main ones.
    """
    seed_salt = _SALT[CONT_TARGET]
    with open(_OUT_JSON) as fh:
        main = json.load(fh)
    main_t = main["targets"][CONT_TARGET]

    probe = build_target(CONT_TARGET, 0.5, d=D, dtype=DTYPE, n_ref=N_REF,
                         n_mu=N_MU)
    sigma = _median_heuristic_sigma(probe, SIGMA_N_SAMPLE,
                                    BASE_SEED + 1234 + seed_salt)
    _control("t0:sigma", {"sigma": sigma},
             {"sigma": main_t["design"]["sigma"]}, TOL_CONTROL)
    target = build_target(CONT_TARGET, sigma, d=D, dtype=DTYPE, n_ref=N_REF,
                          n_mu=N_MU)
    rel = cross_check_target(target, sigma,
                             seed=int(BASE_SEED) + 777 + seed_salt)
    assert rel < 1e-8, f"t1: metric cross-check failed: rel={rel:.2e}"

    z0_by_M = _eval_z0(target, seed_salt, band=50000)
    # Live check against the main record's floor; cheap, and it certifies
    # this process.
    for M in M_EVAL:
        z0 = z0_by_M[M]
        R = z0.shape[0]
        mu0 = target.mu_fn(z0.reshape(R * M, D)).reshape(R, M)
        w_eq = torch.full((R, M), 1.0 / M, dtype=DTYPE)
        fl = mmd_sq_batched(z0, w_eq, mu0, target.c_rho, sigma)
        _control(f"t2:floor:M{M}", {"med": _raw_stats(fl)["median"]},
                 {"med": main_t["references"]["floor"][str(M)]["median"]},
                 TOL_CONTROL)

    val_batches = _val_batches(target, seed_salt)
    partial_path = os.path.join(_LOG_DIR, "partial_contingency.json")
    resumed = {}
    if os.path.exists(partial_path):
        with open(partial_path) as fh:
            resumed = json.load(fh)
        print(f"[resume] found {partial_path} with arms "
              f"{sorted(resumed.get('arms', {}))}")

    result = {
        "design": {
            "target": CONT_TARGET, "sigma": sigma, "n_steps": CONT_STEPS,
            "arms": [[a[0], repr(a[1]), a[2]] for a in CONT_ARMS],
            "n_replicates": len(REP_NET_OFF),
            "amendment": (
                "both arms of each pair extended to 8000 steps under CRN; "
                "sweep judged at the same budget (design: Contingency scope)"
            ),
            "main_record": _OUT_JSON, "head": _git_head(),
        },
        # The references are deterministic and training-independent: carried
        # over from the main record for the stats' cell floors.
        "raw_refs_from_main": ["floor", "reweight", "oracle_rescored"],
        "arms": {}, "curves": {}, "plateau": {}, "wall_s": {}, "raw": {},
        "train_fingerprints": {},
    }
    for rule in ("floor", "reweight", "oracle_rescored"):
        result["raw"][rule] = main_t["raw"][rule]

    for rep in range(len(REP_NET_OFF)):
        for label, budget_input, lr in CONT_ARMS:
            key = f"{label}_r{rep}"
            sd_path = os.path.join(_LOG_DIR, f"net_cont_{key}.pth")
            done = (
                key in resumed.get("arms", {})
                and key in resumed.get("raw", {})
                and os.path.exists(sd_path)
            )
            if done:
                print(f"[resume] {key}: reusing trained arm from partial")
                for field in ("arms", "raw", "curves", "plateau", "wall_s",
                              "train_fingerprints"):
                    result[field][key] = resumed[field][key]
                continue
            net, curve, wall, n_params, fps = _train_arm(
                target, sigma, seed_salt, label, budget_input, rep,
                val_batches, n_steps=CONT_STEPS, lr=lr,
            )
            result["design"]["n_params"] = n_params
            # Cross-record assert: replicate 0's first training batch must
            # be the main run's, which ties the extension to the same
            # trajectory rather than only to the seed construction.
            if rep == 0:
                main_fp = main_t["train_fingerprints"]["intact_r0"]["0"]
                assert fps["0"] == main_fp, (
                    f"contingency r0 first batch differs from the main "
                    f"record: {fps['0']} vs {main_fp}"
                )
            ev = _eval_net(net, target, sigma, z0_by_M)
            result["arms"][key] = {str(M): _raw_stats(ev[M]) for M in M_EVAL}
            result["raw"][key] = {str(M): ev[M].tolist() for M in M_EVAL}
            result["curves"][key] = curve
            result["plateau"][key] = _plateau(curve)
            result["wall_s"][key] = wall
            result["train_fingerprints"][key] = fps
            torch.save(net.state_dict(), sd_path)
            del net
            _atomic_json(partial_path, result)
        # Tripwire: all four arms of the replicate share the stream.
        fp0 = result["train_fingerprints"][f"{CONT_ARMS[0][0]}_r{rep}"]
        for label, _, _ in CONT_ARMS[1:]:
            fp = result["train_fingerprints"][f"{label}_r{rep}"]
            assert fp == fp0, (
                f"CRN VIOLATION at contingency replicate {rep} ({label}): "
                f"{fp} vs {fp0}."
            )
        print(f"[CRN r{rep}] contingency first-batch fingerprints match OK")

    # Paired statistics: each ablated variant vs the extended intact arm.
    stats_block = {}
    n_reps = len(REP_NET_OFF)
    for M in M_EVAL:
        raws = {
            r: torch.tensor(result["raw"][r][str(M)], dtype=DTYPE)
            for r in list(result["raw"])
            if str(M) in result["raw"][r]
        }
        cell_floor = max(
            float(t[t < 0].abs().max()) if (t < 0).any() else 0.0
            for t in raws.values()
        )
        cell = {"cell_floor": cell_floor}
        for label, _, _ in CONT_ARMS[1:]:
            contrast = f"{label}_vs_intact8k"
            per_rep = []
            for r in range(n_reps):
                pr = _paired_cell(
                    raws[f"{label}_r{r}"], raws[f"intact8k_r{r}"],
                    cell_floor, f"cont:{M}:{contrast}:r{r}",
                )
                per_rep.append(pr)
            cell[contrast] = {
                "primary": per_rep[0],
                "replicate_mlr": [p["median_log_ratio"] for p in per_rep],
            }
            signs = {np.sign(v) for p in per_rep
                     if (v := p["median_log_ratio"]) is not None}
            cell[contrast]["replicate_sign_consistent"] = (
                len(signs) == 1
                if all(p["median_log_ratio"] is not None for p in per_rep)
                else None
            )
        stats_block[str(M)] = cell
    result["paired_stats"] = stats_block
    result["plateau_passed"] = {
        key: result["plateau"][key]["passed"] for key in result["plateau"]
    }
    _atomic_json(partial_path, result)
    _atomic_json(_CONT_JSON, result)
    print(f"\nSaved contingency record to {_CONT_JSON}")

    print("\n=== contingency: noexpl8k vs intact8k, per M (r0 | rep mlrs) ===")
    for M in M_EVAL:
        c = stats_block[str(M)]["noexpl8k_vs_intact8k"]
        mlr = c["primary"]["median_log_ratio"]
        print(f"  M={M:>3}: r0 mlr="
              f"{'%.3f' % mlr if mlr is not None else 'CENSORED'} "
              f"reps={[None if v is None else round(v, 3) for v in c['replicate_mlr']]} "
              f"sign_cons={c['replicate_sign_consistent']}")
    print("plateau passed:", result["plateau_passed"])


# ----------------------------------------------------------------- rendering
_C = {
    "floor": "#3a7ca5",     # independent-sample floor
    "reweight": "#4d4d4d",  # solve-only lever (a second, darker grey; the
    #                         solve grey belongs to the oracle)
    "oracle": "#7f7f7f",    # solve grey: per-instance descent reference
    "intact": "#ff7f0e",    # ours: the deployed full-grid net
    "noexpl": "#a63603",    # burnt orange: trained without the budget input
}
_LABEL = {
    "floor": "i.i.d. floor",
    "reweight": "reweight only (solve at $z_0$)",
    "oracle": "per-instance descent (reference)",
    "intact": "full-grid net (deployed)",
    "noexpl": "net trained without the budget input",
}


def _panel(ax, res):
    m_vals = M_EVAL
    n_reps = res["design"]["n_replicates"]
    for rule, ls, mk in (("floor", "--", "o"), ("reweight", "-.", "s"),
                         ("oracle", ":", "^")):
        med = [res["references"][rule][str(M)]["median"] for M in m_vals]
        ax.plot(m_vals, med, color=_C[rule], marker=mk, markersize=4,
                linewidth=1.4, linestyle=ls, label=_LABEL[rule], zorder=2)
    for arm in ARMS:
        med0 = [res["arms"][f"{arm}_r0"][str(M)]["median"] for M in m_vals]
        ax.plot(m_vals, med0, color=_C[arm], linewidth=1.7,
                label=_LABEL[arm], marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        reps = np.array(
            [[res["arms"][f"{arm}_r{r}"][str(M)]["median"] for M in m_vals]
             for r in range(n_reps)]
        )
        ax.fill_between(m_vals, reps.min(axis=0), reps.max(axis=0),
                        color=_C[arm], alpha=0.18, linewidth=0, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(m_vals)
    ax.set_xticklabels([str(m) for m in m_vals])
    ax.set_xlabel(r"node budget $M$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(TARGET_TITLE[res["design"]["target"]], fontsize=10)


def _render(all_results):
    with plt.rc_context(
        {"pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
         "font.size": 9}
    ):
        fig, axes = plt.subplots(1, len(TARGETS),
                                 figsize=(4.6 * len(TARGETS), 4.0),
                                 sharey=False)
        if len(TARGETS) == 1:
            axes = [axes]
        for ax, tname in zip(axes, TARGETS):
            _panel(ax, all_results["targets"][tname])
        axes[0].set_ylabel(r"$\mathrm{MMD}^2(Q, \rho)$ (held out)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
                   loc="lower center", bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(
            "Budget-channel ablation: the full-grid net without the "
            "explicit budget input", fontsize=11, y=1.0,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        os.makedirs(_FIG_DIR, exist_ok=True)
        for ext in ("pdf", "png"):
            path = os.path.join(_FIG_DIR, f"dq_budget_channel_ablation.{ext}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.close(fig)


def _print_summary(all_results):
    for tname in all_results["targets"]:
        res = all_results["targets"][tname]
        print(f"\n=== {tname}: paired noexpl vs intact (r0), per M ===")
        for M in M_EVAL:
            c = res["paired_stats"][str(M)]
            pc = c["noexpl_vs_intact"]
            mlr = pc["median_log_ratio"]
            wf = pc["win_frac"]
            print(
                f"  M={M:>3}: win_frac="
                f"{'%.3f' % wf if wf is not None else 'n/a'} "
                f"CI={pc['win_ci']}, median log-ratio="
                f"{'%.3f' % mlr if mlr is not None else 'CENSORED'}, "
                f"usable={pc['n_usable']}/{pc['n']}, "
                f"solve_dominated={c['solve_dominated']}, "
                f"verdict_allowed={c.get('directional_verdict_allowed')}"
            )


# ----------------------------------------------------------------------- main
def _provenance(timing: dict | None) -> dict:
    return {
        "head": _git_head(),
        "e6_record": _E6_JSON,
        "ray_kmax": int(RAY_KMAX),
        "move_py_note": (
            "move_descend_batched carries the post-record ray-probe "
            "augmentation (490af60); its own (clamped) values are asserted "
            "one-sided (new <= tracked) only, and the numerical-floor "
            "instrument reads the un-clamped oracle_rescored rule instead"
        ),
        "palette_note": (
            "reweight #4d4d4d is a second grey outside the canonical "
            "4-color map (solve grey #7f7f7f is the oracle's)"
        ),
        "timing_probe": timing,
    }


def main(phase: str = "all", fallback: bool = False):
    torch.set_default_dtype(DTYPE)  # LOAD-BEARING, first statement: net.py's
    # log-M feature is created in the DEFAULT dtype before its cast, so
    # without this the conditioning is silently float32-rounded.

    if phase == "visualize":
        with open(_OUT_JSON) as fh:
            all_results = json.load(fh)
        print(f"Loaded {_OUT_JSON} (visualize-only; no training).")
        _render(all_results)
        return

    if phase == "contingency":
        os.makedirs(_LOG_DIR, exist_ok=True)
        run_contingency()
        return

    if phase == "assemble":
        # Stitch a killed run's partials into the sidecar and figure. The
        # gates are finalized here too, since they need both targets' cells.
        all_results = {"provenance": _provenance(None), "targets": {}}
        for tname in TARGETS:
            p = os.path.join(_LOG_DIR, f"partial_{tname}.json")
            with open(p) as fh:
                all_results["targets"][tname] = json.load(fh)
            print(f"[assemble] loaded {p}")
        _finalize_gates(all_results)
        _atomic_json(_OUT_JSON, all_results)
        print(f"Saved numbers to {_OUT_JSON}")
        _render(all_results)
        _print_summary(all_results)
        return

    os.makedirs(_LOG_DIR, exist_ok=True)
    os.makedirs(_DIAG_DIR, exist_ok=True)
    with open(_E6_JSON) as fh:
        e6 = json.load(fh)

    reps = FALLBACK_REPS if fallback else {t: len(REP_NET_OFF) for t in TARGETS}
    print(f"[arch-6 budget-channel ablation] targets={TARGETS} d={D}; "
          f"replicates={reps}; HEAD {_git_head()}")
    probe = build_target(TARGETS[0], 0.5, d=D, dtype=DTYPE, n_ref=N_REF,
                         n_mu=N_MU)
    sigma_probe = _median_heuristic_sigma(
        probe, SIGMA_N_SAMPLE, BASE_SEED + 1234 + _SALT[TARGETS[0]]
    )
    timing = _timing_probe(
        build_target(TARGETS[0], sigma_probe, d=D, dtype=DTYPE, n_ref=N_REF,
                     n_mu=N_MU),
        sigma_probe,
    )
    timing["projected_total_h"] = _project_and_report(timing, reps)

    all_results = {"provenance": _provenance(timing), "targets": {}}
    t_all = time.time()
    for tname in TARGETS:
        all_results["targets"][tname] = run_target(tname, e6, reps[tname])
    all_results["provenance"]["total_wall_s"] = time.time() - t_all

    _finalize_gates(all_results)
    # Re-dump the partials so each carries its finalized gates too.
    for tname in TARGETS:
        _atomic_json(os.path.join(_LOG_DIR, f"partial_{tname}.json"),
                     all_results["targets"][tname])
    _atomic_json(_OUT_JSON, all_results)
    print(f"\nSaved numbers to {_OUT_JSON}")
    _render(all_results)
    _print_summary(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "visualize", "assemble", "contingency"),
        default="all",
        help="'all' runs the full ablation (resuming from partials if "
        "present); 'visualize' re-renders from the saved sidecar; "
        "'assemble' stitches a killed run's partials into the sidecar; "
        "'contingency' runs the registered gaussian 8000-step + LR-sweep "
        "contingency (needs the main sidecar on disk).",
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="the design's >24h fallback schedule: 2 gaussian + 3 gmm "
        "replicates.",
    )
    args = parser.parse_args()
    main(phase=args.phase, fallback=args.fallback)
