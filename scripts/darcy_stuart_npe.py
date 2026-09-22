"""Train the conditional flow that serves as the Darcy reference
``rho_y = p(xi | y)``.

:class:`~qfield.models.hint_flow.HINTFlow` is the recursive coupling block of
Kruse, Detommaso, Koethe & Scheichl (arXiv:1905.10687), their Eq. (6)
binary-tree of affine couplings with the dense triangular Jacobian,
conditioned by CONCATENATING ``y`` into every coupling subnetwork. That is the
conditional-INN scheme of Ardizzone et al. (2019b), not the ``y``-lane /
``x``-lane construction of the HINT paper's own Fig. 3.

With ``y`` entering only as a conditioner the network defines ``p(xi | y)``
directly, so training minimizes ``-log p(xi_i | y_i)`` on the joint pairs; no
marginal ``p(y)`` is modelled and none is needed. One joint sample per
observation suffices: the conditional expectation being fitted is an
expectation over the joint, so a single ``xi`` per ``y`` is unbiased, and the
network pools information ACROSS observations.

The joint construction supplies two validation targets that are exact:

  1. **Calibration.** Each held-out pair ``(xi_true, y)`` comes from the joint,
     so ``xi_true`` is distributed as ``rho_y`` EXACTLY. The rank of any
     projection of ``xi_true`` among posterior samples is therefore uniform,
     and every central credible band covers at its nominal rate. Measured along
     the operator's own resolved and blind directions, this catches a posterior
     that is too tight, too broad, or displaced.
  2. **The misfit**, reported and not gated. By joint exchangeability
     ``G(xi') - y`` is distributed exactly ``N(0, sigma_y^2 I_m)``, marginally
     over ``y``, so ``chi^2 / m`` is exactly ``chi^2_m / m``, median 0.9799 at
     ``m = 33``, and its null band is simulated at the sample size used. It is
     reported with that band and a z-score.

The held-out conditional NLL is reported beside these and is the control for
EARLY STOPPING, a relative comparison between checkpoints of one model that
needs no absolute target. It is not itself a gate: the best achievable value is
``h(xi | y) / d``, and while ``h(xi)`` and ``h(y | xi)`` are exact, the marginal
``h(y)`` has no closed form. A model ignoring ``y`` scores
``0.5 log(2 pi e) = 1.4189``, so anything below that is information gained.

Reference:
  - Kruse, Detommaso, Koethe & Scheichl, "HINT: Hierarchical Invertible Neural
    Transport for Density Estimation and Bayesian Inference" (arXiv:1905.10687)
    -- the recursive coupling block, Eq. (6).
  - Ardizzone, Lueth, Kruse, Rother & Koethe, "Guided image generation with
    conditional invertible neural networks" (2019) -- the concatenation
    conditioning deployed here.
  - Beskos, Girolami, Lan, Farrell & Stuart (2017), sec. 4.2 -- the benchmark.

Run (train, then validate; resumes from the latest checkpoint if present):
    CUDA_VISIBLE_DEVICES=0 python scripts/darcy_stuart_npe.py
Validate only, from the best checkpoint:
    python scripts/darcy_stuart_npe.py --phase visualization
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import argparse
import hashlib
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch

from qfield.dataset.darcy_stuart_diagnostics import (
    calibration,
    calibration_null,
    chi2_misfit,
    chi2_null_band,
    information_split,
    invertibility,
)
from qfield.dataset.darcy_stuart_kl_prior import StuartKLPrior
from qfield.dataset.darcy_stuart_op import DarcyForward, beskos_sensors
from qfield.dataset.darcy_stuart_subspace import operator_subspace
from qfield.models.hint_flow import HINTFlow

from _darcy_npe import load_npe, sample_batched, standardize  # noqa: E402

BANK = os.path.join(datadir("darcy_stuart"), "darcy_stuart_bank_300k.h5")
CKPT_ROOT = os.path.join(datadir("checkpoints"), "darcy_stuart_npe")
DIAG = os.path.join(datadir("records"), "darcy_stuart_npe.json")

# CONFIG IS THE RUN'S IDENTITY: every field that changes WHAT IS TRAINED goes
# into the directory name. Without this a fixed checkpoint path means changing
# --lr or --seed silently resumes the previous run's weights, optimizer state
# and step counter; and if the stored step already exceeds n_steps, training
# no-ops and the OLD model is reported as the new result. Architecture changes
# at least fail loudly on load_state_dict; every other change fails silently,
# which is worse.
_IDENTITY = ("bank", "n_val", "n_hidden", "n_flow_layers", "depth",
             "n_mlp_layers", "batch_size", "n_steps", "lr", "weight_decay",
             "warmup", "grad_clip", "seed")
# Read from the CHECKPOINT at render time; everything else is the CLI's, so a
# validation knob can be changed without retraining.
_FROM_CKPT = ("bank", "n_val", "n_hidden", "n_flow_layers", "depth",
              "n_mlp_layers")


def run_dir(cfg: dict) -> str:
    """``data/checkpoints/darcy_stuart_npe/<hash>`` for this configuration."""
    ident = json.dumps({k: cfg[k] for k in _IDENTITY}, sort_keys=True)
    tag = hashlib.sha1(ident.encode()).hexdigest()[:10]
    return os.path.join(CKPT_ROOT, tag)

# The flow is small (133-d vectors through MLPs), so 1 GB is a generous
# ceiling; below it the run stays on CPU.
_MIN_FREE_GB = 1.0
# GATE THRESHOLDS ARE SIMULATED FROM THE NULL, NOT CHOSEN. Every gate here has
# a null distribution that is exactly known under the joint construction (see
# ``qfield/dataset/darcy_stuart_diagnostics``), so its acceptance region is
# computed at the sample size actually used rather than picked. A FIXED band is
# not a gate, because the statistic concentrates inside it as the sample grows;
# power must rise with data, and only a null-derived interval does.
#
# The misfit's null rests on joint exchangeability, ``(xi', y) =_d (xi, y)``,
# which holds MARGINALLY OVER y only. Hence the misfit uses one sample per
# observation over many observations: its variance is between observations, not
# within, which is also what makes the null exact.
_ALPHA = 0.01                       # two-sided false-failure rate per gate
_NULL_REPS = 400                    # replicates for the coverage null
_KS_SLACK = 2.0                     # KS limit as a multiple of the null's p99.5


def pick_device(min_free_gb: float = _MIN_FREE_GB) -> str:
    """``"cuda"`` only if free GPU memory exceeds ``min_free_gb``; else CPU.

    Reads free memory WITHOUT allocating.
    """
    if not torch.cuda.is_available():
        print("[gpu] CUDA unavailable -> CPU", flush=True)
        return "cpu"
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[gpu] mem_get_info failed ({exc}) -> CPU", flush=True)
        return "cpu"
    print(f"[gpu] free={free / 1e9:.2f} GB / total={total / 1e9:.2f} GB "
          f"(need {min_free_gb:.2f} GB)", flush=True)
    if free / 1e9 >= min_free_gb:
        return "cuda"
    print("[gpu] insufficient free memory -> CPU", flush=True)
    return "cpu"


def default_cfg() -> dict:
    return {
        "bank": BANK,
        "n_val": 20_000,         # held out by INDEX, so the split is stable
        # The target is easy: the operator resolves 10 of 100 directions, so
        # p(xi | y) is the prior contracted in a rank-10 block, and a larger
        # flow overfits. The HINT paper limits the hierarchy depth to 2 or 3 in
        # practice; depth 2 with a narrower subnetwork is ~0.6M parameters.
        "n_hidden": 128,
        "n_flow_layers": 6,
        "depth": 2,
        "n_mlp_layers": 3,
        # optimization: 3e-4 with early stopping on the held-out NLL; 1e-3 is
        # too hot for an affine-coupling flow here.
        #
        # n_steps SIZES THE COSINE SCHEDULE, so it must be a realistic budget
        # rather than a generous ceiling: too large a value leaves the anneal
        # incomplete, and the model never sees the low-learning-rate phase
        # where a coupling flow usually takes its last bite.
        "batch_size": 512,
        "n_steps": 15_000,
        "lr": 3e-4,
        # AdamW, so this is DECOUPLED. Plain Adam adds ``wd * theta`` to the raw
        # gradient, which the ``loss / d`` normalization below then scales
        # against, making the effective decay d = 100 times what is written.
        "weight_decay": 0.0,
        "warmup": 500,
        "grad_clip": 10.0,
        "val_every": 500,
        "ckpt_every": 500,
        # Stop after this many consecutive validations without improvement.
        "patience": 10,
        "seed": 20260612,
        # validation
        "n_val_obs": 256,        # observations for calibration AND the misfit
        "n_val_draws": 512,      # draws per observation
        # ONE misfit sample per observation. Extra samples within an
        # observation share its single noise realization and buy almost
        # nothing; pooling over OBSERVATIONS is what tightens the statistic.
        # One per observation also makes the per-observation values i.i.d.,
        # which is what makes the null exact and free of forward solves.
        "n_misfit_per_obs": 1,
        "chunk_obs": 32,         # observations per batched sampling call
    }


# --------------------------------------------------------------------------- #
# Data. 300k joint pairs at 133 floats each is ~160 MB, so the bank is loaded
# once rather than streamed.
# --------------------------------------------------------------------------- #
def load_bank(cfg: dict, device: str) -> dict:
    """Split the joint bank and derive the ``y`` standardization.

    Returns RAW ``y`` alongside the standardization rather than pre-scaled
    tensors, so that a consumer holding a trained checkpoint can apply THAT
    checkpoint's ``(mean, std)`` instead of whatever this bank implies. The two
    agree only while the bank is unchanged.
    """
    with h5py.File(cfg["bank"], "r") as f:
        xi = torch.as_tensor(f["xi"][:], dtype=torch.float32)
        y = torch.as_tensor(f["y_obs"][:], dtype=torch.float32)
        attrs = {k: f.attrs[k] for k in f.attrs}
    n_val = int(cfg["n_val"])
    if n_val >= xi.shape[0]:
        raise ValueError(
            f"n_val={n_val} leaves no training data in a bank of "
            f"{xi.shape[0]}"
        )
    if int(attrs.get("n", xi.shape[0])) != xi.shape[0]:
        raise ValueError(
            f"{cfg['bank']} declares n={attrs.get('n')} but holds "
            f"{xi.shape[0]} rows, so it was assembled from more than one "
            "generation run. The held-out split is the LAST rows by index, "
            "which is only safe when the rows are exchangeable; a concatenated "
            "bank would put a whole generation's parameters in the val split."
        )
    # Held out by index (the LAST rows), so the split is identical on every
    # run and on every resume without storing a mask. Safe because the
    # generator draws every row in one i.i.d. call (checked above that this is
    # a single run).
    xi_tr, xi_va = xi[:-n_val], xi[-n_val:]
    y_tr, y_va = y[:-n_val], y[-n_val:]
    # Standardize y per sensor from the TRAIN split only. xi is already
    # N(0, I) by construction, so it is left alone.
    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp_min(1e-8)
    print(
        f"[data] {xi_tr.shape[0]} train / {xi_va.shape[0]} val pairs | "
        f"d={xi.shape[1]} dy={y.shape[1]} | xi std="
        f"{float(xi_tr.std()):.4f} | y std range "
        f"[{float(y_std.min()):.4f}, {float(y_std.max()):.4f}]",
        flush=True,
    )
    return {
        "xi_tr": xi_tr.to(device), "y_tr_raw": y_tr.to(device),
        "xi_va": xi_va.to(device), "y_va_raw": y_va.to(device),
        "y_mean": y_mean.to(device), "y_std": y_std.to(device),
        "attrs": attrs,
    }


def build_model(cfg: dict, d: int, dy: int, device: str) -> HINTFlow:
    torch.manual_seed(int(cfg["seed"]))
    net = HINTFlow(
        d, n_cond=dy, n_hidden=int(cfg["n_hidden"]),
        n_flow_layers=int(cfg["n_flow_layers"]), depth=int(cfg["depth"]),
        n_mlp_layers=int(cfg["n_mlp_layers"]),
    ).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[model] conditional INN, HINT recursive coupling | d={d} "
          f"n_cond={dy} blocks={cfg['n_flow_layers']} depth={cfg['depth']} "
          f"hidden={cfg['n_hidden']} | params: {n_par:,}", flush=True)
    return net


def eval_nll(net: HINTFlow, xi: torch.Tensor, y: torch.Tensor,
             batch: int = 2048) -> float:
    """Mean held-out conditional NLL in nats per dimension."""
    net.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, xi.shape[0], batch):
            lp = net.log_prob(xi[i:i + batch], y[i:i + batch])
            tot += float(lp.sum())
            n += lp.shape[0]
    net.train()
    return -tot / (n * xi.shape[1])


# --------------------------------------------------------------------------- #
# Validation: is what it samples a POSTERIOR?
#
# The instruments and their exact nulls live in
# ``qfield.dataset.darcy_stuart_diagnostics``. They are estimators of the
# PROBLEM rather than of this script, and the downstream quadrature experiment
# scores its reference with the same instruments.
# --------------------------------------------------------------------------- #
def direction_groups(sub: dict) -> dict:
    """The operator's own directions, split by how strongly the data see them.

    SBC uniformity holds along ANY direction fixed independently of the
    samples, and the full basis costs one matmul, so all ``d`` right singular
    vectors are tested and the grouping is for reporting only:

      * ``resolved``   -- singular value above the noise, where the posterior
                          contracts and a width error shows first;
      * ``marginal``   -- seen but below the noise. The cutoff is a knife edge
                          (``S/sigma_y`` runs ... 1.11, 1.01 | 0.93, 0.82 ...),
                          so these are not qualitatively distinct from the
                          resolved block and must not be called blind;
      * ``exact_null`` -- singular value numerically zero, so the data say
                          NOTHING. A correct posterior is the prior here, and a
                          flow that leaked information shows up nowhere else as
                          clearly.
    """
    s = np.asarray(sub["S"])
    vh = sub["Vh"]
    n_seen = int((s > 1e-10 * s.max()).sum())
    return {
        "resolved": vh[: int(sub["rank"])].T,
        "marginal": vh[int(sub["rank"]) : n_seen].T,
        "exact_null": vh[n_seen:].T,
    }


def validate(net: HINTFlow, cfg: dict, xi_va: torch.Tensor,
             y_raw: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor,
             attrs: dict, device: str) -> dict:
    """Held-out NLL plus the gates; returns every reading and any failures."""
    y_std_cond = standardize(y_raw, y_mean, y_std)
    nll = eval_nll(net, xi_va, y_std_cond)
    n_obs = min(int(cfg["n_val_obs"]), xi_va.shape[0])
    n_draw = int(cfg["n_val_draws"])
    # A dedicated generator, so the reported numbers do not depend on how many
    # batches the training loop consumed.
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 3)
    draws = sample_batched(
        net, y_std_cond[:n_obs], n_draw,
        chunk_obs=int(cfg["chunk_obs"]), generator=gen,
    )

    n_grid = int(attrs["N"])
    sigma_y, m = float(attrs["sigma_y"]), int(attrs["n_sensors"])
    kl = StuartKLPrior(n_grid, K=int(attrs["K"]), alpha=float(attrs["alpha"]),
                       s=float(attrs["s"]), sigma=float(attrs["sigma"]))
    fwd = DarcyForward(n_grid)
    sensors, _ = beskos_sensors(n_grid, m)
    sub = operator_subspace(fwd, kl, sensors, sigma_y=sigma_y, exact=True)
    groups = direction_groups(sub)

    out: dict = {
        "val_nll_per_dim": nll,
        # The BEST a y-ignoring model can do -- not what one scores. Anything
        # below it is information the model took from the observation.
        "nll_per_dim_prior_bound": 0.5 * math.log(2 * math.pi * math.e),
        "operator_rank": int(sub["rank"]),
        "n_val_obs": n_obs,
        "n_val_draws": n_draw,
        "groups": {},
    }
    failures: list[str] = []
    lines = [
        f"\n[validate] held-out NLL {nll:.4f} nats/dim (the best a y-ignoring "
        f"model can do is {out['nll_per_dim_prior_bound']:.4f})"
    ]

    for name, dirs in groups.items():
        if dirs.shape[1] == 0:
            continue
        cal = calibration(draws, xi_va[:n_obs], dirs)
        # ONE NULL PER LEVEL. A coverage statistic's spread is largest where
        # p(1 - p) is, i.e. at the 50% band, so it is 1.7x wider there than at
        # the 90% band -- judging the 50% deviation against the 90% null makes
        # a correct posterior look broken.
        limits = {
            lvl: calibration_null(
                n_obs, dirs.shape[1], n_draw, lvl, _NULL_REPS,
                int(cfg["seed"]) + 11,
            )["max_dev_p995"]
            for lvl in (0.50, 0.90)
        }
        spread = information_split(draws, dirs)
        limit = limits[0.90]
        cal["null_max_dev50_p995"] = limits[0.50]
        cal["null_max_dev_p995"] = limit
        cal["std_min"] = float(min(spread))
        cal["std_mean"] = float(np.mean(spread))
        cal["std_max"] = float(max(spread))
        out["groups"][name] = cal
        lines.append(
            f"[calibration] {name:11s} n={cal['n_dirs']:3d}  "
            f"cov50={cal['cov50']:.3f} cov90={cal['cov90']:.3f} "
            f"(nominal {cal['nominal90']:.3f})  worst dir "
            f"{cal['max_dev90']:.3f} vs null p99.5 {limit:.3f}  "
            f"KS={cal['ks']:.4f}\n"
            f"              per-direction std [not gated]: "
            f"{cal['std_min']:.3f} .. {cal['std_mean']:.3f} .. "
            f"{cal['std_max']:.3f}"
        )
        for lvl, key, nom_key in ((0.50, "max_dev50", "nominal50"),
                                  (0.90, "max_dev90", "nominal90")):
            if cal[key] > limits[lvl]:
                failures.append(
                    f"{name}: at the {lvl:.0%} band the worst single direction "
                    f"deviates {cal[key]:.3f} from its nominal "
                    f"{cal[nom_key]:.3f} coverage, past that level's null "
                    f"99.5th percentile {limits[lvl]:.3f}. The MEAN over "
                    "directions would have hidden this -- one broken direction "
                    "in ten is diluted below any mean tolerance."
                )
        if cal["ks"] > _KS_SLACK * limit:
            failures.append(
                f"{name}: pooled rank distribution is {cal['ks']:.4f} from "
                f"uniform in KS distance, past {_KS_SLACK:.0f}x the null's "
                f"{limit:.3f}."
            )

    mis = chi2_misfit(draws, y_raw[:n_obs].double().cpu().numpy(), kl, fwd,
                      sensors, sigma_y, n_per_obs=int(cfg["n_misfit_per_obs"]))
    band = chi2_null_band(n_obs * int(cfg["n_misfit_per_obs"]), m,
                          alpha=_ALPHA, seed=int(cfg["seed"]) + 13)
    out["misfit"] = {**{k: v for k, v in mis.items() if k != "values"},
                     **{f"null_{k}": v for k, v in band.items()}}
    z = (mis["median"] - band["target"]) / max(band["sd"], 1e-30)
    out["misfit"]["z"] = float(z)
    lines.append(
        f"[misfit, REPORTED not gated] median chi^2/dof {mis['median']:.4f} "
        f"| exact null band [{band['lo']:.4f}, {band['hi']:.4f}] around "
        f"{band['target']:.4f} (sd {band['sd']:.4f}, z={z:+.2f}, "
        f"{mis['n_obs']} obs x {mis['n_per_obs']})"
    )

    # The reference is CONSUMED by sampling, so the inverse map is the property
    # that matters -- log_prob never exercises the division by the coupling
    # scale and stays healthy-looking even when sample() has drifted.
    inv = invertibility(net, y_std_cond, 256, int(cfg["seed"]) + 17)
    out["invertibility_max_err"] = inv
    lines.append(f"[invertibility] max round-trip error {inv:.2e}")
    if not inv < 1e-3:
        failures.append(
            f"the flow does not round-trip ({inv:.2e}); sample() cannot be "
            "trusted even though log_prob may look healthy."
        )

    out["validated"] = int(not failures)
    out["failures"] = failures
    print("\n".join(lines), flush=True)
    for msg in failures:
        print(f"  [FAIL] {msg}", flush=True)
    return out


def save_ckpt(path: str, net, opt, sched, step: int, best: float, cfg: dict,
              y_mean, y_std, n_stale: int, gen_state) -> None:
    """Write the checkpoint atomically.

    ``os.replace`` is atomic, so a kill mid-write cannot leave a torn file (it
    is not a guarantee against a machine crash -- there is no fsync). The
    architecture is NOT duplicated into flat fields: ``cfg`` already carries it,
    and a second copy is a second thing to keep in sync. ``n_stale`` and the
    minibatch generator travel with it so a resume CONTINUES rather than
    restarts -- without them a relaunched run burns another full patience
    window past its own best before stopping again.
    """
    tmp = path + ".tmp"
    torch.save(
        {"model": net.state_dict(), "opt": opt.state_dict(),
         "sched": sched.state_dict(), "step": step, "best_val": best,
         "cfg": cfg, "y_mean": y_mean, "y_std": y_std,
         "n_in": net.n_in, "n_cond": net.n_cond,
         "n_stale": n_stale, "gen_state": gen_state},
        tmp,
    )
    os.replace(tmp, path)


def train(cfg: dict) -> None:
    ckpt_dir = run_dir(cfg)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[run] {ckpt_dir}", flush=True)
    device = pick_device()
    bank = load_bank(cfg, device)
    xi_tr, xi_va = bank["xi_tr"], bank["xi_va"]
    y_mean, y_std = bank["y_mean"], bank["y_std"]
    y_tr = standardize(bank["y_tr_raw"], y_mean, y_std)
    y_va = standardize(bank["y_va_raw"], y_mean, y_std)
    net = build_model(cfg, xi_tr.shape[1], y_tr.shape[1], device)
    # AdamW: DECOUPLED decay. Plain Adam adds ``wd * theta`` to the raw
    # gradient, which the ``loss / d`` normalization then scales against, so the
    # effective decay would be d = 100 times what the config says.
    opt = torch.optim.AdamW(
        net.parameters(), lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    n_steps, warm = int(cfg["n_steps"]), int(cfg["warmup"])

    def lr_lambda(s: int) -> float:
        if s < warm:
            return (s + 1) / warm
        frac = (s - warm) / max(1, n_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg["seed"]) + 7)

    step, best, n_stale = 0, float("inf"), 0
    latest = os.path.join(ckpt_dir, "latest.pth")
    if os.path.isfile(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best = int(ck["step"]), float(ck["best_val"])
        n_stale = int(ck.get("n_stale", 0))
        if ck.get("gen_state") is not None:
            gen.set_state(ck["gen_state"])
        print(f"[resume] step {step}, best val NLL {best:.4f}, "
              f"{n_stale} stale validations", flush=True)

    batch, d = int(cfg["batch_size"]), xi_tr.shape[1]
    n_tr = xi_tr.shape[0]
    n_skipped = 0
    t0 = time.time()
    net.train()
    while step < n_steps:
        idx = torch.randint(0, n_tr, (batch,), generator=gen).to(device)
        opt.zero_grad(set_to_none=True)
        loss = -(net.log_prob(xi_tr[idx], y_tr[idx]).mean()) / d
        # Affine-coupling flows can spike to a non-finite loss or gradient on an
        # unlucky batch (the exp scale). The STEP is skipped rather than letting
        # one NaN poison every parameter through Adam.
        finite = bool(torch.isfinite(loss))
        if finite:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), float(cfg["grad_clip"])
            )
            if torch.isfinite(gnorm):
                opt.step()
            else:
                finite = False
                opt.zero_grad(set_to_none=True)
        n_skipped += int(not finite)
        sched.step()
        step += 1

        # Deliberately OUTSIDE the finite branch: a skipped step must still be
        # able to close the run, or a final non-finite batch would end training
        # with no last validation and no final checkpoint.
        if step % int(cfg["val_every"]) == 0 or step == n_steps:
            v = eval_nll(net, xi_va, y_va)
            shown = float(loss.detach()) if finite else float("nan")
            print(
                f"  step {step:>6d}/{n_steps}  train={shown:.4f}  "
                f"val={v:.4f} nats/dim  lr={sched.get_last_lr()[0]:.2e}  "
                f"skipped={n_skipped}  {(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
            if v < best:
                best, n_stale = v, 0
                save_ckpt(os.path.join(ckpt_dir, "best.pth"), net, opt, sched,
                          step, best, cfg, y_mean, y_std, n_stale,
                          gen.get_state())
            else:
                n_stale += 1
                if n_stale >= int(cfg["patience"]):
                    print(
                        f"[early stop] held-out NLL has not improved on "
                        f"{n_stale} consecutive validations; best {best:.4f} "
                        "nats/dim. The best checkpoint is kept.",
                        flush=True,
                    )
                    break
        if step % int(cfg["ckpt_every"]) == 0:
            save_ckpt(latest, net, opt, sched, step, best, cfg, y_mean, y_std,
                      n_stale, gen.get_state())

    save_ckpt(latest, net, opt, sched, step, best, cfg, y_mean, y_std, n_stale,
              gen.get_state())
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min | best val NLL "
          f"{best:.4f} nats/dim | non-finite steps skipped {n_skipped}",
          flush=True)


def visualize(cfg: dict) -> None:
    """Validate the best checkpoint of this configuration's run."""
    device = pick_device()
    ckpt_dir = run_dir(cfg)
    net, y_mean, y_std, ck_cfg = load_npe(ckpt_dir, device)
    # ARCHITECTURE AND SPLIT come from the checkpoint; every validation knob
    # stays the CLI's. Merging the checkpoint over everything, the inverse of
    # this, silently discards --n_val_obs and its neighbours.
    cfg = {**cfg, **{k: ck_cfg[k] for k in _FROM_CKPT}}
    bank = load_bank(cfg, device)
    # The stored standardization is authoritative; the bank's is a cross-check.
    for name, stored, derived in (("y_mean", y_mean, bank["y_mean"]),
                                  ("y_std", y_std, bank["y_std"])):
        if not torch.allclose(stored, derived, rtol=1e-5, atol=1e-8):
            raise ValueError(
                f"the checkpoint's {name} differs from the one this bank "
                f"implies (max |delta| "
                f"{float((stored - derived).abs().max()):.3e}). The bank has "
                "changed under the model; conditioning it on the wrong scale "
                "raises no error and silently stops being the posterior."
            )
    out = validate(net, cfg, bank["xi_va"], bank["y_va_raw"], y_mean, y_std,
                   bank["attrs"], device)
    ck = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location="cpu",
                    weights_only=False)
    out["step"] = int(ck["step"])
    out["best_val_nll_per_dim"] = float(ck["best_val"])
    out["run_dir"] = ckpt_dir
    # Always write into the RUN's own directory, so a sweep's runs cannot
    # overwrite each other's readings. Promote to the shared diagnostics
    # directory only on a PASS, so a failing run never leaves numbers where a
    # consumer looks for them.
    with open(os.path.join(ckpt_dir, "diagnostics.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved to {os.path.join(ckpt_dir, 'diagnostics.json')}", flush=True)
    if not out["failures"]:
        os.makedirs(os.path.dirname(DIAG), exist_ok=True)
        with open(DIAG, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Promoted to {DIAG}", flush=True)
    if out["failures"]:
        raise AssertionError(
            f"{len(out['failures'])} reference check(s) failed; this NPE is "
            "not usable as rho."
        )


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("train", "visualization"),
                    default="train")
    for k, v in cfg.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in cfg}
    torch.set_num_threads(4)
    if a.phase == "train":
        train(cfg)
    visualize(cfg)


if __name__ == "__main__":
    main()
