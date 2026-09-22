"""Multi-training-seed re-training of the designed-quadrature toy amortizers.

Covers the Gaussian + GMM dimension sweep (isotropic AND posterior-whitened)
and the ``dq_mmd_vs_M`` panels (Gaussian d=2, GMM d=10, banana d=2). For each
(target, d, kernel) config the same trainer
(:class:`scripts.designed_quadrature_gaussian_amortized.DesignedQuadratureAmortized`)
is re-run across several TRAINING seeds (the net-init / training-RNG seed
``args.seed``; ``base_seed`` -- the data / held-out generator -- is held
FIXED so every seed is scored on the SAME held-out sets and only the trained
net differs). It then reports, per (config, M), the across-seed warm/floor and
warm/oracle spread.

This is purely a re-training + collection step (CPU only). It writes the
consolidated per-seed + across-seed JSON to
``data/multiseed/dq_toy_seeds.json``; ``render_dq_figures.py`` reads it back
to draw the multi-seed confidence bands.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/dq_multiseed_train.py
    CUDA_VISIBLE_DEVICES="" python scripts/dq_multiseed_train.py --seeds 0,1,2
    CUDA_VISIBLE_DEVICES="" python scripts/dq_multiseed_train.py --only mmd_vs_M
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import os  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from projorg import checkpointsdir, datadir

from designed_quadrature_gaussian_amortized import (  # noqa: E402
    DesignedQuadratureAmortized,
)

OUT_DIR = datadir("multiseed")
OUT_JSON = os.path.join(OUT_DIR, "dq_toy_seeds.json")

# Held-out / data seed: FIXED across training seeds so every trained net
# is scored on the SAME held-out M-sample sets (the only thing that varies is
# the training seed -> the trained net).
BASE_SEED = 20260612

# Training seeds (the net-init + training-RNG seed).
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# The node-budget grid.
M_LIST = [32, 64, 128, 256, 512]

# The earlier node-budget grid, kept because it is part of the run identity:
# this script writes its OWN ``experiment`` string instead of letting projorg
# hash the config, so ``m_list`` is not in the name and a bare grid change
# would leave the [cached] branch returning an old-grid results.json under the
# new grid's name. The fix is to token the identity whenever the grid is not
# this legacy one, so legacy names still resolve.
LEGACY_M_LIST = [4, 8, 16, 32, 64]

# Shared net + optim hyperparameters.
_COMMON = dict(
    m_list=M_LIST,
    sigma=0.5,
    sigma_mode="median",
    bandwidth="median",
    sigma_n_sample=4000,
    n_ref=20000,
    n_mu=4000,
    jitter=1e-10,
    hidden=64,
    n_blocks=3,
    n_heads=4,
    cond_hidden=32,
    delta_scale=0.5,
    batch_size=64,
    lr=0.0003,
    eval_repeats=256,
    oracle_lr=0.05,
    oracle_iters=150,
    base_seed=BASE_SEED,
    gpu_id=-1,
    phase="train",
    upload=0,
    whiten=0,
    whiten_bandwidth="sqrt_d",
)

# The sweep configs. ``tag`` is the row label; ``family`` groups for the
# dimension-sweep render; ``kernel`` is "iso" or "whiten"; ``n_steps`` is 3500
# for the GMM and the banana and 4000 for the rest. The dimension sweep reads
# warm/floor at M=64; mmd_vs_M reads all M.
SWEEP_CONFIGS = {
    # --- dimension sweep: isotropic SE kernel --------------------------------
    "iso_gaussian_d2": dict(family="gaussian", kernel="iso", target="gaussian", d=2, n_steps=4000),
    "iso_gaussian_d4": dict(family="gaussian", kernel="iso", target="gaussian", d=4, n_steps=4000),
    "iso_gaussian_d8": dict(family="gaussian", kernel="iso", target="gaussian", d=8, n_steps=4000),
    "iso_gaussian_d16": dict(family="gaussian", kernel="iso", target="gaussian", d=16, n_steps=4000),
    "iso_gaussian_d32": dict(family="gaussian", kernel="iso", target="gaussian", d=32, n_steps=4000),
    "iso_gmm_d10": dict(family="gmm", kernel="iso", target="gmm_cncv", d=10, n_steps=3500),
    "iso_gmm_d32": dict(family="gmm", kernel="iso", target="gmm_cncv", d=32, n_steps=3500),
    "iso_gmm_d64": dict(family="gmm", kernel="iso", target="gmm_cncv", d=64, n_steps=3500),
    # --- dimension sweep: posterior-whitened (Mahalanobis) kernel ------------
    "whiten_gaussian_d2": dict(family="gaussian", kernel="whiten", target="gaussian", d=2, n_steps=4000),
    "whiten_gaussian_d4": dict(family="gaussian", kernel="whiten", target="gaussian", d=4, n_steps=4000),
    "whiten_gaussian_d8": dict(family="gaussian", kernel="whiten", target="gaussian", d=8, n_steps=4000),
    "whiten_gaussian_d16": dict(family="gaussian", kernel="whiten", target="gaussian", d=16, n_steps=4000),
    "whiten_gaussian_d32": dict(family="gaussian", kernel="whiten", target="gaussian", d=32, n_steps=4000),
    "whiten_gmm_d10": dict(family="gmm", kernel="whiten", target="gmm_cncv", d=10, n_steps=3500),
    "whiten_gmm_d32": dict(family="gmm", kernel="whiten", target="gmm_cncv", d=32, n_steps=3500),
    "whiten_gmm_d64": dict(family="gmm", kernel="whiten", target="gmm_cncv", d=64, n_steps=3500),
    # --- mmd_vs_M panels (the three representative iso targets) ---------------
    # iso_gaussian_d2 and iso_gmm_d10 above are reused for the mmd_vs_M render.
    "iso_banana_d2": dict(family="banana", kernel="iso", target="banana", d=2, n_steps=3500),
}

# Which configs each figure needs (for the --only filter / render planning).
DIM_SWEEP_KEYS = [k for k in SWEEP_CONFIGS if k.startswith(("iso_", "whiten_")) and "banana" not in k]
MMD_VS_M_KEYS = ["iso_gaussian_d2", "iso_gmm_d10", "iso_banana_d2"]


def _build_args(
    cfg_key: str, cfg: dict, seed: int, bandwidth: str = "median_mmd"
) -> SimpleNamespace:
    """Assemble the trainer args namespace for one (config, seed)."""
    a = dict(_COMMON)
    a.update(
        target=cfg["target"],
        d=cfg["d"],
        n_steps=cfg["n_steps"],
        whiten=1 if cfg["kernel"] == "whiten" else 0,
        seed=int(seed),
        bandwidth=str(bandwidth),
    )
    # A unique experiment dir per (config, seed) -- distinct checkpoints, no
    # collision between seeds (different experiment_name).
    # The bandwidth RULE is part of the identity: a non-legacy rule gets its
    # own token so the [cached] branch can never reuse a legacy-rule run's
    # results.json under the new rule. Legacy names carry no token, so
    # ``--bandwidth median`` still resolves them. (whiten rows ignore
    # ``bandwidth`` entirely -- they use ``whiten_bandwidth`` -- but the token
    # keeps their identities distinct all the same.)
    # The node-budget grid is part of the identity for the same reason: the
    # grid decides what a run measured, and without a token the two grids'
    # runs would share one directory name and the [cached] branch would return
    # whichever landed first. Legacy names again carry no token, so
    # ``--m-list 4,8,16,32,64`` still resolves them.
    a["experiment"] = (
        f"dq_ms_{cfg_key}_seed-{seed}_target-{a['target']}_d-{a['d']}"
        f"_whiten-{a['whiten']}_steps-{a['n_steps']}"
    )
    if str(bandwidth) != "median":
        a["experiment"] += f"_bw-{bandwidth}"
    if list(a["m_list"]) != LEGACY_M_LIST:
        a["experiment"] += "_m-" + "-".join(str(m) for m in a["m_list"])
    return SimpleNamespace(**a)


def _med_iqr(vals):
    arr = np.asarray(vals, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan"), float("nan")]
    return [
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
    ]


def _mean_std(vals):
    arr = np.asarray(vals, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan"), 0]
    return [float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)]


def _train_one(
    cfg_key: str, cfg: dict, seed: int, bandwidth: str = "median_mmd"
) -> dict:
    """Train one (config, seed); return its results.json dict (cached)."""
    args = _build_args(cfg_key, cfg, seed, bandwidth)
    res_path = os.path.join(checkpointsdir(args.experiment), "results.json")
    if os.path.isfile(res_path):
        with open(res_path) as fh:
            res = json.load(fh)
        print(f"  [cached] {cfg_key} seed={seed} ({res_path})")
        return res
    t0 = time.time()
    exp = DesignedQuadratureAmortized(args)
    exp.train()  # trains + evaluates held-out + writes results.json
    with open(res_path) as fh:
        res = json.load(fh)
    print(f"  [done] {cfg_key} seed={seed} in {time.time() - t0:.0f}s")
    return res


def _collect(cfg_key: str, cfg: dict, per_seed_results: dict) -> dict:
    """Per-M across-seed warm/floor + warm/oracle median+IQR and mean+std.

    For each M, gathers the per-seed warm/floor (and warm/oracle, and the raw
    floor/warm/oracle MMD^2 medians) across the available seeds, and summarizes
    the spread two ways (median+IQR, mean+std) so the render and the notes can
    use either.
    """
    seeds = sorted(per_seed_results.keys())
    per_M = {}
    for M in M_LIST:
        wf, wo, fl, wa, orc = [], [], [], [], []
        for s in seeds:
            row = next(
                (r for r in per_seed_results[s]["rows"] if r["M"] == M), None
            )
            if row is None:
                continue
            wf.append(row["warm_over_floor"])
            wo.append(row.get("warm_over_oracle", float("nan")))
            fl.append(row["floor"][0])
            wa.append(row["warm"][0])
            orc.append(row["oracle"][0])
        per_M[str(M)] = {
            "warm_over_floor": {
                "median_iqr": _med_iqr(wf),
                "mean_std_n": _mean_std(wf),
                "per_seed": {s: wf[i] for i, s in enumerate(seeds) if i < len(wf)},
            },
            "warm_over_oracle": {
                "median_iqr": _med_iqr(wo),
                "mean_std_n": _mean_std(wo),
            },
            "floor_mmd2": {"median_iqr": _med_iqr(fl), "mean_std_n": _mean_std(fl)},
            "warm_mmd2": {"median_iqr": _med_iqr(wa), "mean_std_n": _mean_std(wa)},
            "oracle_mmd2": {"median_iqr": _med_iqr(orc), "mean_std_n": _mean_std(orc)},
            # All seeds beat the floor at this M?
            "warm_below_floor_all_seeds": bool(np.all(np.asarray(wf) < 1.0)) if wf else False,
            "warm_below_floor_worst": float(np.max(wf)) if wf else float("nan"),
        }
    sigmas = [per_seed_results[s].get("sigma") for s in seeds]
    return {
        "family": cfg["family"],
        "kernel": cfg["kernel"],
        "target": cfg["target"],
        "d": cfg["d"],
        "n_steps": cfg["n_steps"],
        "seeds": [int(s) for s in seeds],
        "sigma_per_seed": sigmas,
        "per_M": per_M,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument(
        "--only", default="all",
        choices=["all", "dim_sweep", "mmd_vs_M"],
        help="restrict to the dimension-sweep configs or the mmd_vs_M configs",
    )
    ap.add_argument(
        "--bandwidth", default="median_mmd",
        choices=["median", "median_mmd"],
        help="MMD-ruler bandwidth rule: 'median_mmd' = the true median "
        "heuristic (the paper's rule after the 2026-08-06 switch, default); "
        "'median' = the legacy SVGD rule, kept only to resolve pre-switch "
        "runs. Never mix the two in one output JSON.",
    )
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.replace(" ", "").split(",")]
    if a.only == "dim_sweep":
        keys = DIM_SWEEP_KEYS
    elif a.only == "mmd_vs_M":
        keys = MMD_VS_M_KEYS
    else:
        keys = list(SWEEP_CONFIGS.keys())

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[multiseed] seeds={seeds}  configs={len(keys)}  -> {OUT_JSON}")

    out = {}
    if os.path.isfile(OUT_JSON):
        with open(OUT_JSON) as fh:
            out = json.load(fh)
        # NEVER mix bandwidth rules inside one output JSON (the render reads
        # it whole). An older file carries no "bandwidth" key -> treat as
        # legacy "median". On mismatch, set the old record aside untouched
        # and start fresh.
        prev_bw = out.get("bandwidth", "median")
        if prev_bw != a.bandwidth:
            aside = OUT_JSON.replace(".json", f"_bw-{prev_bw}.json")
            os.replace(OUT_JSON, aside)
            print(
                f"[multiseed] existing {OUT_JSON} is bandwidth={prev_bw!r}, "
                f"requested {a.bandwidth!r} -- set aside to {aside}"
            )
            out = {}
        # And NEVER mix node-budget grids inside one output JSON, for the same
        # reason and by the same rule. A record written at the old grid carries
        # m_list=[4,...,64]; merging this run's rows into it would put two
        # budget grids under one "m_list" key.
        prev_ms = [int(m) for m in out.get("m_list", LEGACY_M_LIST)]
        if out and prev_ms != list(M_LIST):
            tok = "-".join(str(m) for m in prev_ms)
            aside = OUT_JSON.replace(".json", f"_m-{tok}.json")
            os.replace(OUT_JSON, aside)
            print(
                f"[multiseed] existing {OUT_JSON} is m_list={prev_ms}, "
                f"requested {list(M_LIST)} -- set aside to {aside}"
            )
            out = {}
    summaries = out.get("configs", {})

    t_all = time.time()
    for cfg_key in keys:
        cfg = SWEEP_CONFIGS[cfg_key]
        print(f"\n=== {cfg_key}  ({cfg['target']} d={cfg['d']} {cfg['kernel']}) ===")
        per_seed = {}
        for s in seeds:
            res = _train_one(cfg_key, cfg, s, a.bandwidth)
            per_seed[str(s)] = res
        summaries[cfg_key] = _collect(cfg_key, cfg, per_seed)
        # Persist after each config (long job; keep partials safe).
        out = {
            "experiment": "designed_quadrature_toy_multiseed",
            "bandwidth": a.bandwidth,
            "base_seed_fixed": BASE_SEED,
            "seeds": seeds,
            "m_list": M_LIST,
            "metric_of_record": (
                "across-training-seed warm/floor and warm/oracle MMD^2 ratio "
                "(median+IQR and mean+std) per (config, M); base_seed (held-out "
                "draws) fixed so only the training seed varies"
            ),
            "configs": summaries,
        }
        with open(OUT_JSON, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  [persisted] {cfg_key} -> {OUT_JSON}")

    print(f"\n[multiseed] all done in {time.time() - t_all:.0f}s -> {OUT_JSON}")

    # Compact stdout verdict table.
    print("\n=== across-seed warm/floor (median [q25,q75]) at M=64 ===")
    for cfg_key in keys:
        s = summaries[cfg_key]
        m64 = s["per_M"]["64"]["warm_over_floor"]["median_iqr"]
        worst = s["per_M"]["64"]["warm_below_floor_worst"]
        allb = s["per_M"]["64"]["warm_below_floor_all_seeds"]
        print(
            f"  {cfg_key:>20}: {m64[0]:.3f} [{m64[1]:.3f},{m64[2]:.3f}]  "
            f"worst-seed={worst:.3f}  all<floor={allb}"
        )


if __name__ == "__main__":
    main()
