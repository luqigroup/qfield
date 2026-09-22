"""Every method at one shared reference-sample budget per observation.

This script fixes one number: ``n``, the reference samples a method may
consume for a single observation. Every method receives exactly the same ``n``
samples and must return an ``M``-node rule.

  * kernel thinning, Stein thinning  -- select ``M`` of the ``n``;
  * herding, recombination, SBQ      -- the pool IS the ``n``;
  * ours                             -- ``M`` of the ``n`` are the seeds, and
                                        the kernel mean is estimated from the
                                        same ``n``.

Ours is reported as three arms, so the two levers separate:

  * ``rw``   reweighting only: the seeds where they are, weights solved;
  * ``mv``   moving only:      the returned nodes under equal weights;
  * ``ours`` combined:         the returned nodes with the weights solved.

Everything is scored against an independent bank the size of the deployed
scoring bank, with that bank's own discretization subtracted, so a curve that
sits at the resolution is reported as unresolved rather than as depth.

The training cost is outside this sweep. Within a fixed per-observation sample
budget this is a like-for-like comparison; what the amortization buys is that
our fit is paid once for the family while theirs recurs at every observation,
and that is accounted separately.

    LV_FRAME=new LV_FLOW_TAG=_newframe LV_TAG=_condfix \
        python scripts/lv_unified_budget.py
"""
from __future__ import annotations
from projorg import datadir  # noqa: E402

import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

import dq_vs_thinning as T  # noqa: E402
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)
from qfield.designed_quadrature.target import (  # noqa: E402
    sampled_bank_target,
)
from qfield.designed_quadrature.weights import (  # noqa: E402
    constrained_weights,
)
from qfield.models.hint_flow import HINTFlow  # noqa: E402
from lv_peers import stein_thin  # noqa: E402

OUT = datadir("lotka_volterra")
DTYPE = torch.float64
D = 4
BASE_SEED = 20260612
SIGMA = float(os.environ.get("LV_SIGMA", 2.5544))
M = int(os.environ.get("LV_M", 32))
TAG = os.environ.get("LV_TAG", "_condfix")
SCORE_L = int(os.environ.get("LV_SCORE_L", 65536))
N_OBS = int(os.environ.get("LV_N_OBS", 8))
N_REP = int(os.environ.get("LV_N_REP", 4))
# The shared budget, in powers of two so kernel thinning's halving is exact,
# and capped at 4,096: the pooled peers (herding, recombination, the greedy
# search) cost O(M n) weight solves, so at 16,384 a single cell runs past ten
# minutes and the sweep past a day. That cost is reported beside the accuracy
# rather than avoided by handing those peers a smaller pool.
N_GRID = [int(x) for x in os.environ.get(
    "LV_N_GRID", "64,256,1024,4096").split(",")]
# Sharding, for the cluster. A task may be restricted to one observation and
# one budget, the unit of work that keeps the expensive large-``n`` cells
# inside a short walltime. Unset means everything here, the local path.
SHARD_OBS = os.environ.get("LV_OBS")
SHARD_N = os.environ.get("LV_ONLY_N")
SHARD_OUT = os.environ.get("LV_SHARD_OUT")


def _score(z: torch.Tensor, w: torch.Tensor, tsc) -> float:
    K = torch.exp(-torch.cdist(z, z) ** 2 / (2 * SIGMA ** 2))
    return float(w @ K @ w - 2 * w @ tsc.mu_fn(z) + tsc.c_rho)


def main() -> None:
    global N_GRID
    t0 = time.time()
    fk = torch.load(os.path.join(OUT, "flow_newframe.pth"),
                    map_location="cpu", weights_only=False)
    flow = HINTFlow(n_in=D, n_cond=fk["n_cond"], n_hidden=fk["n_hidden"],
                    n_flow_layers=fk["n_layers"]).to(DTYPE)
    flow.load_state_dict(fk["state"])
    flow.eval()

    qk = torch.load(os.path.join(OUT, f"quadfield{TAG}.pth"),
                    map_location="cpu", weights_only=False)
    net = ConditionalQuadratureAmortizer(
        d=D, hidden=qk["hidden"], n_blocks=qk["n_blocks"],
        n_heads=qk["n_heads"], delta_scale=qk["delta_scale"],
        cond_kind="vector", cond_vec_dim=qk["cond_dim"]).to(DTYPE)
    net.load_state_dict(qk["state"])
    net.eval()

    rf = np.load(os.path.join(OUT, "train_reframed.npz"))
    keep = rf["finite"]
    C_all = torch.tensor(
        (np.hstack([rf["feat"][keep],
                    rf["refine"][keep], rf["frame"][keep]]) - fk["cond_mean"])
        / fk["cond_std"], dtype=DTYPE)
    # Held-out observations: the tail of the family, never trained on.
    C = C_all[-N_OBS:]
    print(f"[setup] M={M} sigma={SIGMA} checkpoint={TAG!r} "
          f"{N_OBS} held-out observations x {N_REP} replicates, "
          f"scored on {SCORE_L} independent draws", flush=True)
    print(f"[setup] shared budgets n = {N_GRID}", flush=True)

    obs_ids = ([int(SHARD_OBS)] if SHARD_OBS is not None
               else list(range(N_OBS)))
    if SHARD_N is not None:
        N_GRID = [int(SHARD_N)]
    print(f"[shard] observations {obs_ids}, budgets {N_GRID}", flush=True)

    ARMS = ["floor", "rw", "mv", "ours", "thinning", "stein",
            "herding", "sbq", "recombine"]
    acc = {n: {a: [] for a in ARMS} for n in N_GRID}
    secs = {n: {a: 0.0 for a in ARMS} for n in N_GRID}
    res_acc = {n: [] for n in N_GRID}

    for oi in obs_ids:
        c = C[oi:oi + 1]
        with torch.no_grad():
            bsc = flow.sample(SCORE_L, c.repeat_interleave(SCORE_L, 0),
                              dtype=DTYPE).reshape(SCORE_L, D)
        tsc = sampled_bank_target(bsc, SIGMA, name=f"score{oi}")
        resolution = float((1.0 - tsc.c_rho) / SCORE_L)

        for rep in range(N_REP):
            gen = torch.Generator().manual_seed(
                BASE_SEED + 900000 + 1000 * oi + rep)
            for n in N_GRID:
                if n < M:
                    continue
                # The shared information: n samples, once, for every method.
                with torch.no_grad():
                    pool = flow.sample(n, c.repeat_interleave(n, 0),
                                       dtype=DTYPE).reshape(n, D)
                tgt = sampled_bank_target(pool, SIGMA, name=f"o{oi}n{n}")
                res_acc[n].append(resolution)

                # ---- ours: the first M of the pool are the seeds ----------
                z0 = pool[:M].clone()
                wu = torch.full((M,), 1.0 / M, dtype=DTYPE)
                acc[n]["floor"].append(_score(z0, wu, tsc) - resolution)

                _t = time.time()
                wr = constrained_weights(z0, tgt.mu_fn(z0), SIGMA)
                secs[n]["rw"] += time.time() - _t
                acc[n]["rw"].append(_score(z0, wr, tsc) - resolution)

                _t = time.time()
                with torch.no_grad():
                    ze = net(z0.unsqueeze(0), c)[0]
                _t_fwd = time.time() - _t
                secs[n]["mv"] += _t_fwd
                acc[n]["mv"].append(_score(ze, wu, tsc) - resolution)

                _t = time.time()
                we = constrained_weights(ze, tgt.mu_fn(ze), SIGMA)
                secs[n]["ours"] += _t_fwd + (time.time() - _t)
                acc[n]["ours"].append(_score(ze, we, tsc) - resolution)

                # ---- the peers, on the same pool ------------------------
                T.THINNING_INPUT = {M: n}
                T.HERDING_POOL = n
                T.SBQ_POOL = n
                T.RECOMBINE_POOL = n
                T.RECOMBINE_NYSTROM = max(2, n // 2)
                for name, fn in (
                    ("thinning", lambda: T.thinning_arm(tgt, M, SIGMA, gen, 0)),
                    # herding is tuned over its three modes, since an untuned
                    # peer is not a fair comparison.
                    ("herding", lambda: min(
                        (T.herding_arm(tgt, M, SIGMA, gen, md)
                         for md in T.HERDING_MODES),
                        key=lambda o: _score(o["z"], o["w"], tsc))),
                    ("sbq", lambda: T.sbq_arm(tgt, M, SIGMA, gen, 0)),
                    ("recombine", lambda: T.recombine_arm(tgt, M, SIGMA, gen)),
                ):
                    try:
                        _t = time.time()
                        o = fn()
                        secs[n][name] += time.time() - _t
                        z, w = o["z"], o["w"]
                        acc[n][name].append(_score(z, w, tsc) - resolution)
                    except Exception as exc:  # an arm that cannot run at this n
                        acc[n][name].append(float("nan"))
                        if oi == 0 and rep == 0:
                            print(f"    [n={n}] {name}: {type(exc).__name__} "
                                  f"{exc}", flush=True)
                # Stein thinning reads the score, which the others do not.
                try:
                    _t = time.time()
                    p = pool.clone().requires_grad_(True)
                    lp = flow.log_prob(p, c.repeat_interleave(n, 0))
                    sc_ = torch.autograd.grad(lp.sum(), p)[0].detach()
                    zs = stein_thin(pool, sc_, M, SIGMA)
                    secs[n]["stein"] += time.time() - _t
                    acc[n]["stein"].append(
                        _score(zs, wu, tsc) - resolution)
                except Exception as exc:
                    acc[n]["stein"].append(float("nan"))
                    if oi == 0 and rep == 0:
                        print(f"    [n={n}] stein: {type(exc).__name__} {exc}",
                              flush=True)
        print(f"  observation {oi} done "
              f"({time.time() - t0:.0f} s)", flush=True)

    out = {
        "what": ("every method at ONE shared per-observation reference-draw "
                 "budget n; ours split into reweight-only, move-only and "
                 "combined. Training cost is outside this sweep."),
        "M": M, "sigma": SIGMA, "checkpoint": f"quadfield{TAG}.pth",
        "n_obs": N_OBS, "n_replicates": N_REP, "score_L": SCORE_L,
        "n_grid": N_GRID, "cells": {},
    }
    print(f"\n{'n':>7s} " + "".join(f"{a:>11s}" for a in ARMS))
    for n in N_GRID:
        if n < M:
            continue
        row = {}
        line = f"{n:7d} "
        for a in ARMS:
            v = np.array(acc[n][a], dtype=float)
            v = v[np.isfinite(v)]
            row[a] = float(np.median(v)) if v.size else None
            line += f"{row[a]:11.3e}" if v.size else f"{'-':>11s}"
        row["resolution"] = float(np.median(res_acc[n]))
        row["secs_per_cell"] = {a: secs[n][a] / max(1, N_OBS * N_REP)
                                for a in ARMS if secs[n][a] > 0}
        out["cells"][str(n)] = row
        print(line, flush=True)

    print(f"\ndepth below the floor (floor / arm), median over "
          f"{N_OBS}x{N_REP} cells\n")
    print(f"{'n':>7s} " + "".join(f"{a:>11s}" for a in ARMS[1:]))
    for n in N_GRID:
        if n < M:
            continue
        r = out["cells"][str(n)]
        line = f"{n:7d} "
        for a in ARMS[1:]:
            line += (f"{r['floor'] / r[a]:10.1f}x"
                     if r.get(a) and r[a] > 0 else f"{'-':>11s}")
        print(line, flush=True)

    if SHARD_OUT:
        out["shard"] = {"observations": obs_ids, "budgets": N_GRID}
        out["raw"] = {str(n): {a: acc[n][a] for a in ARMS} for n in N_GRID}
        out["raw_resolution"] = {str(n): res_acc[n] for n in N_GRID}
        os.makedirs(os.path.dirname(SHARD_OUT), exist_ok=True)
        with open(SHARD_OUT, "w") as fh:
            json.dump(out, fh)
        print(f"\n[shard done] {time.time() - t0:.0f} s -> {SHARD_OUT}",
              flush=True)
        return

    dst = os.path.join(datadir("records"), "lv_unified_budget.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[done] {time.time() - t0:.0f} s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
