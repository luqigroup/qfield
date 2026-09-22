"""Lotka-Volterra peer ladder.

Every method in the compression peer set is per-instance: SBQ, kernel
herding, kernel thinning, Stein thinning and recombination each run a fresh
optimization for every observation, while the amortized map is one forward
pass. The panel reports three things together:

  * the amortized output, one forward pass and two kernel-mean reads;
  * the per-instance peers as published, their own output, without the
    constrained weights bolted on (thinning and herding are designed to
    return equal-weight sets, so a reweighted variant is neither their method
    nor this one);
  * the finetuned output, the amortized output plus `K` steps of per-instance
    descent, with `K` chosen so the kernel-mean spend matches the peer's.
    SBQ reads the kernel mean at `SBQ_POOL` pool points, a `K`-step descent
    at `M` nodes reads `M(2K+1)`, so `K = (SBQ_POOL/M - 1)/2`.

The control is cold start versus warm start: the identical descent, the
identical budget, differing only in where it begins.

Stein thinning is given the exact score of the reference, by autodiff through
the flow's own log-density, while the map here sees samples only.

Writes `lv_peer_ladder.json` into the records directory.

    python scripts/lv_peer_ladder.py
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


from qfield.dataset import lotka_volterra as lv  # noqa: E402
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)
from qfield.designed_quadrature.move import move_descend  # noqa: E402
from qfield.designed_quadrature.target import (  # noqa: E402
    sampled_bank_target,
)
from qfield.designed_quadrature.weights import (  # noqa: E402
    constrained_weights_batched,
)
from qfield.models.hint_flow import HINTFlow  # noqa: E402
from lv_peers import stein_thin  # noqa: E402
from lv_train_flow import OUT  # noqa: E402

import dq_vs_thinning as T  # noqa: E402

torch.set_num_threads(3)

DIAG = datadir("records")
D = lv.D_X
BASE_SEED = 20260612
M_LIST = [32, 64, 128]
L_BANK = 16384
L_SCORE = 65536
N_REP = 4
JITTER = 1e-8


def main() -> None:
    t0 = time.time()
    tag = os.environ.get("LV_TAG", "_d05")
    ck = torch.load(os.path.join(
        OUT, f"flow{os.environ.get('LV_FLOW_TAG', '')}.pth"),
        weights_only=False)
    flow = HINTFlow(n_in=D, n_cond=ck["n_cond"], n_hidden=ck["n_hidden"],
                    n_flow_layers=ck["n_layers"]).to(torch.float64)
    flow.load_state_dict(ck["state"])
    flow.eval()
    cm, cs = ck["cond_mean"], ck["cond_std"]
    qk = torch.load(os.path.join(OUT, f"quadfield{tag}.pth"),
                    weights_only=False)
    sigma = float(qk["sigma"])
    net = ConditionalQuadratureAmortizer(
        d=D, hidden=qk["hidden"], n_blocks=qk["n_blocks"],
        n_heads=qk["n_heads"], delta_scale=qk["delta_scale"],
        cond_kind="vector", cond_vec_dim=qk["cond_dim"]).to(torch.float64)
    net.load_state_dict(qk["state"])
    net.eval()

    t = np.load(os.path.join(OUT, "test.npz"), allow_pickle=True)
    R = np.load(os.path.join(OUT, "test_refined.npz"))["refine"]
    # Under the flow-moment frame the conditioner carries the frame, so the
    # test conditioners must match how the flow was trained.
    if os.environ.get("LV_FRAME") == "new":
        R = np.hstack([R, np.load(
            os.path.join(OUT, "test_reframed.npz"))["frame"]])
    ch = np.load(os.path.join(OUT, "chains.npz"))
    out: dict = {
        "what": "Lotka-Volterra peer ladder: amortized emission, per-instance "
                "peers as published, and the emission finetuned at MATCHED "
                "kernel-mean spend, with cold-vs-warm start as the control.",
        "checkpoint": f"quadfield{tag}.pth",
        "delta_scale": float(qk["delta_scale"]),
        "sigma": sigma, "bank_L": L_BANK, "score_L": L_SCORE, "scoring": "held-out, offset-corrected", "n_replicates": N_REP,
        "sbq_pool": int(T.SBQ_POOL), "oracle_iters": int(T.ORACLE_ITERS),
        "stein_note": "given the EXACT reference score by autodiff through "
                      "the flow; our map sees samples only",
        "peer_note": "peers are their own published output; our constrained "
                     "weights are NOT applied to them",
        "cells": {},
    }
    print(f"[setup] {tag} delta_scale={qk['delta_scale']} sigma={sigma:.4f}",
          flush=True)
    print(f"{'cell':<12}{'M':>4}{'K':>3}{'iid':>10}{'emission':>10}"
          f"{'warm+K':>10}{'cold+K':>10}{'SBQ':>10}{'stein':>10}"
          f"{'herding':>10}{'thin':>10}{'oracle':>10}", flush=True)

    for k in ch["cells"]:
        name = str(t["names"][k])
        cvec = (np.hstack([lv.summary(t["y"][k]), R[k]]) - cm) / cs
        C1 = torch.tensor(cvec[None], dtype=torch.float64)
        torch.manual_seed(int(BASE_SEED + 31 * k))
        with torch.no_grad():
            bank = flow.sample(L_BANK, C1.repeat(L_BANK, 1),
                               dtype=torch.float64)
        with torch.no_grad():
            bsc = flow.sample(L_SCORE, C1.repeat(L_SCORE, 1),
                              dtype=torch.float64)
        # Fit every arm against `bank` and score every arm against the
        # independent `bsc`. Same-bank scoring lets any arm that re-solves
        # weights reproduce the particular sample it was graded on, and the
        # bias is asymmetric: signed weights re-solve, equal-weight peers
        # cannot. The scoring bank is four times the fitting one because the
        # compared quantities sit below a 16k bank's own discretization.
        tgt = sampled_bank_target(bank, sigma, name=f"lv_{name}")
        tsc = sampled_bank_target(bsc, sigma, name=f"lv_{name}_score")
        res = float((1.0 - float(tsc.c_rho)) / L_SCORE)

        def sc(z, w):
            return T._score(z, w, tsc, sigma) - res
        zp = bank.clone().requires_grad_(True)
        score = torch.autograd.grad(
            flow.log_prob(zp, C1.repeat(L_BANK, 1)).sum(), zp)[0].detach()
        out["cells"][name] = {}

        for M in M_LIST:
            K = max(1, int(round((T.SBQ_POOL / M - 1) / 2)))
            acc: dict[str, list] = {}
            gen = torch.Generator().manual_seed(int(BASE_SEED + 7 * k + M))
            for r in range(N_REP):
                g = torch.Generator().manual_seed(
                    int(BASE_SEED + 977 * k + 13 * M + r))
                z0 = tgt.sampler(M, g)
                w0 = torch.full((M,), 1.0 / M, dtype=torch.float64)
                wr = constrained_weights_batched(
                    z0[None], tgt.mu_fn(z0)[None], sigma, JITTER)[0]
                with torch.no_grad():
                    ze = net(z0[None], C1)[0]
                we = constrained_weights_batched(
                    ze[None], tgt.mu_fn(ze)[None], sigma, JITTER)[0]
                cand = [(T._score(z0, w0, tgt, sigma), z0, w0),
                        (T._score(z0, wr, tgt, sigma), z0, wr),
                        (T._score(ze, we, tgt, sigma), ze, we)]
                b = min(cand, key=lambda e: e[0])       # selected on fit
                acc.setdefault("iid", []).append(sc(z0, w0))
                acc.setdefault("reweight", []).append(sc(z0, wr))
                acc.setdefault("emission", []).append(sc(b[1], b[2]))
                zw, _ = move_descend(ze, tgt.mu_fn, tgt.c_rho, sigma,
                                     T.ORACLE_LR, K, JITTER)
                zc_, _ = move_descend(z0, tgt.mu_fn, tgt.c_rho, sigma,
                                      T.ORACLE_LR, K, JITTER)
                zo, _ = move_descend(z0, tgt.mu_fn, tgt.c_rho, sigma,
                                     T.ORACLE_LR, T.ORACLE_ITERS, JITTER)
                for nm_, zz in (("finetuned_warm", zw), ("finetuned_cold", zc_),
                                ("oracle", zo)):
                    acc.setdefault(nm_, []).append(
                        sc(zz, T._reweight(zz, tgt, sigma)))
                npool = min(M * M, L_BANK)
                sel = torch.randperm(L_BANK, generator=g)[:npool]
                zs_ = stein_thin(bank[sel], score[sel], M, sigma)
                acc.setdefault("stein", []).append(sc(zs_, w0))
                th = T.thinning_arm(tgt, M, sigma, g, swaps=0)
                acc.setdefault("thinning", []).append(sc(th["z"], th["w"]))
            # herding is tuned over its modes on the fit bank, and the
            # winner is then scored held out
            hv = []
            for mode in T.HERDING_MODES:
                try:
                    o = T.herding_arm(tgt, M, sigma, gen, mode)
                    hv.append((o["mmd_sq"], o))
                except Exception:
                    pass
            if hv:
                o = min(hv, key=lambda e: e[0])[1]
                acc["herding"] = [sc(o["z"], o["w"])]
            for arm, fn in (("sbq", lambda: T.sbq_arm(tgt, M, sigma, gen, 0)),
                            ("recombine", lambda: T.recombine_arm(
                                tgt, M, sigma, gen))):
                try:
                    o = fn()
                    acc[arm] = [sc(o["z"], o["w"])]
                except Exception as e:
                    print(f"      {arm}: {type(e).__name__}", flush=True)
            med = {a: float(np.median(v)) for a, v in acc.items() if v}
            med["resolution"] = res
            med["resolved"] = {a: bool(med[a] > 3.0 * res) for a in acc if acc[a]}
            med["K_finetune"] = K
            med["cost_draws_ours"] = M
            med["cost_draws_sbq"] = int(T.SBQ_POOL)
            med["cost_mu_points_emission"] = 2 * M
            med["cost_mu_points_sbq"] = int(T.SBQ_POOL)
            # Measured, not derived: the formula M(2K+1) omits the frozen
            # gradient read and the RAY_KMAX+1 ray probes that move_descend
            # adds so its return is covered by the ray certificate, which
            # cost 24M points whatever K is.
            for nm_, it_ in (("finetune", K), ("oracle", T.ORACLE_ITERS)):
                hk = T.CountingHook(tgt.mu_fn, "mu")
                z_probe = tgt.sampler(M, torch.Generator().manual_seed(0))
                move_descend(z_probe, hk, tgt.c_rho, sigma, T.ORACLE_LR,
                             it_, JITTER)
                med[f"cost_mu_points_{nm_}"] = int(hk.n_points)
            out["cells"][name][str(M)] = med
            print(f"{name:<12}{M:>4}{K:>3}{med['iid']:>10.2e}"
                  f"{med['emission']:>10.2e}{med['finetuned_warm']:>10.2e}"
                  f"{med['finetuned_cold']:>10.2e}{med.get('sbq',np.nan):>10.2e}"
                  f"{med['stein']:>10.2e}{med.get('herding',np.nan):>10.2e}"
                  f"{med['thinning']:>10.2e}{med['oracle']:>10.2e}", flush=True)

    os.makedirs(DIAG, exist_ok=True)
    with open(os.path.join(DIAG, f"lv_peer_ladder{os.environ.get('LV_OUT_TAG','')}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] {time.time()-t0:.0f} s -> "
          f"paper/figures/diagnostics/lv_peer_ladder.json", flush=True)


if __name__ == "__main__":
    main()
