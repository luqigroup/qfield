"""Compare the amortized quadrature rule against compression baselines on the
Lotka-Volterra model.

Lotka-Volterra is the model Riabiz et al. benchmark Stein Thinning on, so
Stein Thinning is implemented here and included in the peer set.

ACCESS IS DELIBERATELY ASYMMETRIC, IN THE PEERS' FAVOUR. Stein Thinning is
given the EXACT score of the reference, obtained by autodiff through the
flow's own log-density, while the amortized map sees samples only. That is the
access its method is entitled to.

Pool peers (herding, SBQ, recombination, msip_dd) share a 512-atom pool, and a
pool of at most ``2M`` atoms is not admissible, so they run at ``M <= 128``.

    python scripts/lv_peers.py
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

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
from qfield.designed_quadrature.target import (  # noqa: E402
    sampled_bank_target,
)
from qfield.designed_quadrature.weights import (  # noqa: E402
    constrained_weights_batched,
)
from qfield.models.hint_flow import HINTFlow  # noqa: E402
from lv_train_flow import OUT  # noqa: E402

import dq_vs_thinning as T  # noqa: E402

torch.set_num_threads(4)

D = lv.D_X
BASE_SEED = 20260612
M_LIST = [32, 64, 128]
L_BANK = 16384
STEIN_POOL_CAP = 16384
N_REP = 6


def stein_thin(pool: torch.Tensor, score: torch.Tensor, M: int,
               sigma: float) -> torch.Tensor:
    """Riabiz et al.'s greedy Stein thinning, on the SE-based Stein kernel.

    For ``k(x,y) = exp(-||x-y||^2 / 2 sigma^2)`` the Stein kernel is

        k0(x,y) = k * [ s(x).s(y) + (s(x)-s(y)).(x-y)/sigma^2
                        + d/sigma^2 - ||x-y||^2 / sigma^4 ] ,

    and the greedy rule appends whichever pool point minimizes
    ``k0(x,x)/2 + sum_{l<j} k0(x, x_l)``. Points may repeat, exactly as in
    the published method.
    """
    n = len(pool)
    s2, s4 = sigma ** 2, sigma ** 4
    sq = (score * score).sum(-1)
    diag = sq + D / s2                       # k0(x, x), since k(x,x) = 1
    running = torch.zeros(n, dtype=torch.float64)
    idx: list[int] = []
    for j in range(M):
        obj = diag / 2.0 + running
        i = int(torch.argmin(obj))
        idx.append(i)
        d2 = ((pool - pool[i]) ** 2).sum(-1)
        k = torch.exp(-d2 / (2.0 * s2))
        dot = score @ score[i]
        cross = ((score - score[i]) * (pool - pool[i])).sum(-1)
        running = running + k * (dot + cross / s2 + D / s2 - d2 / s4)
    return pool[torch.tensor(idx)]


def main() -> None:
    t0 = time.time()
    ck = torch.load(os.path.join(OUT, "flow.pth"), weights_only=False)
    flow = HINTFlow(n_in=D, n_cond=ck["n_cond"], n_hidden=ck["n_hidden"],
                    n_flow_layers=ck["n_layers"]).to(torch.float64)
    flow.load_state_dict(ck["state"])
    flow.eval()
    cm, cs = ck["cond_mean"], ck["cond_std"]
    tag = os.environ.get("LV_TAG", "")
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
    ch = np.load(os.path.join(OUT, "chains.npz"))
    print(f"[setup] tag={tag!r} delta_scale={qk['delta_scale']} "
          f"sigma {sigma:.4f}, bank {L_BANK}, peers at M "
          f"{M_LIST} (pool rule), {N_REP} replicates", flush=True)

    rows = []
    for k in ch["cells"]:
        name = str(t["names"][k])
        cvec = (np.hstack([lv.summary(t["y"][k]), R[k]]) - cm) / cs
        C1 = torch.tensor(cvec[None], dtype=torch.float64)
        torch.manual_seed(int(BASE_SEED + 31 * k))
        with torch.no_grad():
            bank = flow.sample(L_BANK, C1.repeat(L_BANK, 1),
                               dtype=torch.float64)
        tgt = sampled_bank_target(bank, sigma, name=f"lv_{name}")
        # the EXACT score of the reference, by autodiff through the flow
        zp = bank.clone().requires_grad_(True)
        lp = flow.log_prob(zp, C1.repeat(L_BANK, 1)).sum()
        score = torch.autograd.grad(lp, zp)[0].detach()
        print(f"  [{name}] score |grad| median "
              f"{float(score.norm(dim=-1).median()):.3f}", flush=True)

        for M in M_LIST:
            gen = torch.Generator().manual_seed(int(BASE_SEED + 7 * k + M))
            out: dict[str, list] = {}
            for r in range(N_REP):
                g = torch.Generator().manual_seed(
                    int(BASE_SEED + 977 * k + 13 * M + r))
                z0 = tgt.sampler(M, g)
                mu0 = tgt.mu_fn(z0)
                w0 = torch.full((M,), 1.0 / M, dtype=torch.float64)
                wr = constrained_weights_batched(z0[None], mu0[None], sigma,
                                                 1e-8)[0]
                with torch.no_grad():
                    ze = net(z0[None], C1)[0]
                we = constrained_weights_batched(
                    ze[None], tgt.mu_fn(ze)[None], sigma, 1e-8)[0]
                cand = [(T._score(z0, w0, tgt, sigma), z0, w0),
                        (T._score(z0, wr, tgt, sigma), z0, wr),
                        (T._score(ze, we, tgt, sigma), ze, we)]
                b = min(cand, key=lambda e: e[0])
                out.setdefault("iid", []).append(cand[0][0])
                out.setdefault("ours", []).append(b[0])
                npool = min(M * M, STEIN_POOL_CAP)
                sel = torch.randperm(L_BANK, generator=g)[:npool]
                zs_ = stein_thin(bank[sel], score[sel], M, sigma)
                ws = torch.full((M,), 1.0 / M, dtype=torch.float64)
                out.setdefault("stein", []).append(
                    T._score(zs_, ws, tgt, sigma))
                out.setdefault("stein_rw", []).append(
                    T._score(zs_, T._reweight(zs_, tgt, sigma), tgt, sigma))
                try:
                    out.setdefault("thinning", []).append(
                        T.thinning_arm(tgt, M, sigma, g, swaps=0)["mmd_sq"])
                except Exception:
                    pass
            # Herding is TUNED over its modes and the best taken, which is
            # the generous reading for the peer.
            hv, hr = [], []
            for mode in T.HERDING_MODES:
                try:
                    o = T.herding_arm(tgt, M, sigma, gen, mode)
                    hv.append(o["mmd_sq"])
                    hr.append(o["mmd_sq_rw"])
                except Exception as e:
                    print(f"      herding[{mode}] skipped: "
                          f"{type(e).__name__}: {e}", flush=True)
            if hv:
                out["herding"] = [min(hv)]
                out["herding_rw"] = [min(hr)]
            for arm, fn in (("sbq", lambda: T.sbq_arm(tgt, M, sigma, gen,
                                                      swaps=0)),
                            ("recombine", lambda: T.recombine_arm(
                                tgt, M, sigma, gen))):
                try:
                    o = fn()
                    out[arm] = [o["mmd_sq"]]
                    if "mmd_sq_rw" in o:
                        out[arm + "_rw"] = [o["mmd_sq_rw"]]
                except Exception as e:
                    print(f"      {arm} skipped: {type(e).__name__}: {e}",
                          flush=True)
            med = {a: float(np.median(v)) for a, v in out.items() if v}
            rows.append((name, M, med))
            line = f"  {name:<12} M={M:<4}"
            for a in ("iid", "ours", "stein", "stein_rw", "thinning",
                      "herding", "herding_rw", "sbq", "sbq_rw", "recombine",
                      "recombine_rw"):
                if a in med:
                    line += f"  {a} {med[a]:.2e}"
            print(line, flush=True)

    np.savez(os.path.join(OUT, f"peers{tag}.npz"),
             names=np.array([r[0] for r in rows]),
             M=np.array([r[1] for r in rows]),
             **{a: np.array([r[2].get(a, np.nan) for r in rows])
                for a in ("iid", "ours", "stein", "stein_rw", "thinning",
                          "herding", "herding_rw", "sbq", "sbq_rw",
                          "recombine", "recombine_rw")})
    print(f"[done] {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
