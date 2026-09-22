"""Evaluation of the Lotka--Volterra emissions on a held-out scoring bank.

`c_rho` and `mu` must be estimated from the same sample. Estimating them from
different samples leaves a per-cell constant in their difference, which at
large `M` exceeds the floor and drives the squared discrepancy negative. The
bank is therefore treated as the reference and `c_rho` is the V-statistic
with the diagonal included, which makes `MMD^2 = a^T K a >= 0` by
construction.

The weights are fitted against `bank_fit` and every reported number is scored
against an independent `bank_score`. Fitting and scoring on one bank lets the
weight solve fit that particular sample, which scores far below the bank's
own discretization error and reports depth that is not integration. The
residual offset is then (1 - c_rho)/L, common to every arm including the
floor, and it is the resolution limit: a depth is reported only where the
emission clears it.

    python scripts/lv_evaluate2.py
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
from qfield.designed_quadrature.weights import (  # noqa: E402
    constrained_weights_batched,
)
from qfield.models.hint_flow import HINTFlow  # noqa: E402
from lv_train_flow import unpack, OUT  # noqa: E402

torch.set_num_threads(4)

D = lv.D_X
BASE_SEED = 20260612
M_LIST = [32, 64, 128, 256, 512]
L_BANK = 16384
N_REP = 12
JITTER = 1e-8


def _mu(z, bank, i2):
    out = torch.zeros(len(z), dtype=torch.float64)
    for k in range(0, len(bank), 8192):
        out += torch.exp(-(torch.cdist(z, bank[k:k + 8192]) ** 2) * i2).sum(-1)
    return out / len(bank)


def _c_rho(bank, i2):
    """V-statistic self-affinity of the DISCRETE bank measure."""
    tot = 0.0
    for k in range(0, len(bank), 1024):
        tot += float(torch.exp(
            -(torch.cdist(bank[k:k + 1024], bank) ** 2) * i2).sum())
    return tot / (len(bank) ** 2)


def _mmd2(z, w, bank, c, i2):
    G = torch.exp(-(torch.cdist(z, z) ** 2) * i2)
    return float(w @ G @ w - 2.0 * w @ _mu(z, bank, i2) + c)


def main() -> None:
    t0 = time.time()
    ck = torch.load(os.path.join(
        OUT, f"flow{os.environ.get('LV_FLOW_TAG', '')}.pth"),
        weights_only=False)
    flow = HINTFlow(n_in=D, n_cond=ck["n_cond"], n_hidden=ck["n_hidden"],
                    n_flow_layers=ck["n_layers"]).to(torch.float64)
    flow.load_state_dict(ck["state"])
    flow.eval()
    cm, cs, zs = ck["cond_mean"], ck["cond_std"], ck["z_std"]
    tag = os.environ.get("LV_TAG", "")
    qk = torch.load(os.path.join(OUT, f"quadfield{tag}.pth"),
                    weights_only=False)
    sigma = float(qk["sigma"])
    i2 = 1.0 / (2.0 * sigma ** 2)
    net = ConditionalQuadratureAmortizer(
        d=D, hidden=qk["hidden"], n_blocks=qk["n_blocks"],
        n_heads=qk["n_heads"], delta_scale=qk["delta_scale"],
        cond_kind="vector", cond_vec_dim=qk["cond_dim"]).to(torch.float64)
    net.load_state_dict(qk["state"])
    net.eval()

    t = np.load(os.path.join(OUT, "test.npz"), allow_pickle=True)
    R = np.load(os.path.join(OUT, "test_refined.npz"))["refine"]
    # Under LV_FRAME=new the conditioner carries the flow-moment frame, so
    # the test conditioners must be built the same way the flow was trained.
    NEWF = os.environ.get("LV_FRAME") == "new"
    if NEWF:
        R = np.hstack([R, np.load(os.path.join(OUT,
                                               "test_reframed.npz"))["frame"]])
    x_hat, Lh = unpack(R[:, :15] if NEWF else R)
    ch = np.load(os.path.join(OUT, "chains.npz"))
    d = np.load(os.path.join(OUT, "train_refined.npz"))
    ok = d["converged"]
    if NEWF:
        rf = np.load(os.path.join(OUT, "train_reframed.npz"))
        kk = rf["finite"]
        Fm = np.hstack([rf["feat"][kk], rf["refine"][kk],
                        rf["frame"][kk]]).mean(0)
    else:
        Fm = np.hstack([d["feat"][ok], d["refine"][ok]]).mean(0)
    print(f"[setup] sigma {sigma:.4f}, fit/score banks {L_BANK} each, "
          f"{N_REP} replicates", flush=True)

    rows = []
    for k in ch["cells"]:
        name = str(t["names"][k])
        cvec = (np.hstack([lv.summary(t["y"][k]), R[k]]) - cm) / cs
        C1 = torch.tensor(cvec[None], dtype=torch.float64)
        torch.manual_seed(int(BASE_SEED + 31 * k))
        with torch.no_grad():
            bfit = flow.sample(L_BANK, C1.repeat(L_BANK, 1),
                               dtype=torch.float64)
            bsc = flow.sample(L_BANK, C1.repeat(L_BANK, 1),
                              dtype=torch.float64)
        c_sc = _c_rho(bsc, i2)
        res = (1.0 - c_sc) / L_BANK
        Cc = np.vstack(ch[f"cell{k}"])
        zc = torch.tensor(np.einsum("ji,nj->ni", Lh[k], Cc - x_hat[k]) / zs,
                          dtype=torch.float64)
        c_ch = _c_rho(zc, i2)
        other = ch["cells"][0] if k != ch["cells"][0] else ch["cells"][1]
        # The whole conditioner is swapped, R[other] included. Swapping only
        # the physical features would leave this member's own Gauss--Newton
        # part in place, so a null result would not be informative: the map
        # could have been reading only the part that was never shuffled.
        c_shuf = (np.hstack([lv.summary(t["y"][other]), R[other]]) - cm) / cs
        c_const = (Fm - cm) / cs
        print(f"  [{name}] c_rho score-bank {c_sc:.6f}, resolution "
              f"(1-c_rho)/L = {res:.2e}", flush=True)

        for M in M_LIST:
            acc: dict[str, list] = {a: [] for a in
                                    ("iid", "reweight", "emission", "shuffled",
                                     "constant", "iid_chain", "emission_chain")}
            for r in range(N_REP):
                g = torch.Generator().manual_seed(
                    int(BASE_SEED + 977 * k + 13 * M + r))
                z0 = bfit[torch.randint(L_BANK, (M,), generator=g)]
                w0 = torch.full((M,), 1.0 / M, dtype=torch.float64)
                wr = constrained_weights_batched(
                    z0[None], _mu(z0, bfit, i2)[None], sigma, JITTER)[0]
                acc["iid"].append(_mmd2(z0, w0, bsc, c_sc, i2))
                acc["reweight"].append(_mmd2(z0, wr, bsc, c_sc, i2))
                acc["iid_chain"].append(_mmd2(z0, w0, zc, c_ch, i2))
                for arm, cvv in (("emission", cvec),
                                 ("shuffled", c_shuf),
                                 ("constant", c_const)):
                    cb = torch.tensor(cvv[None], dtype=torch.float64)
                    with torch.no_grad():
                        z = net(z0[None], cb)[0]
                    w = constrained_weights_batched(
                        z[None], _mu(z, bfit, i2)[None], sigma, JITTER)[0]
                    # Selection happens on the fitting bank; the emission is
                    # then scored on the held-out one. The additive constant
                    # is common to all three candidates, so it cancels in the
                    # argmin and is passed as zero.
                    cand = [(_mmd2(z0, w0, bfit, 0.0, i2), z0, w0),
                            (_mmd2(z0, wr, bfit, 0.0, i2), z0, wr),
                            (_mmd2(z, w, bfit, 0.0, i2), z, w)]
                    bi = int(np.argmin([e[0] for e in cand]))
                    b = cand[bi]
                    acc[arm].append(_mmd2(b[1], b[2], bsc, c_sc, i2))
                    if arm == "emission":
                        acc.setdefault("pick_iid", []).append(1.0 * (bi == 0))
                        acc.setdefault("pick_reweight", []).append(
                            1.0 * (bi == 1))
                        acc.setdefault("pick_moved", []).append(1.0 * (bi == 2))
                        acc.setdefault("worse_than_seeds", []).append(
                            1.0 * (acc["emission"][-1] > acc["iid"][-1]))
                    if arm == "emission":
                        acc["emission_chain"].append(
                            _mmd2(b[1], b[2], zc, c_ch, i2))
            med = {a: float(np.median(v)) for a, v in acc.items()
                   if not a.startswith(("pick_", "worse_"))}
            for a in ("pick_iid", "pick_reweight", "pick_moved",
                      "worse_than_seeds"):
                med[a] = float(np.mean(acc[a])) if a in acc else float("nan")
            med["floor"] = (1.0 - c_sc) / M
            med["resolution"] = res
            med["depth_ok"] = float(med["emission"] > 3.0 * res)
            rows.append((name, M, med))
            dep = med["floor"] / max(med["emission"], 1e-14)
            print(f"  {name:<12} M={M:<4} floor {med['floor']:.3e}  "
                  f"iid {med['iid']:.3e}  rw {med['reweight']:.3e}  "
                  f"emis {med['emission']:.3e}  depth {dep:8.2f}x"
                  f"  moved {med['pick_moved']:.2f} rw {med['pick_reweight']:.2f}"
                  f" worse {med['worse_than_seeds']:.2f}"
                  f"{'' if med['depth_ok'] else ' (below resolution)'}",
                  flush=True)

    np.savez(os.path.join(
        OUT, f"evaluation2{os.environ.get('LV_OUT_TAG', tag)}.npz"),
             names=np.array([r[0] for r in rows]),
             M=np.array([r[1] for r in rows]),
             **{a: np.array([r[2][a] for r in rows]) for a in rows[0][2]})
    print(f"[done] {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
