"""Train the conditional flow that is the Lotka--Volterra rung's reference.

The flow is the reference the quadrature field is trained against and the
thing whose kernel mean the criterion reads.

The flow does not model ``x``. It models the Laplace-whitened residual

    z = L^T (x - x_hat) ,     H = L L^T ,

with ``x_hat`` and ``H`` taken from the refined summary, which is a function
of the observation alone. Two problems are solved by the same change. The
posterior is razor-sharp, a standard deviation of a few thousandths inside a
unit-scale prior, so a flow asked to model ``x`` directly must place a needle
to four significant figures at a location that moves over the whole prior;
asked to model ``z`` it starts from a Laplace approximation that is already
right and learns only the correction. And the coordinates in which the
criterion is evaluated are these same coordinates, so the reference, the
criterion and the quadrature all live in one frame, computable at test time
from ``y``.

Members whose refinement missed the dominant basin are excluded by the
chi-square statistic of the data alone, which restricts the family and is
decidable at test time.

    python scripts/lv_train_flow.py
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
from qfield.models.hint_flow import HINTFlow  # noqa: E402

torch.set_num_threads(2)

OUT = datadir("lotka_volterra")
DTYPE = torch.float64
BASE_SEED = 20260612
D = lv.D_X
N_HIDDEN = 64
N_LAYERS = 6
WD = 1e-4
# The whitened residual is already close to standard normal for most of the
# family. A small tail, where refinement passes the chi-square test but the
# Laplace curvature is still wrong, carries a per-coordinate standard
# deviation in the hundreds and drags the global scale with it; training on
# that tail makes the flow under-disperse.
Z_CAP = 10.0
BATCH = 512
EPOCHS = 120
LR = 1e-3
VAL_FRAC = 0.1
PATIENCE = 25


def unpack(refine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``x_hat`` and the Cholesky factor ``L`` from the summary."""
    n = refine.shape[0]
    x_hat = refine[:, :D]
    L = np.zeros((n, D, D))
    iu = np.tril_indices(D)
    L[:, iu[0], iu[1]] = refine[:, D:D + len(iu[0])]
    di = np.arange(D)
    L[:, di, di] = np.exp(L[:, di, di])
    return x_hat, L


def whiten(x: np.ndarray, x_hat: np.ndarray, L: np.ndarray) -> np.ndarray:
    return np.einsum("nji,nj->ni", L, x - x_hat)


def main() -> None:
    t0 = time.time()
    # LV_FRAME=new whitens by the flow's own conditional moments instead of
    # the Laplace ones (scripts/lv_reframe.py). Under the same outlier filter
    # the new frame standardizes better, but it inherits the flow's errors
    # where the flow is weakest.
    _frame = os.environ.get("LV_FRAME", "laplace")
    if _frame == "new":
        rf = np.load(os.path.join(OUT, "train_reframed.npz"))
        X, F, R = rf["x"], rf["feat"], rf["refine"]
        ok = rf["finite"]
        X, F, R = X[ok], F[ok], R[ok]
        Z_PRE, FRAME = rf["z2"][ok], rf["frame"][ok]
    else:
        Z_PRE, FRAME = None, None
    d = np.load(os.path.join(OUT, "train_refined.npz"))
    ok = d["converged"]
    if _frame != "new":
        X, F, R = d["x"][ok], d["feat"][ok], d["refine"][ok]
    print(f"[data] {len(X)} converged members of {len(ok)} "
          f"({100 * ok.mean():.1f}%)", flush=True)

    if Z_PRE is not None:
        Z = Z_PRE
        R = np.hstack([R, FRAME])     # the frame is part of the conditioner
    else:
        x_hat, L = unpack(R)
        Z = whiten(X, x_hat, L)
    keep = np.all(np.isfinite(Z), axis=1) & (np.abs(Z).max(1) < Z_CAP)
    print(f"[data] whitened residual: kept {keep.sum()}, "
          f"|z| p50 {np.median(np.abs(Z[keep])):.3f} "
          f"p99 {np.percentile(np.abs(Z[keep]), 99):.3f}  "
          f"(a perfect Laplace fit would give standard normal)", flush=True)
    Z, C = Z[keep], np.hstack([F[keep], R[keep]])
    # The Laplace whitening systematically under-covers. A fixed global
    # rescaling puts the flow near the right scale at initialization; the
    # per-observation part of that miscalibration is what it then learns.
    zs = Z.std(0)
    Z = Z / zs
    print(f"[data] global z sd {np.round(zs, 3)} divided out", flush=True)

    cm, cs = C.mean(0), C.std(0) + 1e-9
    C = (C - cm) / cs
    n_val = int(VAL_FRAC * len(Z))
    g = torch.Generator().manual_seed(BASE_SEED)
    perm = torch.randperm(len(Z), generator=g).numpy()
    va, tr = perm[:n_val], perm[n_val:]
    Zt = torch.tensor(Z, dtype=DTYPE)
    Ct = torch.tensor(C, dtype=DTYPE)

    torch.manual_seed(BASE_SEED)
    net = HINTFlow(n_in=D, n_cond=C.shape[1], n_hidden=N_HIDDEN,
                   n_flow_layers=N_LAYERS).to(DTYPE)
    npar = sum(p.numel() for p in net.parameters())
    print(f"[net] HINTFlow d={D} cond={C.shape[1]} -> {npar/1e3:.1f}k "
          f"parameters (a raw 4800-d conditioner would have needed 31M)",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best, bad, best_state = np.inf, 0, None
    for ep in range(EPOCHS):
        net.train()
        idx = tr[torch.randperm(len(tr), generator=g).numpy()]
        tot = 0.0
        for k in range(0, len(idx), BATCH):
            b = idx[k:k + BATCH]
            loss = -net.log_prob(Zt[b], Ct[b]).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(-net.log_prob(Zt[va], Ct[va]).mean())
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad >= PATIENCE:
            print(f"  epoch {ep:3d}  train {tot/len(idx):+.4f}  "
                  f"val {vl:+.4f}  best {best:+.4f}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop at epoch {ep}", flush=True)
            break

    net.load_state_dict(best_state)
    torch.save({"state": best_state, "cond_mean": cm, "cond_std": cs,
                "z_std": zs,
                "n_cond": C.shape[1], "n_hidden": N_HIDDEN,
                "n_layers": N_LAYERS, "val_nll": best},
               os.path.join(OUT, f"flow{os.environ.get('LV_FLOW_TAG','')}.pth"))
    with torch.no_grad():
        base = float(0.5 * (Zt[va] ** 2).sum(1).mean()) + 0.5 * D * np.log(2 * np.pi)
    lj = float(np.log(zs).sum())
    print(f"[done] best val NLL {best + lj:+.4f} in the original frame; "
          f"a standard normal on the same data scores {base + lj:+.4f}, "
          f"so the flow buys {base - best:+.3f} nats "
          f"in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
