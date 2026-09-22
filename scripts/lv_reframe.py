"""Rebuild the quadrature frame from the flow's own conditional moments.

The Laplace frame whitens by a single Gauss--Newton linearization, which does
not describe a posterior that curves over its own width.

This module replaces that frame with the trained flow's own conditional
moments, which are equally computable from the observation at test time.
Sampling the flow gives `z`-moments in the Laplace frame, and the affine lift
`x = x_hat + z L^{-1}` carries them across in closed form:

    m = x_hat + mean(z) L^{-1} ,     C = L^{-T} cov(z) L^{-1} ,

so the new frame is `z' = R^{-1}(x - m)` with `C = R R^T`. The Laplace frame
bootstraps a better one, and nothing here needs the truth.

    python scripts/lv_reframe.py
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
from lv_train_flow import unpack, OUT  # noqa: E402

torch.set_num_threads(4)

D = lv.D_X
BASE_SEED = 20260612
N_MOMENT = 2048
RIDGE = 1e-10
CHUNK = 256


def frames_from_flow(flow, cond, x_hat, L, zs):
    """``(m, R)`` per member: the flow's conditional mean and a Cholesky
    factor of its covariance, lifted from the Laplace frame to ``x``."""
    n = cond.shape[0]
    m = np.empty((n, D))
    R = np.empty((n, D, D))
    for a in range(0, n, CHUNK):
        cb = cond[a:a + CHUNK]
        b = cb.shape[0]
        rep = cb.repeat_interleave(N_MOMENT, 0)
        with torch.no_grad():
            z = flow.sample(b * N_MOMENT, rep,
                            dtype=torch.float64).reshape(b, N_MOMENT, D)
        z = (z * torch.tensor(zs)).numpy()
        mz = z.mean(1)
        Sz = np.einsum("bni,bnj->bij", z - mz[:, None], z - mz[:, None]) \
            / (N_MOMENT - 1)
        for i in range(b):
            Li = np.linalg.inv(L[a + i])          # x = x_hat + z @ L^{-1}
            m[a + i] = x_hat[a + i] + mz[i] @ Li
            C = Li.T @ Sz[i] @ Li
            C[np.diag_indices(D)] += RIDGE
            R[a + i] = np.linalg.cholesky(C)
    return m, R


def main() -> None:
    t0 = time.time()
    ck = torch.load(os.path.join(OUT, "flow.pth"), weights_only=False)
    flow = HINTFlow(n_in=D, n_cond=ck["n_cond"], n_hidden=ck["n_hidden"],
                    n_flow_layers=ck["n_layers"]).to(torch.float64)
    flow.load_state_dict(ck["state"])
    flow.eval()
    cm, cs, zs = ck["cond_mean"], ck["cond_std"], ck["z_std"]

    d = np.load(os.path.join(OUT, "train_refined.npz"))
    ok = d["converged"]
    X, F, Rr = d["x"][ok], d["feat"][ok], d["refine"][ok]
    x_hat, L = unpack(Rr)
    cond = torch.tensor((np.hstack([F, Rr]) - cm) / cs, dtype=torch.float64)
    print(f"[train] {len(X)} members, {N_MOMENT} samples each for the moments",
          flush=True)
    m, R = frames_from_flow(flow, cond, x_hat, L, zs)
    iu = np.tril_indices(D)
    Rl = R.copy()
    Rl[:, np.arange(D), np.arange(D)] = np.log(
        np.clip(Rl[:, np.arange(D), np.arange(D)], 1e-300, None))
    frame = np.hstack([m, Rl[:, iu[0], iu[1]]])
    Z2 = np.linalg.solve(R, (X - m)[:, :, None])[:, :, 0]   # z' = R^{-1}(x-m)
    fin = np.all(np.isfinite(Z2), axis=1)
    print(f"[train] new frame: E||z'||^2 = "
          f"{np.mean((Z2[fin] ** 2).sum(1)):.3f} (target {D}), per-coord sd "
          f"{np.round(Z2[fin].std(0), 3)}", flush=True)
    np.savez_compressed(os.path.join(OUT, "train_reframed.npz"),
                        x=X, feat=F, refine=Rr, frame=frame, z2=Z2,
                        finite=fin)

    t = np.load(os.path.join(OUT, "test.npz"), allow_pickle=True)
    Rt = np.load(os.path.join(OUT, "test_refined.npz"))["refine"]
    xh_t, L_t = unpack(Rt)
    ct = torch.tensor(
        (np.stack([np.hstack([lv.summary(t["y"][k]), Rt[k]])
                   for k in range(len(Rt))]) - cm) / cs, dtype=torch.float64)
    mt, Rtt = frames_from_flow(flow, ct, xh_t, L_t, zs)
    Rtl = Rtt.copy()
    Rtl[:, np.arange(D), np.arange(D)] = np.log(
        np.clip(Rtl[:, np.arange(D), np.arange(D)], 1e-300, None))
    np.savez_compressed(os.path.join(OUT, "test_reframed.npz"),
                        m=mt, R=Rtt,
                        frame=np.hstack([mt, Rtl[:, iu[0], iu[1]]]))

    ch = np.load(os.path.join(OUT, "chains.npz"))
    print(f"\n[test] the frame check that motivated this, on the chains:")
    print(f"{'member':<12}{'OLD z mean':>26}{'OLD z sd':>22}"
          f"{'NEW z mean':>26}{'NEW z sd':>22}")
    for k in ch["cells"]:
        Xc = np.vstack(ch[f"cell{k}"])
        zo = np.einsum("ji,nj->ni", L_t[k], Xc - xh_t[k])
        zn = np.linalg.solve(Rtt[k], (Xc - mt[k]).T).T
        print(f"{str(t['names'][k]):<12}{str(np.round(zo.mean(0), 2)):>26}"
              f"{str(np.round(zo.std(0), 2)):>22}"
              f"{str(np.round(zn.mean(0), 2)):>26}"
              f"{str(np.round(zn.std(0), 2)):>22}", flush=True)
    print(f"[done] {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
