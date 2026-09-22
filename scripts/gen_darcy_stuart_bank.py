"""Joint-sample bank for the Stuart/Beskos elliptic groundwater problem.

Samples ``(xi, y)`` jointly
from the model -- ``xi ~ N(0, I_d)`` whitened KL, ``u = sum_k amp_k xi_k
phi_k``, one exact elliptic solve, 33 sparse head readings, Gaussian noise --
and stores them. No MAP reconstruction is computed, which is what makes a
large bank cheap (about 3 ms per sample).

Both the clean and the noisy observation are stored: ``y_clean`` is the
noiseless forward reading and ``y_obs = y_clean + sigma_y * eps``. Keeping only
the noisy one would throw away the ability to separate observation noise from
reconstruction error afterwards.

The operator's resolved/blind split is stored beside the samples: the
linearized Jacobian's right singular vectors rank the KL directions by how
strongly 33 sensors see them, which is the likelihood-informed subspace the
quadrature works in. It is a property of the operator and the sensor layout
alone, so it is computed once here rather than per run.

Reference:
  - Beskos, Girolami, Lan, Farrell & Stuart, "Geometric MCMC for
    Infinite-Dimensional Inverse Problems" (2016), sec. 4.2 -- the problem,
    the KL prior (Eqs. 37-38), the sensor layout, and the forward (Eq. 40).

Run:
    CUDA_VISIBLE_DEVICES="" python scripts/gen_darcy_stuart_bank.py
    CUDA_VISIBLE_DEVICES="" python scripts/gen_darcy_stuart_bank.py --smoke
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import argparse
import os
import time

import h5py
import numpy as np

from qfield.dataset.darcy_stuart_kl_prior import StuartKLPrior
from qfield.dataset.darcy_stuart_op import DarcyForward, beskos_sensors
from qfield.dataset.darcy_stuart_subspace import operator_subspace

OUT = os.path.join(datadir("darcy_stuart"), "darcy_stuart_bank.h5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=40)          # Stuart's grid
    ap.add_argument("--K", type=int, default=10)          # d = K^2 = 100 modes
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--s", type=float, default=1.1)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--sigma_y", type=float, default=0.01)
    ap.add_argument("--n_sensors", type=int, default=33)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.n = 200
    out = a.out or (OUT.replace(".h5", "_smoke.h5") if a.smoke else OUT)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    kl = StuartKLPrior(a.N, K=a.K, alpha=a.alpha, s=a.s, sigma=a.sigma)
    fwd = DarcyForward(a.N)
    sensors, pos = beskos_sensors(a.N, a.n_sensors)
    sub = operator_subspace(fwd, kl, sensors, sigma_y=a.sigma_y)
    print(f"[setup] N={a.N} d={kl.d} sensors={a.n_sensors} sigma_y={a.sigma_y}"
          f"  operator rank={sub['rank']} "
          f"(resolved={sub['rank']}, blind={kl.d - sub['rank']})", flush=True)

    rng = np.random.default_rng(a.seed)
    xi = rng.standard_normal((a.n, kl.d))
    y_clean = np.empty((a.n, a.n_sensors))
    t0 = time.time()
    for i in range(a.n):
        y_clean[i] = fwd.observe(fwd.solve(kl.reconstruct(xi[i])), sensors)
        if (i + 1) % max(1, a.n // 20) == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{a.n}  ({el:.0f}s, "
                  f"eta {el / (i + 1) * (a.n - i - 1):.0f}s)", flush=True)
    # The noise is sampled ONCE and stored with the clean reading beside it,
    # so a consumer can re-noise at a different sigma_y without re-solving.
    y_obs = y_clean + a.sigma_y * rng.standard_normal((a.n, a.n_sensors))

    with h5py.File(out, "w") as f:
        for k, v in dict(N=a.N, K=a.K, d=kl.d, alpha=a.alpha, s=a.s,
                         sigma=a.sigma, sigma_y=a.sigma_y,
                         n_sensors=a.n_sensors, n=a.n, seed=a.seed,
                         rank=int(sub["rank"]),
                         source="stuart_beskos_2016_via_sbimad").items():
            f.attrs[k] = v
        f.create_dataset("xi", data=xi.astype(np.float32))
        f.create_dataset("y_clean", data=y_clean.astype(np.float32))
        f.create_dataset("y_obs", data=y_obs.astype(np.float32))
        f.create_dataset("amp", data=kl.amp.astype(np.float32))
        f.create_dataset("sv", data=sub["S"].astype(np.float32))
        f.create_dataset("resolved", data=sub["resolved"].astype(np.float32))
        f.create_dataset("blind", data=sub["blind"].astype(np.float32))
        f.create_dataset("sensor_ix", data=np.asarray(sensors[0]))
        f.create_dataset("sensor_iy", data=np.asarray(sensors[1]))
        f.create_dataset("sensor_x", data=np.asarray(pos[0]))
        f.create_dataset("sensor_y", data=np.asarray(pos[1]))
    snr = np.median(np.std(y_clean, axis=0)) / a.sigma_y
    print(f"[done] {a.n} joint samples in {time.time() - t0:.0f}s  "
          f"median per-sensor SNR {snr:.1f}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
