"""Refine the Lotka--Volterra summary until it is near-sufficient.

The fixed physical features of ``lotka_volterra.summary`` identify the orbit
but do not determine it to posterior precision: the spread of ``x`` given
those fourteen numbers is a few tenths, while the spread of ``x`` given the
full trajectory is a few thousandths. Conditioning a flow on them alone would
approximate a measure some hundreds of times broader than the posterior a
reference chain targets.

This pass therefore appends a Gauss--Newton refinement, started from a ridge
prediction built on the physical features rather than from the prior mean.
The start is what makes it work: period mismatch in an oscillatory model
fills the landscape with spurious optima, and from the prior mean the
iteration usually lands in one of them.

Convergence is judged by a statistic of the data alone, never the truth:
chi-square per degree of freedom of the fitted trajectory. A spurious basin
misfits the data by orders of magnitude, so this separates cleanly and
stays computable at test time.

Emits ``train_refined.npz`` and ``test_refined.npz`` under
``data/lotka_volterra/``.

    python scripts/lv_refine_summary.py
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import os
import sys
import time
from multiprocessing import Pool

import numpy as np


from qfield.dataset import lotka_volterra as lv  # noqa: E402

OUT = datadir("lotka_volterra")
N_REFINE = 50_000
BASE_SEED = 20260612
N_WORKERS = 5
GN_ITERS = 25
GN_CAP = 0.5
CHI2_OK = 2.0
N_RETRY = 3
D = lv.D_X
_RIDGE: dict[str, np.ndarray] = {}


def _q(F: np.ndarray) -> np.ndarray:
    return np.hstack([F, F ** 2])


def _predict(f: np.ndarray) -> np.ndarray:
    z = (_q(f[None]) - _RIDGE["m"]) / _RIDGE["s"]
    return (z @ _RIDGE["W"])[0]


def _gn(y: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Damped Gauss--Newton; returns the iterate, its precision, its chi2."""
    x = x0.copy()
    H = np.eye(D)
    for _ in range(GN_ITERS):
        u, J = lv.simulate(x, sensitivities=True)
        if J is None or not lv.accept(u):
            return x, H, np.inf
        Jf = J.reshape(-1, D)
        H = Jf.T @ Jf / lv.NOISE_SD ** 2 + np.eye(D)
        step = np.linalg.solve(H, Jf.T @ (y - u).ravel() / lv.NOISE_SD ** 2 - x)
        n = float(np.linalg.norm(step))
        if n > GN_CAP:
            step *= GN_CAP / n
        x = x + step
        if n < 1e-11:
            break
    u, _ = lv.simulate(x)
    if not lv.accept(u):
        return x, H, np.inf
    chi2 = float(((y - u) ** 2).sum() / lv.NOISE_SD ** 2 / y.size)
    return x, H, chi2


def _refine(y: np.ndarray, f: np.ndarray, seed: int) -> np.ndarray:
    """The appended summary: x_hat, log-Cholesky of H, and log10 chi2."""
    rng = np.random.default_rng(seed)
    x0 = _predict(f)
    best = _gn(y, x0)
    tries = 0
    while best[2] > CHI2_OK and tries < N_RETRY:
        cand = _gn(y, x0 + 0.5 * rng.standard_normal(D))
        if cand[2] < best[2]:
            best = cand
        tries += 1
    x_hat, H, chi2 = best
    Lc = np.linalg.cholesky(H) if np.all(np.isfinite(H)) else np.eye(D)
    Lc[np.diag_indices(D)] = np.log(np.clip(np.diag(Lc), 1e-12, None))
    return np.concatenate([x_hat, Lc[np.tril_indices(D)],
                           [np.log10(min(chi2, 1e12))]])


def _train_one(i: int):
    rng = np.random.default_rng(BASE_SEED + i)
    x = rng.standard_normal(D)
    u, _ = lv.simulate(x)
    if not lv.accept(u):
        return None
    y = lv.observe(u, rng)
    f = lv.summary(y)
    return i, x, f, _refine(y, f, BASE_SEED + 7_000_000 + i)


def main() -> None:
    t0 = time.time()
    d = np.load(os.path.join(OUT, "train.npz"))
    X, F = d["x"], d["s"]
    n_fit = 20_000
    A = _q(F[:n_fit])
    m, s = A.mean(0), A.std(0) + 1e-9
    Z = (A - m) / s
    W = np.linalg.solve(Z.T @ Z + 1e1 * np.eye(Z.shape[1]), Z.T @ X[:n_fit])
    _RIDGE.update(m=m, s=s, W=W)
    np.savez(os.path.join(OUT, "ridge.npz"), m=m, s=s, W=W)
    pred = ((_q(F[n_fit:]) - m) / s) @ W
    print(f"[ridge] fitted on {n_fit}; held-out median "
          f"||x0 - x|| = {np.median(np.linalg.norm(pred - X[n_fit:], axis=1)):.3f}",
          flush=True)

    print(f"[refine] {N_REFINE} members on {N_WORKERS} workers", flush=True)
    with Pool(N_WORKERS, initializer=_RIDGE.update, initargs=(dict(m=m, s=s, W=W),)) as p:
        got = [g for g in p.map(_train_one, range(N_REFINE), chunksize=50)
               if g is not None]
    Xr = np.array([g[1] for g in got])
    Fr = np.array([g[2] for g in got])
    Rr = np.array([g[3] for g in got])
    chi2 = 10.0 ** Rr[:, -1]
    ok = chi2 < CHI2_OK
    err = np.linalg.norm(Rr[ok, :D] - Xr[ok], axis=1)
    print(f"[refine] converged {ok.sum()}/{len(ok)} ({100 * ok.mean():.1f}%) "
          f"in {time.time() - t0:.0f} s", flush=True)
    print(f"[refine] where converged, ||x_hat - x||: median "
          f"{np.median(err):.2e}  p90 {np.percentile(err, 90):.2e}", flush=True)
    np.savez_compressed(os.path.join(OUT, "train_refined.npz"),
                        x=Xr, feat=Fr, refine=Rr, converged=ok)

    t = np.load(os.path.join(OUT, "test.npz"), allow_pickle=True)
    R = []
    for k in range(t["x"].shape[0]):
        y = t["y"][k]
        R.append(_refine(y, lv.summary(y), BASE_SEED + 8_000_000 + k))
    R = np.array(R)
    print("[test] per-observation refinement:", flush=True)
    for k, nm in enumerate(t["names"]):
        c = 10.0 ** R[k, -1]
        print(f"        {str(nm):<16} chi2/dof {c:>10.3f}  "
              f"{'OK ' if c < CHI2_OK else 'MISS'}  "
              f"||x_hat-x|| {np.linalg.norm(R[k, :D] - t['x'][k]):.2e}",
              flush=True)
    np.savez_compressed(os.path.join(OUT, "test_refined.npz"), refine=R)
    print(f"[done] {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
