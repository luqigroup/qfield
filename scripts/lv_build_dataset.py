"""Build the Lotka--Volterra training family and the nine test observations.

Emits, under ``data/lotka_volterra/``:

  * ``train.npz`` -- ``x`` (log rates) and ``s`` (the fixed 14-number
    summary of the observation) for the accepted members of ``N_TRAIN``
    prior samples, plus the rejected count. These are the pairs the
    conditional flow is trained on.
  * ``test.npz``  -- nine held-out observations kept in FULL, since the
    reference chains need the likelihood: eight from the family and
    stratified across the Laplace posterior-sharpness deciles, plus
    Riabiz et al.'s own ``theta* = (0.67, 1.33, 1, 1)`` as a named cell.
    Each carries its Laplace covariance, which preconditions its chain.

The acceptance predicate defines the family, so a chain scored against this
bank must apply the identical predicate. The family is heterogeneous enough
that a pooled statistic can pass while its sharp decile fails, which is why
the test set is stratified rather than sampled.

    python scripts/lv_build_dataset.py
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
N_TRAIN = 100_000
N_TEST_POOL = 400
N_TEST = 8
BASE_SEED = 20260612
N_WORKERS = 5


def _train_one(i: int):
    rng = np.random.default_rng(BASE_SEED + i)
    x = rng.standard_normal(lv.D_X)
    u, _ = lv.simulate(x)
    if not lv.accept(u):
        return None
    return x, lv.summary(lv.observe(u, rng))


def _test_one(i: int):
    """A candidate test member, with its Laplace posterior for stratifying."""
    rng = np.random.default_rng(BASE_SEED + 900_000 + i)
    x = rng.standard_normal(lv.D_X)
    u, _ = lv.simulate(x)
    if not lv.accept(u):
        return None
    y = lv.observe(u, rng)
    p = lv.laplace_posterior(x, y)
    if p is None:
        return None
    sd = float(np.exp(np.mean(np.log(np.sqrt(np.diag(p[1]))))))
    return x, y, p[1], sd


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    print(f"[train] {N_TRAIN} prior draws on {N_WORKERS} workers", flush=True)
    with Pool(N_WORKERS) as pool:
        got = pool.map(_train_one, range(N_TRAIN), chunksize=200)
    keep = [g for g in got if g is not None]
    X = np.array([g[0] for g in keep])
    S = np.array([g[1] for g in keep])
    n_rej = N_TRAIN - len(keep)
    print(f"[train] accepted {len(keep)}/{N_TRAIN}, rejected {n_rej} "
          f"({100 * n_rej / N_TRAIN:.2f}%) in {time.time() - t0:.0f} s",
          flush=True)
    np.savez_compressed(os.path.join(OUT, "train.npz"), x=X, s=S,
                        n_drawn=N_TRAIN, n_rejected=n_rej)

    print(f"[test] {N_TEST_POOL} candidates, stratifying by posterior sd",
          flush=True)
    with Pool(N_WORKERS) as pool:
        cand = [c for c in pool.map(_test_one, range(N_TEST_POOL),
                                    chunksize=10) if c is not None]
    sds = np.array([c[3] for c in cand])
    order = np.argsort(sds)
    qs = np.linspace(0.03, 0.97, N_TEST)
    pick = [order[min(len(order) - 1, int(q * len(order)))] for q in qs]
    seen: list[int] = []
    for p in pick:
        if p not in seen:
            seen.append(int(p))

    xs = [cand[p][0] for p in seen]
    ys = [cand[p][1] for p in seen]
    cv = [cand[p][2] for p in seen]
    sd = [cand[p][3] for p in seen]

    # Riabiz et al.'s own dataset, as its own named cell.
    xr = np.log(np.array(lv.RIABIZ_THETA))
    ur, _ = lv.simulate(xr)
    yr = lv.observe(ur, np.random.default_rng(BASE_SEED + 12345))
    pr = lv.laplace_posterior(xr, yr)
    xs.append(xr)
    ys.append(yr)
    cv.append(pr[1])
    sd.append(float(np.exp(np.mean(np.log(np.sqrt(np.diag(pr[1])))))))

    names = [f"family_q{int(q * 100):02d}" for q in qs[:len(seen)]] + ["riabiz"]
    np.savez_compressed(os.path.join(OUT, "test.npz"),
                        x=np.array(xs), y=np.array(ys), cov=np.array(cv),
                        sd=np.array(sd), names=np.array(names))
    print(f"[test] {len(xs)} observations; posterior sd "
          f"{min(sd):.2e} .. {max(sd):.2e} (spread {max(sd) / min(sd):.0f}x)",
          flush=True)
    for n, s in zip(names, sd):
        print(f"        {n:<16} geo-mean posterior sd {s:.3e}", flush=True)
    print(f"[done] {time.time() - t0:.0f} s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
