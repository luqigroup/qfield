"""Gate 1a: does the trained flow agree with a reference MCMC chain?

Every arm is scored against a held-out chain as well as against the flow, so
the reference the quadrature is scored against is itself checked.

The chains start at the generating parameter. Riabiz et al. report their
Gelman-Rubin diagnostic failing to cross threshold in seven of eight
Lotka-Volterra cells at two million iterations. Period mismatch fills the
landscape with spurious optima, overdispersed chains fall into them and
never leave, and those basins sit up to 4e6 nats below the generating
parameter, so they carry no mass. Preconditioned by the Laplace covariance
and started in the basin that holds the mass, four chains reach R-hat 1.005
in 3,500 steps.

The verdict is three-valued. UNRESOLVED means the rung reports flow-scored
numbers only and drops every claim quantifying over the true posterior.

    python scripts/lv_gate1a.py
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402


from qfield.dataset import lotka_volterra as lv  # noqa: E402
from qfield.metrics_mmd import mmd_sq_empirical  # noqa: E402
from qfield.models.hint_flow import HINTFlow  # noqa: E402

torch.set_num_threads(2)

OUT = datadir("lotka_volterra")
D = lv.D_X
BASE_SEED = 20260612
N_STEPS = 20_000
N_BURN = 10_000
THIN = 4
N_CHAINS = 4
CHI2_OK = 2.0
N_DIRS = 16
N_FLOW = 4000
# Measured on the reference in the whitened frame. This is ONE MINUS c_rho,
# not c_rho. It is frame-dependent, so it and the bandwidth are both read
# from the environment: the flow-moment frame measures 0.3786 at sigma
# 2.5544, the Laplace frame 0.4035 at sigma 2.5308. Quoting one frame's
# constant against the other's kernel is an error.
ONE_MINUS_C_RHO = float(os.environ.get("LV_ONE_MINUS_C_RHO", 0.4035))
FLOOR_512 = ONE_MINUS_C_RHO / 512
# The criterion is read with the kernel the rung deploys, a pooled median
# heuristic over reference samples, not one refitted per cell to the chain,
# which would make the ruler a function of the thing being measured.
SIGMA_DEPLOYED = float(os.environ.get("LV_SIGMA", 2.5308))


def _logpost(x: np.ndarray, y: np.ndarray) -> float:
    u, _ = lv.simulate(x)
    if not lv.accept(u):
        return -np.inf
    r = (y - u).ravel()
    return float(-0.5 * x @ x - 0.5 * r @ r / lv.NOISE_SD ** 2)


def _chain_job(a):
    """One preconditioned random-walk chain at one observation."""
    k, c, x_true, y, cov = a
    rng = np.random.default_rng(BASE_SEED + 5_000 * k + c)
    L = np.linalg.cholesky(cov) * (2.4 / np.sqrt(D))
    x = x_true + np.linalg.cholesky(cov) @ rng.standard_normal(D)
    f = _logpost(x, y)
    out = np.empty((N_STEPS, D))
    acc = 0
    for i in range(N_STEPS):
        p = x + L @ rng.standard_normal(D)
        fp = _logpost(p, y)
        if np.log(rng.random()) < fp - f:
            x, f, acc = p, fp, acc + 1
        out[i] = x
    return k, c, out[N_BURN::THIN], acc / N_STEPS


def _rhat(chains: list[np.ndarray], rng) -> float:
    C = np.array(chains)
    m, n = C.shape[0], C.shape[1]
    dirs = rng.standard_normal((N_DIRS, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    best = 0.0
    for d in dirs:
        p = C @ d
        B = n * p.mean(1).var(ddof=1)
        W = p.var(1, ddof=1).mean()
        best = max(best, float(np.sqrt(((n - 1) / n * W + B / n) / W)))
    return best


def _mmd_u(a: np.ndarray, b: np.ndarray, sigma: float) -> float:
    """Unbiased (U-statistic) MMD^2.

    The biased V-statistic carries an O(1/n) positive bias that, at these
    sample sizes, is the same size as the discrepancies being measured.
    """
    A = torch.tensor(a, dtype=torch.float64)
    B = torch.tensor(b, dtype=torch.float64)
    inv2s = 1.0 / (2.0 * sigma ** 2)
    n, m = len(A), len(B)
    Kaa = torch.exp(-(torch.cdist(A, A) ** 2) * inv2s)
    Kbb = torch.exp(-(torch.cdist(B, B) ** 2) * inv2s)
    Kab = torch.exp(-(torch.cdist(A, B) ** 2) * inv2s)
    return float((Kaa.sum() - Kaa.diag().sum()) / (n * (n - 1))
                 + (Kbb.sum() - Kbb.diag().sum()) / (m * (m - 1))
                 - 2.0 * Kab.mean())


def main() -> None:
    t0 = time.time()
    ck = torch.load(os.path.join(
        OUT, f"flow{os.environ.get('LV_FLOW_TAG', '')}.pth"),
        weights_only=False)
    net = HINTFlow(n_in=D, n_cond=ck["n_cond"], n_hidden=ck["n_hidden"],
                   n_flow_layers=ck["n_layers"]).to(torch.float64)
    net.load_state_dict(ck["state"])
    net.eval()
    cm, cs, zs = ck["cond_mean"], ck["cond_std"], ck["z_std"]

    t = np.load(os.path.join(OUT, "test.npz"), allow_pickle=True)
    R = np.load(os.path.join(OUT, "test_refined.npz"))["refine"]
    NEWF = os.environ.get("LV_FRAME") == "new"
    # Capture the chi-square column before the frame is appended: it lives
    # at index 14 of the 15-wide refine record, and R[:, -1] stops meaning it
    # the moment anything is hstacked on.
    chi2 = 10.0 ** R[:, D + D * (D + 1) // 2]
    if NEWF:
        rf = np.load(os.path.join(OUT, "test_reframed.npz"))
        m_new, R_new = rf["m"], rf["R"]
        R = np.hstack([R, rf["frame"]])
    from lv_train_flow import unpack  # noqa: E402
    x_hat, Lh = unpack(R[:, :15] if NEWF else R)
    cells = [k for k in range(len(chi2)) if chi2[k] < CHI2_OK]
    print(f"[cells] {len(cells)} of {len(chi2)} test observations pass the "
          f"chi-square restriction: {[str(t['names'][k]) for k in cells]}",
          flush=True)

    jobs = [(k, c, t["x"][k], t["y"][k], t["cov"][k])
            for k in cells for c in range(N_CHAINS)]
    print(f"[chains] {len(jobs)} chains x {N_STEPS} steps on 5 workers",
          flush=True)
    with Pool(5) as pool:
        res = pool.map(_chain_job, jobs)
    by: dict[int, list] = {k: [] for k in cells}
    accs: dict[int, list] = {k: [] for k in cells}
    for k, c, s, a in res:
        by[k].append(s)
        accs[k].append(a)
    print(f"[chains] done in {time.time() - t0:.0f} s", flush=True)

    rng = np.random.default_rng(BASE_SEED)
    print(f"\n  floor at M=512 is {FLOOR_512:.3e}; a reference is material "
          f"if its discrepancy is a tenth of that")
    print(f"\n{'cell':<14}{'R-hat':>8}{'chain-chain':>13}{'LAPLACE':>12}"
          f"{'flow':>12}{'flow/floor':>12}   verdict")
    rows = []
    for k in cells:
        ch = by[k]
        rh = _rhat(ch, rng)
        # everything is compared in the frame the quadrature will use
        def wh(X, _k=k):
            if NEWF:                       # z = R^{-1}(x - m), the new frame
                return np.linalg.solve(R_new[_k], (X - m_new[_k]).T).T / zs
            return np.einsum("ji,nj->ni", Lh[_k], X - x_hat[_k]) / zs
        def _iat(x):
            n = len(x)
            x = x - x.mean()
            f = np.fft.rfft(x, 2 * n)
            ac = np.fft.irfft(f * np.conj(f))[:n]
            ac /= ac[0]
            tot = 1.0
            for q in range(1, n // 2):
                if ac[2 * q] + ac[2 * q + 1] <= 0:
                    break
                tot += 2 * (ac[2 * q] + ac[2 * q + 1])
            return max(1.0, tot)

        # The U-statistic is unbiased only for independent samples; on raw
        # chain output the autocorrelation puts chain-vs-chain at 3-12e-4,
        # where two chains from the same target must score zero.
        th = int(np.ceil(max(_iat(ch[0][:, j]) for j in range(D))))
        ch = [c[::th] for c in ch]
        zc = [wh(c) for c in ch]
        cond = torch.tensor(
            ((np.hstack([lv.summary(t["y"][k]), R[k]]) - cm) / cs)[None],
            dtype=torch.float64).repeat(N_FLOW, 1)
        with torch.no_grad():
            zf = net.sample(N_FLOW, cond, dtype=torch.float64).numpy()
        pooled = np.vstack(zc)
        sig = SIGMA_DEPLOYED
        rl = rng.standard_normal((N_FLOW, D))   # the Laplace reference is
        #                                         N(0, I) in the whitened frame
        sub = [c[rng.choice(len(c), min(4000, len(c)), False)] for c in zc]
        fc = float(np.mean([_mmd_u(zf, c, sig) for c in sub]))
        lc = float(np.mean([_mmd_u(rl, c, sig) for c in sub]))
        cc = float(np.mean([_mmd_u(sub[i], sub[j], sig)
                            for i in range(N_CHAINS)
                            for j in range(i + 1, N_CHAINS)]))
        # Materiality, not detectability. A reference is good enough if its
        # discrepancy from the truth is small against the smallest quantity
        # the rung reports, the independent-sample floor at the largest
        # budget. Comparing instead to chain reproducibility would make the
        # verdict a function of chain length: chain-chain shrinks as 1/n
        # while reference-chain plateaus, so a long enough chain fails every
        # reference and a short enough one passes every reference.
        ratio = fc / FLOOR_512
        if rh >= 1.01:
            v = "UNRESOLVED (chains)"
        elif fc <= 0.1 * FLOOR_512:
            v = "PASS"
        elif fc > FLOOR_512:
            v = "FAIL"
        else:
            v = "UNRESOLVED"
        rows.append((str(t["names"][k]), rh, lc, fc, cc, ratio, v))
        print(f"{str(t['names'][k]):<14}{rh:>8.4f}{cc:>13.2e}{lc:>12.2e}"
              f"{fc:>12.2e}{ratio:>12.2f}   {v}", flush=True)

    # The chains are the evaluation reference, so they are kept rather than
    # recomputed. Seeds are fixed, so a lost file is reproducible exactly.
    np.savez_compressed(os.path.join(
        OUT, f"chains{os.environ.get('LV_OUT_TAG', '')}.npz"),
                        **{f"cell{k}": np.array(by[k]) for k in cells},
                        cells=np.array(cells))
    np.savez(os.path.join(OUT, f"gate1a{os.environ.get('LV_OUT_TAG', '')}.npz"),
             names=np.array([r[0] for r in rows]),
             rhat=np.array([r[1] for r in rows]),
             laplace_chain=np.array([r[2] for r in rows]),
             flow_chain=np.array([r[3] for r in rows]),
             chain_chain=np.array([r[4] for r in rows]),
             ratio=np.array([r[5] for r in rows]),
             verdict=np.array([r[6] for r in rows]))
    n_pass = sum(r[6] == "PASS" for r in rows)
    print(f"\n[gate 1a] {n_pass}/{len(rows)} PASS, "
          f"{sum(r[6] == 'FAIL' for r in rows)} FAIL, "
          f"{sum('UNRESOLVED' in r[6] for r in rows)} UNRESOLVED "
          f"in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
