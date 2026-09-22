"""How long a pCN chain, and how much thinning, the Darcy posterior needs.

The settings of Stuart (2016) are ``n_steps=11000``, ``n_burn=1000``,
``beta=0.08``, ``thin=5``, giving 2000 stored states per observation. The use
here is different: the stored states have to behave like an i.i.d. sample from
the posterior, because they become the reference bank a designed quadrature is
scored against, and a bank of correlated states has a smaller effective size
than its count suggests. Thinning below the autocorrelation time buys storage,
not information.

This script runs one chain, measures the integrated autocorrelation time
``tau`` per coordinate, and reports:

  * ``tau`` and the effective sample size of the stored states;
  * the thinning interval at which successive stored states are ~uncorrelated;
  * the chain length needed for a target number of effective samples.

The pCN proposal is in the learned flow's latent, not in KL space: ``w`` is
perturbed by the preconditioned Crank-Nicolson kernel, which leaves ``N(0, I)``
invariant, and the prior enters through ``xi = flow.inverse(w)``. That is why
the acceptance ratio is the likelihood ratio alone.

CPU-only, no GPU. Run:
    CUDA_VISIBLE_DEVICES="" python scripts/darcy_stuart_pcn_acf.py
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qfield.dataset.darcy_stuart_kl_prior import (  # noqa: E402
    StuartKLPrior,
)
from qfield.dataset.darcy_stuart_op import (  # noqa: E402
    DarcyForward,
    beskos_sensors,
)
from qfield.samplers.pcn_latent import (  # noqa: E402
    load_flow,
    pcn_chain,
)

FLOW = os.path.join(datadir("darcy_stuart"), "oracle.pth")
OUT = os.path.join(datadir("records"), "darcy_stuart_pcn_acf.json")

# Stuart (2016) sec. 4.2 problem constants.
N_GRID, K_MODES, ALPHA, S_EXP, SIGMA = 40, 10, 0.0, 1.1, 1.0
N_SENSORS, SIGMA_Y = 33, 0.01
BETA = 0.08          # tuned value, acceptance ~60-70%
N_STEPS = 40_000     # long enough to measure tau, not the production length
N_BURN = 2_000
BASE_SEED = 20260612


def integrated_tau(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time by Sokal's automatic windowing.

    ``x`` is one scalar chain. The sum of the normalized autocorrelation is
    truncated at the first ``W`` with ``W >= c * tau(W)`` -- the standard fix
    for the fact that the tail of an empirical ACF is pure noise and summing
    all of it diverges.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    if acf[0] <= 0:
        return float("nan")
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    for w in range(1, n):
        if w >= c * taus[w]:
            return float(taus[w])
    return float(taus[-1])


def main() -> None:
    kl = StuartKLPrior(N_GRID, K=K_MODES, alpha=ALPHA, s=S_EXP, sigma=SIGMA)
    fwd = DarcyForward(N_GRID)
    sensors, _ = beskos_sensors(N_GRID, N_SENSORS)
    flow, mean, std = load_flow(FLOW, "cpu")

    # One held-out truth and its survey, from the problem's own prior.
    rng = np.random.default_rng(BASE_SEED)
    xi_true = rng.standard_normal(kl.d)
    y_clean = fwd.observe(fwd.solve(kl.reconstruct(xi_true)), sensors)
    y_obs = y_clean + SIGMA_Y * rng.standard_normal(N_SENSORS)

    print(f"[setup] d={kl.d}  sensors={N_SENSORS}  sigma_y={SIGMA_Y}  "
          f"beta={BETA}  steps={N_STEPS}  burn={N_BURN}", flush=True)
    xi, acc = pcn_chain(
        flow, mean, std, kl, fwd, sensors, y_obs, SIGMA_Y,
        n_steps=N_STEPS, n_burn=N_BURN, beta=BETA, thin=1,
        device="cpu", seed=BASE_SEED, progress=True,
    )
    print(f"[chain] {xi.shape[0]} stored draws, acceptance {acc:.3f}", flush=True)

    taus = np.array([integrated_tau(xi[:, k]) for k in range(kl.d)])
    tau_med, tau_max = float(np.median(taus)), float(np.nanmax(taus))
    n_kept = xi.shape[0]
    print(f"\n[autocorrelation] integrated tau over {kl.d} coordinates:")
    print(f"    median {tau_med:.1f}   90th pct {np.percentile(taus, 90):.1f}"
          f"   max {tau_max:.1f}")
    print(f"    ESS of the {n_kept} post-burn draws: "
          f"{n_kept / tau_max:.0f} (worst coordinate), "
          f"{n_kept / tau_med:.0f} (median)")
    print(f"\n[thinning] successive draws are ~uncorrelated at thin >= "
          f"{int(np.ceil(tau_max)):d} (worst coordinate).")
    for target in (2048, 8192, 32768):
        need = int(np.ceil(target * tau_max)) + N_BURN
        print(f"    {target:>6d} effective draws needs ~{need:,} steps "
              f"(~{need * 3.1e-3 / 60:.1f} min at 3.1 ms/solve)")

    rec = {
        "beta": BETA, "n_steps": N_STEPS, "n_burn": N_BURN,
        "acceptance": float(acc), "d": int(kl.d),
        "tau_median": tau_med, "tau_p90": float(np.percentile(taus, 90)),
        "tau_max": tau_max,
        "ess_worst": float(n_kept / tau_max),
        "recommended_thin": int(np.ceil(tau_max)),
        "sigma_y": SIGMA_Y, "n_sensors": N_SENSORS,
        "note": ("thin at the WORST coordinate's tau, not the median: the "
                 "bank is scored by a kernel that sees every coordinate"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
