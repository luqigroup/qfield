"""pCN audit of the Darcy reference flow.

WHAT THIS MEASURES. The guarantee is relative to the reference the user
supplies: the quadrature integrates the trained flow ``rho_y``, not the
unknown true posterior. This audit measures that split on the groundwater
problem, reporting the emission's MMD^2 to the flow (the guarantee's own
currency), the flow's MMD^2 to exact-forward pCN chains (the reference's own
error), and the emission and floor rescored against those chains.

THE PROTOCOL (all numbers in the emitted record):
  * beta = 0.25, the step size with the best effective sample size per solve;
  * 4 chains per observation, from OVERDISPERSED starts: two prior samples
    (the prior is 2-8x wider than the posterior along every operator-resolved
    direction, and IS the posterior along the blind ones) and two prior
    samples scaled by 2.5, harsher than the prior everywhere;
  * 130,000 steps per chain, 10,000 burned, leaving 120,000 post-burn, about
    107x the measured autocorrelation-time upper bound at beta = 0.25 against
    the 20-40x rule of thumb. tau is RE-MEASURED from these chains and the
    achieved multiple recorded;
  * thin 25, giving 4,800 kept states per chain and 19,200 pooled per
    observation;
  * 8 held-out observations at M = 64.

POWER CERTIFICATE. The audit is powered iff the chain-vs-chain MMD^2, which
compares independent chains on the same posterior and so is a pure noise
floor, sits well below the flow-vs-pCN gap it brackets. Both the worst
PAIRWISE gap (single chain against single chain) and the POOLED-HALVES gap
(chains 0+1 against 2+3, the resolution matching the pooled bank used for
scoring) are recorded, with the achieved ratios.

RESUMABLE BY CONSTRUCTION: each finished chain is saved immediately, each
observation's metrics to a partial JSON, and a rerun skips whatever exists.

CPU-only, about 5-6 h under load:
    nice -n 10 python scripts/darcy_stuart_pcn_audit_powered.py
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from _ckpt import checkpoint_path  # noqa: E402
from _darcy_npe import NPE_ROOT, load_npe, sample_batched, standardize  # noqa: E402
from darcy_stuart_pcn_acf import integrated_tau  # noqa: E402

from projorg import configsdir, datadir
from projorg.config import read_config  # noqa: E402

from qfield.dataset.darcy_stuart_kl_prior import StuartKLPrior  # noqa: E402
from qfield.dataset.darcy_stuart_op import (  # noqa: E402
    DarcyForward,
    beskos_sensors,
)
from qfield.designed_quadrature import (  # noqa: E402
    constrained_weights,
    mmd_sq,
    sampled_bank_target,
    select_emission,
)
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)
from qfield.samplers.pcn_latent import pcn_chain  # noqa: E402

DTYPE = torch.float64
torch.set_num_threads(4)

CONFIG_FILE = "dq_darcy_stuart.json"
# The compute script's extended ignore list. It has to be mirrored here, or
# the rebuilt run name points at a directory that does not exist.
IGNORE_EXTRA = ("recon_M", "n_recon", "eval_repeats", "n_pcn_audit")
PARTIAL_DIR = os.path.join(datadir("darcy_stuart"), "pcn_audit_powered")
OUT = os.path.join(datadir("records"), "darcy_stuart_pcn_audit_powered.json")

# ---- the protocol ---------------------------------------------------------
BETA = 0.25            # best effective sample size per solve
N_STEPS = 130_000      # per chain, including burn-in
N_BURN = 10_000        # ~9x the beta-0.25 tau_max upper bound
THIN = 25              # kept spacing << tau; storage only
N_CHAINS = 4           # >= 4, overdispersed starts
N_OBS = 8              # held-out observations
M = 64                 # node budget (m_list[1])
OVERDISP = 2.5         # start scale for chains 2 and 3
SEED_CHAIN = 4_000_000  # offset so no earlier stream is reused
SEED_START = 5_000_000


def _cross_mean(tgt, x: torch.Tensor, chunk: int = 2048) -> float:
    """``mean_j mu_tgt(x_j)`` in row chunks (bounds the cdist transient)."""
    tot, n = 0.0, int(x.shape[0])
    for i in range(0, n, chunk):
        tot += float(tgt.mu_fn(x[i : i + chunk]).sum())
    return tot / n


def _gap(t_a, t_b, bank_b: torch.Tensor) -> float:
    """MMD^2 between two bank-defined measures, exact for the banks."""
    return max(
        float(t_a.c_rho) + float(t_b.c_rho) - 2.0 * _cross_mean(t_a, bank_b),
        0.0,
    )


def _chain_file(o: int, c: int) -> str:
    return os.path.join(PARTIAL_DIR, f"obs{o}_chain{c}.npz")


def _obs_file(o: int) -> str:
    return os.path.join(PARTIAL_DIR, f"obs{o}.json")


def main() -> None:
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    cfg = read_config(os.path.join(configsdir(), CONFIG_FILE))
    base_seed = int(cfg["base_seed"])
    d, dy = int(cfg["d"]), int(cfg["dy"])
    assert int(list(map(int, str(cfg["m_list"]).split(",")))[1]) == M

    # The run's checkpoint, by config identity, never by glob or mtime.
    ck_path = checkpoint_path(CONFIG_FILE, ignore_extra=IGNORE_EXTRA)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sigma = float(ck["results"]["sigma"])
    print(f"[ckpt] {ck_path}\n[sigma] {sigma:.6f}", flush=True)

    # Flow FIRST (float32 params must materialize float32), float64 after.
    npe, y_mean, y_std, npe_cfg = load_npe(
        os.path.join(NPE_ROOT, str(cfg["npe_run"])), "cpu"
    )
    npe = npe.float()
    torch.set_default_dtype(DTYPE)

    with h5py.File(npe_cfg["bank"], "r") as f:
        attrs = {k: f.attrs[k] for k in f.attrs}
    kl = StuartKLPrior(int(attrs["N"]), K=int(attrs["K"]),
                       alpha=float(attrs["alpha"]), s=float(attrs["s"]),
                       sigma=float(attrs["sigma"]))
    fwd = DarcyForward(int(attrs["N"]))
    sensors, _ = beskos_sensors(int(attrs["N"]), int(attrs["n_sensors"]))
    sigma_y = float(attrs["sigma_y"])

    # The eval observations: fresh simulator samples, rng base_seed + 777000,
    # rows 0..N_OBS-1. The noise block is generated AFTER the forward loop
    # over all rows, exactly as in DQDarcyStuart.__init__, so the streams line
    # up row for row.
    n_eval = int(cfg["n_eval_obs"])
    rng = np.random.default_rng(base_seed + 777_000)
    xi_ev = rng.standard_normal((n_eval, d))
    y_ev = np.stack([
        fwd.observe(fwd.solve(kl.reconstruct(x)), sensors) for x in xi_ev
    ]) + sigma_y * rng.standard_normal((n_eval, int(attrs["n_sensors"])))
    y_ev_raw = torch.as_tensor(y_ev, dtype=torch.float32)
    y_ev_std = standardize(y_ev_raw, y_mean, y_std)

    # The trained amortizer, rebuilt from the config and the stored weights.
    net = ConditionalQuadratureAmortizer(
        d=d, hidden=int(cfg["hidden"]), n_blocks=int(cfg["n_blocks"]),
        n_heads=int(cfg["n_heads"]), cond_hidden=int(cfg["cond_hidden"]),
        delta_scale=float(cfg["delta_scale"]),
        cond_token_dim=int(cfg["cond_token_dim"]),
        cond_hidden_ch=int(cfg["cond_hidden_ch"]),
        cond_kind="vector", cond_vec_dim=dy,
        cond_n_tokens=int(cfg["cond_n_tokens"]),
    ).to(DTYPE)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    def draw_flow(o: int, n: int, gen: torch.Generator) -> torch.Tensor:
        return sample_batched(
            npe, y_ev_std[o : o + 1], n, chunk_obs=32, generator=gen
        ).double().cpu()[0]

    rows = []
    for o in range(N_OBS):
        if os.path.isfile(_obs_file(o)):
            with open(_obs_file(o)) as fh:
                rows.append(json.load(fh))
            print(f"[obs {o}] done already; skipping", flush=True)
            continue
        t_obs = time.time()

        # ---- 4 chains, overdispersed starts, resumable per chain ---------
        chains, accs = [], []
        for c in range(N_CHAINS):
            cf = _chain_file(o, c)
            if os.path.isfile(cf):
                z = np.load(cf)
                chains.append(torch.as_tensor(z["xi"], dtype=DTYPE))
                accs.append(float(z["acceptance"]))
                print(f"  [obs {o} chain {c}] cached "
                      f"({z['xi'].shape[0]} kept)", flush=True)
                continue
            w0 = None
            if c >= 2:
                start_rng = np.random.default_rng(
                    base_seed + SEED_START + 977 * o + c)
                w0 = torch.as_tensor(
                    OVERDISP * start_rng.standard_normal(d), dtype=DTYPE)
            t0 = time.time()
            xi_c, acc_c = pcn_chain(
                None, None, None, kl, fwd, sensors,
                y_ev_raw[o].double().numpy(), sigma_y,
                n_steps=N_STEPS, n_burn=N_BURN, beta=BETA, thin=THIN,
                d=d, seed=base_seed + SEED_CHAIN + 977 * o + 499 * c,
                w0=w0,
            )
            np.savez_compressed(cf, xi=xi_c, acceptance=acc_c)
            chains.append(torch.as_tensor(xi_c, dtype=DTYPE))
            accs.append(float(acc_c))
            print(f"  [obs {o} chain {c}] {xi_c.shape[0]} kept, "
                  f"acc={acc_c:.3f}, start="
                  f"{'overdispersed x%.1f' % OVERDISP if c >= 2 else 'prior'}"
                  f", {(time.time() - t0) / 60:.1f} min", flush=True)

        # ---- tau, re-measured from these chains (kept units x THIN) ------
        taus = np.array([
            integrated_tau(ch[:, k].numpy()) * THIN
            for ch in chains for k in range(d)
        ])
        tau_med = float(np.nanmedian(taus))
        tau_max = float(np.nanmax(taus))

        # ---- chain-vs-chain: the noise floor this audit is read against --
        t_c = [sampled_bank_target(ch, sigma, dtype=DTYPE, replace=True)
               for ch in chains]
        pair_gaps = [
            _gap(t_c[i], t_c[j], chains[j])
            for i in range(N_CHAINS) for j in range(i + 1, N_CHAINS)
        ]
        half_a = torch.cat(chains[:2])
        half_b = torch.cat(chains[2:])
        t_ha = sampled_bank_target(half_a, sigma, dtype=DTYPE, replace=True)
        t_hb = sampled_bank_target(half_b, sigma, dtype=DTYPE, replace=True)
        halves_gap = _gap(t_ha, t_hb, half_b)

        pcn_t = sampled_bank_target(
            torch.cat(chains), sigma, name="pcn", dtype=DTYPE, replace=True)

        # ---- the emission protocol ---------------------------------------
        gen = torch.Generator().manual_seed(base_seed + 60_000 + o)
        bank = draw_flow(o, int(cfg["bank_L"]), gen)
        flow_t = sampled_bank_target(
            bank, sigma, name="darcy_npe", dtype=DTYPE, replace=True)
        z0 = flow_t.sampler(M, gen)
        mu0 = flow_t.mu_fn(z0)
        w_c = constrained_weights(z0, mu0, sigma, float(cfg["jitter"]))
        with torch.no_grad():
            z_n = net(z0.unsqueeze(0),
                      y_ev_std[o].double().unsqueeze(0))[0]
        mu_n = flow_t.mu_fn(z_n)
        w_n = constrained_weights(z_n, mu_n, sigma, float(cfg["jitter"]))
        sel = select_emission(
            [(z0, torch.full((M,), 1.0 / M, dtype=DTYPE)),
             (z0, w_c), (z_n, w_n)],
            flow_t.mu_fn, sigma,
            names=("seeds", "reweight", "move"), fitting_criterion=True,
        )
        z_s, w_s = sel["z"], sel["w"]
        fresh2k = draw_flow(o, 2000, gen)   # the gap estimand

        eq = torch.full((M,), 1.0 / M, dtype=DTYPE)
        rec = {
            "obs": o,
            "acceptance": [round(a, 4) for a in accs],
            "tau_median": tau_med, "tau_max": tau_max,
            "post_burn_over_tau_max": (N_STEPS - N_BURN) / tau_max,
            "emission": sel["name"],
            "emission_vs_flow": max(float(mmd_sq(
                z_s, w_s, flow_t.mu_fn(z_s), flow_t.c_rho, sigma)), 0.0),
            "emission_vs_pcn": max(float(mmd_sq(
                z_s, w_s, pcn_t.mu_fn(z_s), pcn_t.c_rho, sigma)), 0.0),
            "floor_vs_flow": max(float(mmd_sq(
                z0, eq, flow_t.mu_fn(z0), flow_t.c_rho, sigma)), 0.0),
            "floor_vs_pcn": max(float(mmd_sq(
                z0, eq, pcn_t.mu_fn(z0), pcn_t.c_rho, sigma)), 0.0),
            "flow_vs_pcn": _gap(pcn_t, flow_t, bank),
            "flow_vs_pcn_fresh2k": max(
                float(flow_t.c_rho) + float(pcn_t.c_rho)
                - 2.0 * _cross_mean(pcn_t, fresh2k), 0.0),
            "chain_vs_chain_pairwise": pair_gaps,
            "chain_vs_chain_max": max(pair_gaps),
            "chain_vs_chain_halves": halves_gap,
            "secs": time.time() - t_obs,
        }
        rec["power_ratio_pairwise"] = (
            rec["flow_vs_pcn"] / rec["chain_vs_chain_max"])
        rec["power_ratio_halves"] = (
            rec["flow_vs_pcn"] / rec["chain_vs_chain_halves"])
        with open(_obs_file(o), "w") as fh:
            json.dump(rec, fh, indent=2)
        rows.append(rec)
        print(f"[obs {o}] flow-vs-pCN={rec['flow_vs_pcn']:.3e}  "
              f"emission: vs flow={rec['emission_vs_flow']:.3e} "
              f"vs pCN={rec['emission_vs_pcn']:.3e}  "
              f"floor vs pCN={rec['floor_vs_pcn']:.3e}  "
              f"chain noise: max-pair={rec['chain_vs_chain_max']:.3e} "
              f"halves={rec['chain_vs_chain_halves']:.3e}  "
              f"tau_max={tau_max:.0f}  "
              f"({rec['secs'] / 60:.1f} min)", flush=True)

    med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
    summary = {
        "flow_vs_pcn": med("flow_vs_pcn"),
        "emission_vs_flow": med("emission_vs_flow"),
        "emission_vs_pcn": med("emission_vs_pcn"),
        "floor_vs_flow": med("floor_vs_flow"),
        "floor_vs_pcn": med("floor_vs_pcn"),
        "chain_vs_chain_max": med("chain_vs_chain_max"),
        "chain_vs_chain_halves": med("chain_vs_chain_halves"),
        "power_ratio_pairwise": med("power_ratio_pairwise"),
        "power_ratio_halves": med("power_ratio_halves"),
        "tau_median": med("tau_median"), "tau_max": med("tau_max"),
        "post_burn_over_tau_max": med("post_burn_over_tau_max"),
        "emission_over_floor_vs_pcn":
            med("emission_vs_pcn") / med("floor_vs_pcn"),
        "emission_over_floor_vs_flow":
            med("emission_vs_flow") / med("floor_vs_flow"),
        "gap_over_emission_margin":
            med("flow_vs_pcn") / med("emission_vs_flow"),
    }
    record = {
        "protocol": {
            "beta": BETA, "n_steps": N_STEPS, "n_burn": N_BURN,
            "thin": THIN, "n_chains": N_CHAINS, "n_obs": N_OBS, "M": M,
            "kept_per_chain": (N_STEPS - N_BURN) // THIN,
            "overdispersed_starts":
                f"chains 0-1 prior draws, chains 2-3 prior x {OVERDISP}",
            "beta_source": "darcy_stuart_pcn_beta.json best_beta",
            "sigma": sigma, "sigma_y": sigma_y,
            "base_seed": base_seed,
            "seed_offsets": {"chain": SEED_CHAIN, "start": SEED_START},
            "checkpoint": ck_path,
        },
        "rows": rows,
        "summary": summary,
        "pilot": {
            "file": "dq_darcy_stuart results.json pcn_audit",
            "protocol": "2 chains x 10k post-burn at beta=0.08, thin 5",
            "why_underpowered": "chain_vs_chain ~0.05-0.07 EXCEEDED the "
                                "flow_vs_pcn gap ~0.014-0.019 it brackets",
        },
        "note": (
            "MCMC as an audit of the reference, never the reference. All "
            "quantities are squared MMD at the deployed bandwidth. "
            "flow_vs_pcn is the reference flow's own distance to the exact-"
            "forward posterior; emission_vs_flow is the quadrature's error "
            "against the reference it is handed (the guarantee's currency); "
            "emission_vs_pcn and floor_vs_pcn rescore emission and i.i.d. "
            "floor against the chains. chain_vs_chain is the audit's noise "
            "floor: powered iff it sits well below flow_vs_pcn "
            "(power_ratio_*)."
        ),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nSaved to {OUT}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
