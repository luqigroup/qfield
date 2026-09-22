"""Aggregate the CNF-tomography designed-quadrature warm start over 3 training
seeds and re-render the MMD-vs-M figure with a 3-seed confidence band.

The single-seed run of `scripts/dq_cnf_tomography.py` (seed 0) is joined by two
full-pipeline re-runs (re-sample the CNF bank and re-train the conditioned
amortizer at fresh `base_seed` / `seed`). This script collects the per-`M`
floor / warm / per-instance-optimum (each a median MMD^2 over the 64 held-out
observations) across the three seeds, reports the across-seed median and
spread, and re-plots the MMD-vs-M panel with a 3-seed min-max band on each arm.

It re-runs no pipeline; it only reads the three saved `results.json` files and
re-plots the figure. Run:

    python scripts/dq_cnf_tomography_multiseed.py

Writes:
    dq_cnf_tomography_mmd_vs_M.{pdf,png}   (3-seed band overlay)

and prints the per-seed warm/floor and the aggregated per-`M` interval.
"""

from __future__ import annotations
from projorg import datadir, plotsdir  # noqa: E402

import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = datadir("checkpoints")
FIGDIR = plotsdir("paper")

# The three full-pipeline seeds (re-sampled bank plus re-trained amortizer at
# the listed base_seed/seed). These are the experiment-directory names for
# r_min=32, whitened, sqrt_r.
_SEED_DIRS = {
    0: (
        "dq_cnf_tomography_subspace-pca_n_img-64_n_angles-60_phi_deg-60.0_"
        "sigma_obs-0.05_n_detectors-90_prior_delta-0.5_prior_alpha-2.0_"
        "prior_tau-1.0_rank_cap-32_r_energy-0.9_r_min-32_m_list-32-64-128-256-"
        "512_whiten-1_white.6a07204b489ef83a4007c3482455fac5043804c0"
    ),
    1: (
        "dq_cnf_tomography_subspace-pca_n_img-64_n_angles-60_phi_deg-60.0_"
        "sigma_obs-0.05_n_detectors-90_prior_delta-0.5_prior_alpha-2.0_"
        "prior_tau-1.0_rank_cap-32_r_energy-0.9_r_min-32_m_list-32-64-128-256-"
        "512_whiten-1_white.84d99dca8c65d2ea4314fc0fe30b89ecdc07b9be"
    ),
    2: (
        "dq_cnf_tomography_subspace-pca_n_img-64_n_angles-60_phi_deg-60.0_"
        "sigma_obs-0.05_n_detectors-90_prior_delta-0.5_prior_alpha-2.0_"
        "prior_tau-1.0_rank_cap-32_r_energy-0.9_r_min-32_m_list-32-64-128-256-"
        "512_whiten-1_white.561f3c405f6f0e6e17da7d8dba8efe6a5fa2befd"
    ),
}

# Per-arm palette and labels: floor slate-blue, warm orange, opt grey (the
# per-instance optimum, started from z0).
_PALETTE = {"floor": "#3a7ca5", "warm": "#ff7f0e", "opt": "#7f7f7f"}
_LABELS = {
    "floor": "i.i.d. floor",
    "warm": "warm start (1 fwd pass)",
    "opt": "per-instance descent",
}
_ARMS = ("floor", "warm", "opt")


def _load() -> dict[int, dict]:
    out = {}
    for s, d in _SEED_DIRS.items():
        p = os.path.join(CKPT, d, "results.json")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"seed {s} results.json missing: {p}")
        out[s] = json.load(open(p))
    return out


def _per_seed_rows(res: dict) -> dict[int, dict]:
    """`{M: row}` keyed by node budget for one seed's results."""
    return {r["M"]: r for r in res["rows"]}


def aggregate(res: dict[int, dict]):
    """Across-seed per-`M` median / min / max / std for each arm + warm/floor.

    Each seed contributes one number per (M, arm): the per-M median MMD^2 over
    the 64 held-out observations (``row[arm][0]``). The across-seed statistics
    are computed over those 3 seed medians.
    """
    seeds = sorted(res)
    rows = {s: _per_seed_rows(res[s]) for s in seeds}
    ms = [r["M"] for r in res[seeds[0]]["rows"]]
    agg = {}
    for M in ms:
        entry = {}
        for arm in _ARMS:
            vals = [rows[s][M][arm][0] for s in seeds]
            entry[arm] = {
                "median": st.median(vals),
                "min": min(vals),
                "max": max(vals),
                "std": st.pstdev(vals),
                "per_seed": vals,
            }
        wf = [rows[s][M]["warm_over_floor"] for s in seeds]
        entry["warm_over_floor"] = {
            "median": st.median(wf),
            "min": min(wf),
            "max": max(wf),
            "std": st.pstdev(wf),
            "per_seed": wf,
        }
        agg[M] = entry
    return ms, agg, rows


def report(ms, agg, rows):
    seeds = sorted(rows)
    print("=== CNF tomography (E-CNF) -- 3-seed CI ===")
    print("per-seed warm/floor by M:")
    for s in seeds:
        line = f"  seed {s}: " + " ".join(
            f"M{M}={rows[s][M]['warm_over_floor']:.3f}" for M in ms
        )
        print(line)
    print("\nacross-seed per-M median MMD^2 [min, max]:")
    for M in ms:
        e = agg[M]
        parts = []
        for arm in _ARMS:
            a = e[arm]
            parts.append(
                f"{arm}={a['median']:.3e}[{a['min']:.3e},{a['max']:.3e}]"
            )
        wf = e["warm_over_floor"]
        print(
            f"  M={M:>3}: " + " ".join(parts)
            + f" || warm/floor med={wf['median']:.3f} "
            f"[{wf['min']:.3f},{wf['max']:.3f}] std={wf['std']:.4f}"
        )
    # Every seed must beat the floor and reach the optimum at every M.
    all_below = all(
        rows[s][M]["warm"][0] < rows[s][M]["floor"][0]
        for s in seeds for M in ms
    )
    all_at_opt = all(
        rows[s][M]["warm"][0] >= rows[s][M]["opt"][0] - 1e-12
        for s in seeds for M in ms
    )
    wf_all = [rows[s][M]["warm_over_floor"] for s in seeds for M in ms]
    print(
        f"\nROBUSTNESS: warm<floor every (seed,M)={all_below}; "
        f"warm>=opt every (seed,M)={all_at_opt}; "
        f"warm/floor range over all (seed,M)=[{min(wf_all):.3f},"
        f"{max(wf_all):.3f}] (headline ~0.35-0.38)."
    )
    return all_below, all_at_opt


def plot(ms, agg):
    """MMD-vs-M with a 3-seed min-max band (median line) per arm."""
    with plt.rc_context(
        {"pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
         "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9}
    ):
        fig, axm = plt.subplots(1, 1, figsize=(4.6, 3.6))
        for arm in _ARMS:
            med = [agg[M][arm]["median"] for M in ms]
            lo = [agg[M][arm]["min"] for M in ms]
            hi = [agg[M][arm]["max"] for M in ms]
            axm.fill_between(
                ms, lo, hi, color=_PALETTE[arm], alpha=0.18, linewidth=0,
            )
            axm.plot(
                ms, med, color=_PALETTE[arm], marker="o", markersize=4,
                linewidth=1.5, label=_LABELS[arm],
            )
        axm.set_xscale("log", base=2)
        axm.set_yscale("log")
        axm.set_xticks(ms)
        axm.set_xticklabels([str(m) for m in ms])
        axm.set_xlabel(r"node budget $M$")
        axm.set_ylabel(r"$\mathrm{MMD}^2(Q, \rho)$")
        axm.spines["top"].set_visible(False)
        axm.spines["right"].set_visible(False)
        # Note the band semantics in the legend title.
        leg = axm.legend(
            frameon=False, fontsize=8,
            title="median over 3 seeds; band = seed min--max",
        )
        leg.get_title().set_fontsize(7)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            path = os.path.join(FIGDIR, f"dq_cnf_tomography_mmd_vs_M.{ext}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved {path}", flush=True)
        plt.close(fig)


if __name__ == "__main__":
    res = _load()
    ms, agg, rows = aggregate(res)
    report(ms, agg, rows)
    plot(ms, agg)
