"""Six-panel below-floor composite, one full-width figure.

    row 1 (closed form)   Gaussian d=2   |  GMM d=10  |  banana d=2
    row 2 (the ladder)    tomography     |  trained   |  groundwater
                          (exact cond.)  |  flow      |  (Darcy)

Read left to right and top to bottom, the reference grows harder to supply.

This is a pure render step over saved records. It trains nothing, samples
nothing and touches no checkpoint weights; it reads ``results.json``
artifacts:

  * top row: ``scripts/render_dq_figures.py``'s ``CANON`` rows, i.e.
    ``data/checkpoints/designed_quadrature_*_amortized*/results.json``,
    resolved by the (target, d, sigma, m_list) pin that module defines. The
    pin is imported rather than copied, so the two cannot come to sit on
    different grids.
  * bottom left: ``scripts/dq_tomography.py``'s run.
  * bottom middle: the three training-seed runs of
    ``scripts/dq_cnf_tomography_multiseed.py`` (min-max band across seeds).
  * bottom right: ``scripts/dq_darcy_stuart.py``'s run, read on the held-out
    bank B, with the shuffled-conditioner null control.

Marker sizes are uniform and conditioning-limited cells are drawn open
rather than silently ridged.

Usage:
    python scripts/render_composite_floor.py
"""

from __future__ import annotations
from projorg import datadir, plotsdir  # noqa: E402

import glob
import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _figstyle import apply_house_style, despine  # noqa: E402
from dq_cnf_tomography_multiseed import _SEED_DIRS  # noqa: E402
from render_dq_figures import (  # noqa: E402
    MMD_VS_M_ROWS,
    _load_canon,
    _load_multiseed,
    _ms_lookup,
)

CKPT_ROOT = datadir("checkpoints")
FIG_DIR = plotsdir("paper")
STEM = "dq_floor_composite"

# The node-budget grid. Every panel must resolve on it; a rung that does not
# is an error rather than a panel drawn on whatever grid its record holds.
M_LIST = [32, 64, 128, 256, 512]

# The checkpoint-directory globs for the three ladder rungs. Each carries the
# grid token, because runs of the same experiment on another grid share the
# readable name prefix and differ only in the hash tail.
_GRID = "m_list-32-64-128-256-512"
TOMO_GLOB = f"dq_tomography_n_img-16_*{_GRID}*"
DARCY_GLOB = f"dq_darcy_stuart_npe_*{_GRID}*"

# Semantic palette, shared with the per-rung scripts.
C_FLOOR = "#3a7ca5"      # slate-blue, dashed: the i.i.d.-MC floor
C_REWEIGHT = "#2a9d8f"   # teal: the free lever at the untouched seeds
C_OURS = "#ff7f0e"       # orange: move and reweight
C_ORACLE = "#7f7f7f"     # grey, dotted: the per-instance descent
C_NULL = "#b06fa8"       # mauve: the shuffled-conditioner null control

LBL_FLOOR = "i.i.d. floor"
LBL_REWEIGHT = "reweight only"
LBL_OURS = "move and reweight"
LBL_ORACLE = "per-instance descent"
LBL_NULL = "shuffled conditioner"

# Uniform marker sizes, never scaled by density or weight.
MS = 3.4
MS_OURS = 3.8
MS_OPEN = 7.0


# --------------------------------------------------------------------------
# Record loading. One loader per rung; each raises rather than returning a
# blank panel.
# --------------------------------------------------------------------------
def _one_results(pattern: str, what: str) -> dict:
    """The single ``results.json`` under ``CKPT_ROOT/pattern``."""
    hits = sorted(glob.glob(os.path.join(CKPT_ROOT, pattern, "results.json")))
    if len(hits) != 1:
        raise RuntimeError(
            f"{what}: expected exactly one deployed-grid record under "
            f"{pattern!r}, found {len(hits)}: {hits}. Pin the glob further "
            "rather than letting glob order pick the rung."
        )
    with open(hits[0]) as fh:
        res = json.load(fh)
    ms = [int(r["M"]) for r in res["rows"]]
    if ms != M_LIST:
        raise RuntimeError(
            f"{what}: record is on grid {ms}, not the deployed {M_LIST}."
        )
    return res


def _load_cnf() -> dict:
    """Across-training-seed median / min / max per arm, per budget.

    Reads the same three full-pipeline seed runs
    ``scripts/dq_cnf_tomography_multiseed.py`` aggregates, and reproduces its
    statistic: each seed contributes one number per (M, arm), that seed's
    per-M median over the held-out observations, and the band is the minimum
    to maximum over the three seed medians.
    """
    per_seed = {}
    for s, d in _SEED_DIRS.items():
        p = os.path.join(CKPT_ROOT, d, "results.json")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"CNF rung: seed {s} record missing: {p}")
        with open(p) as fh:
            per_seed[s] = {int(r["M"]): r for r in json.load(fh)["rows"]}
    for s, rows in per_seed.items():
        if sorted(rows) != M_LIST:
            raise RuntimeError(
                f"CNF rung: seed {s} is on grid {sorted(rows)}, not {M_LIST}."
            )
    out = {}
    for arm in ("floor", "warm", "opt"):
        med, lo, hi = [], [], []
        for M in M_LIST:
            vals = [per_seed[s][M][arm][0] for s in sorted(per_seed)]
            med.append(st.median(vals))
            lo.append(min(vals))
            hi.append(max(vals))
        out[arm] = (med, lo, hi)
    out["_n_seeds"] = len(per_seed)
    return out


# --------------------------------------------------------------------------
# Drawing.
# --------------------------------------------------------------------------
def _arm(ax, m, med, lo, hi, color, label, *, style="solid", band=True,
         marker="o", ms=MS, lw=1.4, z=3):
    """One arm: median line plus (optionally) its spread band."""
    ls = {"solid": "-", "dashed": (0, (5, 3)), "dotted": (0, (1, 2))}[style]
    if band and lo is not None:
        ax.fill_between(m, lo, hi, color=color, alpha=0.16, lw=0, zorder=z - 1)
    ax.plot(m, med, color=color, lw=lw, ls=ls, marker=marker, ms=ms,
            label=label, zorder=z)


def _open_marks(ax, m, y, flags, color):
    """Draw conditioning-limited cells open.

    A cell whose seed Gram is conditioned past the limit has a rank-deficient
    weight solve, so which weight vector each arm lands on is decided by
    solve noise rather than by method. The value still carries digits, so it
    is drawn open-faced.
    """
    for M_, v, bad in zip(m, y, flags):
        if bad:
            ax.plot([M_], [v], marker="o", ms=MS_OPEN, mfc="white", mec=color,
                    mew=1.3, ls="none", zorder=8)


def _finish(ax, title, *, ylabel=None, xlabel=True):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(M_LIST)
    ax.set_xticklabels([str(v) for v in M_LIST])
    # Only the bottom row is labelled: the budget grid is the same in all
    # six panels, so a second copy of the axis name costs vertical space and
    # buys nothing.
    if xlabel:
        ax.set_xlabel(r"node budget $M$")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.5)
    despine(ax)


def _panel_closed_form(ax, res, title, ylabel, ms_cfg=None):
    """A closed-form target: four arms, with open marks on limited cells.

    With ``ms_cfg`` (a multiseed config block) the curve is the
    across-training-seed median with the IQR shaded, as
    ``render_dq_figures.render_mmd_vs_M`` draws it.
    """
    rows = res["rows"]
    m = [int(r["M"]) for r in rows]
    if m != M_LIST:
        raise RuntimeError(f"{title}: record on grid {m}, not {M_LIST}.")
    flags = [bool(r.get("cond_limited")) for r in rows]
    _arm(ax, m,
         [r["floor"][0] for r in rows],
         [min(r["floor"][1], r["floor"][2]) for r in rows],
         [max(r["floor"][1], r["floor"][2]) for r in rows],
         C_FLOOR, LBL_FLOOR, style="dashed", z=3)
    if all("reweight" in r for r in rows):
        _arm(ax, m, [r["reweight"][0] for r in rows], None, None,
             C_REWEIGHT, LBL_REWEIGHT, band=False, marker="s", ms=3.1, z=4)
    else:
        raise RuntimeError(
            f"{title}: the record carries no reweight arm, so the panel would "
            "draw three of the caption's four. Re-run the target's trainer."
        )
    _arm(ax, m, [r["oracle"][0] for r in rows], None, None,
         C_ORACLE, LBL_ORACLE, style="dotted", band=False, ms=3.0, z=4)
    ours = [r["warm"][0] for r in rows]
    ours_lo = ours_hi = None
    if ms_cfg is not None:
        med, lo, hi = [], [], []
        for M_ in m:
            blk = ms_cfg["per_M"].get(str(M_))
            if blk is None:
                raise RuntimeError(
                    f"{title}: the multiseed record has no budget {M_}, so "
                    "this panel would be single-seed while the others are "
                    "banded. Complete the record or drop the band."
                )
            md, q25, q75 = blk["warm_mmd2"]["median_iqr"]
            med.append(md)
            lo.append(min(q25, q75))
            hi.append(max(q25, q75))
        ours, ours_lo, ours_hi = med, lo, hi
    _arm(ax, m, ours, ours_lo, ours_hi, C_OURS, LBL_OURS,
         band=ours_lo is not None, ms=MS_OURS, lw=1.8, z=6)
    _open_marks(ax, m, ours, flags, C_OURS)
    _finish(ax, title, ylabel=ylabel, xlabel=False)


def _panel_tomography(ax, res, ylabel):
    """The exact conditional rung: limited-angle tomography."""
    rows = res["rows"]
    m = [int(r["M"]) for r in rows]
    flags = [bool(r.get("cond_limited")) for r in rows]
    _arm(ax, m, [r["floor"][0] for r in rows],
         [min(r["floor"][1], r["floor"][2]) for r in rows],
         [max(r["floor"][1], r["floor"][2]) for r in rows],
         C_FLOOR, LBL_FLOOR, style="dashed", z=3)
    _arm(ax, m, [r["reweight"][0] for r in rows], None, None,
         C_REWEIGHT, LBL_REWEIGHT, band=False, marker="s", ms=3.1, z=4)
    _arm(ax, m, [r["oracle"][0] for r in rows], None, None,
         C_ORACLE, LBL_ORACLE, style="dotted", band=False, ms=3.0, z=4)
    ours = [r["warm"][0] for r in rows]
    _arm(ax, m, ours, None, None, C_OURS, LBL_OURS, band=False, ms=MS_OURS,
         lw=1.8, z=6)
    _open_marks(ax, m, ours, flags, C_OURS)
    _finish(ax, "exact conditional (tomography)", ylabel=ylabel)


def _panel_cnf(ax, agg, ylabel):
    """The trained-flow rung, banded across the three full-pipeline seeds."""
    m = M_LIST
    fl, fl_lo, fl_hi = agg["floor"]
    _arm(ax, m, fl, fl_lo, fl_hi, C_FLOOR, LBL_FLOOR, style="dashed", z=3)
    op, op_lo, op_hi = agg["opt"]
    _arm(ax, m, op, op_lo, op_hi, C_ORACLE, LBL_ORACLE, style="dotted", ms=3.0,
         z=4)
    ou, ou_lo, ou_hi = agg["warm"]
    _arm(ax, m, ou, ou_lo, ou_hi, C_OURS, LBL_OURS, ms=MS_OURS, lw=1.8, z=6)
    # This rung's records carry no reweight arm, so the panel draws three of
    # the figure's five curves.
    print(
        "  [ARM MISSING] trained-flow rung: its records carry no reweight "
        "arm, so that panel draws floor / emission / per-instance only."
    )
    _finish(ax, "trained flow (same problem)", ylabel=ylabel)


def _panel_darcy(ax, res, ylabel):
    """The groundwater capstone, read on the held-out bank B.

    Bank A's ``ours <= floor`` holds by construction, so bank B is the
    falsifiable column and is what ``scripts/dq_darcy_stuart.py`` plots. The
    per-instance arm stays the in-bank descent, which is optimistic by
    construction.
    """
    rows = res["rows"]
    m = [int(r["M"]) for r in rows]
    flags = [bool(r.get("cond_limited")) for r in rows]
    for key, color, label, style, marker, ms in (
        ("floor_heldout", C_FLOOR, LBL_FLOOR, "dashed", "o", MS),
        ("reweight_heldout", C_REWEIGHT, LBL_REWEIGHT, "solid", "s", 3.1),
        ("null_heldout", C_NULL, LBL_NULL, "solid", "^", 3.1),
        ("opt", C_ORACLE, LBL_ORACLE, "dotted", "o", 3.0),
    ):
        _arm(ax, m, [r[key][0] for r in rows],
             [min(r[key][1], r[key][2]) for r in rows],
             [max(r[key][1], r[key][2]) for r in rows],
             color, label, style=style, marker=marker, ms=ms,
             band=(key == "floor_heldout"), z=4)
    ours = [r["ours_heldout"][0] for r in rows]
    _arm(ax, m, ours, None, None, C_OURS, LBL_OURS, band=False, ms=MS_OURS,
         lw=1.8, z=6)
    _open_marks(ax, m, ours, flags, C_OURS)
    _finish(ax, "groundwater (Darcy), held-out bank", ylabel=ylabel)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    apply_house_style()

    canon = _load_canon()
    missing = [k for k in MMD_VS_M_ROWS if canon.get(k) is None]
    if missing:
        raise RuntimeError(
            f"composite: no canonical run for {missing}. The caption claims "
            "the emission is below the floor at EVERY budget on references of "
            "every kind; a blank panel would ship that claim with a hole in "
            "it. Pin the row to a deployed-grid checkpoint, or retrain it."
        )
    tomo = _one_results(TOMO_GLOB, "tomography rung")
    darcy = _one_results(DARCY_GLOB, "groundwater rung")
    cnf = _load_cnf()
    ms = _load_multiseed()
    if ms is None:
        raise RuntimeError(
            "composite: no data/multiseed/dq_toy_seeds.json. The supplement's "
            "error-bar section states the closed-form panels' emission is a "
            "training-seed median with an IQR band; drawing them single-seed "
            "would falsify it."
        )
    print(
        f"[load] closed form {list(MMD_VS_M_ROWS)}; tomography, "
        f"{cnf['_n_seeds']}-seed trained flow, groundwater -- all on {M_LIST}"
    )

    # (family, kernel, d) keys into the multiseed record, in MMD_VS_M_ROWS
    # order; the same lookup render_dq_figures.render_mmd_vs_M uses.
    ms_keys = (("gaussian", "iso", 2), ("gmm", "iso", 10), ("banana", "iso", 2))
    ms_cfgs = [_ms_lookup(ms, *k) for k in ms_keys]
    if any(c is None for c in ms_cfgs):
        absent = [k for k, c in zip(ms_keys, ms_cfgs) if c is None]
        raise RuntimeError(
            f"composite: the multiseed record has no config for {absent}, so "
            "some closed-form panels would be banded and others not -- which "
            "is how a 2-of-3 record comes to read as 3-of-3."
        )
    print(
        f"[multiseed] {len(ms.get('seeds', []))} training seeds -> IQR band "
        "on every closed-form emission curve"
    )

    ylab = r"$\mathrm{MMD}^2(Q,\rho)$"
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 3.95))
    titles = ("Gaussian, $d=2$", "GMM, $d=10$", "banana, $d=2$")
    for j, (key, title) in enumerate(zip(MMD_VS_M_ROWS, titles)):
        _panel_closed_form(axes[0, j], canon[key], title,
                           ylab if j == 0 else None, ms_cfg=ms_cfgs[j])
    _panel_tomography(axes[1, 0], tomo, ylab)
    _panel_cnf(axes[1, 1], cnf, None)
    _panel_darcy(axes[1, 2], darcy, None)

    # One shared legend for six panels: no panel is large enough to carry its
    # own without covering a curve, and five of the arms are common to the
    # figure. Deduplicated by label, in draw order.
    handles, labels = [], []
    for ax in axes.ravel():
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(labels),
               loc="lower center", bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{STEM}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
