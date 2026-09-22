"""Plot the emission's squared discrepancy relative to the floor against the
node count, one panel, every reference of the ladder. The floor is the line at
one; below it is better.

Reads the saved record ``dq_floor_composite.json`` and never retrains. Cells
the record flags as ill-conditioned (``cond_limited``) are drawn open and read
as values. Bands are drawn only where the record carries the ratio's own band
over training seeds (the three closed forms); the trained flow's record has
the emission's and the floor's seed ranges separately, and the tomography
exact rung and the groundwater rung are single-seed, so those carry none.

    python scripts/render_depth_ratio.py
"""
from __future__ import annotations
from projorg import datadir, plotsdir  # noqa: E402

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _figstyle import apply_house_style, despine  # noqa: E402

DIAG = datadir("records")
FIG_DIR = plotsdir("paper")
STEM = "dq_depth_ratio"
C_FLOOR = "#3a7ca5"

# (record key, legend label, colour) -- the ladder's order, easiest first.
REFS = (
    ("Gaussian, d = 2", r"Gaussian, $d=2$", "#7570b3"),
    ("GMM, d = 10", r"mixture, $d=10$", "#e7298a"),
    ("banana, d = 2", r"banana, $d=2$", "#1b9e77"),
    ("tomography", "tomography, exact", "#66a61e"),
    ("trained_flow", "tomography, flow", "#a6761d"),
    ("groundwater", "groundwater, flow", "#222222"),
)
# Vertical nudges (log10 units) for the right-end labels of curves whose
# ends nearly coincide.
NUDGE = {"GMM, d = 10": 0.22, "tomography": -0.22}


def _series(panel: dict, key: str):
    """Return (M, median ratio, lo, hi, open) for one reference."""
    ms = list(panel["m_list"])
    if key in ("Gaussian, d = 2", "GMM, d = 10", "banana, d = 2"):
        r = panel["warm_over_floor_multiseed"]
        med, lo, hi = [x[0] for x in r], [x[1] for x in r], [x[2] for x in r]
        opn = list(panel["cond_limited"])
    elif key == "tomography":
        med, lo, hi = list(panel["warm_over_floor"]), None, None
        opn = list(panel["cond_limited"])
    elif key == "trained_flow":
        # The record carries min-max of the emission and of the floor across
        # seeds SEPARATELY, not of their ratio, so no band is drawn here.
        f = panel["floor"]["median"]
        med = [w / x for w, x in zip(panel["warm"]["median"], f)]
        lo, hi = None, None
        opn = [False] * len(ms)
    elif key == "groundwater":
        med, lo, hi = list(panel["ours_over_floor_heldout"]), None, None
        opn = list(panel["cond_limited"])
    else:
        raise KeyError(key)
    return ms, med, lo, hi, opn


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    apply_house_style()
    with open(os.path.join(DIAG, "dq_floor_composite.json")) as fh:
        rec = json.load(fh)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    m_all = rec["m_list"]
    ax.axhline(1.0, color=C_FLOOR, lw=1.2, ls="--", zorder=1,
               label="floor (independent samples)")
    for key, label, col in REFS:
        ms, med, lo, hi, opn = _series(rec["panels"][key], key)
        ax.plot(ms, med, color=col, lw=1.5, zorder=3)
        if lo is not None:
            ax.fill_between(ms, lo, hi, color=col, alpha=0.22, lw=0, zorder=2)
        for m, v, o in zip(ms, med, opn):
            ax.plot(m, v, "o", ms=4.0, color=col, mfc="white" if o else col,
                    mew=1.1, zorder=4)
        y_lab = med[-1] * 10 ** NUDGE.get(key, 0.0)
        ax.annotate(label, xy=(ms[-1], med[-1]), xytext=(ms[-1] * 1.13, y_lab),
                    fontsize=6.6, color=col, va="center", ha="left",
                    annotation_clip=False)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(m_all)
    ax.set_xticklabels([str(m) for m in m_all])
    ax.set_xlim(m_all[0] / 1.15, m_all[-1] * 2.9)
    ax.set_xlabel(r"node count $M$")
    ax.set_ylabel(r"$\mathrm{MMD}^2$ relative to the floor")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left",
              handlelength=1.8)
    despine(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{STEM}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
