"""The Lotka--Volterra benchmark, in three panels.

Left, at matched node count: squared discrepancy against the node count for
the returned quadrature, the reweighted samples, independent samples and every
per-observation construction, median over the members the scoring bank
resolves at that node count (the same subset for every arm). Middle, at
matched information: squared discrepancy against the number of reference
samples every method receives. Right: seconds per observation against that
number.

Reads ``lv_peer_ladder_final.json`` (matched node count) and
``lv_unified_budget.json`` (matched information); never retrains.

    python scripts/render_lv_benchmark.py
"""
from __future__ import annotations
from projorg import datadir, plotsdir  # noqa: E402

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _figstyle import C_OURS, C_SOLVE, apply_house_style, despine  # noqa: E402

DIAG = datadir("records")
FIG_DIR = plotsdir("paper")
STEM = "lv_benchmark"
C_IID = "#3a7ca5"
C_RW = C_SOLVE
C_PEER = {"stein": "#c94c4c", "thinning": "#8e6bb2", "herding": "#c97b3a",
          "recombine": "#6b8e23", "sbq": "#222222"}
LABEL = {"iid": "independent samples", "emission": "the returned quadrature",
         "reweight": "reweighted samples", "ours": "the returned quadrature",
         "floor": "independent samples",
         "stein": "Stein thinning", "thinning": "kernel thinning",
         "herding": "kernel herding", "recombine": "recombination",
         "sbq": "greedy search (BQ weights)"}


def _colour(arm: str) -> str:
    if arm in ("emission", "ours"):
        return C_OURS
    if arm in ("iid", "floor"):
        return C_IID
    if arm == "reweight":
        return C_RW
    return C_PEER[arm]


def _width(arm: str) -> float:
    return 2.2 if arm in ("emission", "ours") else 1.5 if arm == "reweight" \
        else 1.2


def _style(arm: str) -> str:
    return "--" if arm == "sbq" else "-"


def _z(arm: str) -> int:
    return 5 if arm == "sbq" else 3 if arm in ("emission", "ours") else 2


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    apply_house_style()
    L = json.load(open(os.path.join(DIAG, "lv_peer_ladder_final.json")))
    U = json.load(open(os.path.join(DIAG, "lv_unified_budget.json")))
    cells = list(L["cells"])
    budgets = [32, 64, 128]
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.0, 2.35))

    # ---- left: matched node count --------------------------------------
    # One common subset per node count: the members the bank resolves for the
    # returned quadrature. Every arm is resolved on it, so the medians compare
    # the same members.
    subset = {M: [c for c in cells if L["cells"][c][str(M)]["resolved"]["emission"]]
              for M in budgets}
    for arm in ("iid", "stein", "thinning", "herding", "recombine", "sbq",
                "reweight", "emission"):
        med = [float(np.median([L["cells"][c][str(M)][arm]
                                for c in subset[M]])) for M in budgets]
        col = _colour(arm)
        ax.plot(budgets, med, _style(arm), marker="o", color=col,
                lw=_width(arm), ms=4.2, zorder=_z(arm), label=LABEL[arm])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel(r"node count $M$")
    ax.set_ylabel(r"$\mathrm{MMD}^2$ to the reference")
    ax.set_title("matched node count", fontsize=8.5, color="0.25")
    handles, labels = ax.get_legend_handles_labels()
    despine(ax)

    # ---- middle and right: matched information -------------------------
    ns = sorted(int(n) for n in U["cells"])
    for arm in ("floor", "stein", "thinning", "herding", "recombine", "sbq",
                "ours"):
        vals = [U["cells"][str(n)][arm] for n in ns]
        bx.plot(ns, vals, _style(arm), marker="o", color=_colour(arm),
                lw=_width(arm), ms=3.8, zorder=_z(arm))
    bx.set_xscale("log", base=2)
    bx.set_yscale("log")
    bx.set_xticks(ns)
    bx.set_xticklabels([str(n) for n in ns])
    bx.set_xlabel(r"reference samples $n$")
    bx.set_ylabel(r"$\mathrm{MMD}^2$ to the reference")
    bx.set_title(r"matched information, $M=32$", fontsize=8.5, color="0.25")
    despine(bx)
    for arm in ("stein", "thinning", "herding", "recombine", "sbq", "ours"):
        secs = [U["cells"][str(n)]["secs_per_cell"][arm] for n in ns]
        cx.plot(ns, secs, _style(arm), marker="o", color=_colour(arm),
                lw=_width(arm), ms=3.8, zorder=_z(arm))
    cx.set_xscale("log", base=2)
    cx.set_yscale("log")
    cx.set_xticks(ns)
    cx.set_xticklabels([str(n) for n in ns])
    cx.set_xlabel(r"reference samples $n$")
    cx.set_ylabel("seconds per observation")
    cx.set_title("cost", fontsize=8.5, color="0.25")
    despine(cx)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.36, top=0.90,
                        wspace=0.38)
    fig.legend(handles, labels, fontsize=7.2, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               columnspacing=1.4, handlelength=1.6, labelspacing=0.3)
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{STEM}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
