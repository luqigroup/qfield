"""Certified fraction against certification-split size, per node count.

The fraction of node sets whose comparison between the two solved candidates
is certified, against the size of the certification split, one curve per node
count, on the groundwater rung.

Reads only ``dq_darcy_certification.json`` and never recomputes a slack. The
rate plotted is the deployed head's, with its 95% Wilson interval as
recorded. The vertical line marks the size of the bank the rung already
spends on its kernel-mean estimate at the three smaller node counts.

    python scripts/render_certification_rate.py
"""
from __future__ import annotations
from projorg import datadir, plotsdir  # noqa: E402

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _figstyle import apply_house_style, despine  # noqa: E402

DIAG = datadir("records")
FIG_DIR = plotsdir("paper")
STEM = "dq_certification_rate"
C_SOLVE = "#7f7f7f"
# One orange ramp, dark for the largest node count.
RAMP = ("#fdae6b", "#fd8d3c", "#ff7f0e", "#d9650a", "#a64b05")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    apply_house_style()
    with open(os.path.join(DIAG, "dq_darcy_certification.json")) as fh:
        rec = json.load(fh)
    by_m = defaultdict(list)
    for row in rec["rates"]:
        sel = row.get("deployed_selection") or row["split_selection"]
        by_m[int(row["M"])].append((int(row["L2"]), float(sel["rate"]),
                                    tuple(sel["rate_ci95"])))
    bank = int(rec["draw_ledger"].get("bank_L", 8192)) if isinstance(
        rec.get("draw_ledger"), dict) else 8192
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    ax.axvline(bank, color=C_SOLVE, lw=1.0, ls=":", zorder=1)
    ax.text(bank * 1.15, 0.52, "kernel-mean\nbank", fontsize=6.8,
            color=C_SOLVE, ha="left", va="center")
    for col, m in zip(RAMP, sorted(by_m)):
        rows = sorted(by_m[m])
        l2 = [r[0] for r in rows]
        rate = [r[1] for r in rows]
        lo = [r[2][0] for r in rows]
        hi = [r[2][1] for r in rows]
        ax.fill_between(l2, lo, hi, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(l2, rate, "-o", color=col, lw=1.4, ms=3.6, zorder=3,
                label=rf"$M={m}$")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel(r"certification split size $L_2$")
    ax.set_ylabel("certified fraction")
    ax.legend(frameon=False, fontsize=7, loc="lower right",
              handlelength=1.6, labelspacing=0.3)
    despine(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{STEM}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
