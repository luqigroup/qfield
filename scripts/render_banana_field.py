"""One forward pass of the quadrature field on the banana reference.

The samples handed in (floor blue) are joined to the nodes the pass carries
them to, and the nodes are coloured by their solved weight relative to the
equal weight.

Reuses the checkpoint lookup and helpers of render_dq_clever_figures.py, which
resolves the banana run by a glob that raises on ambiguity; one forward pass on
the CPU, no training. Markers are uniform in size, and the sign and magnitude
of the weights are carried by colour.

    python scripts/render_banana_field.py
"""
from __future__ import annotations
from projorg import plotsdir  # noqa: E402

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from _figstyle import apply_house_style  # noqa: E402
from render_dq_clever_figures import (  # noqa: E402
    BASE_SEED, C_FLOOR, GRID_TOKEN, _banana_density, _density_backdrop,
    _emit_one_set, _find_ckpt, _load_net_target,
)

FIG_DIR = plotsdir("paper")
STEM = "dq_banana_field"
M = 32


def main() -> None:
    apply_house_style()
    ck = _find_ckpt(
        "designed_quadrature_banana2_amortized_target-banana_d-2_"
        f"*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*"
    )
    if ck is None:
        raise RuntimeError("banana2 amortizer checkpoint (deployed grid) not found")
    net, tgt, sigma, _ = _load_net_target(ck, "banana", 2)
    z0, w_eq, w0, zd, wd = _emit_one_set(
        net, tgt, sigma, 2, M, seed=BASE_SEED + 50000 + 1000 * M + 7
    )
    print(f"[banana] M={M} |w|_1={np.abs(wd).sum():.2f} "
          f"ratio range {(wd * M).min():.2f} .. {(wd * M).max():.2f}")

    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    xg = np.linspace(-9.0, 9.0, 320)
    yg = np.linspace(-4.0, 8.0, 320)
    xx, yy = np.meshgrid(xg, yg)
    _density_backdrop(ax, xx, yy, _banana_density(xx, yy), levels=16)
    for a, b in zip(z0, zd):
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="0.35", lw=0.7,
                alpha=0.8, zorder=2)
    ax.scatter(z0[:, 0], z0[:, 1], s=16, color=C_FLOOR, edgecolors="white",
               linewidths=0.4, zorder=3, label="samples")
    ratio = wd * M
    vmax = max(1.5, float(np.abs(ratio).max()))
    norm = TwoSlopeNorm(vmin=min(float(ratio.min()), 0.0), vcenter=1.0, vmax=vmax)
    sc = ax.scatter(zd[:, 0], zd[:, 1], c=ratio, cmap="PRGn", norm=norm, s=26,
                    edgecolors="black", linewidths=0.4, zorder=4, label="nodes")
    # The colourbar axis is linear in the ratio, so label its two ends and the
    # equal weight; the map itself is centred at the equal weight.
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03,
                      ticks=[norm.vmin, 1.0, norm.vmax])
    cb.ax.set_yticklabels([f"{norm.vmin:.1f}", "1", f"{norm.vmax:.0f}"])
    cb.set_label("weight / equal weight", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", handletextpad=0.3,
              markerscale=1.2)
    ax.set_xlim(xg[0], xg[-1])
    ax.set_ylim(yg[0], yg[-1])
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{STEM}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
