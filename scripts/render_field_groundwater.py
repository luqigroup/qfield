"""One forward pass of the quadrature field on one held-out groundwater
posterior.

Four panels. The inverse problem: the true log-permeability field and the
sensors that observe the pressure it induces. One forward pass: the samples
the field is handed, over the reference posterior's mass, each joined to the
node it is carried to. The weights the closed-form solve attaches to the
nodes, relative to the equal weight, sign and magnitude by colour. And the
quadrature's own output on the problem, the weighted posterior mean of the
field. The two quadrature panels are drawn in the two directions along which
the pass moves the samples most.

Reads the trained groundwater checkpoint through projorg's own naming (the
config file, never a glob), samples the reference flow once for the bank and
runs one forward pass on the CPU. No training, no evaluation sweep.

    python scripts/render_field_groundwater.py
"""
from __future__ import annotations
from projorg import datadir, plotsdir  # noqa: E402

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from _figstyle import C_OURS, apply_house_style, despine  # noqa: E402

# projorg builds the run's identity from the config; phase is ignored in the
# name, so "visualize" resolves to the trained run's directory.
sys.argv = [sys.argv[0], "--phase", "visualize"]
from projorg import plotsdir, setup_environment
from dq_darcy_stuart import CONFIG_FILE, DTYPE, DQDarcyStuart  # noqa: E402
from qfield.designed_quadrature import (  # noqa: E402
    constrained_weights, mmd_sq, select_emission,
)
from qfield.dataset.darcy_stuart_kl_prior import StuartKLPrior  # noqa: E402
from qfield.dataset.darcy_stuart_op import beskos_sensors  # noqa: E402

FIG_DIR = plotsdir("paper")
STEM = "dq_field_groundwater"
C_IID = "#3a7ca5"
OBS = 0     # the first held-out observation
M = 64      # node count


def main() -> None:
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload",
                         "recon_M", "n_recon", "eval_repeats",
                         "n_pcn_audit"],
        sequence_args_and_types=[("m_list", int)],
    )
    exp = DQDarcyStuart(args)
    exp.load_checkpoint()
    exp.sigma = float(exp.results["sigma"])
    net = exp._build_net()
    net.load_state_dict(exp.ckpt["state_dict"])
    net.eval()

    gen = torch.Generator().manual_seed(int(args.base_seed) + 7000 + OBS)
    tgt, bank = exp._eval_target(OBS, gen)
    z0 = tgt.sampler(M, gen)
    mu0 = tgt.mu_fn(z0)
    w_eq = torch.full((M,), 1.0 / M, dtype=DTYPE)
    w_c = constrained_weights(z0, mu0, exp.sigma, args.jitter)
    y = exp.y_ev[OBS].double().cpu().unsqueeze(0)
    with torch.no_grad():
        z_n = net(z0.unsqueeze(0), y)[0]
    mu_n = tgt.mu_fn(z_n)
    w_n = constrained_weights(z_n, mu_n, exp.sigma, args.jitter)
    sel = select_emission([(z0, w_eq), (z0, w_c), (z_n, w_n)], tgt.mu_fn,
                          exp.sigma, names=("seeds", "reweight", "move"),
                          fitting_criterion=True)
    for name, (z, w) in zip(("seeds", "reweight", "move"),
                            ((z0, w_eq), (z0, w_c), (z_n, w_n))):
        print(f"[{name}] MMD^2 = "
              f"{mmd_sq(z, w, tgt.mu_fn(z), tgt.c_rho, exp.sigma):.3e}")
    print(f"[selected] {sel['name']}; |w|_1 = {float(w_n.abs().sum()):.2f}; "
          f"ratio range {float((w_n * M).min()):.2f} .. "
          f"{float((w_n * M).max()):.2f}")

    # Coordinates: the two directions along which the forward pass moves the
    # samples most (leading right-singular vectors of the displacement), so
    # the motion is visible; the grey mass is still the posterior's, in the
    # same two directions.
    ctr = bank.mean(0)
    _, _, vt = torch.linalg.svd(z_n - z0, full_matrices=False)
    P = vt[:2].T

    def proj(z):
        return ((z - ctr) @ P).numpy()

    pb, p0, pn = proj(bank), proj(z0), proj(z_n)
    kde = gaussian_kde(pb.T)
    lo, hi = pb.min(0), pb.max(0)
    pad = 0.08 * (hi - lo)
    gx, gy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], 160),
                         np.linspace(lo[1] - pad[1], hi[1] + pad[1], 160))
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

    apply_house_style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    # Five columns: four panels and a narrow one for the weight colourbar, so
    # the bar never crowds the last panel; every panel square and the same
    # height, so the titles share one line.
    fig = plt.figure(figsize=(7.1, 2.05))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.06, 1],
                          wspace=0.55, left=0.05, right=0.99, bottom=0.2,
                          top=0.88)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 2, 4)]
    cax = fig.add_subplot(gs[0, 3])
    for ax in axes:
        ax.set_box_aspect(1)

    # ---- the problem: the unknown field and the sensors ------------------
    kl = StuartKLPrior(int(exp.attrs["N"]), K=int(exp.attrs["K"]),
                       alpha=float(exp.attrs["alpha"]),
                       s=float(exp.attrs["s"]), sigma=float(exp.attrs["sigma"]))
    truth = kl.reconstruct(exp.xi_true_ev[OBS].double().numpy())
    _, (sx, sy) = beskos_sensors(int(exp.attrs["N"]),
                                 int(exp.attrs["n_sensors"]))
    z_s, w_s = sel["z"], sel["w"]
    fields = kl.reconstruct(z_s.numpy())                       # (M, N, N)
    q_mean = (w_s.numpy()[:, None, None] * fields).sum(0)
    vmax_f = float(max(np.abs(truth).max(), np.abs(q_mean).max()))

    def field_panel(ax, f, title):
        ax.imshow(f, cmap="RdBu_r", vmin=-vmax_f, vmax=vmax_f, origin="lower",
                  extent=(0, 1, 0, 1), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=8.5, color="0.25")

    field_panel(axes[0], truth, "true field, sensors")
    axes[0].plot(np.asarray(sx, dtype=float), np.asarray(sy, dtype=float),
                 "^", ms=3.0, color="k", mfc="white", mew=0.7, zorder=3)
    field_panel(axes[3], q_mean, "quadrature mean")

    # ---- one forward pass: samples joined to their nodes -----------------
    for ax in axes[1:3]:
        ax.contourf(gx, gy, dens, levels=7, cmap="Greys", alpha=0.85,
                    zorder=0)
        despine(ax)
        ax.set_xlabel("displacement direction 1")
        ax.set_xlim(gx.min(), gx.max()); ax.set_ylim(gy.min(), gy.max())
    axes[1].set_ylabel("displacement direction 2")
    axes[2].set_yticklabels([])
    ax = axes[1]
    for a, b in zip(p0, pn):
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="0.25", lw=0.8,
                zorder=2)
    ax.plot(p0[:, 0], p0[:, 1], "o", ms=3.4, color=C_IID, mec="white",
            mew=0.35, zorder=3)
    ax.plot(pn[:, 0], pn[:, 1], "o", ms=3.4, color=C_OURS, mec="white",
            mew=0.35, zorder=4)
    ax.set_title("samples to nodes", fontsize=8.5, color="0.25")

    # ---- the solved weights, centred on the equal weight -----------------
    ax = axes[2]
    ratio = (w_n * M).numpy()
    lo_r, hi_r = float(min(ratio.min(), 0.0)), float(max(ratio.max(), 1.5))
    norm = TwoSlopeNorm(vmin=lo_r, vcenter=1.0, vmax=hi_r)
    sc = ax.scatter(pn[:, 0], pn[:, 1], c=ratio, cmap="PRGn", norm=norm,
                    s=18, edgecolors="white", linewidths=0.35, zorder=4)
    cb = fig.colorbar(sc, cax=cax, ticks=[0.0, 1.0])
    cb.ax.set_yticklabels(["0", "equal"])
    cb.set_label("weight / equal weight", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    cax.set_box_aspect(12)
    ax.set_title("solved weights", fontsize=8.5, color="0.25")

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{STEM}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
