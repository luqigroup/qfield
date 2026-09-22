"""Three designed-quadrature figures, rendered from saved checkpoints.

Built from saved checkpoints and saved results: no training and no GPU. Reuses
``scripts/_figstyle.py`` (apply_house_style, despine, the palette, the gold
star for a converged win point) and the trained toy amortizers under
``data/checkpoints/`` for the Gaussian, GMM and banana targets.

  1. ``dq_teaser.{pdf,png}`` -- on the banana density: the same M i.i.d.
     samples, which visibly miss the curved ridge, against the trained net's
     designed weighted nodes hugging the crest, plus a coupled
     MMD-against-M panel on the banana.

  2. ``dq_mechanism.{pdf,png}`` -- 1x3 on the planar d=2 5-mode GMM: (a) the
     i.i.d. floor nodes; (b) reweight-only, the same node coordinates with
     alpha set by the constrained weight, where weights alone cannot populate
     a starved mode; (c) move plus reweight, the moved nodes redistributed
     across the modes.

  3. ``dq_lever_split.{pdf,png}`` -- 1x2 (banana and GMM d=10): MMD^2 against
     M with four curves, the i.i.d. floor, reweight-only, move plus reweight
     and the per-instance oracle, with the area between reweight-only and
     move plus reweight shaded as the move gain. The reweight-only MMD is
     computed at evaluation time only: load the saved i.i.d. nodes per repeat,
     apply the constrained unit-sum weights and score the existing MMD, with
     no retraining.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/render_dq_clever_figures.py
"""

from __future__ import annotations
from projorg import datadir, plotsdir  # noqa: E402

# Pin CPU before importing torch.
import os  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from _figstyle import (  # noqa: E402
    C_OURS,
    C_SOLVE,
    GOLD,
    apply_house_style,
    despine,
    mode_stars,
    scatter_nodes,
)

from qfield.designed_quadrature import (  # noqa: E402
    build_target,
    constrained_weights,
    constrained_weights_batched,
    mmd_sq,
    mmd_sq_batched,
    move_descend,
)
from qfield.designed_quadrature.net import QuadratureAmortizer  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
DTYPE = torch.float64

CKPT_ROOT = datadir("checkpoints")
FIG_DIR = plotsdir("paper")

# Semantic palette, one concept to one colour, held fixed across the three
# figures.
C_FLOOR = "#3a7ca5"   # slate blue: the i.i.d. floor band and its samples
C_WARM = C_OURS       # orange: move plus reweight
C_ORACLE = C_SOLVE    # neutral grey: the per-instance optimum
C_REWEIGHT = "#5a8a94"  # desaturated teal: reweight only
C_GAIN = "0.55"         # neutral grey: the move-gain band fill

# Net architecture, constant across the checkpoints and recorded in each
# checkpoint's directory name.
NET_KW = dict(hidden=64, n_blocks=3, n_heads=4, cond_hidden=32, delta_scale=0.5)

# Evaluation harness constants.
BASE_SEED = 20260612
EVAL_REPEATS = 256
JITTER = 1e-10
ORACLE_LR = 0.05
ORACLE_ITERS = 150
# The node-budget grid.
M_LIST = [32, 64, 128, 256, 512]

# The projorg identity token of a run trained on that grid. Every checkpoint
# glob below carries it, since runs on other grids match the same prefix.
GRID_TOKEN = "m_list-32-64-128-256-512"


# --------------------------------------------------------------------------
# Checkpoint loading.
# --------------------------------------------------------------------------
def _find_ckpt(pattern: str) -> str | None:
    """Return the single checkpoint.pth path matching a directory glob.

    Raises when the glob matches more than one run: several runs of one target
    share a directory prefix and differ only in the bandwidth rule or the
    node-budget grid, so sort order would otherwise decide which bandwidth and
    which grid a figure is drawn at. Zero hits returns ``None``, so the
    caller's skip path still works.
    """
    hits = [
        h for h in sorted(glob.glob(os.path.join(CKPT_ROOT, pattern)))
        if os.path.isfile(os.path.join(h, "checkpoint.pth"))
    ]
    if len(hits) > 1:
        names = "\n  ".join(os.path.basename(h) for h in hits)
        raise RuntimeError(
            f"ambiguous checkpoint glob {pattern!r}:\n  {names}\n"
            "pin the glob to one bandwidth rule and one node-budget grid."
        )
    return os.path.join(hits[0], "checkpoint.pth") if hits else None


def _load_net_target(ckpt_path: str, target_name: str, d: int):
    """Load (net, target, sigma) from a saved checkpoint and rebuild its target.

    The checkpoint stores ``{"results", "state_dict"}``. The SE bandwidth
    ``sigma`` is the recorded median-heuristic value, so the rebuilt target
    has the same ``mu_fn`` and ``c_rho`` the run trained and evaluated against.
    """
    ck = torch.load(ckpt_path, weights_only=False)
    res = ck["results"]
    sigma = float(res["sigma"])
    net = QuadratureAmortizer(d=d, **NET_KW).to(DTYPE)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    # Rebuild the sampled-bank target at the run's recorded bank sizes rather
    # than build_target's defaults, so the mu and c_rho banks stay the ones
    # the run scored against.
    meta = res.get("target_meta") or {}
    tgt = build_target(
        target_name, sigma, d=d, dtype=DTYPE,
        n_ref=int(meta.get("n_ref", 20000)),
        n_mu=int(meta.get("n_mu", 4000)),
    )
    return net, tgt, sigma, res


# --------------------------------------------------------------------------
# Densities for the viridis backdrops.
# --------------------------------------------------------------------------
def _banana_density(xx, yy, s1=3.0, b=0.1):
    """Analytic banana density on a grid: x1~N(0,s1^2), x2|x1~N(b(x1^2-s1^2),1)."""
    px1 = np.exp(-(xx**2) / (2 * s1**2)) / (np.sqrt(2 * np.pi) * s1)
    mean2 = b * (xx**2 - s1**2)
    px2 = np.exp(-((yy - mean2) ** 2) / 2.0) / np.sqrt(2 * np.pi)
    return px1 * px2


def _gmm_density(xx, yy, means, covs, weights):
    """2-D GMM density on a grid from explicit means / covs / weights."""
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (P, 2)
    dens = np.zeros(pts.shape[0])  # flat accumulator (P,)
    for mu, cov, w in zip(means, covs, weights):
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        diff = pts - mu[None, :]
        q = np.einsum("pi,ij,pj->p", diff, inv, diff)
        dens += w * np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
    return dens.reshape(xx.shape)


def _density_backdrop(ax, xx, yy, dens, *, levels=14, floor_frac=0.04):
    """Light density backdrop on white: a pale Blues fill and thin contours.

    The density is context and has to recede behind the markers, so it is a
    very pale, low-alpha sequential fill on white axes plus a few thin contour
    lines; the dark-edged nodes then read on top and empty space stays white.
    ``floor_frac`` sets where the lowest fill band starts, as a fraction of the
    density peak; a smaller value gives compact, well-separated modes a larger
    visible skirt, still from the computed density and only at lower contours.
    """
    ax.set_facecolor("white")
    # Pale sequential fill; the near-zero lowest band is dropped so the floor
    # stays white.
    dmax = float(np.nanmax(dens))
    flv = np.linspace(float(floor_frac) * dmax, dmax, levels)
    ax.contourf(xx, yy, dens, levels=flv, cmap="Blues", alpha=0.30, zorder=0)
    # A few thin grey line contours give the modes/ridge a readable outline.
    clv = np.linspace(0.10 * dmax, dmax, max(5, levels // 2))
    ax.contour(
        xx, yy, dens, levels=clv, colors="#5b7fa6",
        linewidths=0.7, alpha=0.7, zorder=1,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------
# Evaluation-only legs: floor, warm, reweight-only, oracle.
# --------------------------------------------------------------------------
_LEG_CACHE = os.path.join(CKPT_ROOT, "_dq_clever_leg_cache")


def _eval_legs(net, tgt, sigma, d, M, *, with_oracle=False, cache_tag=""):
    """Held-out median MMD^2 of floor, warm and reweight-only, plus the oracle.

    Replays the evaluation harness of the training script: the same held-out
    generator seed ``BASE_SEED + 50000 + 1000 M``, the same ``EVAL_REPEATS``,
    the same closed-form weights and batched MMD. It adds the reweight-only
    leg, the i.i.d. floor nodes scored with the closed-form
    unit-sum-constrained weights instead of equal weights, at evaluation time
    and with no retraining.

    The result, a median and interquartile range per leg, is cached on disk
    keyed by (target, d, sigma, M, with_oracle, harness constants), so the slow
    banana per-instance oracle is not recomputed when the same legs are reused.
    The cache is a memo of a deterministic computation at identical seeds.
    """
    # "same_bank_eval" marks that the arms are scored on mu_fn_eval where the
    # target has one, so legs cached under a cross-bank estimand are never
    # replayed here. Bank sizes join the key because the banks are rebuilt
    # from the checkpoint's target_meta, so an identical sigma no longer
    # implies identical banks; ``cache_tag`` lets callers key in the
    # checkpoint's identity, so a retrain at the same sigma cannot replay the
    # old warm leg.
    meta = getattr(tgt, "meta", {}) or {}
    key = json.dumps(
        [tgt.name, d, round(float(sigma), 10), int(M), bool(with_oracle),
         EVAL_REPEATS, BASE_SEED, ORACLE_LR, ORACLE_ITERS, JITTER,
         "same_bank_eval", meta.get("n_ref"), meta.get("n_mu"),
         str(cache_tag)],
        sort_keys=True,
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    os.makedirs(_LEG_CACHE, exist_ok=True)
    cpath = os.path.join(_LEG_CACHE, f"{tgt.name}_d{d}_M{M}_{h}.json")
    if os.path.isfile(cpath):
        with open(cpath) as fh:
            return {k: tuple(v) for k, v in json.load(fh).items()}

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(BASE_SEED) + 50000 + 1000 * int(M))
    R = EVAL_REPEATS
    z0 = torch.stack([tgt.sampler(int(M), gen) for _ in range(R)], dim=0)
    # Score every leg on the evaluation estimand where the target has one,
    # the same-bank companion of c_rho, and on mu_fn otherwise.
    mu_e = getattr(tgt, "mu_fn_eval", None) or tgt.mu_fn
    mu0 = mu_e(z0.reshape(R * M, d)).reshape(R, M)

    # floor: equal weights.
    w_eq = torch.full((R, M), 1.0 / M, dtype=DTYPE)
    floor = torch.clamp(
        mmd_sq_batched(z0, w_eq, mu0, tgt.c_rho, sigma), min=0.0
    )
    # reweight-only: the same floor nodes, with the closed-form unit-sum
    # constrained weights.
    w_rw = constrained_weights_batched(z0, mu0, sigma, JITTER)
    reweight = torch.clamp(
        mmd_sq_batched(z0, w_rw, mu0, tgt.c_rho, sigma), min=0.0
    )
    # warm: one forward pass and the closed-form weights.
    with torch.no_grad():
        z = net(z0)
        mu = mu_e(z.reshape(R * M, d)).reshape(R, M)
        w = constrained_weights_batched(z, mu, sigma, JITTER)
        warm = torch.clamp(
            mmd_sq_batched(z, w, mu, tgt.c_rho, sigma), min=0.0
        )

    out = {
        "floor": _median_iqr(floor),
        "reweight": _median_iqr(reweight),
        "warm": _median_iqr(warm),
    }
    if with_oracle:
        from qfield.designed_quadrature import move_descend_batched

        # Descent on the training estimand, the one the amortizer was trained
        # against; the endpoint is scored on the evaluation estimand like
        # every other leg.
        zd, _ = move_descend_batched(
            z0, tgt.mu_fn, tgt.c_rho, sigma, ORACLE_LR, ORACLE_ITERS, JITTER
        )
        with torch.no_grad():
            zd = zd.detach()
            mu_d = mu_e(zd.reshape(R * M, d)).reshape(R, M)
            w_d = constrained_weights_batched(zd, mu_d, sigma, JITTER)
            oracle = torch.clamp(
                mmd_sq_batched(zd, w_d, mu_d, tgt.c_rho, sigma), min=0.0
            )
        out["oracle"] = _median_iqr(oracle)
    with open(cpath, "w") as fh:
        json.dump({k: list(v) for k, v in out.items()}, fh)
    return out


def _median_iqr(t):
    t = t[torch.isfinite(t)]
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


# --------------------------------------------------------------------------
# One held-out node set (positions and weights) for the scatter panels.
# --------------------------------------------------------------------------
def _emit_one_set(net, tgt, sigma, d, M, *, seed):
    """One held-out i.i.d. sample -> (z0, w_eq), (z_designed, w_designed).

    Returns the floor nodes with equal weights and the net's designed nodes
    with the closed-form constrained weights, for the same seed sample, so the
    two are the before and after of the same M points. All numpy.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z0 = tgt.sampler(int(M), gen)  # (M, d)
    # Solve the displayed weights from the evaluation estimand, the same-bank
    # pair, so the scatter's alphas depict the arm the MMD panel scores.
    mu_e = getattr(tgt, "mu_fn_eval", None) or tgt.mu_fn
    mu0 = mu_e(z0)
    w0 = constrained_weights(z0, mu0, sigma, JITTER)  # reweight-only weights
    with torch.no_grad():
        z = net(z0.unsqueeze(0)).squeeze(0)  # (M, d)
        mu = mu_e(z)
        w = constrained_weights(z, mu, sigma, JITTER)
    return (
        z0.numpy(),
        np.full(M, 1.0 / M),
        w0.numpy(),
        z.numpy(),
        w.numpy(),
    )


def _save(fig, stem):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(FIG_DIR, f"{stem}.{ext}"), dpi=300, bbox_inches="tight"
        )
    print(f"  wrote {stem}.pdf / {stem}.png")


# --------------------------------------------------------------------------
# Figure 1: the banana before and after, plus a budget panel.
# --------------------------------------------------------------------------
def render_teaser():
    apply_house_style()
    # Pinned to the median-MMD run on this node-budget grid; runs at another
    # bandwidth rule or another grid match the same prefix.
    ck = _find_ckpt(
        "designed_quadrature_banana2_amortized_target-banana_d-2_"
        f"*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*"
    )
    if ck is None:
        print("  [MISSING] banana2 amortizer checkpoint -- teaser skipped")
        return
    net, tgt, sigma, _ = _load_net_target(ck, "banana", 2)
    M = 16

    # One held-out sample for the scatter.
    z0, w_eq, w0, zd, wd = _emit_one_set(
        net, tgt, sigma, 2, M, seed=BASE_SEED + 50000 + 1000 * M + 7
    )

    fig = plt.figure(figsize=(9.4, 3.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # -- (a) banana density, i.i.d. (open) against designed (filled) nodes --
    xg = np.linspace(-9.0, 9.0, 320)
    yg = np.linspace(-4.0, 8.0, 320)
    xx, yy = np.meshgrid(xg, yg)
    dens = _banana_density(xx, yy)
    _density_backdrop(ax_a, xx, yy, dens, levels=16)
    # connector segments: every i.i.d. node flows to its designed position.
    for a, bxy in zip(z0, zd):
        ax_a.plot(
            [a[0], bxy[0]], [a[1], bxy[1]],
            color="0.45", lw=0.6, alpha=0.5, zorder=2, solid_capstyle="round",
        )
    # i.i.d. samples: open blue circles, uniform size and alpha.
    ax_a.scatter(
        z0[:, 0], z0[:, 1], s=40, facecolors="none", edgecolors=C_FLOOR,
        linewidths=1.2, marker="o", zorder=3, alpha=0.9,
    )
    # designed nodes: filled orange circles, uniform size and alpha, since the
    # right panel carries the quantitative comparison.
    scatter_nodes(
        ax_a, zd, C_WARM, marker="o", s=42, alpha=0.92, edge="black",
        edge_lw=0.5, zorder=4,
    )
    # one dark label on the light backdrop.
    moves = np.linalg.norm(zd - z0, axis=1)
    j = int(np.argmax(moves))
    ax_a.annotate(
        "every node\nrepositioned",
        xy=(0.5 * (z0[j, 0] + zd[j, 0]), 0.5 * (z0[j, 1] + zd[j, 1])),
        xytext=(z0[j, 0] - 1.0, z0[j, 1] + 2.6),
        fontsize=7.5, color="0.15", ha="center", va="center",
        arrowprops=dict(
            arrowstyle="->", color="0.30", lw=1.0,
            connectionstyle="arc3,rad=0.25",
        ),
        zorder=6,
    )
    ax_a.set_xlim(xg[0], xg[-1])
    ax_a.set_ylim(yg[0], yg[-1])
    # legend proxies (shape carries the distinction).
    h_iid = plt.Line2D(
        [], [], marker="o", ls="none", mfc="none", mec=C_FLOOR,
        mew=1.2, ms=6, label="i.i.d. draws",
    )
    h_des = plt.Line2D(
        [], [], marker="o", ls="none", mfc=C_WARM, mec="black",
        ms=6, label="designed nodes (ours)",
    )
    ax_a.legend(
        handles=[h_iid, h_des], frameon=True, fontsize=7.5, loc="upper right",
        framealpha=0.85, edgecolor="none",
    )
    ax_a.set_title(
        "the same draws, repositioned + reweighted", fontsize=9.5, pad=4
    )

    # -- (b) MMD^2 against M on the same target (banana) -------------------
    # Only the i.i.d. floor against the quadrature; the per-instance curve is
    # drawn in the detailed figures instead.
    legs = {M_: _eval_legs(net, tgt, sigma, 2, M_, with_oracle=False,
                           cache_tag=os.path.getmtime(ck))
            for M_ in M_LIST}
    m = M_LIST
    fl_md = [legs[M_]["floor"][0] for M_ in m]
    fl_lo = [min(legs[M_]["floor"][1], legs[M_]["floor"][2]) for M_ in m]
    fl_hi = [max(legs[M_]["floor"][1], legs[M_]["floor"][2]) for M_ in m]
    wa_md = [legs[M_]["warm"][0] for M_ in m]

    ax_b.fill_between(m, fl_lo, fl_hi, color=C_FLOOR, alpha=0.18, lw=0, zorder=1)
    ax_b.plot(
        m, fl_md, color=C_FLOOR, lw=1.5, ls=(0, (5, 3)), marker="o", ms=3.5,
        label="i.i.d. floor", zorder=2,
    )
    ax_b.plot(
        m, wa_md, color=C_WARM, lw=2.0, marker="o", ms=4.0,
        label="the quadrature", zorder=4,
    )
    win_m = [M_ for M_ in m if legs[M_]["warm"][0] < legs[M_]["floor"][0]]
    win_v = [legs[M_]["warm"][0] for M_ in win_m]
    ax_b.scatter(
        win_m, win_v, marker="*", s=95, c=GOLD, edgecolors="black",
        linewidths=0.5, zorder=6, label="beats the floor",
    )
    ax_b.set_xscale("log", base=2)
    ax_b.set_yscale("log")
    ax_b.set_xticks(m)
    ax_b.set_xticklabels([str(v) for v in m])
    ax_b.set_xlabel(r"node budget $M$")
    ax_b.set_ylabel(r"$\mathrm{MMD}^2(Q,\rho)$")
    ax_b.set_title("below the floor at every budget", fontsize=9.5, pad=4)
    ax_b.legend(frameon=False, fontsize=7.5, loc="lower left")
    despine(ax_b)

    _save(fig, "dq_teaser")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: the mechanism. Reweighting cannot rescue a starved mode; moving
# can.
# --------------------------------------------------------------------------
# The planar d=2 5-mode GMM, the same mixture as target._build_gmm(2), so the
# mode-coverage reading renders directly in the plane with no projection.
_GMM2_MEANS = 2.0 * np.array(
    [[6.2, -6.0], [-4.0, 5.0], [7.0, 3.0], [-6.5, -4.5], [1.0, 7.0]]
)
_GMM2_COVS = np.array(
    [
        [[1.5, 0.1], [0.1, 0.5]],
        [[2.0, -0.6], [-0.6, 0.5]],
        [[0.7, 0.4], [0.4, 1.2]],
        [[1.3, -0.5], [-0.5, 0.9]],
        [[0.6, 0.35], [0.35, 1.6]],
    ]
)
_GMM2_W = np.full(5, 0.2)


def render_mechanism():
    apply_house_style()
    # Moving relocates nodes across a low-density gap onto a mode the seed
    # missed, where reweighting, locked to the seed coordinates, cannot. That
    # is a property of the move lever, here the per-instance MMD optimum
    # (move_descend from z0), and it appears only in the bandwidth window
    # where the kernel resolves the modes yet still senses mass across the
    # gap. This figure is pedagogical, so it uses an illustrative bandwidth in
    # that window: sigma sits between the mode width, about 1, and the
    # inter-mode spacing, about 11 to 20. At the data-driven median-MMD
    # bandwidth, about 20 on this scene, the kernel is smooth across the whole
    # mixture and the same descent corrects global moments instead of planting
    # a node on the starved mode; at a much smaller sigma the kernel is blind
    # across the gap and the move is merely local.
    from qfield.designed_quadrature.target import gmm_target
    M = 16
    sigma = 5.0  # illustrative: mode width << sigma < inter-mode spacing
    tgt = gmm_target(sigma, 2)

    def _move(z0t):
        zd, _ = move_descend(z0t, tgt.mu_fn, tgt.c_rho, sigma, 0.05, 300, JITTER)
        return zd.detach()

    # Pick a held-out seed that under-samples at least one mode and where the
    # move lever then covers it.
    best = None
    for k in range(60):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(BASE_SEED + 90000 + k)
        z0t = tgt.sampler(M, gen)
        z0n = z0t.numpy()
        d2 = ((z0n[:, None, :] - _GMM2_MEANS[None, :, :]) ** 2).sum(-1)
        counts = np.bincount(d2.argmin(1), minlength=5)
        empties = np.where(counts == 0)[0]
        if empties.size == 0:
            continue
        zdt = _move(z0t)
        zdn = zdt.numpy()
        dmove = np.sqrt(
            ((zdn[:, None, :] - _GMM2_MEANS[None, empties, :]) ** 2).sum(-1)
        ).min(0)
        covered = int((dmove < 2.5).sum())
        score = (covered, int(empties.size))
        if best is None or score > best[0]:
            best = (score, (z0t, z0n, zdt, zdn, counts))
    z0t, z0, zdt, zd, seed_counts = best[1]
    w0 = constrained_weights(z0t, tgt.mu_fn(z0t), sigma, JITTER).detach().numpy()
    wd = constrained_weights(zdt, tgt.mu_fn(zdt), sigma, JITTER).detach().numpy()

    # Shared backdrop, on a tight frame: the modes are compact and far apart,
    # so a generous pad buries them in white space.
    pad = 2.0
    allp = np.concatenate([z0, zd, _GMM2_MEANS], axis=0)
    xlo, xhi = allp[:, 0].min() - pad, allp[:, 0].max() + pad
    ylo, yhi = allp[:, 1].min() - pad, allp[:, 1].max() + pad
    xg = np.linspace(xlo, xhi, 300)
    yg = np.linspace(ylo, yhi, 300)
    xx, yy = np.meshgrid(xg, yg)
    dens = _gmm_density(xx, yy, _GMM2_MEANS, _GMM2_COVS, _GMM2_W)

    # Starved mode (fewest i.i.d. seed nodes) for the annotations.
    starved = int(seed_counts.argmin())
    sx, sy = _GMM2_MEANS[starved]
    rad = 2.4

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    titles = ["(a) i.i.d. seed", "(b) reweight", "(c) move + reweight"]
    for ax in axes:
        _density_backdrop(ax, xx, yy, dens, levels=14, floor_frac=0.015)
        mode_stars(ax, _GMM2_MEANS, s=150)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    # (a) i.i.d. seed nodes: open blue circles, uniform size and alpha.
    axes[0].scatter(
        z0[:, 0], z0[:, 1], s=58, facecolors="none", edgecolors=C_FLOOR,
        linewidths=1.2, marker="o", zorder=3, alpha=0.9,
    )
    # (b) reweight-only: the same coordinates, filled orange, alpha = |weight|.
    scatter_nodes(
        axes[1], z0, C_WARM, marker="o", s=60, weights=w0, edge="black",
        edge_lw=0.5, zorder=3, alpha_floor=0.30, alpha_ceil=0.98,
    )
    # (c) move plus reweight: the move-lever optimum, where nodes cross the
    # low-density gap onto the starved mode, with a grey displacement arrow
    # from each seed to its destination.
    for a, bxy in zip(z0, zd):
        axes[2].annotate(
            "", xy=(bxy[0], bxy[1]), xytext=(a[0], a[1]),
            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0, alpha=0.9),
            zorder=2,
        )
    scatter_nodes(
        axes[2], zd, C_WARM, marker="o", s=60, weights=wd, edge="black",
        edge_lw=0.5, zorder=4, alpha_floor=0.45, alpha_ceil=0.98,
    )

    # Ring and annotate the starved mode across the three panels.
    for ax, msg in zip(
        axes,
        [
            "starved by\nthe seed",
            "reweighting\ncannot fill it\n(nodes are fixed)",
            "moving carries nodes\nacross the gap onto it",
        ],
    ):
        ring = plt.Circle(
            (sx, sy), rad, fill=False, edgecolor="0.20", lw=1.1,
            ls="--", zorder=5,
        )
        ax.add_patch(ring)
        ax.annotate(
            msg, xy=(sx, sy - rad), xytext=(sx, sy - rad - 2.2),
            fontsize=8.0, color="0.10", ha="center", va="top", zorder=6,
        )

    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=9.5, pad=4)

    fig.tight_layout()
    _save(fig, "dq_mechanism")
    plt.close(fig)


# --------------------------------------------------------------------------
def _lever_panel(ax, legs, title, show_ylabel, *, subnote):
    m = M_LIST
    fl_md = [legs[M_]["floor"][0] for M_ in m]
    rw_md = [legs[M_]["reweight"][0] for M_ in m]
    wa_md = [legs[M_]["warm"][0] for M_ in m]
    or_md = [legs[M_]["oracle"][0] for M_ in m]

    # i.i.d. floor: the dashed median line only, with no band.
    ax.plot(
        m, fl_md, color=C_FLOOR, lw=1.5, ls=(0, (5, 3)), marker="o", ms=3.5,
        label="i.i.d. floor", zorder=2,
    )
    # the move gain: a neutral grey hatch between reweight-only and move plus
    # reweight, which spans two hues and so reads as a band.
    ax.fill_between(
        m, wa_md, rw_md, facecolor="none", edgecolor=C_GAIN, hatch="////",
        lw=0.0, alpha=0.9, zorder=1, label="the move gain",
    )
    # reweight-only: desaturated teal, open markers.
    ax.plot(
        m, rw_md, color=C_REWEIGHT, lw=1.6, ls="-", marker="o", ms=4.0,
        mfc="white", mec=C_REWEIGHT, mew=1.3, label="reweight only", zorder=3,
    )
    # move plus reweight: full orange, filled.
    ax.plot(
        m, wa_md, color=C_WARM, lw=2.0, marker="o", ms=4.2,
        label="move + reweight (ours)", zorder=4,
    )
    # the per-instance best: a grey dotted lower envelope.
    ax.plot(
        m, or_md, color=C_ORACLE, lw=1.4, ls=":", marker="o", ms=3.2,
        label="per-instance optimum", zorder=3,
    )
    # a gold star on each move-plus-reweight point that beats the floor.
    win_m = [M_ for M_ in m if legs[M_]["warm"][0] < legs[M_]["floor"][0]]
    win_v = [legs[M_]["warm"][0] for M_ in win_m]
    ax.scatter(
        win_m, win_v, marker="*", s=90, c=GOLD, edgecolors="black",
        linewidths=0.5, zorder=6, label="beats the floor",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(m)
    ax.set_xticklabels([str(v) for v in m])
    ax.set_xlabel(r"node budget $M$")
    if show_ylabel:
        ax.set_ylabel(r"$\mathrm{MMD}^2(Q,\rho)$")
    ax.set_title(title, fontsize=9.5, pad=4)
    # one-line per-panel sub-annotation.
    ax.text(
        0.5, 0.02, subnote, transform=ax.transAxes, fontsize=7.5,
        color="0.15", ha="center", va="bottom",
    )
    despine(ax)


def render_lever_split():
    apply_house_style()
    # Pinned to the median-MMD runs on this node-budget grid. A checkpoint
    # that is absent prints [MISSING] and skips, so a panel never mixes two
    # bandwidth rules or two budget grids.
    ck_ban = _find_ckpt(
        "designed_quadrature_banana2_amortized_target-banana_d-2_"
        f"*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*"
    )
    ck_gmm = _find_ckpt(
        "designed_quadrature_gmmcncv10_amortized_target-gmm_cncv_d-10_"
        f"*{GRID_TOKEN}_*bandwidth-median_mmd_whiten-*"
    )
    if ck_ban is None or ck_gmm is None:
        miss = "banana2" if ck_ban is None else "gmmcncv10"
        print(f"  [MISSING] {miss} checkpoint -- lever_split skipped")
        return

    net_b, tgt_b, sig_b, _ = _load_net_target(ck_ban, "banana", 2)
    net_g, tgt_g, sig_g, _ = _load_net_target(ck_gmm, "gmm_cncv", 10)

    print("  [lever] evaluating banana legs (incl. reweight-only + oracle)")
    legs_b = {M_: _eval_legs(net_b, tgt_b, sig_b, 2, M_, with_oracle=True,
                             cache_tag=os.path.getmtime(ck_ban))
              for M_ in M_LIST}
    print("  [lever] evaluating GMM d=10 legs")
    legs_g = {M_: _eval_legs(net_g, tgt_g, sig_g, 10, M_, with_oracle=True,
                             cache_tag=os.path.getmtime(ck_gmm))
              for M_ in M_LIST}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    _lever_panel(
        axes[0], legs_b, "banana, $d=2$", show_ylabel=True,
        subnote="curved ridge: reweighting alone\nalready pulls off the floor",
    )
    _lever_panel(
        axes[1], legs_g, "GMM, $d=10$", show_ylabel=False,
        subnote="d=10 modes: the move carries the\nsmall-budget gain; reweighting\nstrengthens with M",
    )
    axes[0].legend(frameon=False, fontsize=7.0, loc="lower left")
    fig.tight_layout()
    _save(fig, "dq_lever_split")
    plt.close(fig)


def main():
    print("[render] FIG 1 teaser (banana)")
    render_teaser()
    print("[render] FIG 2 mechanism (GMM d=2 5-mode)")
    render_mechanism()
    print("[render] FIG 3 lever split (banana + GMM d=10)")
    render_lever_split()
    print("done.")


if __name__ == "__main__":
    main()
