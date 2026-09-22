"""Figures for the designed-quadrature warm start.

Uses the shared style helpers of ``scripts/_figstyle.py``: vector PDF and
300-dpi PNG, type-42 fonts, ASCII minus. One concept, one colour: the designed
warm start is orange (``C_OURS``), the independent-sample floor a slate dashed
band, the per-instance descent neutral grey, and a point below the floor
carries a gold star (``GOLD``).

  1. ``dq_dimension_sweep.{pdf,png}`` -- two panels. Left: the warm/floor
     MMD^2 ratio at a fixed budget (M = 64) against ambient dimension ``d``
     for two kernels, the isotropic squared-exponential kernel rising toward
     the floor and the posterior-whitened (Mahalanobis) kernel with the
     dimension-aware sigma = sqrt(d) bandwidth staying below it. Gaussian
     sweep (d in {2, 4, 8, 16, 32}) and 2-component GMM sweep (d in {10, 32,
     64}); a ratio below one is below the independent-sample floor, marked by
     a dashed reference at one. Right: the isotropic node Gram condition
     number tending to one as ``d`` grows, read from
     ``dq_kernel_wall_gram.json``, which is the mechanism the whitened kernel
     removes.

  2. ``dq_mmd_vs_M.{pdf,png}`` -- a 1x3 panel for three references (Gaussian
     d = 2, GMM d = 10, banana d = 2): MMD^2 against node budget ``M``
     (log-log) over {32, ..., 512}, with four arms -- the independent-sample
     floor (slate dashed band), reweight only (teal, the solve at the
     untouched seeds), move and reweight (orange, with an across-training-seed
     interquartile band), and the per-instance descent (grey, dotted).
     Conditioning-limited cells carry an open mark.

This is a pure render step: it reads the per-run ``results.json`` written by
``scripts/designed_quadrature_gaussian_amortized.py`` and trains nothing.

Usage:
    python scripts/render_dq_figures.py
"""

from __future__ import annotations
from projorg import datadir, gitdir, plotsdir  # noqa: E402

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _figstyle import C_OURS, C_SOLVE, GOLD, apply_house_style, despine  # noqa: E402

CKPT_ROOT = datadir("checkpoints")
FIG_DIR = plotsdir("paper")
GRAM_JSON = os.path.join(FIG_DIR, "diagnostics", "dq_kernel_wall_gram.json")
# Multi-training-seed summary written by scripts/dq_multiseed_train.py. When
# present, the dimension-sweep and mmd-vs-M figures draw across-seed bands
# (interquartile range over the training seed).
MULTISEED_JSON = os.path.join(datadir("multiseed"), "dq_toy_seeds.json")

# The two node-budget grids, each written down once.
M_LIST = [32, 64, 128, 256, 512]
LEGACY_M_LIST = [4, 8, 16, 32, 64]

# Fixed node budget at which the dimension sweep is read, and the node-budget
# grid it is read on. Both grids contain 64, so the grid is stated explicitly.
SWEEP_M = 64
SWEEP_GRID = LEGACY_M_LIST

# Dimension sets per family. Degenerate low-dimensional GMM points, such as a
# d = 2 mixture, are excluded so the GMM sweep matches the whitened arm's
# coverage.
SWEEP_DIMS = {
    "gaussian": {2, 4, 8, 16, 32},
    "gmm": {10, 32, 64},
}

# One concept, one colour.
C_FLOOR = "#3a7ca5"   # slate -- the independent-sample floor band and line.
C_WARM = C_OURS       # orange -- the designed warm start.
C_REWEIGHT = "#2a9d8f"  # teal -- the reweight-only arm.
C_ORACLE = C_SOLVE      # neutral grey, dotted -- the per-instance descent.

# Fully-trained runs, keyed by (target tag in results.json, d,
# median-heuristic sigma, node-budget grid). The sigma together with the
# n_steps >= 3000 selector separates a full run from the smoke and
# fixed-sigma runs that share a directory prefix. The m_list column is part
# of the key because sigma is not: the median heuristic reads a sample of the
# target, so a rerun on a different node-budget grid carries the same sigma.
CANON = [
    # label,           family,     target,     d,   sigma,   m_list
    ("Gaussian, d = 2",  "gaussian", "gaussian", 2,  0.6712, M_LIST),
    ("Gaussian, d = 4",  "gaussian", "gaussian", 4,  0.1840, M_LIST),
    ("Gaussian, d = 8",  "gaussian", "gaussian", 8,  0.2735, M_LIST),
    ("Gaussian, d = 16", "gaussian", "gaussian", 16, 0.3962, M_LIST),
    ("GMM, d = 10",      "gmm",      "gmm_cncv", 10, 9.0206, M_LIST),
    ("GMM, d = 32",      "gmm",      "gmm_cncv", 32, 12.5192, M_LIST),
    ("GMM, d = 64",      "gmm",      "gmm_cncv", 64, 16.0039, M_LIST),
    ("banana, d = 2",    "banana",   "banana",   2,  3.4267, M_LIST),
]

# The rows the MMD-vs-M figure is drawn from. ``render_mmd_vs_M`` raises on a
# missing row rather than drawing a blank panel.
MMD_VS_M_ROWS = ("Gaussian, d = 2", "GMM, d = 10", "banana, d = 2")


def _load_all(globs: list[str]) -> list[dict]:
    """Read every ``results.json`` under the given checkpoint globs."""
    cache = []
    seen = set()
    for g in globs:
        for d in glob.glob(os.path.join(CKPT_ROOT, g)):
            f = os.path.join(d, "results.json")
            if not os.path.exists(f) or f in seen:
                continue
            seen.add(f)
            try:
                with open(f) as fh:
                    cache.append(json.load(fh))
            except Exception:  # noqa: BLE001 -- skip partial runs.
                continue
    return cache


def _load_canon() -> dict:
    """Return {label: results-dict} for the trained runs named in ``CANON``.

    Selects, among all ``designed_quadrature_*_amortized*`` checkpoints, the
    run whose (target, d, sigma, m_list) matches a ``CANON`` entry and which
    is fully trained (n_steps >= 3000), rather than the n_steps = 5 smoke run
    or the fixed-sigma = 0.5 run sharing the directory prefix.

    Raises on ambiguity. The row's ``m_list`` is part of the key because sigma
    is not: the median heuristic reads a target sample, so a rerun at a
    different node-budget grid reproduces the same sigma and a (target, d,
    sigma) match would be decided by ``glob`` order.
    """
    cache = _load_all(["designed_quadrature_*_amortized*"])
    out = {}
    for label, family, target, d_, sigma, m_list in CANON:
        hits = [
            r for r in cache
            if r.get("target") == target
            and r.get("d") == d_
            and isinstance(r.get("sigma"), float)
            and abs(r["sigma"] - sigma) < 1e-3
            and r.get("n_steps", 0) >= 3000
            and [int(m) for m in r.get("m_list", [])] == list(m_list)
        ]
        if len(hits) > 1:
            raise RuntimeError(
                f"ambiguous canonical run for {label}: {len(hits)} runs match "
                f"(target={target}, d={d_}, sigma~{sigma}, m_list={m_list}); "
                "pin the row further before rendering."
            )
        match = hits[0] if hits else None
        if match is None:
            print(
                f"  [MISSING] no canonical run for {label} "
                f"(sigma~{sigma}, m_list={m_list})"
            )
            continue
        match = dict(match)
        match["_label"] = label
        match["_family"] = family
        out[label] = match
    return out


def _is_whitened(r: dict) -> bool:
    """True if the run used the posterior-whitened (Mahalanobis) kernel."""
    return str(r.get("target_meta", {}).get("kind", "")).startswith("whitened")


def _family(r: dict) -> str:
    """``"gaussian"`` or ``"gmm"`` for a sweep run (banana excluded)."""
    t = r.get("target", "")
    if t == "gaussian":
        return "gaussian"
    if t in ("gmm", "gmm_cncv"):
        return "gmm"
    return "other"


def _load_sweep() -> dict:
    """Return isotropic and whitened (d, warm/floor) sweep points per family.

    Reads both the isotropic-kernel sweep runs (``dq_gaussian_d*`` /
    ``dq_gmm_d*`` and the ``designed_quadrature_*_amortized`` set) and the
    whitened-kernel runs (``dq_w_*``). Returns
    ``{family: {"iso": [(d, ratio), ...], "whiten": [(d, ratio), ...]}}``
    at the fixed budget ``M = SWEEP_M``, fully-trained runs only.
    """
    cache = _load_all(
        [
            "dq_gaussian_d*",
            "dq_gmm_d*",
            "dq_w_*",
            "designed_quadrature_*_amortized*",
        ]
    )
    out = {
        "gaussian": {"iso": {}, "whiten": {}},
        "gmm": {"iso": {}, "whiten": {}},
    }
    for r in cache:
        fam = _family(r)
        if fam not in out:
            continue
        if r.get("n_steps", 0) < 3000:
            continue
        # Exclude the fixed-sigma = 0.5 runs: the median-heuristic sweep runs
        # never resolve to sigma = 0.5. Without this the highest-n_steps
        # selector below could pick one of them over the median-heuristic run
        # and misreport the isotropic curve.
        if abs(float(r.get("sigma", 0.0)) - 0.5) < 1e-9:
            continue
        # The gmm-family sweep is the 2-component gmm_cncv reference, not the
        # 5-mode `gmm` target; _family() lumps both together, so select the
        # target here to keep the isotropic and whitened arms comparable.
        if fam == "gmm" and r.get("target") != "gmm_cncv":
            continue
        if not isinstance(r.get("d"), int):
            continue
        if r["d"] not in SWEEP_DIMS[fam]:
            continue
        # One node-budget grid per sweep. Both grids contain SWEEP_M = 64, so
        # two runs of one (family, arm, d) can both carry a readable row and
        # the most-trained tiebreak below, at equal n_steps, would resolve by
        # cache order. The sweep is a fixed-budget cut, so select the grid.
        if [int(m) for m in r.get("m_list", [])] != SWEEP_GRID:
            continue
        row = next((x for x in r.get("rows", []) if x.get("M") == SWEEP_M), None)
        if row is None or "warm_over_floor" not in row:
            continue
        arm = "whiten" if _is_whitened(r) else "iso"
        d_ = r["d"]
        # One run per (family, arm, d); prefer the most-trained on collision.
        store = out[fam][arm]
        if d_ not in store or r.get("n_steps", 0) >= store[d_][1]:
            store[d_] = (row["warm_over_floor"], r.get("n_steps", 0))
    # Flatten to sorted (d, ratio) lists.
    flat = {}
    for fam in out:
        flat[fam] = {
            arm: sorted((d_, v[0]) for d_, v in out[fam][arm].items())
            for arm in ("iso", "whiten")
        }
    return flat


# --------------------------------------------------------------------------
# Multi-training-seed bands. The summary written by
# scripts/dq_multiseed_train.py holds, per config (kernel x family x d), the
# across-training-seed warm/floor median and interquartile range at every M.
# It is read for the dimension sweep (warm/floor at M = SWEEP_M) and for the
# mmd-vs-M panels.
# --------------------------------------------------------------------------
def _load_multiseed() -> dict | None:
    """Return the parsed multiseed summary, or None if not yet produced."""
    if not os.path.exists(MULTISEED_JSON):
        return None
    try:
        with open(MULTISEED_JSON) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _ms_dim_sweep(ms: dict, m_budget: int) -> dict:
    """Across-seed warm/floor (median, q25, q75) per (family, arm, d) at M.

    Returns ``{family: {arm: [(d, med, q25, q75, n_seeds), ...]}}`` for the
    isotropic / whitened sweep configs, sorted by d, restricted to SWEEP_DIMS.
    """
    out = {
        "gaussian": {"iso": [], "whiten": []},
        "gmm": {"iso": [], "whiten": []},
    }
    for cfg in ms.get("configs", {}).values():
        fam, arm, d_ = cfg["family"], cfg["kernel"], cfg["d"]
        if fam not in out or arm not in ("iso", "whiten"):
            continue
        if d_ not in SWEEP_DIMS.get(fam, set()):
            continue
        block = cfg["per_M"].get(str(m_budget))
        if block is None:
            continue
        med, q25, q75 = block["warm_over_floor"]["median_iqr"]
        n = block["warm_over_floor"]["mean_std_n"][2]
        out[fam][arm].append((d_, med, q25, q75, n))
    for fam in out:
        for arm in out[fam]:
            out[fam][arm].sort(key=lambda t: t[0])
    return out


def _ms_covers_sweep(ms: dict | None) -> bool:
    """True if the multiseed record can supply the dimension-sweep bands.

    A record may hold only the three ``mmd_vs_M`` configs, while the sweep
    needs the isotropic and whitened arms across ``SWEEP_DIMS``. Handing such
    a record to ``render_dimension_sweep`` does not fail; it draws two
    isolated points and no whitened curve. This predicate lets the caller skip
    the sweep instead.
    """
    if not ms:
        return False
    got = _ms_dim_sweep(ms, SWEEP_M)
    for fam in ("gaussian", "gmm"):
        # both arms, and more than a single point, or there is no curve.
        if len(got[fam]["iso"]) < 2 or len(got[fam]["whiten"]) < 2:
            return False
    return True


def _ms_lookup(ms: dict, family: str, kernel: str, d: int) -> dict | None:
    """The multiseed config block matching (family, kernel, d), or None."""
    for cfg in ms.get("configs", {}).values():
        if cfg["family"] == family and cfg["kernel"] == kernel and cfg["d"] == d:
            return cfg
    return None


# --------------------------------------------------------------------------
# Figure 1: dimension sweep -- the kernel wall and its whitened-kernel fix.
#   (left)  warm/floor ratio at M = SWEEP_M against d, isotropic vs whitened.
#   (right) the isotropic node-Gram condition number tending to one.
# --------------------------------------------------------------------------
# Isotropic curve grey, whitened curve orange. Gaussian circle and solid, GMM
# square and dashed.
C_ISO = C_SOLVE       # grey -- the isotropic baseline that walls.

FAM_STYLE = {
    "gaussian": dict(marker="o", label="Gaussian"),
    "gmm": dict(marker="s", label="GMM"),
}


def render_dimension_sweep(
    sweep: dict, m_budget: int = SWEEP_M, ms: dict | None = None
) -> None:
    """Two-panel dimension sweep: the kernel wall and the whitened fix.

    When ``ms`` (the multiseed summary) is given, the left panel's curves use
    the across-training-seed median warm/floor and shade the interquartile
    range over the training seed as a band. Without ``ms`` the single-seed
    ``sweep`` points are drawn instead.
    """
    apply_house_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.4))

    ms_sweep = _ms_dim_sweep(ms, m_budget) if ms else None
    n_seeds = 0
    if ms:
        n_seeds = len(ms.get("seeds", []))

    # ---- (left) warm/floor vs d: isotropic (walls) vs whitened (holds). ---
    for fam, st in FAM_STYLE.items():
        if ms_sweep is not None:
            iso_m = ms_sweep[fam]["iso"]
            wht_m = ms_sweep[fam]["whiten"]
            if iso_m:
                d_ = [p[0] for p in iso_m]
                axL.fill_between(
                    d_, [p[2] for p in iso_m], [p[3] for p in iso_m],
                    color=C_ISO, alpha=0.16, lw=0, zorder=1,
                )
                axL.plot(
                    d_, [p[1] for p in iso_m], color=C_ISO,
                    lw=1.7, ms=6.0, ls="--", marker=st["marker"], zorder=3,
                    markeredgecolor="black", markeredgewidth=0.4,
                    label=f"{st['label']}, isotropic",
                )
            if wht_m:
                d_ = [p[0] for p in wht_m]
                axL.fill_between(
                    d_, [p[2] for p in wht_m], [p[3] for p in wht_m],
                    color=C_WARM, alpha=0.18, lw=0, zorder=2,
                )
                axL.plot(
                    d_, [p[1] for p in wht_m], color=C_WARM,
                    lw=2.0, ms=6.5, ls="-", marker=st["marker"], zorder=5,
                    markeredgecolor="black", markeredgewidth=0.45,
                    label=f"{st['label']}, whitened",
                )
            continue
        iso = sweep.get(fam, {}).get("iso", [])
        wht = sweep.get(fam, {}).get("whiten", [])
        if iso:
            axL.plot(
                [p[0] for p in iso], [p[1] for p in iso], color=C_ISO,
                lw=1.7, ms=6.0, ls="--", marker=st["marker"], zorder=3,
                markeredgecolor="black", markeredgewidth=0.4,
                label=f"{st['label']}, isotropic",
            )
        if wht:
            axL.plot(
                [p[0] for p in wht], [p[1] for p in wht], color=C_WARM,
                lw=2.0, ms=6.5, ls="-", marker=st["marker"], zorder=5,
                markeredgecolor="black", markeredgewidth=0.45,
                label=f"{st['label']}, whitened",
            )

    axL.axhline(
        1.0, color=C_FLOOR, lw=1.4, ls=(0, (5, 3)), zorder=2,
        label="i.i.d. floor",
    )
    axL.set_xscale("log", base=2)
    axL.set_yscale("log")
    if ms_sweep is not None:
        all_d = sorted(
            {p[0] for fam in ms_sweep.values() for arm in fam.values()
             for p in arm}
        )
    else:
        all_d = sorted(
            {p[0] for fam in sweep.values() for arm in fam.values() for p in arm}
        )
    axL.set_xticks(all_d)
    axL.set_xticklabels([str(v) for v in all_d])
    axL.set_xlabel(r"ambient dimension $d$")
    axL.set_ylabel(r"warm / floor $\mathrm{MMD}^2$ at $M=%d$" % m_budget)
    _title = "the whitened kernel holds the margin"
    if ms_sweep is not None and n_seeds > 1:
        _title += f" ({n_seeds} training seeds; band = IQR)"
    axL.set_title(_title, fontsize=10)
    # A plain label in the empty upper-left corner, where at small d every
    # curve is far below the line. No arrow: its tail crossed the rising
    # isotropic curve and the d = 10 markers.
    axL.text(
        0.02, 0.90, "below 1: beats i.i.d. floor",
        transform=axL.transAxes, fontsize=7.0, color="0.35",
        ha="left", va="top",
    )
    despine(axL)
    axL.legend(frameon=False, fontsize=7.2, loc="lower left", ncol=1)

    # ---- (right) the kernel-wall diagnostic: cond(K) -> 1 as d grows. ------
    try:
        with open(GRAM_JSON) as fh:
            gram = json.load(fh)
    except Exception:  # noqa: BLE001
        gram = {}
    gram_fam = {"gaussian": "gaussian", "gmm": "gmm_cncv"}
    for fam, st in FAM_STYLE.items():
        block = gram.get(gram_fam[fam], {})
        if not block:
            continue
        pts = sorted((int(k), v["cond"]) for k, v in block.items())
        axR.plot(
            [p[0] for p in pts], [p[1] for p in pts], color=C_ISO,
            lw=1.7, ms=6.0, marker=st["marker"], zorder=3,
            markeredgecolor="black", markeredgewidth=0.4,
            label=f"{st['label']}, isotropic",
        )
    axR.axhline(
        1.0, color=C_FLOOR, lw=1.4, ls=(0, (5, 3)), zorder=2,
        label=r"$K=I$",
    )
    axR.set_xscale("log", base=2)
    axR.set_yscale("log")
    if all_d:
        axR.set_xticks(all_d)
        axR.set_xticklabels([str(v) for v in all_d])
    axR.set_xlabel(r"ambient dimension $d$")
    axR.set_ylabel(r"isotropic node Gram $\mathrm{cond}(K)$")
    axR.set_title("the kernel wall: the Gram tends to $I$", fontsize=10)
    despine(axR)
    axR.legend(frameon=False, fontsize=7.2, loc="upper right")

    fig.tight_layout()
    _save(fig, "dq_dimension_sweep")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: MMD^2 vs M for three representative targets (1 x 3).
# --------------------------------------------------------------------------
def _panel(
    ax, res: dict, title: str, show_ylabel: bool, ms_cfg: dict | None = None
) -> None:
    """Log-log MMD^2 against M: floor band, warm (orange), descent (grey).

    When ``ms_cfg`` (a multiseed config block) is given, the warm-start curve
    is drawn at the across-training-seed median warm MMD^2 with the
    interquartile range over the training seed shaded as a band. The floor and
    the descent do not depend on the training seed, since the held-out sets
    are fixed through ``base_seed``, so they keep their single curve with the
    median and interquartile range over those sets.
    """
    rows = res["rows"]
    m = [r["M"] for r in rows]

    fl_md = [r["floor"][0] for r in rows]
    fl_lo = [min(r["floor"][1], r["floor"][2]) for r in rows]
    fl_hi = [max(r["floor"][1], r["floor"][2]) for r in rows]
    or_md = [r["oracle"][0] for r in rows]
    wa_md = [r["warm"][0] for r in rows]
    # Older records carry no reweight arm; draw what is there and report the
    # gap rather than inventing the arm.
    rw_md = (
        [r["reweight"][0] for r in rows] if all("reweight" in r for r in rows)
        else None
    )
    if rw_md is None:
        print(
            f"  [ARM MISSING] '{title}': the record carries no reweight arm, "
            "so this panel draws 3 of the caption's 4. Re-run the target's "
            "trainer (records written after 2026-08-10 carry it)."
        )

    # Independent-sample floor: shaded interquartile band and dashed median.
    ax.fill_between(m, fl_lo, fl_hi, color=C_FLOOR, alpha=0.18, lw=0, zorder=1)
    ax.plot(
        m, fl_md, color=C_FLOOR, lw=1.5, ls=(0, (5, 3)), marker="o", ms=3.5,
        label="i.i.d. floor", zorder=2,
    )
    # The constrained solve at the untouched independent-sample nodes.
    if rw_md is not None:
        ax.plot(
            m, rw_md, color=C_REWEIGHT, lw=1.5, marker="s", ms=3.2,
            label="reweight only", zorder=3,
        )
    # Per-instance descent from the independent-sample seed, not a refinement
    # of the returned quadrature, and a budgeted descent rather than an
    # optimum: other arms can cross below it at large M.
    ax.plot(
        m, or_md, color=C_ORACLE, lw=1.3, ls=(0, (1, 2)), marker="o", ms=3.0,
        label="per-instance descent", zorder=3,
    )

    # The returned quadrature, in orange. With a multiseed block, the
    # across-seed median and interquartile band; otherwise the single-seed
    # line.
    warm_label = "move and reweight"
    if ms_cfg is not None:
        ms_m, ms_md, ms_lo, ms_hi = [], [], [], []
        for M_ in m:
            blk = ms_cfg["per_M"].get(str(M_))
            if blk is None:
                continue
            md, q25, q75 = blk["warm_mmd2"]["median_iqr"]
            ms_m.append(M_)
            ms_md.append(md)
            ms_lo.append(min(q25, q75))
            ms_hi.append(max(q25, q75))
        if ms_m:
            ax.fill_between(
                ms_m, ms_lo, ms_hi, color=C_WARM, alpha=0.22, lw=0, zorder=3,
            )
            ax.plot(
                ms_m, ms_md, color=C_WARM, lw=1.9, marker="o", ms=4.0,
                label=warm_label, zorder=4,
            )
            wa_md = ms_md  # use the median for the win-star check below
            m = ms_m
            fl_md = [
                next(r["floor"][0] for r in rows if r["M"] == M_) for M_ in ms_m
            ]
    else:
        ax.plot(
            m, wa_md, color=C_WARM, lw=1.9, marker="o", ms=4.0,
            label=warm_label, zorder=4,
        )

    # A cell whose seed Gram is conditioned past the limit has a
    # rank-deficient weight solve, so many weight vectors fit almost equally
    # and which one each arm lands on is decided by solve noise rather than by
    # method. The values still carry digits above the noise floor, so they are
    # drawn, but open-faced, to be read as values and not as an ordering.
    lim = {r["M"] for r in rows if r.get("cond_limited")}
    if lim:
        for M_, w in zip(m, wa_md):
            if M_ in lim:
                ax.plot(
                    [M_], [w], marker="o", ms=7.5, mfc="white",
                    mec=C_WARM, mew=1.4, ls="none", zorder=7,
                )

    # Gold star on every warm point below the floor median. A
    # conditioning-limited cell gets no star, since the comparison there is
    # not an ordering.
    win_m = [M_ for M_, w, f in zip(m, wa_md, fl_md) if w < f and M_ not in lim]
    win_v = [
        w for M_, w, f in zip(m, wa_md, fl_md) if w < f and M_ not in lim
    ]
    if win_m:
        ax.scatter(
            win_m, win_v, marker="*", s=95, c=GOLD, edgecolors="black",
            linewidths=0.5, zorder=8,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(m)
    ax.set_xticklabels([str(v) for v in m])
    ax.set_xlabel(r"node budget $M$")
    if show_ylabel:
        ax.set_ylabel(r"$\mathrm{MMD}^2(Q,\rho)$")
    ax.set_title(title, fontsize=9.5)
    despine(ax)


def render_mmd_vs_M(runs: dict, ms: dict | None = None) -> None:
    """1x3 references: Gaussian d=2, GMM d=10, banana d=2.

    When the multiseed summary ``ms`` is present, each panel's warm-start curve
    shows the across-training-seed median and interquartile band.
    """
    apply_house_style()
    picks = [
        # CANON label,        title,           (family, kernel, d) for ms lookup
        ("Gaussian, d = 2", "Gaussian, $d=2$", ("gaussian", "iso", 2)),
        ("GMM, d = 10", "GMM, $d=10$", ("gmm", "iso", 10)),
        ("banana, d = 2", "banana, $d=2$", ("banana", "iso", 2)),
    ]
    missing = [k for k, _t, _f in picks if runs.get(k) is None]
    if missing:
        raise RuntimeError(
            f"fig:mmd-vs-M cannot be drawn: no canonical run for {missing}. "
            "The caption claims the emission is below the floor at EVERY "
            "budget on references of every kind; a blank panel would ship "
            "that claim with a hole in it. Pin the row to a checkpoint that "
            "exists on the deployed grid, or retrain the target."
        )
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
    n_seeds = len(ms.get("seeds", [])) if ms else 0
    banded = 0
    for i, (key, title, fkd) in enumerate(picks):
        res = runs.get(key)
        ms_cfg = _ms_lookup(ms, *fkd) if ms else None
        if ms and ms_cfg is None:
            print(
                f"  [NO BAND] '{title}': the multiseed record has no "
                f"{fkd} config, so this panel is single-seed while the "
                "others are banded."
            )
        banded += ms_cfg is not None
        _panel(axes[i], res, title, show_ylabel=(i == 0), ms_cfg=ms_cfg)
    # One shared legend, on the first panel.
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    # The suptitle describes the whole figure, so it is written only when
    # every panel carries the band.
    if ms and n_seeds > 1 and banded == len(picks):
        fig.suptitle(
            f"emission band = IQR over {n_seeds} training seeds",
            fontsize=8.5, y=1.02, color="0.35",
        )
    elif ms and n_seeds > 1:
        print(
            f"  [SUPTITLE OMITTED] only {banded}/{len(picks)} panels carry a "
            "multi-seed band; the blanket sentence would overclaim."
        )
    fig.tight_layout()
    _save(fig, "dq_mmd_vs_M")
    plt.close(fig)


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  wrote {stem}.pdf / {stem}.png")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = _load_canon()
    print(f"[load] {len(runs)}/{len(CANON)} canonical runs found")
    sweep = _load_sweep()
    for fam in ("gaussian", "gmm"):
        iso = sweep[fam]["iso"]
        wht = sweep[fam]["whiten"]
        print(
            f"[sweep] {fam}: iso d={[p[0] for p in iso]}  "
            f"whitened d={[p[0] for p in wht]}"
        )
    ms = _load_multiseed()
    if ms is not None:
        print(
            f"[multiseed] loaded {len(ms.get('configs', {}))} configs, "
            f"{len(ms.get('seeds', []))} training seeds "
            f"-> drawing across-seed IQR bands"
        )
    else:
        print("[multiseed] no dq_toy_seeds.json -- single-seed figures")
    if _ms_covers_sweep(ms):
        print("[render] dimension sweep (kernel wall + whitened fix)")
        render_dimension_sweep(sweep, ms=ms)
    else:
        # Redrawing the sweep from a record that does not cover it would
        # replace the existing figure with a degenerate two-point panel, so
        # the skip is reported rather than silent.
        print(
            "[render] dimension sweep SKIPPED: "
            f"{os.path.relpath(MULTISEED_JSON, gitdir())} does not carry the "
            "sweep configs (needs the isotropic AND whitened arms across "
            "SWEEP_DIMS). The shipped artifact is left untouched; re-point "
            "MULTISEED_JSON at a record that covers the sweep to redraw it."
        )
    print("[render] MMD^2 vs M (1x3)")
    render_mmd_vs_M(runs, ms=ms)
    print("done.")


if __name__ == "__main__":
    main()
