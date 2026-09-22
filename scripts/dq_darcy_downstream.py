"""Darcy expensive-integrand panel: a PDE expectation priced in forward solves.

Three functionals of the solved pressure field, each costing exactly one
``DarcyForward.solve`` per node, are integrated against the reference measure
by five arms at matched node budget, and the record prices error in the
expectation against forward solves spent. The gold is the reference flow's own
measure. Everything is counted in solves, never in wall-clock time.

Arms per (observation, M, repeat), transcribed from the deployed evaluator's
loop (``dq_darcy_stuart.py``, ``_evaluate``): the same bank seeds, the same
dedupe-with-multiplicity, the same standardized-y conditioning, the same
``select_emission(names=("seeds", "reweight", "move"), fitting_criterion=True)``:

  iid       -- multiplicity-weighted bank rows (the floor; the M solves).
  reweight  -- the same rows with constrained solved weights, at no extra
               solves; the improvement is in the kernel's own norm, and the
               per-functional direction is what this panel measures.
  emission  -- the safeguarded selection.
  oracle    -- per-instance move-descend, R_ORACLE repeats only, the descent
               being the run's cost pole.
  plugin    -- the point-estimate plug-in: the field at the calibration
               block's posterior-mean latent, one solve.

Statistics. The primary contrast is emission against iid on the smoothed tail
exceedance. Observations are replicates, so the statistics are pooled paired
ones plus a per-observation sign consistency, with one Holm family over the
15 QoI x M cells and censored cells kept at p = 1. A cell counts as
gold-resolved when the separation of the two median absolute errors exceeds
GOLD_MULT times the gold CLT half-width. tau and b come from a seed-disjoint
calibration block drawn before the gold, so the estimand is a fixed integrand.
The matched-scale positive control is iid-at-M against iid-at-2M through the
identical pipeline and must recover a factor near sqrt(2). The between-arm
variance contrast is the gold-free secondary. Raw per-repeat estimates persist
so bias and variance decompose; safeguard ties drop from the sign statistics
with n_ties recorded, and a selection-frequency table ships per cell. The
iid-equivalent multiplier is computed only in gold-resolved cells, after the
iid slope is verified near -1/2, with a bootstrap interval, and binds to the
reweight arm only.

The sweep takes hours, so run it under tmux. Re-render from the sidecar with
``--phase visualize``. Resume is per observation: partials are written
atomically and a relaunch reuses the finished observations.

Outputs: ``paper/figures/dq_darcy_downstream.{pdf,png}`` and
``paper/figures/diagnostics/dq_darcy_downstream.json``.
"""

from __future__ import annotations
from projorg import configsdir, logsdir, plotsdir  # noqa: E402

# Pin CPU before importing torch.
import os  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sps  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from _ckpt import checkpoint_path  # noqa: E402
from _figstyle import apply_house_style  # noqa: E402
# The statistics machinery is reused rather than copied.
from dq_budget_channel_ablation import _holm, _paired_cell  # noqa: E402
from dq_darcy_stuart import DQDarcyStuart  # noqa: E402

from qfield.dataset.darcy_stuart_kl_prior import StuartKLPrior  # noqa: E402
from qfield.dataset.darcy_stuart_op import DarcyForward  # noqa: E402
from qfield.designed_quadrature import (  # noqa: E402
    constrained_weights,
    move_descend,
)
from qfield.designed_quadrature.safeguard import select_emission  # noqa: E402

DTYPE = torch.float64
torch.set_num_threads(4)

_FIG_DIR = plotsdir("paper")
_DIAG_DIR = os.path.join(_FIG_DIR, "diagnostics")
_LOG_DIR = logsdir("darcy_downstream")
_OUT_JSON = os.path.join(_DIAG_DIR, "dq_darcy_downstream.json")
_CONFIG = configsdir("dq_darcy_stuart.json")
_CKPT_IGNORE = ("recon_M", "n_recon", "eval_repeats", "n_pcn_audit")

# ---- constants ------------------------------------------------------------
N_OBS = 8            # the first 8 of the evaluator's own fresh observations
R = 256              # repeats for the solved arms
R_ORACLE = 8         # oracle repeats; the descent is the run's cost pole
N_GOLD = 131072      # from N_gold >= M_max (1.96 GOLD_MULT / (0.674 f_min))^2
N_CAL = 4096         # seed-disjoint calibration block, drawn before the gold
GOLD_MULT = 2.0      # censoring multiplier (f_min = 0.5)
QOIS = ("pressure", "tail", "flux")
PRIMARY_QOI = "tail"
B_BRIDGE_DIV = 4     # the b/4 bridge cell recorded in the sidecar
BOOT_B = 10000
BOOT_SEED = 20260612
ALPHA = 0.05
SLOPE_TOL = 0.1      # iid slope must be within this of -1/2 to invert
# The unobserved probe location, a fixed interior grid point. The Beskos
# sensor ring is near-boundary (default radius (N-1)/(2N) ~ 0.49 about the
# center); the probe at physical ~(0.77, 0.77) sits ~0.14 from the nearest
# ring sensor, so it is interior and unobserved.
PROBE_FRAC = (0.75, 0.75)
# Seed bands disjoint from the evaluator's own: the calibration block, the
# gold halves and the 2M positive control all live at 900_000 and above.
CAL_OFF = 900_000
GOLD_OFF = 910_000
CTRL_OFF = 930_000


# ------------------------------------------------------------------ helpers
def _atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _status(line: str) -> None:
    try:
        with open(os.path.join(_LOG_DIR, "status.txt"), "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _boot_seed(tag: str) -> int:
    return BOOT_SEED ^ zlib.crc32(tag.encode())


def _config_args() -> argparse.Namespace:
    """The deployed run's config as a namespace (projorg-free construction).

    Types come straight from the JSON; the one sequence field (``m_list``)
    is parsed exactly as ``setup_environment`` would.
    """
    with open(_CONFIG) as fh:
        cfg = json.load(fh)
    cfg["m_list"] = [int(v) for v in str(cfg["m_list"]).split(",")]
    return argparse.Namespace(**cfg)


class QoiEvaluator:
    """The three functionals of one solved pressure field.

    Each call to :meth:`values` costs exactly one ``DarcyForward.solve``, the
    accounting unit of the whole panel. A row-level cache keyed on the latent's
    bytes exploits the safeguard's fallback rate, where the selected nodes are
    usually the seed rows themselves at the top budget, and so cuts repeated
    solves without changing any number.
    """

    def __init__(self, attrs: dict, tau_b: dict | None = None):
        self.N = int(attrs["N"])
        self.kl = StuartKLPrior(
            self.N, K=int(attrs["K"]), alpha=float(attrs["alpha"]),
            s=float(attrs["s"]), sigma=float(attrs["sigma"]),
        )
        self.fwd = DarcyForward(self.N)
        i = min(int(PROBE_FRAC[0] * self.N), self.N - 1)
        j = min(int(PROBE_FRAC[1] * self.N), self.N - 1)
        self.probe_ij = (i, j)
        self.mid = self.N // 2
        self.tau_b = tau_b or {}
        self.n_solves = 0
        self._cache: dict[bytes, tuple[float, float]] = {}

    def _raw(self, xi: np.ndarray,
             cache: bool = True) -> tuple[float, float]:
        """(pressure at the probe, midline flux) for one latent, in one solve.

        ``cache=False`` on the gold and calibration paths: their latents are
        never hit twice, and retaining them grows the cache without bound.
        A non-finite solve returns (nan, nan), and the pair is dropped at the
        statistics stage.
        """
        key = None
        if cache:
            key = hashlib.sha1(np.ascontiguousarray(xi).tobytes()).digest()
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        u = self.kl.reconstruct(xi)
        p = self.fwd.solve(u)
        self.n_solves += 1
        if not np.all(np.isfinite(p)):
            out = (float("nan"), float("nan"))
            if key is not None:
                self._cache[key] = out
            return out
        press = float(p[self.probe_ij])
        # Midline Darcy flux across the x2 = 1/2 line (differences along
        # axis 1), using the operator's own face conductance: the arithmetic
        # mean of the two cell permeabilities over the grid spacing h
        # (darcy_stuart_op.py builds Kx = 0.5 (K_a + K_b) / h^2), averaged
        # over rows.
        K = np.exp(u) if u.ndim == 2 else np.exp(u.reshape(self.N, self.N))
        h = 1.0 / (self.N - 1)
        kf = 0.5 * (K[:, self.mid - 1] + K[:, self.mid])
        flux = float(
            np.mean(kf * (p[:, self.mid - 1] - p[:, self.mid])) / h
        )
        out = (press, flux)
        if key is not None:
            self._cache[key] = out
        return out

    def values(self, xi: np.ndarray, obs_key: str,
               cache: bool = True) -> dict[str, float]:
        press, flux = self._raw(xi, cache=cache)
        tau, b = self.tau_b[obs_key]
        out = {
            "pressure": press,
            "tail": float(sps.logistic.cdf((press - tau) / b)),
            "flux": flux,
            "tail_bridge": float(
                sps.logistic.cdf((press - tau) / (b / B_BRIDGE_DIV))
            ),
        }
        return out

    def batch_means(self, xis: np.ndarray, weights: np.ndarray,
                    obs_key: str) -> dict[str, float]:
        vals = {q: 0.0 for q in QOIS + ("tail_bridge",)}
        for w, xi in zip(weights, xis):
            v = self.values(np.asarray(xi, dtype=np.float64), obs_key)
            for q in vals:
                vals[q] += float(w) * v[q]
        return vals


# ------------------------------------------------------------- gold + cal
def _draw_latents(exp: DQDarcyStuart, o: int, n: int, seed: int) -> np.ndarray:
    gen = torch.Generator().manual_seed(seed)
    return exp._draw(exp.y_ev[o : o + 1], n, gen)[0].numpy()


def _calibrate(exp, qoi: QoiEvaluator, o: int, base_seed: int) -> dict:
    """tau (the 0.9 quantile) and b from the calibration block.

    The block is drawn before the gold and from a disjoint seed band, so the
    estimand is a fixed integrand and the gold CLT applies to it.
    """
    xis = _draw_latents(exp, o, N_CAL, base_seed + CAL_OFF + o)
    press = np.array([qoi._raw(xi, cache=False)[0] for xi in xis])
    tau = float(np.quantile(press, 0.9))
    b = float(0.5 * press.std())
    mean_xi = xis.mean(axis=0)
    return {"tau": tau, "b": b, "cal_press_std": float(press.std()),
            "plugin_xi": mean_xi.tolist()}


def _gold(exp, qoi: QoiEvaluator, o: int, obs_key: str,
          base_seed: int) -> dict:
    """Two independent halves, with a 3-sigma agreement check between them."""
    halves = []
    for h in (0, 1):
        xis = _draw_latents(exp, o, N_GOLD // 2,
                            base_seed + GOLD_OFF + 1000 * o + h)
        vals = {q: [] for q in QOIS + ("tail_bridge",)}
        for xi in xis:
            v = qoi.values(np.asarray(xi, dtype=np.float64), obs_key,
                           cache=False)
            for q in vals:
                vals[q].append(v[q])
        halves.append({q: np.array(v) for q, v in vals.items()})
    out = {}
    for q in QOIS + ("tail_bridge",):
        a, b = halves[0][q], halves[1][q]
        pooled = np.concatenate([a, b])
        se = float(pooled.std(ddof=1) / np.sqrt(len(pooled)))
        z = abs(a.mean() - b.mean()) / max(
            np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)), 1e-300
        )
        assert z < 3.0, (
            f"gold halves disagree at {z:.1f} sigma for {q} (obs {o}) -- "
            "the gold is not self-consistent; do not trust this record."
        )
        out[q] = {"mean": float(pooled.mean()), "half_width": 1.96 * se,
                  "halves_z": float(z), "kurtosis": float(sps.kurtosis(pooled))}
    return out


# ------------------------------------------------------------------ arms
def _score_observation(exp, net, qoi: QoiEvaluator, o: int, a) -> dict:
    """All arms, all budgets, one observation.

    A transcription of the deployed evaluator's menu loop
    (dq_darcy_stuart.py, _evaluate): the same seed band, the same
    dedupe-with-multiplicity, the same standardized-y conditioning and the
    same safeguarded selection.
    """
    obs_key = str(o)
    y_cpu = exp.y_ev.cpu().to(DTYPE)
    raw = {}
    sel_freq = {}
    oracle_raw = {}
    t0 = time.time()
    for M in exp.m_list:
        L = int(a.bank_L) if int(M) < 256 else 2 * int(a.bank_L)
        gA = torch.Generator().manual_seed(
            int(a.base_seed) + 50_000 + 1000 * int(M) + o)
        gC = torch.Generator().manual_seed(
            int(a.base_seed) + CTRL_OFF + 1000 * int(M) + o)
        tgtA, bankA = exp._eval_target(o, gA, L)
        cell = {arm: {q: [] for q in QOIS + ("tail_bridge",)}
                for arm in ("iid", "reweight", "emission", "iid2m")}
        freq = {"seeds": 0, "reweight": 0, "move": 0}
        orc = {q: [] for q in QOIS}
        for r in range(R):
            idx = torch.multinomial(
                torch.ones(L, dtype=DTYPE), int(M),
                replacement=True, generator=gA)
            uniq, counts = torch.unique(idx, return_counts=True)
            zu = bankA[uniq]
            w_eq = counts.to(DTYPE) / float(M)
            muA_u = tgtA.mu_fn(zu)
            w_c = constrained_weights(zu, muA_u, exp.sigma, a.jitter)
            cnd = y_cpu[o].unsqueeze(0)
            with torch.no_grad():
                z_n = net(zu.unsqueeze(0), cnd)[0]
                mu_n = tgtA.mu_fn(z_n)
                w_n = constrained_weights(z_n, mu_n, exp.sigma, a.jitter)
            sel = select_emission(
                [(zu, w_eq), (zu, w_c.detach()), (z_n, w_n)],
                tgtA.mu_fn, exp.sigma,
                names=("seeds", "reweight", "move"),
                fitting_criterion=True)
            freq[sel["name"]] += 1
            zu_np = zu.numpy()
            cell["iid"] = _push(cell["iid"], qoi.batch_means(
                zu_np, w_eq.numpy(), obs_key))
            cell["reweight"] = _push(cell["reweight"], qoi.batch_means(
                zu_np, w_c.detach().numpy(), obs_key))
            cell["emission"] = _push(cell["emission"], qoi.batch_means(
                sel["z"].numpy(), sel["w"].numpy(), obs_key))
            # Matched-scale positive control: 2M rows through the same
            # pipeline, from a disjoint generator so the seed band's stream
            # stays in sync.
            idx2 = torch.multinomial(
                torch.ones(L, dtype=DTYPE), 2 * int(M),
                replacement=True, generator=gC)
            u2, c2 = torch.unique(idx2, return_counts=True)
            cell["iid2m"] = _push(cell["iid2m"], qoi.batch_means(
                bankA[u2].numpy(), (c2.to(DTYPE) / float(2 * M)).numpy(),
                obs_key))
            if r < R_ORACLE:
                z_o, _ = move_descend(
                    zu, tgtA.mu_fn, tgtA.c_rho, exp.sigma,
                    float(a.opt_lr), int(a.opt_iters), a.jitter)
                w_o = constrained_weights(
                    z_o, tgtA.mu_fn(z_o), exp.sigma, a.jitter)
                v = qoi.batch_means(z_o.numpy(), w_o.numpy(), obs_key)
                for q in QOIS:
                    orc[q].append(v[q])
        raw[str(M)] = cell
        sel_freq[str(M)] = {k: v / R for k, v in freq.items()}
        oracle_raw[str(M)] = orc
        print(f"  [obs {o}] M={M:>3} done  ({time.time() - t0:.0f} s, "
              f"{qoi.n_solves} solves so far, "
              f"sel={sel_freq[str(M)]})", flush=True)
    return {"raw": raw, "sel_freq": sel_freq, "oracle_raw": oracle_raw}


def _push(cell: dict, vals: dict) -> dict:
    for q in cell:
        cell[q].append(vals[q])
    return cell


# ------------------------------------------------------------- statistics
def _cell_stats(all_obs: list[dict], gold: list[dict], M: str,
                qoi_name: str, bank_L: int) -> dict:
    """Pooled paired statistics for one (QoI, M) cell across observations."""
    err = {arm: [] for arm in ("iid", "reweight", "emission", "iid2m")}
    est = {arm: [] for arm in err}
    per_obs_mlr, per_obs_ties, n_dropped = [], [], 0
    for o, ob in enumerate(all_obs):
        g = gold[o][qoi_name]["mean"]
        ests = {arm: np.array(ob["raw"][M][arm][qoi_name]) for arm in err}
        # A non-finite estimate drops the whole repeat pair across arms,
        # counted per cell.
        finite = np.all([np.isfinite(ests[arm]) for arm in err], axis=0)
        n_dropped += int((~finite).sum())
        for arm in err:
            e = ests[arm][finite] - g
            err[arm].append(np.abs(e))
            est[arm].append(ests[arm][finite])
        a = np.abs(ests["emission"][finite] - g)
        b = np.abs(ests["iid"][finite] - g)
        # Sign consistency runs on the untied pairs: the safeguard's fallback
        # makes emission equal to iid bitwise on most top-budget repeats, and
        # sign(0) would make the statistic vacuous.
        ok = (a > 0) & (b > 0) & (a != b)
        per_obs_ties.append(int((a == b).sum()))
        per_obs_mlr.append(
            float(np.median(np.log(a[ok]) - np.log(b[ok]))) if ok.sum() >= 8
            else None)
    cat = {arm: np.concatenate(err[arm]) for arm in err}
    hw = float(np.mean([gold[o][qoi_name]["half_width"]
                        for o in range(len(gold))]))
    sep = abs(float(np.median(cat["iid"])) - float(np.median(cat["emission"])))
    resolved = bool(sep > GOLD_MULT * hw)
    pc = _paired_cell(
        torch.tensor(cat["emission"], dtype=DTYPE),
        torch.tensor(cat["iid"], dtype=DTYPE), 0.0,
        f"dd:{qoi_name}:{M}:ei")
    pc_rw = _paired_cell(
        torch.tensor(cat["reweight"], dtype=DTYPE),
        torch.tensor(cat["iid"], dtype=DTYPE), 0.0,
        f"dd:{qoi_name}:{M}:ri")
    pc_ctrl = _paired_cell(
        torch.tensor(cat["iid2m"], dtype=DTYPE),
        torch.tensor(cat["iid"], dtype=DTYPE), 0.0,
        f"dd:{qoi_name}:{M}:c2")
    # sqrt(2) recovery: the control's interval must cover the L-adjusted
    # target, since the shared finite-L bank floor degrades the ideal
    # -0.5 log 2 toward zero at the top budgets.
    L_here = bank_L if int(M) < 256 else 2 * bank_L
    tgt = -0.5 * float(np.log((1.0 / int(M) + 1.0 / L_here)
                              / (0.5 / int(M) + 1.0 / L_here)))
    ci = pc_ctrl["log_ratio_ci"]
    pc_ctrl["sqrt2_target"] = tgt
    pc_ctrl["sqrt2_ok"] = (bool(ci[0] <= tgt <= ci[1])
                           if ci is not None else None)
    signs = {np.sign(v) for v in per_obs_mlr if v is not None}
    # Paired iid-equivalent multiplier per contrast: the median of the
    # per-repeat ratios with a bootstrap interval, gated later by the slope
    # and resolution checks in _multiplier.
    mult = {}
    for arm in ("emission", "reweight"):
        a_arr, b_arr = cat[arm], cat["iid"]
        use = (a_arr > 0) & (b_arr > 0) & (a_arr != b_arr)
        if use.sum() >= 32:
            m_eq_r = int(M) * (b_arr[use] / a_arr[use]) ** 2
            rng = np.random.default_rng(
                _boot_seed(f"dd:mult:{qoi_name}:{M}:{arm}"))
            meds = np.median(rng.choice(
                m_eq_r, size=(BOOT_B, len(m_eq_r)), replace=True), axis=1)
            mult[arm] = {
                "median_m_eq": float(np.median(m_eq_r)),
                "ci": [float(np.quantile(meds, ALPHA / 2)),
                       float(np.quantile(meds, 1 - ALPHA / 2))],
                "n_used": int(use.sum()),
            }
        else:
            mult[arm] = None
    out = {
        "paired_multiplier": mult,
        "gold_half_width": hw,
        "median_abs_err": {arm: float(np.median(cat[arm])) for arm in cat},
        "rmse": {arm: float(np.sqrt(np.mean(cat[arm] ** 2))) for arm in cat},
        "bias": {arm: float(np.mean(np.concatenate(est[arm]))
                            - np.mean([gold[o][qoi_name]["mean"]
                                       for o in range(len(gold))]))
                 for arm in est},
        "variance": {arm: float(np.mean([np.var(x, ddof=1)
                                         for x in est[arm]]))
                     for arm in est},
        "separation": sep, "gold_resolved": resolved,
        "emission_vs_iid": pc, "reweight_vs_iid": pc_rw,
        "control_2m_vs_iid": pc_ctrl,
        "per_obs_mlr": per_obs_mlr, "per_obs_ties": per_obs_ties,
        "n_dropped_pairs": n_dropped,
        "obs_sign_consistent": (len(signs) == 1 if all(
            v is not None for v in per_obs_mlr) else None),
    }
    return out


def _multiplier(cells: dict, qoi_name: str) -> dict:
    """The iid-equivalent budget multiplier, gated.

    The paired form (the median of the per-repeat ratios with a bootstrap
    interval, computed in _cell_stats) is reported only where the fitted iid
    slope is near -1/2 and the cell is gold-resolved. Extrapolation past the
    measured grid is flagged. Both contrasts are returned: a claim of fewer
    solves binds to the reweight arm, while the emission's multiplier is a
    counterfactual i.i.d. cost.
    """
    ms = sorted(int(M) for M in cells)
    rmse = [cells[str(M)]["rmse"]["iid"] for M in ms]
    slope = float(np.polyfit(np.log(ms), np.log(rmse), 1)[0])
    ok_slope = bool(abs(slope + 0.5) < SLOPE_TOL)
    out = {"iid_slope": slope, "slope_ok": ok_slope, "per_M": {}}
    for M in ms:
        c = cells[str(M)]
        if not (ok_slope and c["gold_resolved"]):
            out["per_M"][str(M)] = None
            continue
        entry = {}
        for arm in ("emission", "reweight"):
            pm = c["paired_multiplier"][arm]
            entry[arm] = None if pm is None else {
                **pm, "extrapolated": bool(pm["median_m_eq"] > max(ms)),
            }
        out["per_M"][str(M)] = entry
    return out


# ------------------------------------------------------------------ figure
def _render(rec: dict) -> None:
    apply_house_style()
    cells = rec["cells"]
    ms = sorted(int(M) for M in cells[PRIMARY_QOI])
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    # Palette, shared with every other figure: floor #3a7ca5, reweight
    # #2a9d8f, emission #ff7f0e, per-instance descent #7f7f7f.
    colors = {"iid": "#3a7ca5", "reweight": "#2a9d8f",
              "emission": "#ff7f0e", "oracle": "#7f7f7f"}
    labels = {"iid": "independent samples", "reweight": "reweight only",
              "emission": "the emission", "oracle": "per-instance descent"}
    for ax, q in zip(axes, QOIS):
        for arm in ("iid", "reweight", "emission"):
            med = [cells[q][str(M)]["median_abs_err"][arm] for M in ms]
            ax.plot(ms, med, marker="o", markersize=4, linewidth=1.5,
                    color=colors[arm], label=labels[arm])
        orc = [rec["oracle_median_abs_err"][q].get(str(M)) for M in ms]
        if all(v is not None for v in orc):
            ax.plot(ms, orc, marker="^", markersize=4, linewidth=1.2,
                    linestyle=":", color=colors["oracle"],
                    label=labels["oracle"])
        hw = [cells[q][str(M)]["gold_half_width"] for M in ms]
        ax.fill_between(ms, 0, hw, color="0.85", zorder=0,
                        label="reference resolution")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ms)
        ax.set_xticklabels([str(m) for m in ms])
        ax.set_xlabel("node budget $M$ (= forward solves per estimate)")
        ax.set_title({"pressure": "pressure at the probe",
                      "tail": "smoothed tail exceedance",
                      "flux": "midline flux"}[q], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("median absolute error")
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, frameon=False, fontsize=8, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        path = os.path.join(_FIG_DIR, f"dq_darcy_downstream.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved to {path}")
    plt.close(fig)


# ------------------------------------------------------------------- main
def main(phase: str = "all") -> None:
    if phase == "visualize":
        with open(_OUT_JSON) as fh:
            rec = json.load(fh)
        _render(rec)
        return

    os.makedirs(_LOG_DIR, exist_ok=True)
    os.makedirs(_DIAG_DIR, exist_ok=True)
    a = _config_args()
    print("[darcy downstream] constructing the deployed evaluator ...",
          flush=True)
    # No set_default_dtype here: the flow must load before the float64
    # default, and DQDarcyStuart.__init__ sets it itself.
    exp = DQDarcyStuart(a)
    ck_path = checkpoint_path(os.path.basename(_CONFIG),
                              ignore_extra=_CKPT_IGNORE)
    ck = torch.load(ck_path, weights_only=False)
    # sigma is read from the run record and never recomputed, since only
    # train() computes it. Two checks follow: the checkpoint's copy must
    # match the sibling results.json, and it must match the deployed value.
    rec_sigma = float(ck["results"]["sigma"])
    sib = os.path.join(os.path.dirname(ck_path), "results.json")
    if os.path.isfile(sib):
        with open(sib) as fh:
            assert abs(float(json.load(fh)["sigma"]) - rec_sigma) < 1e-12, \
                "checkpoint sigma disagrees with its sibling results.json"
    assert abs(rec_sigma - 14.1489) < 1e-3, (
        f"sigma {rec_sigma!r} is not the registered deployed value 14.1489 "
        "-- wrong run resolved?")
    exp.sigma = rec_sigma
    net = exp._build_net()
    net.load_state_dict(ck["state_dict"])
    net.eval()

    qoi = QoiEvaluator(exp.attrs)
    # Timing probe: price a solve AND one oracle descent before committing.
    t0 = time.time()
    probe_xi = np.zeros(exp.d)
    qoi._raw(probe_xi)
    solve_ms = (time.time() - t0) * 1000
    projected_h = (
        N_OBS * (3.2 * sum(exp.m_list) * R + N_GOLD + N_CAL) * solve_ms
        / 1000 / 3600
        + N_OBS * R_ORACLE * 130 / 3600  # measured ~128 s descent per grid
    )
    msg = (f"PROJECTED <= ~{projected_h:.1f} h (an UPPER bound: the solve "
           f"term ignores the row cache, which caps distinct bank rows at "
           f"L per cell; solve {solve_ms:.1f} ms, descents priced at the "
           f"feasibility measurement)")
    print(f"[timing] {msg}", flush=True)
    _status(msg)

    # Provenance stamp: a partial written under different constants must not
    # be silently reused on resume.
    stamp = {"n_gold": N_GOLD, "n_cal": N_CAL, "repeats": R,
             "r_oracle": R_ORACLE, "gold_mult": GOLD_MULT,
             "probe": list(qoi.probe_ij), "sigma": rec_sigma}
    all_obs, gold, cal = [], [], []
    for o in range(N_OBS):
        part = os.path.join(_LOG_DIR, f"partial_obs{o}.json")
        if os.path.exists(part):
            with open(part) as fh:
                blob = json.load(fh)
            assert blob.get("stamp") == stamp, (
                f"partial obs {o} carries a different provenance stamp "
                f"({blob.get('stamp')} vs {stamp}); delete it or match the "
                "constants -- never mix records.")
            # The gold self-consistency assertion ran when the partial was
            # written; halves_z persists in the blob.
            print(f"[resume] obs {o} reused from partial", flush=True)
        else:
            solves0 = qoi.n_solves
            c = _calibrate(exp, qoi, o, int(a.base_seed))
            qoi.tau_b[str(o)] = (c["tau"], c["b"])
            g = _gold(exp, qoi, o, str(o), int(a.base_seed))
            sc = _score_observation(exp, net, qoi, o, a)
            blob = {"stamp": stamp, "cal": c, "gold": g,
                    "n_solves_obs": qoi.n_solves - solves0, **sc}
            _atomic_json(part, blob)
            # The cache holds nothing reusable across observations, so it is
            # released rather than allowed to grow.
            qoi._cache.clear()
        qoi.tau_b[str(o)] = (blob["cal"]["tau"], blob["cal"]["b"])
        cal.append(blob["cal"])
        gold.append(blob["gold"])
        all_obs.append(blob)

    # The point-estimate plug-in: one solve per observation, at the
    # calibration block's mean latent. The distinguishability check is
    # scoped to the nonlinear QoIs.
    plugin = []
    for o in range(N_OBS):
        v = qoi.values(np.array(cal[o]["plugin_xi"]), str(o))
        entry = {}
        for q in QOIS:
            err = abs(v[q] - gold[o][q]["mean"])
            hw = gold[o][q]["half_width"]
            entry[q] = {
                "value": v[q], "abs_err": err,
                "distinguished": (bool(err > GOLD_MULT * hw)
                                  if q in ("tail", "flux") else None),
                "within_gold_resolution": bool(err <= GOLD_MULT * hw),
            }
        plugin.append(entry)

    cells = {q: {} for q in QOIS}
    for q in QOIS:
        for M in (str(m) for m in exp.m_list):
            cells[q][M] = _cell_stats(all_obs, gold, M, q, int(a.bank_L))
    # The b/4 bridge cell ships in the sidecar, outside the Holm family.
    bridge = {M: _cell_stats(all_obs, gold, M, "tail_bridge", int(a.bank_L))
              for M in (str(m) for m in exp.m_list)}
    # One Holm family: the 15 QoI x M cells of the primary contrast, with
    # censored cells kept at p = 1.
    pvals = {}
    for q in QOIS:
        for M in cells[q]:
            c = cells[q][M]
            pvals[f"{q}:{M}"] = (c["emission_vs_iid"]["sign_p"]
                                 if c["gold_resolved"] else 1.0)
    holm = _holm(pvals)
    for key, rej in holm.items():
        q, M = key.split(":")
        c = cells[q][M]
        c["holm_reject"] = bool(rej)
        c["directional_verdict_allowed"] = bool(
            rej and c["gold_resolved"] and c["obs_sign_consistent"] is True)

    oracle_med = {q: {} for q in QOIS}
    for q in QOIS:
        for M in (str(m) for m in exp.m_list):
            vals = np.concatenate([
                np.abs(np.array(ob["oracle_raw"][M][q])
                       - gold[o][q]["mean"])
                for o, ob in enumerate(all_obs)])
            oracle_med[q][M] = float(np.median(vals))

    n_solves_total = sum(
        int(ob.get("n_solves_obs", 0)) for ob in all_obs
    ) + N_OBS  # + one plugin solve per observation
    rec = {
        "design": {
            "n_obs": N_OBS, "repeats": R, "r_oracle": R_ORACLE,
            "n_gold": N_GOLD, "n_cal": N_CAL, "gold_mult": GOLD_MULT,
            "probe_ij": qoi.probe_ij, "b_bridge_div": B_BRIDGE_DIV,
            "primary_qoi": PRIMARY_QOI, "sigma": rec_sigma,
            "m_list": exp.m_list, "config": _CONFIG,
            "n_solves_total": n_solves_total,
            "caveats": (
                "Per-repeat errors treat the gold mean as exact: every CI "
                "and win fraction is conditional on the gold, with the "
                "gold_resolved censoring as the guard; the between-arm "
                "variance contrast is the gold-free secondary. The gold "
                "self-consistency family is 32 tests (4 QoIs x 8 obs) at "
                "3 sigma. Raw per-repeat estimates live in "
                "logs/darcy_downstream/partial_obs*.json (kept, "
                "gitignored); the sidecar's summaries re-derive from them "
                "while they are retained."
            ),
        },
        "cells": cells,
        "bridge": bridge,
        "plugin": plugin,
        "oracle_median_abs_err": oracle_med,
        "multiplier": {q: _multiplier(cells[q], q) for q in QOIS},
        "holm_family": {"pvals": pvals, "rejections": holm},
        "selection_freq": {str(o): ob["sel_freq"]
                           for o, ob in enumerate(all_obs)},
        "gold": [{q: gold[o][q] for q in QOIS + ("tail_bridge",)}
                 for o in range(N_OBS)],
        "cal": cal,
    }
    _atomic_json(_OUT_JSON, rec)
    print(f"\nSaved numbers to {_OUT_JSON} "
          f"({n_solves_total} forward solves spent, "
          f"{qoi.n_solves} in this process)")
    _render(rec)

    print("\n=== emission vs iid (paired, pooled over observations) ===")
    for q in QOIS:
        for M in sorted(cells[q], key=int):
            c = cells[q][M]
            pc = c["emission_vs_iid"]
            mlr = pc["median_log_ratio"]
            print(f"  {q:>8} M={M:>3}: "
                  f"mlr={'%.3f' % mlr if mlr is not None else 'n/a'} "
                  f"win={pc['win_frac']} resolved={c['gold_resolved']} "
                  f"verdict={c.get('directional_verdict_allowed')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "visualize"),
                        default="all")
    args = parser.parse_args()
    main(phase=args.phase)
