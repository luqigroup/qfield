"""The Lotka--Volterra downstream integrand: expectations priced in solves.

The benchmark's quadratures are scored in the kernel discrepancy elsewhere;
this script adds the quantity-of-interest currency on the same trained
field. For each held-out member the safeguarded quadrature is emitted
exactly as ``lv_evaluate2.py`` emits it (same checkpoints, same banks, same
seeds, so the shipped selection ledger reproduces as the first twelve
repeats), each node is pushed through one extended solve of the system, and
three functionals of the trajectory are integrated against the reference
flow by five arms at matched node budget. Accuracy is priced in ODE solves.
The gold is the flow's own measure, so every claim this record licenses
integrates the reference, never the true posterior.

QoIs, one solve per node: ``pred`` (prey population at t = 35, ten time
units past the data), ``exceed`` (a logistic in the windowed prey peak,
with threshold and scale fixed from a calibration block drawn before the
gold; the registered primary), and ``tpeak`` (time of the first interior
prey peak, refined below the grid). A trajectory with no interior peak
yields tpeak = NaN, a flag distinct from solver failure, and a member whose
calibration block or gold shows any no-peak trajectory is censored from the
tpeak cells. The unsmoothed indicator ``exceed_sharp`` is recorded as a
sidecar outside the Holm family, beside the emitted weights' l1 norm.

Arms per (member, node count, repeat): ``iid`` (equal weights on the seed
samples, M solves), ``reweight`` (same nodes, solved weights, zero extra
solves), ``emission`` (the safeguarded selection; deployed it costs M
solves total), ``iid2m`` (2M fresh bank rows, the sqrt(2) positive
control), and ``plugin`` (one solve at the frame mean). Statistics follow
``dq_darcy_downstream.py``: pooled paired contrasts with members as
replicates, one Holm family over the fifteen QoI-by-budget cells with
censored cells at p = 1, errors normalized per member by the gold standard
deviation, and the paired iid-equivalent multiplier gated by the -1/2 iid
slope and gold resolution.

Needs a trained flow and field (``lv_train_flow.py``,
``lv_quadrature_rung.py``); checkpoints are not distributed. The run is a
few hours on one CPU and is resumable: one atomic partial per member,
stamped with every constant and the checkpoint identities, and

    python scripts/lv_downstream.py --phase distill

re-derives the record from finished partials without a single solve.
Writes the per-member partials under ``logs/lv_downstream/``, the raw
per-repeat arrays beside the checkpoints, and ``lv_downstream.json`` into
the records directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zlib
from multiprocessing import Pool

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
import torch  # noqa: E402

from projorg import datadir, logsdir  # noqa: E402

from qfield.dataset import lotka_volterra as lv  # noqa: E402
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)
from qfield.designed_quadrature.weights import (  # noqa: E402
    constrained_weights_batched,
)
from qfield.models.hint_flow import HINTFlow  # noqa: E402
from dq_budget_channel_ablation import _holm, _paired_cell  # noqa: E402
from lv_train_flow import OUT  # noqa: E402

torch.set_num_threads(4)

DIAG = datadir("records")
LOG_DIR = logsdir("lv_downstream")

D = lv.D_X
BASE_SEED = 20260612
M_LIST = (32, 64, 128, 256, 512)
L_BANK = 16384
JITTER = 1e-8
# Env overrides let a reduced run finish in seconds; every override lands
# in the stamp, so a reduced partial can never be mistaken for the full
# run's.
N_REP = int(os.environ.get("LV_DS_NREP", 64))
N_CAL = int(os.environ.get("LV_DS_NCAL", 4096))
N_GOLD = int(os.environ.get("LV_DS_NGOLD", 131072))
N_WORKERS = int(os.environ.get("LV_DS_WORKERS", 5))
GOLD_MULT = 2.0
T_STAR = 35.0
N_T_EXT = 3360
RTOL, ATOL = 1e-8, 1e-10            # lv.simulate's own tolerances
GUARD_RTOL, GUARD_ATOL = 1e-10, 1e-12
N_GUARD = 8
BOOT_B = 10000
BOOT_SEED = 20260612
ALPHA = 0.05
SLOPE_TOL = 0.1
CAL_OFF = 2_000_000                 # bands disjoint from every seed the
GOLD_OFF = 2_100_000                # other lv_ scripts derive from the
CTRL_OFF = 2_200_000                # same base
QOIS = ("pred", "exceed", "tpeak")
PRIMARY_QOI = "exceed"
ARMS = ("iid", "reweight", "emission", "iid2m")

_T_EXT = np.linspace(0.0, T_STAR, N_T_EXT)
_DT = _T_EXT[1] - _T_EXT[0]
_N_W = int((_T_EXT <= lv.T_END).sum())


# ------------------------------------------------------------- solve worker
def _first_peak(u1w: np.ndarray) -> float:
    """Time of the first interior local maximum, quadratically refined.

    NaN when the windowed trajectory has no interior peak, a flag kept
    distinct from solver failure.
    """
    hits = np.flatnonzero((u1w[1:-1] > u1w[:-2]) & (u1w[1:-1] >= u1w[2:]))
    if len(hits) == 0:
        return float("nan")
    i = int(hits[0]) + 1
    a, b, c = float(u1w[i - 1]), float(u1w[i]), float(u1w[i + 1])
    den = a - 2.0 * b + c
    off = 0.0 if den == 0.0 else min(max(0.5 * (a - c) / den, -0.5), 0.5)
    return float(_DT * (i + off))


def _solve_raw(x, rtol: float = RTOL, atol: float = ATOL):
    """One extended log-state solve -> (pred, P, tpeak); NaNs on failure.

    Module-level and numpy/scipy only, so it is picklable and a Pool worker
    never imports anything torch touches.
    """
    th = np.exp(np.asarray(x, dtype=np.float64))
    v0 = np.log(np.asarray(lv.U0, dtype=np.float64))
    try:
        sol = solve_ivp(lv._rhs_log, (0.0, T_STAR), v0, t_eval=_T_EXT,
                        args=(th,), method="LSODA", rtol=rtol, atol=atol)
    except Exception:
        return (float("nan"), float("nan"), float("nan"))
    if (not sol.success or sol.y.shape[1] != N_T_EXT
            or not np.all(np.isfinite(sol.y))):
        return (float("nan"), float("nan"), float("nan"))
    u1 = np.exp(sol.y[0])
    u1w = u1[:_N_W]
    return (float(u1[-1]), float(u1w.max()), _first_peak(u1w))


def _solve_raw_guard(x):
    return _solve_raw(x, rtol=GUARD_RTOL, atol=GUARD_ATOL)


class SolveCache:
    """Parent-side dedupe over exact float64 node coordinates.

    Seed and control nodes are rows of the 16,384-row fit bank drawn with
    replacement, so most arm solves are cache hits; moved nodes are unique.
    """

    def __init__(self, pool: Pool) -> None:
        self.pool = pool
        self.d: dict[bytes, tuple[float, float, float]] = {}
        self.n_solved = 0

    def eval(self, X: np.ndarray) -> list[tuple[float, float, float]]:
        keys = [hashlib.sha1(np.ascontiguousarray(x).tobytes()).digest()
                for x in X]
        miss_i = [i for i, kk in enumerate(keys) if kk not in self.d]
        seen: dict[bytes, int] = {}
        todo = []
        for i in miss_i:
            if keys[i] not in seen:
                seen[keys[i]] = len(todo)
                todo.append(np.array(X[i], dtype=np.float64))
        if todo:
            got = self.pool.map(_solve_raw, todo, chunksize=64)
            self.n_solved += len(todo)
            for kk, j in seen.items():
                self.d[kk] = got[j]
        return [self.d[kk] for kk in keys]


# ------------------------------------------------------------- small utils
def _scrub(o):
    """Non-finite floats -> None, recursively (tracked JSON must be strict;
    partials keep their NaNs, which numpy needs back)."""
    if isinstance(o, dict):
        return {kk: _scrub(v) for kk, v in o.items()}
    if isinstance(o, list):
        return [_scrub(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _atomic_json(path: str, obj, scrub: bool = False) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_scrub(obj) if scrub else obj, fh, indent=1)
    os.replace(tmp, path)


def _boot_seed(tag: str) -> int:
    return BOOT_SEED ^ zlib.crc32(tag.encode())


def _stamp() -> dict:
    # Checkpoint identity by file size + mtime: a retrained map under the
    # same tag must invalidate old partials.
    ck = {}
    for label, fn in (("flow", f"flow{os.environ.get('LV_FLOW_TAG', '_newframe')}.pth"),
                      ("quadfield", f"quadfield{os.environ.get('LV_TAG', '_final')}.pth")):
        st = os.stat(os.path.join(OUT, fn))
        ck[label] = [int(st.st_size), int(st.st_mtime)]
    return {"n_rep": N_REP, "n_cal": N_CAL, "n_gold": N_GOLD,
            "m_list": list(M_LIST), "bank_L": L_BANK, "t_star": T_STAR,
            "n_t_ext": N_T_EXT, "rtol": RTOL, "atol": ATOL,
            "gold_mult": GOLD_MULT, "jitter": JITTER,
            "cal_off": CAL_OFF, "gold_off": GOLD_OFF, "ctrl_off": CTRL_OFF,
            "lv_tag": os.environ.get("LV_TAG", "_final"),
            "lv_flow_tag": os.environ.get("LV_FLOW_TAG", "_newframe"),
            "checkpoints": ck}


# ------------------------------------------------------------- model loading
def load_models():
    """The deployed flow and field, with wrong-run guards."""
    ftag = os.environ.get("LV_FLOW_TAG", "_newframe")
    fpath = os.path.join(OUT, f"flow{ftag}.pth")
    if not os.path.exists(fpath):
        raise SystemExit(
            "no trained flow at " + fpath + "; checkpoints are not "
            "distributed -- train with scripts/lv_train_flow.py and "
            "scripts/lv_quadrature_rung.py first.")
    ck = torch.load(fpath, weights_only=False)
    flow = HINTFlow(n_in=D, n_cond=ck["n_cond"], n_hidden=ck["n_hidden"],
                    n_flow_layers=ck["n_layers"]).to(torch.float64)
    flow.load_state_dict(ck["state"])
    flow.eval()
    cm, cs = ck["cond_mean"], ck["cond_std"]
    zs = np.asarray(ck["z_std"], dtype=np.float64)
    tag = os.environ.get("LV_TAG", "_final")
    qk = torch.load(os.path.join(OUT, f"quadfield{tag}.pth"),
                    weights_only=False)
    sigma = float(qk["sigma"])
    assert int(ck["n_cond"]) == 43 and int(qk["cond_dim"]) == 43, \
        "not the new-frame checkpoints (cond dim != 43)"
    assert abs(sigma - 2.5544) < 1e-3, f"wrong-run sigma {sigma}"
    net = ConditionalQuadratureAmortizer(
        d=D, hidden=qk["hidden"], n_blocks=qk["n_blocks"],
        n_heads=qk["n_heads"], delta_scale=qk["delta_scale"],
        cond_kind="vector", cond_vec_dim=qk["cond_dim"]).to(torch.float64)
    net.load_state_dict(qk["state"])
    net.eval()
    return flow, cm, cs, zs, net, sigma, 1.0 / (2.0 * sigma ** 2)


def load_members(cm, cs):
    """Conditioners + frame lift data for the eight evaluated members."""
    t = np.load(os.path.join(OUT, "test.npz"), allow_pickle=True)
    R = np.hstack([np.load(os.path.join(OUT, "test_refined.npz"))["refine"],
                   np.load(os.path.join(OUT, "test_reframed.npz"))["frame"]])
    rf = np.load(os.path.join(OUT, "test_reframed.npz"))
    cells = [int(k) for k in np.load(os.path.join(OUT, "chains.npz"))["cells"]]
    names = {k: str(t["names"][k]) for k in cells}
    cvecs = {k: (np.hstack([lv.summary(t["y"][k]), R[k]]) - cm) / cs
             for k in cells}
    return names, cells, cvecs, np.asarray(rf["m"], dtype=np.float64), \
        np.asarray(rf["R"], dtype=np.float64)


def lift(z: torch.Tensor, k: int, zs: np.ndarray, m: np.ndarray,
         Rch: np.ndarray) -> np.ndarray:
    """Flow sample space -> x = log theta: ``x = m_k + (z * zs) @ R_k.T``.

    Inverts the frame ``z' = R^{-1}(x - m)`` of ``lv_reframe.py`` after
    undoing the flow's stored ``z_std``; never the Laplace matrices.
    """
    return m[k] + (z.detach().numpy() * zs) @ Rch[k].T


def _frame_assert(bfit: torch.Tensor, zs: np.ndarray, k: int,
                  name: str) -> None:
    """Whitened-units gate on the lift: a transpose or forgotten-z_std
    error trips it by orders of magnitude. The bounds are loose because
    ``test_reframed``'s moments come from an earlier flow, so a correct
    lift still shows offsets up to ~0.15 and variances up to ~2 on the
    broadest member."""
    zb = bfit.numpy() * zs
    mu, var = np.abs(zb.mean(0)), zb.var(0)
    assert mu.max() <= 0.5 and var.min() >= 0.25 and var.max() <= 4.0, \
        f"frame gate failed on {name}: |mean|={mu}, var={var}"


# ------------------------------------------------------------- QoI algebra
def _estimates(raw: list[tuple[float, float, float]], w: np.ndarray,
               tau: float, b: float) -> dict:
    """Weighted QoI estimates over one node set, with failure accounting.

    Solver failure (all-NaN raw) poisons every QoI of this arm-repeat;
    a no-peak node (finite P, NaN tpeak) poisons tpeak only.
    """
    pred = np.array([r[0] for r in raw])
    P = np.array([r[1] for r in raw])
    tpeak = np.array([r[2] for r in raw])
    n_fail = int((~np.isfinite(P)).sum())
    n_nopeak = int((np.isfinite(P) & ~np.isfinite(tpeak)).sum())
    ex = 1.0 / (1.0 + np.exp(-(P - tau) / b))
    sharp = np.where(np.isfinite(P), (P > tau).astype(np.float64), np.nan)
    out = {"n_fail": n_fail, "n_nopeak": n_nopeak}
    for q, vals in (("pred", pred), ("exceed", ex), ("tpeak", tpeak),
                    ("exceed_sharp", sharp)):
        out[q] = float(w @ vals) if np.all(np.isfinite(vals)) else float("nan")
    return out


# ------------------------------------------------------------- member phases
def _sample_flow(flow, C1, n: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    with torch.no_grad():
        return flow.sample(n, C1.repeat(n, 1), dtype=torch.float64)


def calibrate(flow, C1, k, zs, m, Rch, cache) -> dict:
    """tau, b and the no-peak fraction from a seed-disjoint block drawn
    before the gold, so the estimand is a fixed integrand."""
    z = _sample_flow(flow, C1, N_CAL, BASE_SEED + CAL_OFF + 31 * k)
    raw = cache.eval(lift(z, k, zs, m, Rch))
    P = np.array([r[1] for r in raw])
    tp = np.array([r[2] for r in raw])
    fin = np.isfinite(P)
    tau = float(np.quantile(P[fin], 0.9))
    b = float(0.5 * P[fin].std())
    assert b > 0.0, "degenerate calibration spread"
    return {"tau": tau, "b": b, "cal_P_std": float(P[fin].std()),
            "n_fail": int((~fin).sum()),
            "nopeak_frac": float((fin & ~np.isfinite(tp)).mean())}


def gold(flow, C1, k, zs, m, Rch, cache, tau, b) -> dict:
    """Two independent halves, a 3-sigma agreement gate, and a
    solver-resolution guard at tighter tolerances."""
    halves, guard_x = [], None
    for h in (0, 1):
        z = _sample_flow(flow, C1, N_GOLD // 2,
                         BASE_SEED + GOLD_OFF + 1000 * k + h)
        X = lift(z, k, zs, m, Rch)
        if h == 0:
            guard_x = X[:N_GUARD].copy()
        raw = cache.eval(X)
        P = np.array([r[1] for r in raw])
        fin = np.isfinite(P)
        vals = {"pred": np.array([r[0] for r in raw])[fin],
                "exceed": 1.0 / (1.0 + np.exp(-(P[fin] - tau) / b)),
                "tpeak": np.array([r[2] for r in raw])[fin],
                "exceed_sharp": (P[fin] > tau).astype(np.float64)}
        vals["n_fail"] = int((~fin).sum())
        halves.append(vals)
    guard = [_solve_raw_guard(x) for x in guard_x]
    base = cache.eval(guard_x)
    cache.n_solved += len(guard_x)          # guard re-solves are real solves
    out = {"n_fail": halves[0]["n_fail"] + halves[1]["n_fail"],
           "n_nopeak": int(sum((~np.isfinite(hh["tpeak"])).sum()
                               for hh in halves))}
    for qi, q in enumerate(QOIS + ("exceed_sharp",)):
        a = halves[0][q][np.isfinite(halves[0][q])]
        c = halves[1][q][np.isfinite(halves[1][q])]
        # A censored member's tpeak can leave a half with 0 or 1 finite
        # values, and a NaN z must not abort a member whose solves are
        # already spent.
        if min(len(a), len(c)) < 2:
            out[q] = None
            continue
        pooled = np.concatenate([a, c])
        se = float(pooled.std(ddof=1) / np.sqrt(len(pooled)))
        zdis = abs(a.mean() - c.mean()) / max(
            np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c)), 1e-300)
        if q in QOIS:
            assert zdis < 3.0, \
                f"gold halves disagree at {zdis:.1f} sigma ({q})"
        gr = np.array([(_estimates([g], np.ones(1), tau, b)[q]
                        - _estimates([bb], np.ones(1), tau, b)[q])
                       for g, bb in zip(guard, base)])
        dmax = float(np.nanmax(np.abs(gr))) if np.any(np.isfinite(gr)) else 0.0
        sd = float(pooled.std(ddof=1))
        out[q] = {"mean": float(pooled.mean()), "sd": sd,
                  "half_width": 1.96 * se, "halves_z": float(zdis),
                  "kurtosis": float(sps.kurtosis(pooled)),
                  "n": len(pooled), "guard_max_df": dmax,
                  "below_solver_resolution": bool(sd < 1e3 * dmax)}
    return out


def _mu(z, bank, i2):
    outv = torch.zeros(len(z), dtype=torch.float64)
    for kk in range(0, len(bank), 8192):
        outv += torch.exp(
            -(torch.cdist(z, bank[kk:kk + 8192]) ** 2) * i2).sum(-1)
    return outv / len(bank)


def _fit_crit(z, w, bank, i2):
    G = torch.exp(-(torch.cdist(z, z) ** 2) * i2)
    return float(w @ G @ w - 2.0 * w @ _mu(z, bank, i2))


def score_member(flow, net, C1, k, name, zs, m, Rch, sigma, i2) -> dict:
    """Everything for one member: cal, gold, all (M, r) cells, plugin."""
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        cache = SolveCache(pool)
        # The fit bank of lv_evaluate2.py, bit-identical: bfit is the
        # first draw after this seed.
        torch.manual_seed(int(BASE_SEED + 31 * k))
        with torch.no_grad():
            bfit = flow.sample(L_BANK, C1.repeat(L_BANK, 1),
                               dtype=torch.float64)
        _frame_assert(bfit, zs, k, name)
        cal = calibrate(flow, C1, k, zs, m, Rch, cache)
        tau, b = cal["tau"], cal["b"]
        gd = gold(flow, C1, k, zs, m, Rch, cache, tau, b)
        print(f"  [{name}] cal tau={tau:.4g} b={b:.4g} "
              f"nopeak={cal['nopeak_frac']:.4f}; gold done "
              f"({cache.n_solved} solves, {time.time() - t0:.0f} s)",
              flush=True)

        raw: dict = {}
        sel: dict = {}
        for M in M_LIST:
            cell = {a: {q: [] for q in QOIS + ("exceed_sharp",)}
                    for a in ARMS}
            cell["counts"] = {a: {"n_fail": 0, "n_nopeak": 0} for a in ARMS}
            fit_vals: list[list[float]] = []
            picks = {"iid": 0, "reweight": 0, "moved": 0}
            pick_seq: list[int] = []
            l1s = []
            for r in range(N_REP):
                g = torch.Generator().manual_seed(
                    int(BASE_SEED + 977 * k + 13 * M + r))
                z0 = bfit[torch.randint(L_BANK, (M,), generator=g)]
                w0 = torch.full((M,), 1.0 / M, dtype=torch.float64)
                wr = constrained_weights_batched(
                    z0[None], _mu(z0, bfit, i2)[None], sigma, JITTER)[0]
                with torch.no_grad():
                    ze = net(z0[None], C1)[0]
                we = constrained_weights_batched(
                    ze[None], _mu(ze, bfit, i2)[None], sigma, JITTER)[0]
                cand = [(_fit_crit(z0, w0, bfit, i2), z0, w0),
                        (_fit_crit(z0, wr, bfit, i2), z0, wr),
                        (_fit_crit(ze, we, bfit, i2), ze, we)]
                bi = int(np.argmin([e[0] for e in cand]))
                picks[("iid", "reweight", "moved")[bi]] += 1
                pick_seq.append(bi)
                g2 = torch.Generator().manual_seed(
                    int(BASE_SEED + CTRL_OFF + 977 * k + 13 * M + r))
                z2 = bfit[torch.randint(L_BANK, (2 * M,), generator=g2)]
                w2 = torch.full((2 * M,), 0.5 / M, dtype=torch.float64)

                raw0 = cache.eval(lift(z0, k, zs, m, Rch))
                rawe = (raw0 if bi < 2
                        else cache.eval(lift(cand[bi][1], k, zs, m, Rch)))
                raw2 = cache.eval(lift(z2, k, zs, m, Rch))
                ests = {
                    "iid": _estimates(raw0, w0.numpy(), tau, b),
                    "reweight": _estimates(raw0, wr.numpy(), tau, b),
                    "emission": _estimates(
                        rawe, cand[bi][2].numpy(), tau, b),
                    "iid2m": _estimates(raw2, w2.numpy(), tau, b),
                }
                for a in ARMS:
                    for q in QOIS + ("exceed_sharp",):
                        cell[a][q].append(ests[a][q])
                    cell["counts"][a]["n_fail"] += ests[a]["n_fail"]
                    cell["counts"][a]["n_nopeak"] += ests[a]["n_nopeak"]
                fit_vals.append([float(e[0]) for e in cand])
                l1s.append(float(cand[bi][2].abs().sum()))
            cell["picks"] = picks
            cell["pick_seq"] = pick_seq
            cell["fit_vals"] = fit_vals
            cell["l1_emission"] = l1s
            raw[str(M)] = cell
            sel[str(M)] = picks
            print(f"  [{name}] M={M:<4} picks {picks}  "
                  f"({cache.n_solved} solves)", flush=True)

        plugin_raw = cache.eval(m[k][None])
        plugin = _estimates(plugin_raw, np.ones(1), tau, b)
        n_solves = cache.n_solved
    return {"name": name, "k": k, "stamp": _stamp(), "cal": cal, "gold": gd,
            "raw": raw, "selection": sel, "plugin": plugin,
            "n_solves": n_solves, "wall_s": time.time() - t0}


# ------------------------------------------------------------- distillation
def _cell_stats(parts: list[dict], M: str, q: str) -> dict:
    """One (QoI, M) cell pooled over members, in gold-sd units, with the
    tpeak member censoring applied."""
    use = [p for p in parts
           if p["gold"].get(q)
           and not p["gold"][q]["below_solver_resolution"]
           and not (q == "tpeak" and (p["cal"]["nopeak_frac"] > 0
                                      or p["gold"]["n_nopeak"] > 0))]
    if not use:
        return {"members_used": 0, "gold_resolved": False}
    err_n = {a: [] for a in ARMS}
    err_r = {a: [] for a in ARMS}
    per_m_mlr, per_m_ties, per_m_res, n_dropped = [], [], [], 0
    for p in use:
        gmean, gsd = p["gold"][q]["mean"], max(p["gold"][q]["sd"], 1e-300)
        ests = {a: np.array(p["raw"][M][a][q]) for a in ARMS}
        finite = np.all([np.isfinite(ests[a]) for a in ARMS], axis=0)
        n_dropped += int((~finite).sum())
        for a in ARMS:
            e = np.abs(ests[a][finite] - gmean)
            err_r[a].append(e)
            err_n[a].append(e / gsd)
        aa = err_n["emission"][-1]
        bb = err_n["iid"][-1]
        # Per-member resolution flag.
        sep_m = (abs(float(np.median(bb)) - float(np.median(aa)))
                 if len(aa) else 0.0)
        per_m_res.append(
            bool(sep_m > GOLD_MULT * 1.96 / np.sqrt(p["gold"][q]["n"])))
        ok = (aa > 0) & (bb > 0) & (aa != bb)
        per_m_ties.append(int((aa == bb).sum()))
        per_m_mlr.append(
            float(np.median(np.log(aa[ok]) - np.log(bb[ok])))
            if ok.sum() >= 8 else None)
    cat = {a: np.concatenate(err_n[a]) for a in ARMS}
    cat_r = {a: np.concatenate(err_r[a]) for a in ARMS}
    n_eff = min(p["gold"][q]["n"] for p in use)
    hw_n = 1.96 / np.sqrt(n_eff)
    sep = abs(float(np.median(cat["iid"]))
              - float(np.median(cat["emission"])))
    resolved = bool(sep > GOLD_MULT * hw_n)
    pcs = {
        "emission_vs_iid": _paired_cell(
            torch.tensor(cat["emission"]), torch.tensor(cat["iid"]),
            0.0, f"lvd:{q}:{M}:ei"),
        "reweight_vs_iid": _paired_cell(
            torch.tensor(cat["reweight"]), torch.tensor(cat["iid"]),
            0.0, f"lvd:{q}:{M}:ri"),
        "control_2m_vs_iid": _paired_cell(
            torch.tensor(cat["iid2m"]), torch.tensor(cat["iid"]),
            0.0, f"lvd:{q}:{M}:c2"),
    }
    # sqrt(2) target with the finite-bank adjustment; the exact per-arm
    # MSE is sigma^2 (1/M + 1/L - 1/(ML)) and the -1/(ML) terms are
    # dropped (< 1e-4 nats at the top budget).
    Mi = int(M)
    tgt = -0.5 * float(np.log((1.0 / Mi + 1.0 / L_BANK)
                              / (0.5 / Mi + 1.0 / L_BANK)))
    ci = pcs["control_2m_vs_iid"]["log_ratio_ci"]
    pcs["control_2m_vs_iid"]["sqrt2_target"] = tgt
    pcs["control_2m_vs_iid"]["sqrt2_ok"] = (
        bool(ci[0] <= tgt <= ci[1]) if ci is not None else None)
    mult = {}
    for a in ("emission", "reweight"):
        na, nb = cat[a], cat["iid"]
        usable = (na > 0) & (nb > 0) & (na != nb)
        if usable.sum() >= 32:
            m_eq = Mi * (nb[usable] / na[usable]) ** 2
            rng = np.random.default_rng(_boot_seed(f"lvd:mult:{q}:{M}:{a}"))
            meds = np.median(rng.choice(
                m_eq, size=(BOOT_B, len(m_eq)), replace=True), axis=1)
            mult[a] = {"median_m_eq": float(np.median(m_eq)),
                       "ci": [float(np.quantile(meds, ALPHA / 2)),
                              float(np.quantile(meds, 1 - ALPHA / 2))],
                       "n_used": int(usable.sum())}
        else:
            mult[a] = None
    signs = {np.sign(v) for v in per_m_mlr if v is not None}
    return {
        "members_used": len(use),
        "member_names": [p["name"] for p in use],
        "gold_half_width_norm": hw_n, "separation_norm": sep,
        "gold_resolved": resolved,
        "median_abs_err_norm": {a: float(np.median(cat[a])) for a in cat},
        "median_abs_err_raw": {a: float(np.median(cat_r[a])) for a in cat_r},
        "rmse_norm": {a: float(np.sqrt(np.mean(cat[a] ** 2))) for a in cat},
        "paired_multiplier": mult, **pcs,
        "per_member_mlr": per_m_mlr, "per_member_ties": per_m_ties,
        "per_member_resolved": per_m_res,
        "n_dropped_pairs": n_dropped,
        "member_sign_consistent": (len(signs) == 1 if all(
            v is not None for v in per_m_mlr) else None),
    }


def _multiplier(cells: dict) -> dict:
    """Gated iid-equivalent multiplier per QoI."""
    out = {}
    for q in QOIS:
        ms = [M for M in map(str, M_LIST) if cells[q][M]["members_used"]]
        if len(ms) < 2:
            out[q] = None
            continue
        rmse = [cells[q][M]["rmse_norm"]["iid"] for M in ms]
        slope = float(np.polyfit(np.log([int(M) for M in ms]),
                                 np.log(rmse), 1)[0])
        ok_slope = bool(abs(slope + 0.5) < SLOPE_TOL)
        per = {}
        for M in ms:
            c = cells[q][M]
            if not (ok_slope and c["gold_resolved"]):
                per[M] = None
                continue
            per[M] = {a: (None if c["paired_multiplier"][a] is None else {
                **c["paired_multiplier"][a],
                "extrapolated": bool(
                    c["paired_multiplier"][a]["median_m_eq"] > max(M_LIST)),
            }) for a in ("emission", "reweight")}
        out[q] = {"iid_slope": slope, "slope_ok": ok_slope, "per_M": per}
    return out


def _paper_crosscheck(parts: list[dict]) -> dict | None:
    """The first twelve repeats reuse ``lv_evaluate2.py``'s exact seeds,
    so their selection frequencies must equal the shipped ledger's.
    Warn-only, and skipped when the ledger is absent."""
    path = os.path.join(OUT, "evaluation2_final.npz")
    if not os.path.exists(path) or N_REP < 12:
        return None
    d = np.load(path, allow_pickle=True)
    out = {}
    for p in parts:
        for M in map(str, M_LIST):
            seq = p["raw"][M].get("pick_seq")
            rows = np.flatnonzero((d["names"] == p["name"])
                                  & (d["M"] == int(M)))
            if seq is None or len(rows) != 1:
                continue
            ours = float(np.mean(np.array(seq[:12]) == 2))
            theirs = float(d["pick_moved"][rows[0]])
            key = f"{p['name']}:{M}"
            out[key] = {"ours_first12_moved": ours, "paper_moved": theirs,
                        "match": bool(abs(ours - theirs) < 1e-12)}
            if not out[key]["match"]:
                print(f"  [warn] selection-ledger mismatch {key}: "
                      f"ours {ours:.4f} vs paper {theirs:.4f}", flush=True)
    return out


def distill(parts: list[dict]) -> dict:
    cells = {q: {str(M): _cell_stats(parts, str(M), q) for M in M_LIST}
             for q in QOIS}
    sidecar = {str(M): _cell_stats(parts, str(M), "exceed_sharp")
               for M in M_LIST}
    holm_p = {}
    for q in QOIS:
        for M in map(str, M_LIST):
            c = cells[q][M]
            holm_p[f"{q}:{M}"] = (
                c["emission_vs_iid"]["sign_p"]
                if c.get("gold_resolved") else 1.0)
    plugin = {p["name"]: {q: (None if p["gold"].get(q) is None else
                              abs(p["plugin"][q] - p["gold"][q]["mean"])
                              / max(p["gold"][q]["sd"], 1e-300))
                          for q in QOIS} for p in parts}
    return {
        "what": ("LV downstream integrand: the same posterior expectation "
                 "UNDER THE REFERENCE FLOW for fewer ODE solves. The gold "
                 "is the flow's own measure (rem:integrate-rho); accuracy "
                 "is priced in solves, wall-clock recorded but never the "
                 "claim. Primary (registered): emission vs iid on 'exceed'. "
                 "The plugin arm evaluates the FRAME MEAN m_k (the old "
                 "flow's moment center), not the reference mean."),
        "design": "docs/DESIGN_lv_downstream.md (plan v3; Gate-2 verdict "
                  "and binding hedges in its final section)",
        "renderer": "scripts/lv_downstream.py",
        "stamp": _stamp(),
        "lift": "x = m_k + (z * z_std) @ R_k.T  (test_reframed.npz m, R)",
        "qois": {"pred": f"prey population at t*={T_STAR}",
                 "exceed": "logistic((P - tau_k)/b_k), P = windowed prey "
                           "peak; PRIMARY",
                 "tpeak": "time of first interior prey peak, sub-grid "
                          "refined; members with any no-peak trajectory "
                          "censored"},
        "caveats": [
            "gold = the reference flow's own measure, not the posterior",
            "sqrt2 target drops the -1/(ML) bank terms (<1e-4 nats)",
            "bank offset shared within a member: pooled control CI "
            "slightly understates uncertainty (1/L <= 3% of 1/M)",
            "emission deployed costs M solves TOTAL; its multiplier is a "
            "counterfactual iid cost at equal M",
            "'fewer solves' binds to the reweight arm",
            "extending the paper's seed formula to 64 repeats creates 277 "
            "exact cross-member/cross-M seed collisions (none within any "
            "Holm cell; the paper's r<12 subset is collision-free); the "
            "coupling is index-pattern only and Holm is dependence-robust",
            "GATE-2 (maths): the M=32 reweight multiplier holds on seven "
            "of eight members; on family_q97 reweighting is below par "
            "(m_eq ~ 11); a member-cluster bootstrap widens the CI to "
            "about [149, 576]",
            "GATE-2 (maths): 'more accurate in all fifteen cells' is a "
            "paired-median/win-fraction statement, never RMSE (pred at "
            "M>=256 has emission RMSE above iid via family_q97's tail, "
            "gold kurtosis 33); member-uniform direction holds in 9 of 15 "
            "cells (exceed 32-256, tpeak all five)",
            "GATE-2 (maths): tpeak emission medians sit at the gold's own "
            "CLT resolution (1.96/sqrt(131072)), so tpeak multipliers are "
            "conservative FLOORS and ship as 'at least'",
        ],
        "cells": cells, "sidecar_exceed_sharp": sidecar,
        "multiplier": _multiplier(cells),
        "paper_selection_crosscheck": _paper_crosscheck(parts),
        "holm": {"family": holm_p, "reject": _holm(holm_p)},
        "selection_freq": {p["name"]: p["selection"] for p in parts},
        "plugin_err_norm": plugin,
        "gold": {p["name"]: p["gold"] for p in parts},
        "cal": {p["name"]: p["cal"] for p in parts},
        "l1_emission_median": {
            p["name"]: {str(M): float(np.median(p["raw"][str(M)]
                                                ["l1_emission"]))
                        for M in M_LIST} for p in parts},
        "n_solves_total": int(sum(p["n_solves"] for p in parts)),
        "wall_s_total": float(sum(p["wall_s"] for p in parts)),
    }


# ------------------------------------------------------------------- main
def main(phase: str) -> None:
    t0 = time.time()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DIAG, exist_ok=True)
    otag = os.environ.get("LV_OUT_TAG", os.environ.get("LV_TAG", "_final"))
    flow, cm, cs, zs, net, sigma, i2 = load_models()
    names, cells, cvecs, m, Rch = load_members(cm, cs)
    print(f"[setup] sigma {sigma:.4f}, members {[names[k] for k in cells]}, "
          f"N_REP={N_REP} N_GOLD={N_GOLD} workers={N_WORKERS}", flush=True)

    parts = []
    for k in cells:
        ppath = os.path.join(LOG_DIR, f"partial_member{k}{otag}.json")
        if os.path.exists(ppath):
            with open(ppath) as fh:
                p = json.load(fh)
            if p.get("stamp") == _stamp():
                print(f"  [{names[k]}] partial exists, stamp matches -- skip",
                      flush=True)
                parts.append(p)
                continue
            print(f"  [{names[k]}] partial STALE (stamp mismatch) -- redo",
                  flush=True)
        if phase == "distill":
            raise SystemExit(f"missing/stale partial for member {k}: {ppath}")
        C1 = torch.tensor(cvecs[k][None], dtype=torch.float64)
        p = score_member(flow, net, C1, k, names[k], zs, m, Rch, sigma, i2)
        _atomic_json(ppath, p)
        parts.append(p)

    rec = distill(parts)
    est = np.full((len(ARMS), len(QOIS) + 1, len(parts), len(M_LIST), N_REP),
                  np.nan)
    for pi, p in enumerate(parts):
        for ai, a in enumerate(ARMS):
            for qi, q in enumerate(QOIS + ("exceed_sharp",)):
                for mi, M in enumerate(M_LIST):
                    est[ai, qi, pi, mi] = p["raw"][str(M)][a][q]
    np.savez(os.path.join(OUT, f"downstream{otag}.npz"), est=est,
             arms=np.array(ARMS), qois=np.array(QOIS + ("exceed_sharp",)),
             members=np.array([p["name"] for p in parts]),
             m_list=np.array(M_LIST))
    # The unsuffixed name is reserved for the deployed configuration; a
    # reduced or variant run gets a tag-suffixed sidecar instead.
    jname = ("lv_downstream.json" if otag == "_final"
             else f"lv_downstream{otag}.json")
    _atomic_json(os.path.join(DIAG, jname), rec, scrub=True)
    print(f"[done] {rec['n_solves_total']} solves, "
          f"{time.time() - t0:.0f} s -> data/records/{jname}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=("all", "distill"))
    main(ap.parse_args().phase)
