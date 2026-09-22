"""Certification sweep for the selected emission on the Darcy problem.

The never-worse guarantee in the sampled-reference regime says that, against a
certification split independent of everything used to fit the arms, the
emission is never worse than the seeds it was handed, up to a computable
per-emission slack, on an explicit high-probability event. This script
measures whether that certificate is INFORMATIVE, by reporting it as a CURVE
in the certification budget. It is EVALUATION ONLY: the trained network is
resolved by config identity and nothing is retrained.

The question it answers: at what certification budget does the slack fall
below the arm gap it has to resolve, and how does that budget move with the
node budget ``M``?

The setting is ``scripts/dq_darcy_stuart.py``: ``rho_y`` is the trained
conditional flow of ``darcy_stuart_npe.py`` for the
Beskos--Girolami--Lan--Farrell--Stuart (2017, sec. 4.2) groundwater problem,
ambient ``d = 100``, held-out observations that are fresh simulator samples
(new to both networks).

THE PROTOCOL.

  * **The arms are the training run's arms.** Per ``(M, observation)`` the
    estimand bank size, the generator schedule, the with-replacement seed
    selection and the dedupe-with-multiplicity are reproduced from
    ``dq_darcy_stuart._evaluate``. Bitwise identity is not claimed: that run
    may have sampled the flow in float32 on the GPU where this samples on the
    CPU, which perturbs the banks below any reported digit.
    :meth:`_verify_arms` puts the rebuilt bank-A depths against the saved
    quartiles at every budget and arm.
  * **The certification split is SAMPLED, not carved.** The reference is a
    trained flow, so fresh samples from it are available at will and the split
    is taken at a data-independent seed offset, independent of every bank used
    to fit or score the arms. Carving it out of the estimand bank instead
    would starve the arms and price an artifact of the bookkeeping rather than
    a property of the certificate.
  * **The split budget is swept**, ``L2`` from a thousand to a million, as
    NESTED PREFIXES of one stream: the certificate at every smaller budget is
    a checkpoint of the same stream, so the curve costs one pass.
  * **All three menu members are scored live on the split** and the emission is
    the argmin there, each pairwise slack paid at ``delta/3``. The DEPLOYED
    selection (argmin on the fitting estimand, which is what the training run
    emits) is certified against its seeds beside it: both selections are
    functions of the estimand bank, the observation and the network alone, so
    both are independent of the split.
  * **Verdicts are DIRECTIONAL** -- CERTIFIED-BETTER / CERTIFIED-WORSE /
    WITHIN-SLACK -- with a COINCIDENCE guard at ``D < 1e-4``. Rates count LIVE
    (non-coincident) cells only and every denominator is printed: a sign read
    off round-off is not a sign.

Because the split is sampled rather than carved, certification costs no
precision: it costs reference samples per observation, plus ``O(M L2)`` kernel
evaluations to contract them.

One hypothesis is tighter than it looks: the guarantee prices ONE emission
against ONE split, while this sweep reuses each observation's split across
every redraw and every budget. Each cell's verdict is therefore a valid
marginal ``1 - delta`` statement, and the reported rate is an average of
marginal certificates, NOT a simultaneous statement over the cells that share
a split. The Bonferroni-corrected slack that WOULD make it simultaneous over
one split's whole service is computed and reported beside it.

Run (CPU only):
    nice -n 15 python scripts/dq_darcy_certification.py
    nice -n 15 python scripts/dq_darcy_certification.py --smoke
Per-observation sidecars make it resumable: re-running picks up where it died.
"""

from __future__ import annotations
from projorg import configsdir, datadir  # noqa: E402

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from projorg.config import read_config  # noqa: E402
from tqdm import tqdm  # noqa: E402

from _ckpt import checkpoint_path, experiment_name  # noqa: E402
from _darcy_npe import NPE_ROOT, load_npe, sample_batched, standardize  # noqa: E402

from qfield.dataset.darcy_stuart_kl_prior import StuartKLPrior  # noqa: E402
from qfield.dataset.darcy_stuart_op import (  # noqa: E402
    DarcyForward,
    beskos_sensors,
)
from qfield.designed_quadrature import (  # noqa: E402
    COINCIDENCE_D,
    constrained_weights,
    gram,
    pair_scale,
    pairwise_slack,
    sampled_bank_target,
    select_emission,
    selection_verdict,
    slack_value,
)
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)

torch.set_num_threads(4)
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

CONFIG_FILE = "dq_darcy_stuart.json"
# ``dq_darcy_stuart.py`` extends the shared ignore list with these render-only
# knobs; mirror it or the rebuilt experiment name does not resolve.
IGNORE_EXTRA = ("recon_M", "n_recon", "eval_repeats", "n_pcn_audit")

OUT = os.path.join(datadir("records"), "dq_darcy_certification.json")
SIDE_DIR = datadir("certsel_darcy")

# Family confidence budget over the three pairwise events.
DELTA = 0.05
# The certification stream is seeded from THIS offset and the observation index
# alone: no atom of any estimand bank, no arm, no network parameter reaches it.
CERT_SEED_OFFSET = 610_000
ARM_NAMES = ("seeds", "reweight", "move")
PAIRS = ((0, 1), (0, 2), (1, 2))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _atomic_json(obj, path: str) -> None:
    """Write JSON via tmp + ``os.replace``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _med_band(vals) -> tuple[float, float, float]:
    """Median and the 10-90% band."""
    t = torch.as_tensor(list(vals), dtype=DTYPE)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    q = torch.quantile(t, torch.tensor([0.10, 0.5, 0.90], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct at rates of exactly 0 or 1.

    A normal-approximation interval collapses to a point there, which is where
    most of these cells land: it would report a rate of 0.000 with zero width
    and make a vacuous certificate look precise.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)


# ---------------------------------------------------------------------------
class DarcyCertification:
    """One certification split per observation, swept in ``L2``."""

    def __init__(self, args):
        self.args = args
        cfg = read_config(configsdir(CONFIG_FILE))
        self.cfg = cfg
        self.d, self.dy = int(cfg["d"]), int(cfg["dy"])
        self.jitter = float(cfg["jitter"])
        self.base_seed = int(cfg["base_seed"])
        self.bank_L = int(cfg["bank_L"])
        self.m_list = [int(v) for v in str(cfg["m_list"]).split(",")]
        self.grid = [int(v) for v in str(args.l2_grid).split(",")]
        if sorted(self.grid) != self.grid or self.grid[0] < 2:
            raise ValueError(f"--l2_grid must be increasing and >= 2: "
                             f"{self.grid}")

        # ---- the trained run, resolved by config identity ------------------
        self.exp = experiment_name(CONFIG_FILE, ignore_extra=IGNORE_EXTRA)
        ck = torch.load(
            checkpoint_path(CONFIG_FILE, ignore_extra=IGNORE_EXTRA),
            weights_only=False, map_location="cpu")
        res = ck["results"]
        # THE BANDWIDTH IS READ, NOT RECOMPUTED. It is the trained run's
        # pooled median heuristic over the TRAINING observations, a functional
        # of data disjoint from everything scored here, which is what the
        # independence hypothesis needs. Recomputing it from anything scored
        # here would put the split inside the ruler.
        self.sigma = float(res["sigma"])
        self.inv2s = 1.0 / (2.0 * self.sigma**2)
        assert int(res["bank_L"]) == self.bank_L
        assert list(res["m_list"]) == self.m_list
        # The deployed depths, so the record can say what the slack is being
        # asked to resolve without a second run. Each saved row carries
        # [median, q25, q75] across observations; the band is kept because the
        # arm check below tests band containment.
        self.deployed = {int(r["M"]): {
            "floor_bankA": r["floor"], "reweight_bankA": r["reweight"],
            "ours_bankA": r["ours"], "move_raw_bankA": r["ours_move_raw"],
            "floor_heldout": r["floor_heldout"][0],
            "ours_heldout": r["ours_heldout"][0],
            "activation": r["activation"]} for r in res["rows"]}
        print(f"[ckpt] {self.exp}\n[ckpt] sigma={self.sigma:.6f} pinned by the "
              f"trained run | bank_L={self.bank_L}", flush=True)

        # ---- the flow (the reference) --------------------------------------
        self.npe, self.y_mean, self.y_std, self.npe_cfg = load_npe(
            os.path.join(NPE_ROOT, str(cfg["npe_run"])), "cpu")
        self.npe = self.npe.float()
        assert self.npe.n_in == self.d and self.npe.n_cond == self.dy

        # ---- the held-out observations, reproduced from the same stream ----
        with h5py.File(self.npe_cfg["bank"], "r") as f:
            self.attrs = {k: f.attrs[k] for k in f.attrs}
        kl = StuartKLPrior(int(self.attrs["N"]), K=int(self.attrs["K"]),
                           alpha=float(self.attrs["alpha"]),
                           s=float(self.attrs["s"]),
                           sigma=float(self.attrs["sigma"]))
        fwd = DarcyForward(int(self.attrs["N"]))
        sensors, _ = beskos_sensors(int(self.attrs["N"]),
                                    int(self.attrs["n_sensors"]))
        n_ev_cfg = int(cfg["n_eval_obs"])
        rng = np.random.default_rng(self.base_seed + 777_000)
        xi_ev = rng.standard_normal((n_ev_cfg, self.d))
        y_ev = np.stack([
            fwd.observe(fwd.solve(kl.reconstruct(x)), sensors) for x in xi_ev
        ]) + float(self.attrs["sigma_y"]) * rng.standard_normal(
            (n_ev_cfg, int(self.attrs["n_sensors"])))
        self.y_ev = standardize(
            torch.as_tensor(y_ev, dtype=torch.float32), self.y_mean,
            self.y_std)
        self.y_cpu = self.y_ev.double()
        self.n_obs = min(int(args.n_obs), n_ev_cfg)
        if int(args.redraws) > int(cfg["eval_repeats"]):
            raise ValueError(
                f"--redraws {args.redraws} exceeds the rung's eval_repeats "
                f"({cfg['eval_repeats']}): beyond it these are no longer the "
                "emissions the paper reports.")
        print(f"[obs] {self.n_obs} of {n_ev_cfg} held-out observations x "
              f"{args.redraws} redraws (a prefix of the rung's own "
              f"{cfg['eval_repeats']})", flush=True)

        # ---- the trained amortizer -----------------------------------------
        self.net = self._build_net()
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()

    def _build_net(self) -> ConditionalQuadratureAmortizer:
        c = self.cfg
        return ConditionalQuadratureAmortizer(
            d=self.d, hidden=int(c["hidden"]), n_blocks=int(c["n_blocks"]),
            n_heads=int(c["n_heads"]), cond_hidden=int(c["cond_hidden"]),
            delta_scale=float(c["delta_scale"]),
            cond_token_dim=int(c["cond_token_dim"]),
            cond_hidden_ch=int(c["cond_hidden_ch"]),
            cond_kind="vector", cond_vec_dim=self.dy,
            cond_n_tokens=int(c["cond_n_tokens"]),
        ).to(DTYPE)

    def _draw(self, o: int, n: int, gen: torch.Generator) -> torch.Tensor:
        """``n`` i.i.d. samples of ``rho_y`` at observation ``o``, float64 CPU.

        Consecutive calls on one generator concatenate into the same stream a
        single call of the summed size would produce, which is what makes the
        ``L2`` grid genuine nested prefixes.
        """
        return sample_batched(
            self.npe, self.y_ev[o : o + 1], int(n), chunk_obs=1,
            generator=gen).double()[0]

    # ---- the arms, reproduced from the training run ------------------------
    def _armsets(self, o: int) -> list[dict]:
        """Every ``(M, redraw)`` arm-set at observation ``o``.

        The generator schedule, bank size, with-replacement multinomial and
        dedupe-with-multiplicity are reproduced from
        ``dq_darcy_stuart._evaluate``. The estimand bank is released as soon
        as the arms are fixed: only the nodes, the weights, the Gram
        quadratics and the pair scales are needed downstream, and holding five
        16k-atom banks at once is what would make this run memory-bound.
        """
        out = []
        for M in self.m_list:
            # The bank is raised at the deep budgets; mirror that exactly.
            L = self.bank_L if int(M) < 256 else 2 * self.bank_L
            gA = torch.Generator().manual_seed(
                self.base_seed + 50_000 + 1000 * int(M) + o)
            bankA = self._draw(o, L, gA)
            tgt = sampled_bank_target(bankA, self.sigma, name="darcy_npe",
                                      dtype=DTYPE, replace=True)
            for k in range(int(self.args.redraws)):
                idx = torch.multinomial(
                    torch.ones(L, dtype=DTYPE), int(M), replacement=True,
                    generator=gA)
                uniq, counts = torch.unique(idx, return_counts=True)
                zu = bankA[uniq].contiguous()
                w_eq = counts.to(DTYPE) / float(M)
                mu_u = tgt.mu_fn(zu)
                w_rw = constrained_weights(zu, mu_u, self.sigma, self.jitter)
                with torch.no_grad():
                    z_n = self.net(zu.unsqueeze(0),
                                   self.y_cpu[o].unsqueeze(0))[0].contiguous()
                mu_n = tgt.mu_fn(z_n)
                w_mv = constrained_weights(z_n, mu_n, self.sigma, self.jitter)
                # The DEPLOYED emission: argmin on the fitting estimand.
                # fitting_criterion=True keeps the runtime assertion live.
                sel_a = select_emission(
                    [(zu, w_eq), (zu, w_rw), (z_n, w_mv)], tgt.mu_fn,
                    self.sigma, names=ARM_NAMES, fitting_criterion=True)
                nodes = (zu, zu, z_n)
                wts = (w_eq, w_rw, w_mv)
                Ku, Kn = gram(zu, self.sigma), gram(z_n, self.sigma)
                quad = [float(w_eq @ Ku @ w_eq), float(w_rw @ Ku @ w_rw),
                        float(w_mv @ Kn @ w_mv)]
                out.append({
                    "M": int(M), "k": int(k), "zu": zu, "z_n": z_n,
                    # rows of the two weight blocks, so one cross-Gram per
                    # DISTINCT node set serves every arm on it
                    "W_u": torch.stack([w_eq, w_rw]), "W_n": w_mv.unsqueeze(0),
                    "quad": quad,
                    "D": [pair_scale(nodes[i], wts[i], nodes[j], wts[j],
                                     self.sigma) for i, j in PAIRS],
                    "idx_A": int(sel_a["index"]),
                    "J_A": [float(v) for v in sel_a["values"]],
                    "S": torch.zeros(3, dtype=DTYPE),
                    "Q": torch.zeros(3, 3, dtype=DTYPE),
                    "off": None,
                    # MMD^2 against the fitting bank, which _verify_arms
                    # compares against the training run's reported depths.
                    "mmd2_A": [float(v) + float(tgt.c_rho)
                               for v in sel_a["values"]],
                    "sel_A_mmd2": float(sel_a["values"][sel_a["index"]])
                    + float(tgt.c_rho),
                })
            del bankA, tgt
        return out

    # ---- are these the training run's arms? -------------------------------
    def _verify_arms(self) -> dict:
        """Check the rebuilt arms against the depths the training run reports.

        The generator schedule, bank size, seed selection and dedupe are
        reproduced from ``dq_darcy_stuart._evaluate``, so the bank-A depths of
        floor, reweight and emitted arm must land where that run put them.

        EXACT identity is not claimable: the saved ``results.json`` stores
        only quartiles across observations, and its flow sampling may have run
        in float32 on the GPU while this runs on the CPU, which perturbs the
        banks below any reported digit. What is checkable is that the median
        over the observations used here falls inside the run's own
        interquartile band, at every budget and every arm.
        """
        n_v = min(int(self.args.n_verify), self.n_obs)
        if n_v <= 0:
            return {}
        # arm index -> the key that arm's bank-A depth is stored under
        SHIPPED = {0: "floor_bankA", 1: "reweight_bankA",
                   2: "move_raw_bankA", 3: "ours_bankA"}
        per = {M: {i: [] for i in SHIPPED} for M in self.m_list}
        for o in range(n_v):
            sets = self._armsets(o)
            for M in self.m_list:
                mine = [a for a in sets if a["M"] == M]
                for i in range(3):
                    per[M][i].append(float(np.mean(
                        [a["mmd2_A"][i] for a in mine])))
                per[M][3].append(float(np.mean(
                    [a["sel_A_mmd2"] for a in mine])))
        out, in_band = {}, True
        for M in self.m_list:
            ship, row = self.deployed[int(M)], {}
            for i, key in SHIPPED.items():
                mine = float(np.median(per[M][i]))
                med, q25, q75 = ship[key]
                hit = bool(q25 <= mine <= q75)
                in_band &= hit
                row[key.replace("_bankA", "")] = {
                    "median_here": mine, "shipped_median": med,
                    "shipped_iqr": [q25, q75], "ratio": mine / med,
                    "inside_shipped_iqr": hit}
            out[str(M)] = row
        print("  [arms] rebuilt on %d observations vs the shipped run: " % n_v
              + " ".join(f"M={M} ours {out[str(M)]['ours']['ratio']:.3f}x"
                         for M in self.m_list)
              + f" | all inside the shipped IQR: {in_band}", flush=True)
        return {"n_obs_checked": n_v, "by_M": out,
                "all_inside_shipped_iqr": in_band,
                "note": (
                    "the median over this run's observation subsample against "
                    "the shipped run's 64-observation median and "
                    "interquartile band. The shipped record stores no "
                    "per-observation values and its flow sampling may have "
                    "run in float32 on the GPU, so band containment is the "
                    "strongest statement available -- not a bitwise identity "
                    "claim, which the record cannot support.")}

    # ---- the split, streamed, with nested checkpoints ----------------------
    def _stream(self, o: int, sets: list[dict]) -> list[dict]:
        """Take the split and certify every arm-set at every ``L2`` checkpoint.

        Nothing of size ``L2`` is materialized: the certificate depends on the
        split only through, per arm-set, the three witness sums ``S`` and their
        ``3 x 3`` second-moment matrix ``Q``, so the stream is consumed in
        blocks and the grid's smaller budgets are exact prefix checkpoints of
        the same stream. Each arm's witness is centered on its own first-block
        mean before accumulation: ``h`` sits near ``c_rho ~ 0.65`` while the
        variance being estimated is ``1e-8``-ish, and the raw second moment
        would spend a third of the mantissa on a constant that cancels.
        """
        gC = torch.Generator().manual_seed(
            self.base_seed + CERT_SEED_OFFSET + int(o))
        recs, done, gi = [], 0, 0
        while gi < len(self.grid):
            take = min(int(self.args.block), self.grid[gi] - done)
            X = self._draw(o, take, gC)
            for a in sets:
                Ku = torch.exp(-(torch.cdist(a["zu"], X) ** 2) * self.inv2s)
                Kn = torch.exp(-(torch.cdist(a["z_n"], X) ** 2) * self.inv2s)
                H = torch.cat([a["W_u"] @ Ku, a["W_n"] @ Kn], dim=0)  # (3, n)
                if a["off"] is None:
                    a["off"] = H.mean(dim=1).clone()
                Hc = H - a["off"].unsqueeze(1)
                a["S"] += Hc.sum(dim=1)
                a["Q"] += Hc @ Hc.T
            done += take
            del X
            if done == self.grid[gi]:
                recs.extend(self._certify(o, sets, done))
                gi += 1
        return recs

    def _certify(self, o: int, sets: list[dict], L2: int) -> list[dict]:
        """The per-emission certificates at one checkpoint."""
        a_simult = int(self.args.redraws) * len(self.m_list) * len(PAIRS)
        out = []
        for a in sets:
            S, Q, off = a["S"], a["Q"], a["off"]
            # J_hat_B(Q_a) = w'Kw - 2 w'mu_B ; w'mu_B = mean_i h_a(X_i).
            hbar = [float(S[i]) / L2 + float(off[i]) for i in range(3)]
            J = [a["quad"][i] - 2.0 * hbar[i] for i in range(3)]
            # Ties to the LAST candidate, matching select_emission.
            idx_b = min(range(3), key=lambda i: (J[i], -i))
            pairs = {}
            for p, (i, j) in enumerate(PAIRS):
                gbar_c = float(S[i] - S[j]) / L2
                sum_g2 = float(Q[i, i] - 2.0 * Q[i, j] + Q[j, j])
                v_hat = (sum_g2 - L2 * gbar_c * gbar_c) / (L2 - 1)
                # A tiny negative reading is cancellation, not a defect; it is
                # clamped and its worst excursion is reported rather than
                # silently absorbed.
                v_neg = min(0.0, v_hat)
                v_hat = max(v_hat, 0.0)
                D = a["D"][p]
                gam = slack_value(v_hat, D, L2, delta=DELTA / 3.0)
                gam_s = slack_value(v_hat, D, L2, delta=DELTA / a_simult)
                dhat = J[i] - J[j]
                rec = {"D": D, "V_hat": v_hat, "delta_hat": dhat,
                       "gamma_hat": gam, "gamma_hat_simultaneous": gam_s,
                       "v_hat_negative_excursion": v_neg}
                rec.update(selection_verdict(dhat, gam, D))
                rec["verdict_simultaneous"] = selection_verdict(
                    dhat, gam_s, D)["verdict"]
                pairs[f"{i}{j}"] = rec
            out.append({
                "obs": int(o), "M": a["M"], "k": a["k"], "L2": int(L2),
                "J_B": J, "idx_B": idx_b, "idx_A": a["idx_A"],
                "J_A": a["J_A"], "pairs": pairs,
                "head_B": self._head(pairs, idx_b, L2),
                "head_A": self._head(pairs, a["idx_A"], L2),
            })
        return out

    @staticmethod
    def _head(pairs: dict, idx: int, L2: int) -> dict:
        """The SELECTED arm against the seeds -- the never-worse comparison.

        ORIENTATION IS THE WHOLE CLAIM. The stored pairs are keyed ``(i, j)``
        with ``i < j``, so pair ``0j`` holds ``Jhat(seeds) - Jhat(arm j)``;
        the never-worse statement is about the OTHER direction,
        ``Jhat(selected) - Jhat(seeds)``, and reporting the stored record
        unflipped prints every win as CERTIFIED-WORSE. ``D``, ``V_hat`` and
        the slack are symmetric in the pair, so the flip is exactly a sign on
        ``delta_hat`` and a re-read of the verdict.

        With the seeds in the menu and the selection an argmin on the split,
        ``delta_hat <= 0`` identically, so CERTIFIED-WORSE cannot fire on this
        pair FOR THE SPLIT SELECTION. It stays reachable on the other two
        pairs and on the DEPLOYED selection, which is an argmin on a different
        estimand and so can be certified worse here.
        """
        if idx == 0:
            # the seeds themselves won: the pair is a rule against itself.
            return {"D": 0.0, "V_hat": 0.0, "delta_hat": 0.0,
                    "gamma_hat": 0.0, "gamma_hat_simultaneous": 0.0,
                    "verdict": "COINCIDENT", "verdict_simultaneous":
                    "COINCIDENT", "certified_better": False,
                    "certified_worse": False, "coincident": True,
                    "ratio": float("nan"), "v_hat_negative_excursion": 0.0}
        rec = dict(pairs[f"0{idx}"])
        rec["delta_hat"] = -rec["delta_hat"]
        rec.update(selection_verdict(rec["delta_hat"], rec["gamma_hat"],
                                     rec["D"]))
        rec["verdict_simultaneous"] = selection_verdict(
            rec["delta_hat"], rec["gamma_hat_simultaneous"], rec["D"])[
                "verdict"]
        return rec

    # ---- the library cross-check ------------------------------------------
    def _cross_check(self) -> dict:
        """Blockwise streaming vs the library entered at the nodes.

        The sweep never materializes a split: it accumulates, per arm, the
        witness sums and their second-moment matrix, block by block, each arm
        centered on its own first-block mean. :func:`pairwise_slack` computes
        the same slack from a materialized split in one shot, so the two are
        asserted equal here. The replay is deliberately multi-block, at a block
        size that does NOT divide the split, so a bug in the accumulation
        rather than only in the algebra would surface.

        It builds its own arms and its own split, so it runs on a resumed
        sweep too, where every observation comes off a sidecar and no stream
        is consumed at all.
        """
        keep_m, keep_r = self.m_list, int(self.args.redraws)
        self.m_list, self.args.redraws = [self.m_list[0]], 2
        try:
            sets = self._armsets(0)
        finally:
            self.m_list, self.args.redraws = keep_m, keep_r
        gX = torch.Generator().manual_seed(self.base_seed + 31_337)
        L2, blk = 3000, 700          # 700 does not divide 3000, on purpose
        X = self._draw(0, L2, gX)
        worst_g = worst_d = 0.0
        n = 0
        for a in sets[: int(self.args.n_cross)]:
            nodes = (a["zu"], a["zu"], a["z_n"])
            wts = (a["W_u"][0], a["W_u"][1], a["W_n"][0])
            S = torch.zeros(3, dtype=DTYPE)
            Q = torch.zeros(3, 3, dtype=DTYPE)
            off = None
            for s in range(0, L2, blk):
                B = X[s : s + blk]
                Ku = torch.exp(-(torch.cdist(a["zu"], B) ** 2) * self.inv2s)
                Kn = torch.exp(-(torch.cdist(a["z_n"], B) ** 2) * self.inv2s)
                H = torch.cat([a["W_u"] @ Ku, a["W_n"] @ Kn], dim=0)
                if off is None:
                    off = H.mean(dim=1).clone()
                Hc = H - off.unsqueeze(1)
                S += Hc.sum(dim=1)
                Q += Hc @ Hc.T
            for p, (i, j) in enumerate(PAIRS):
                gbar_c = float(S[i] - S[j]) / L2
                v = ((float(Q[i, i] - 2 * Q[i, j] + Q[j, j])
                      - L2 * gbar_c**2) / (L2 - 1))
                mine = slack_value(max(v, 0.0), a["D"][p], L2,
                                   delta=DELTA / 3.0)
                lib = pairwise_slack(nodes[i], wts[i], nodes[j], wts[j],
                                     X, self.sigma, delta=DELTA / 3.0)
                sc = max(abs(mine), abs(lib["gamma_hat"]), 1e-300)
                worst_g = max(worst_g, abs(mine - lib["gamma_hat"]) / sc)
                worst_d = max(worst_d, abs(a["D"][p] - lib["D"]))
                n += 1
        if worst_g > 1e-9 or worst_d > 1e-12:
            raise AssertionError(
                "the streamed certificate and the library entered at the "
                f"nodes disagree: gamma max rel {worst_g:.3e}, D max abs "
                f"{worst_d:.3e}. One of the two paths is wrong and no rate "
                "in this record means anything.")
        print(f"  [check] streamed vs library gamma: max rel {worst_g:.2e} "
              f"over {n} pairs at L2={L2} in {blk}-draw blocks", flush=True)
        return {"streamed_vs_library_gamma_max_rel": worst_g,
                "pair_scale_max_abs": worst_d, "n_checked": n, "L2": L2,
                "block": blk,
                "tolerances": {"gamma_rel": 1e-9, "D_abs": 1e-12}}

    # ---- the degenerate-case control --------------------------------------
    def _untrained_control(self) -> dict:
        """Zero-init head => the arms coincide bitwise => the slack is 0.

        Not "small": exactly zero, pathwise, with no probability event. Run on
        the real pipeline -- real flow bank, real split, real solve -- because
        that is where an implementation that quietly perturbs an arm would
        show up and a fixture would not.
        """
        torch.manual_seed(int(self.cfg["seed"]))
        net0, keep = self._build_net(), self.net
        net0.eval()
        self.net, keep_r = net0, int(self.args.redraws)
        keep_grid, self.grid = self.grid, [self.grid[0]]
        self.args.redraws, keep_m = 2, self.m_list
        self.m_list = [self.m_list[0]]
        try:
            sets = self._armsets(0)
            recs = self._stream(0, sets)
            worst = {"D": 0.0, "gamma_hat": 0.0, "V_hat": 0.0}
            dz = max(float((a["z_n"] - a["zu"]).abs().max()) for a in sets)
            for r in recs:
                p = r["pairs"]["12"]        # reweight vs move
                for kk in worst:
                    worst[kk] = max(worst[kk], p[kk])
        finally:
            self.net, self.args.redraws = keep, keep_r
            self.grid, self.m_list = keep_grid, keep_m
        exact = all(v == 0.0 for v in worst.values())
        print(f"  [control] untrained: max|dz|={dz:.1e} D={worst['D']:.1e} "
              f"gamma={worst['gamma_hat']:.1e} -- cs-thm-certsel(4a) "
              f"{'HOLDS EXACTLY' if exact else 'FAILED'}", flush=True)
        if not exact:
            raise AssertionError(
                "cs-thm-certsel(4a): a zero-initialized head must make the "
                "move and reweight arms bitwise equal, giving "
                f"D = V_hat = gamma_hat = 0 exactly; got {worst}")
        return {"M": int(self.m_list[0]), "max_displacement": dz, **worst,
                "degenerate_case_holds_exactly": exact}

    # ---- the sweep --------------------------------------------------------
    def run(self):
        os.makedirs(SIDE_DIR, exist_ok=True)
        t0 = time.time()
        cells = []
        tag = (f"o{self.n_obs}_r{self.args.redraws}_"
               f"L{self.grid[-1]}_{len(self.grid)}")
        for o in tqdm(range(self.n_obs), desc="observations"):
            side = os.path.join(SIDE_DIR, f"{tag}_obs{o:03d}.json")
            if os.path.isfile(side):
                with open(side) as fh:
                    cells.extend(json.load(fh)["cells"])
                print(f"  [resume] observation {o} read from its sidecar",
                      flush=True)
                continue
            t_o = time.time()
            sets = self._armsets(o)
            recs = self._stream(o, sets)
            _atomic_json({"obs": o, "cells": recs}, side)
            cells.extend(recs)
            print(f"  [obs {o}] {len(recs)} certificates | "
                  f"{time.time() - t_o:.0f}s", flush=True)

        rows = self._aggregate(cells)
        out = {
            "what": (
                "cs-thm-certsel / cs-cor-menu(ii) instrumented on the Darcy "
                "rung: the certification split DRAWN fresh from the reference "
                "flow and swept in L2, so the deliverable is the curve of "
                "certification rate against certification budget."),
            "rung": "scripts/dq_darcy_stuart.py (README Group 10c)",
            "checkpoint_experiment": self.exp,
            "protocol": self._protocol(),
            "sigma": self.sigma, "delta_family": DELTA,
            "delta_per_pair": DELTA / 3.0,
            "coincidence_D": COINCIDENCE_D,
            "m_list": self.m_list, "l2_grid": self.grid,
            "n_obs": self.n_obs, "redraws_per_obs": int(self.args.redraws),
            "deployed_depths": self.deployed,
            "rates": rows,
            "draw_ledger": self._ledger(),
            "arm_identity": self._verify_arms(),
            "controls": {"untrained_zero_init": self._untrained_control()},
            "cross_checks": self._cross_check(),
            "wall_secs": time.time() - t0,
        }
        out["verdict"] = self._verdict(rows)
        _atomic_json(out, OUT)
        print(f"\nSaved to {OUT}")
        return out

    # ---- aggregation ------------------------------------------------------
    def _aggregate(self, cells: list[dict]) -> list[dict]:
        rows = []
        for M in self.m_list:
            for L2 in self.grid:
                sub = [c for c in cells if c["M"] == M and c["L2"] == L2]
                if not sub:
                    continue
                row = {"M": int(M), "L2": int(L2), "n_emissions": len(sub)}
                for tag, key in (("split_selection", "head_B"),
                                 ("deployed_selection", "head_A")):
                    h = [c[key] for c in sub]
                    live = [r for r in h if not r["coincident"]]
                    nb = sum(1 for r in live if r["certified_better"])
                    nw = sum(1 for r in live if r["certified_worse"])
                    ns = sum(1 for r in live
                             if r["verdict_simultaneous"] == "CERTIFIED-BETTER")
                    lo, hi = _wilson(nb, len(live))
                    row[tag] = {
                        "n_live": len(live),
                        "n_coincident": len(h) - len(live),
                        "certified_better": nb, "certified_worse": nw,
                        "within_slack": len(live) - nb - nw,
                        "rate": (nb / len(live)) if live else float("nan"),
                        "rate_ci95": [lo, hi],
                        "certified_better_simultaneous": ns,
                        "rate_simultaneous": ((ns / len(live)) if live
                                              else float("nan")),
                        "gamma_hat": _med_band(r["gamma_hat"] for r in live),
                        "abs_delta_hat": _med_band(
                            abs(r["delta_hat"]) for r in live),
                        "D": _med_band(r["D"] for r in live),
                        "stakes_over_slack": _med_band(
                            r["ratio"] for r in live),
                    }
                # every menu pair, since the event is their intersection and
                # CERTIFIED-WORSE can only fire off-headline
                row["pairs_orientation"] = (
                    "delta_hat = Jhat(FIRST named arm) - Jhat(SECOND), so "
                    "CERTIFIED-WORSE counts emissions where the FIRST arm is "
                    "certified worse. The head_A / head_B columns above are "
                    "the opposite orientation (selected minus seeds), which "
                    "is the direction the never-worse claim is stated in.")
                row["pairs"] = {}
                for i, j in PAIRS:
                    pr = [c["pairs"][f"{i}{j}"] for c in sub]
                    live = [r for r in pr if not r["coincident"]]
                    row["pairs"][f"{ARM_NAMES[i]}_vs_{ARM_NAMES[j]}"] = {
                        "n_live": len(live),
                        "certified_better": sum(
                            1 for r in live if r["certified_better"]),
                        "certified_worse": sum(
                            1 for r in live if r["certified_worse"]),
                        "gamma_hat": _med_band(r["gamma_hat"] for r in live),
                        "abs_delta_hat": _med_band(
                            abs(r["delta_hat"]) for r in live),
                    }
                row["selection_agreement"] = float(np.mean(
                    [c["idx_A"] == c["idx_B"] for c in sub]))
                row["selected_on_split"] = {
                    ARM_NAMES[i]: float(np.mean([c["idx_B"] == i
                                                 for c in sub]))
                    for i in range(3)}
                row["max_v_hat_negative_excursion"] = min(
                    [p["v_hat_negative_excursion"]
                     for c in sub for p in c["pairs"].values()])
                rows.append(row)
                hb = row["split_selection"]
                print(f"  [M={M:>3d} L2={L2:>8d}] split-selection certified "
                      f"{hb['certified_better']}/{hb['n_live']} live "
                      f"({hb['rate']:.3f}) | gamma~{hb['gamma_hat'][0]:.2e} "
                      f"|dhat|~{hb['abs_delta_hat'][0]:.2e} "
                      f"stake/slack~{hb['stakes_over_slack'][0]:.3f}",
                      flush=True)
        return rows

    # ---- bookkeeping ------------------------------------------------------
    def _ledger(self) -> dict:
        """The certificate's price, in reference samples per observation."""
        return {
            "currency": (
                "reference draws consumed per observation -- the same ledger "
                "sec 6.3 prices the compression peers against."),
            "certification_draws_per_observation": self.grid,
            "note": (
                "the split is DRAWN from the flow, not carved out of the "
                "estimand bank, so certification costs the arms no precision "
                "at all: the estimand bank is untouched and every number the "
                "rung already reports is unmoved. What it costs is draws, "
                "plus O(M L2) kernel evaluations to contract them. For "
                "reference the rung's own per-observation estimand bank is "
                f"{self.bank_L} draws ({2 * self.bank_L} at the two deepest "
                "budgets)."),
            "one_split_serves": (
                "one drawn split serves every budget and every redraw at its "
                "observation, so the per-observation draw cost is paid ONCE "
                "for all of them -- but see the simultaneity disclosure: the "
                "marginal per-emission verdict is what that reuse licenses, "
                "and the Bonferroni column is what would license the joint "
                "statement."),
        }

    def _protocol(self) -> dict:
        return {
            "reference": (
                "rho_y = the trained conditional flow (darcy_stuart_npe run "
                f"{self.cfg['npe_run']}), a samplable measure. Certification "
                "draws are FRESH draws from it."),
            "arms": (
                "reproduced from dq_darcy_stuart._evaluate: bank "
                "size bank_L (2x at M >= 256), generator base_seed + 50_000 + "
                "1000*M + obs, seeds WITH replacement, deduped with "
                "multiplicity, weights from the unit-sum constrained solve "
                "against that bank."),
            "certification_seed_rule": (
                f"base_seed + {CERT_SEED_OFFSET} + obs -- a data-independent "
                "offset. No estimand-bank atom, arm, or network parameter "
                "reaches the stream."),
            "independence": (
                "the split shares no draw with any estimand bank, the network "
                "never saw this observation, and the bandwidth is a "
                "functional of the TRAINING observations' draws -- so every "
                "quantity used to fit or score the arms is independent of the "
                "split, which is the theorem's hypothesis."),
            "what_the_certificate_does_NOT_repair": (
                "the seeds are rows of the estimand bank (the rung's own "
                "convention), so they are i.i.d. draws of the bank's "
                "empirical measure rather than of rho, and the arms carry "
                "e2-prop-overlap's shrinkage. cs-thm-certsel is indifferent "
                "to how the arms were produced -- it needs only that the "
                "split is independent of them -- so the certificate is valid "
                "AND certifies the rule as fitted, shrinkage included. It "
                "does not launder the fit; it prices the comparison."),
            "rho_is_the_flow": (
                "MMD^2 in the certificate is against the FLOW, not against a "
                "bank: the split is drawn from the flow, so the guarantee is "
                "'never worse than its seeds as an integrator of rho_y'. The "
                "rung's own reported depths are against the bank-as-measure, "
                "a different and weaker object."),
            "l2_grid_is_nested": (
                "the grid's smaller budgets are exact prefix checkpoints of "
                "one stream (verified bit-identical against a single call), "
                "so the whole curve costs one pass."),
            "selection": (
                "TWO selections are certified: (a) argmin of Jhat on the "
                "split over the live three-arm menu -- cs-cor-menu(ii), each "
                "pair at delta/3 under cs-eq-gammahat's own ln(4/.) "
                "numerator; and (b) the DEPLOYED argmin on the fitting "
                "estimand, which is what the rung actually emits and is "
                "equally independent of the split."),
            "simultaneity_disclosure": (
                "cs-thm-certsel prices ONE emission against ONE split. This "
                "sweep reuses each observation's split across every redraw "
                "and budget, so each cell's verdict is a valid MARGINAL "
                "1-delta statement and the reported rate is an average of "
                "marginal certificates, not a simultaneous statement over the "
                "cells sharing a split. The Bonferroni slack that would make "
                "it simultaneous over one split's whole service is reported "
                "as gamma_hat_simultaneous / rate_simultaneous."),
            "bandwidth": (
                "read from the trained run's results.json, never recomputed "
                "from anything scored here."),
        }

    @staticmethod
    def _verdict(rows: list[dict]) -> dict:
        by = {}
        for r in rows:
            by.setdefault(str(r["M"]), {})[str(r["L2"])] = (
                r["split_selection"]["rate"])
        # The smallest grid budget at which the split selection certifies a
        # majority of live emissions.
        bite = {}
        for r in sorted(rows, key=lambda x: (x["M"], x["L2"])):
            m = str(r["M"])
            if m not in bite and r["split_selection"]["rate"] >= 0.5:
                bite[m] = int(r["L2"])
        return {
            "rate_by_M_by_L2": by,
            "smallest_L2_certifying_a_majority": bite,
            "certified_worse_total": sum(
                r[k]["certified_worse"] for r in rows
                for k in ("split_selection", "deployed_selection")),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_obs", type=int, default=24)
    p.add_argument("--redraws", type=int, default=8)
    p.add_argument("--l2_grid", type=str,
                   default="1000,3000,10000,30000,100000,300000,1000000")
    p.add_argument("--block", type=int, default=10000)
    p.add_argument("--n_cross", type=int, default=3)
    p.add_argument("--n_verify", type=int, default=24)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.n_obs, a.redraws = 2, 2
        a.l2_grid = "1000,3000,10000"
        a.n_verify = 2
    t0 = time.time()
    DarcyCertification(a).run()
    print(f"Total wall-time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
