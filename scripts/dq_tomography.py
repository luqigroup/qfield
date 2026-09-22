"""Designed-quadrature warm start for tomography, in the informed subspace.

The problem is limited-angle tomography with a ``d_x = 256`` image, an exact
closed-form Gaussian posterior as the reference ``rho``, and the score-free
informed subspace ``phi_H`` of rank ``r``. The warm start is the
set-equivariant, arbitrary-``M``
:class:`qfield.designed_quadrature.net.QuadratureAmortizer` trained to
MMD-regress onto the exact posterior.

A high ambient ``d`` walls the SE kernel (``K -> I``, so node motion buys
nothing), so the work is done entirely in the ``r``-dimensional informed
subspace:

  1. Build the exact Gaussian tomography posterior ``N(mu_post(y*),
     Sigma_post)`` (closed form; ``Sigma_post`` is ``y*``-independent) and the
     score-free informed subspace ``phi_H`` (rank ``r`` by the participation
     ratio of the LIS spectrum, capped at ``rank_cap``; from
     ``qfield.subspace``). ``phi_H(x) = x @ W`` with ``W =
     Sigma_pr^{-1} U`` (whitened informed coords) and informed directions ``U``
     (columns), satisfying ``W^T U = I`` (an exact lift ``x = U z``).
  2. Project the posterior into the subspace: ``rho_z = N(m_z, S_z)`` with
     ``m_z = W^T mu_post``, ``S_z = W^T Sigma_post W``, a Gaussian in ``R^r``
     built via :func:`qfield.designed_quadrature.target.latent_gaussian_target`
     (the exact SE-kernel closed forms ``mu_fn`` / ``c_rho`` at ``d = r``).
  3. Train the net at ``d = r`` on ``rho_z`` (median-heuristic bandwidth on
     latent samples; ``MMD^2`` objective). On held-out ``y*`` evaluate, per
     node budget ``M``, the warm-start MMD against the i.i.d.-``rho_z`` floor
     and against the per-instance oracle (``move_descend``). Because ``S_z``
     is ``y*``-independent and the SE-MMD is translation-invariant, the
     integration metric is the same across ``y*``.
  4. Back-map the nodes through ``U`` to image space (``x_j = U z_j``) to get
     a weighted-quadrature reconstruction ``sum_j w_j x_j`` and a per-pixel
     UQ ``sqrt(sum_j w_j (x_j - x_bar)^2)`` for a few held-out ``y*``.

Train (build subspace, project posterior, train + eval + recon, save):
    CUDA_VISIBLE_DEVICES="" python scripts/dq_tomography.py \
        --experiment_name dq_tomography
Visualize (re-render figures from the saved artifact):
    python scripts/dq_tomography.py --experiment_name dq_tomography \
        --phase visualization

Reference:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (the MMD-optimal designed
    quadrature; the per-instance oracle the net amortizes).
  - Spantini et al. 2015; Zahm et al. 2022 (the LIS / informed subspace).
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

# Pin CPU before importing torch.
import os  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from projorg import (  # noqa: E402
    checkpointsdir,
    plotsdir,
    setup_environment,
    upload_to_cloud,
)

import qfield.subspace as sub  # noqa: E402
from qfield.dataset.tomography import limited_angle_tomography  # noqa: E402
from qfield.designed_quadrature import (  # noqa: E402
    constrained_weights_batched,
    cross_check_target,
    latent_gaussian_target,
    mmd_sq_batched,
    move_descend_batched,
)
from qfield.designed_quadrature.arms import (  # noqa: E402
    COND_LIMIT,
    eval_row,
    seed_gram_cond,
)
from qfield.designed_quadrature.net import (  # noqa: E402
    QuadratureAmortizer,
)
from qfield.kernels import (  # noqa: E402
    median_heuristic_h_sq,      # the Liu-Wang SVGD rule
    median_heuristic_sigma_sq,  # the median heuristic
)

CONFIG_FILE = "dq_tomography.json"

DTYPE = torch.float64

# Cap the intra-op pool: many small batched solves.
torch.set_num_threads(4)

# Palette: floor slate-blue (independent-sample floor), warm orange (one-pass
# warm start), oracle grey (the per-instance optimum, descended from z0).
_PALETTE = {"floor": "#3a7ca5", "warm": "#ff7f0e", "oracle": "#7f7f7f"}
_LABELS = {
    "floor": "i.i.d. floor",
    "warm": "warm start (1 fwd pass)",
    "oracle": "per-instance optimum (from the i.i.d. seed)",
}


def _median_iqr(vals: list[float]) -> tuple[float, float, float]:
    """``(median, q25, q75)`` over the finite entries of ``vals``."""
    t = torch.tensor(vals, dtype=DTYPE)
    t = t[torch.isfinite(t)]
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


def _draw_batch(target, M: int, B: int, gen: torch.Generator) -> torch.Tensor:
    """A batch of ``B`` i.i.d. ``M``-sample sets from the target."""
    return torch.stack(
        [target.sampler(int(M), gen) for _ in range(int(B))], dim=0
    )


def _emit_mmd_batched(
    net: QuadratureAmortizer,
    z0: torch.Tensor,
    target,
    sigma: float,
    jitter: float,
) -> torch.Tensor:
    """Per-set exact SE-MMD^2 of the net's emitted latent quadrature."""
    B, M, d = z0.shape
    z = net(z0)
    mu = target.mu_fn(z.reshape(B * M, d)).reshape(B, M)
    w = constrained_weights_batched(z, mu, sigma, jitter)
    return mmd_sq_batched(z, w, mu, target.c_rho, sigma)


class DQTomography:
    """One warm-start run on the projected tomography posterior."""

    def __init__(self, args):
        self.args = args
        if args.gpu_id != -1:
            raise ValueError(
                "dq_tomography is CPU-only (small net, r-dim latent); set "
                f"gpu_id=-1, got {args.gpu_id}"
            )
        torch.set_default_dtype(DTYPE)
        self.r = int(args.r)
        self.m_list = list(args.m_list)
        self._build_problem()
        self._build_target()

    # ---- problem + informed subspace + projected posterior --------------
    def _build_problem(self):
        """Assemble the LG tomography problem and the rank-``r`` ``phi_H``.

        Uses :func:`limited_angle_tomography` (the exact construction) and
        :class:`qfield.subspace.InformedSubspace` (the score-free informed
        subspace from the closed-form ``H``). The retained rank is the config
        ``r``.
        """
        a = self.args
        problem, meta = limited_angle_tomography(
            n=a.n_img, n_angles=a.n_angles, phi_deg=a.phi_deg,
            sigma_obs=a.sigma_obs, n_detectors=a.n_detectors,
            prior_delta=a.prior_delta, prior_alpha=a.prior_alpha,
            prior_tau=a.prior_tau, dtype=torch.float32,
        )
        a64, spr64, sobs64 = meta["A"], meta["Sigma_pr"], meta["Sigma_obs"]
        self.dx, self.dy = meta["dx"], meta["dy"]
        self.n_img = a.n_img
        self.problem = problem

        # Score-free informed subspace at the requested rank, plus the
        # participation rank for the record.
        full_H = sub.InformedSubspace.from_H(a64, sobs64, spr64, r=self.dx)
        self.r_participation = sub.participation_rank(
            full_H.eigvals, cap=int(a.rank_cap)
        )
        self.phi_H = sub.InformedSubspace.from_H(a64, sobs64, spr64, r=self.r)
        self.W = self.phi_H._w.double()              # (dx, r): z = x @ W
        self.U = self.phi_H.directions.double()      # (dx, r): lift x = U z
        # Exact (y*-independent) posterior covariance + its latent projection.
        _, sigma_post = problem.posterior(torch.zeros(self.dy))
        self.sigma_post = sigma_post.double()
        S_z = self.W.T @ self.sigma_post @ self.W
        self.S_z = 0.5 * (S_z + S_z.T)               # (r, r) latent post cov
        print(
            f"[problem] dx={self.dx} dy={self.dy} | informed r={self.r} "
            f"(v4 participation rank={self.r_participation}) | "
            f"latent post eig range="
            f"[{float(torch.linalg.eigvalsh(self.S_z).min()):.3e}, "
            f"{float(torch.linalg.eigvalsh(self.S_z).max()):.3e}]"
        )

    def _latent_sigma(self) -> float:
        """SE bandwidth: median heuristic on latent posterior samples.

        ``bandwidth`` selects the rule (the mean is immaterial throughout:
        the median heuristic and the SE-MMD are translation-invariant):

        - ``"median_mmd"``: the median heuristic
          (``median_heuristic_sigma_sq``, no ``2 log(N+1)`` divisor).
        - ``"median"``: the Liu-Wang SVGD rule.
        - a float or ``"x<f>"``: an absolute sigma, or ``f`` times the
          Liu-Wang median.
        """
        a = self.args
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(a.base_seed) + 1234)
        chol = torch.linalg.cholesky(self.S_z)
        z = torch.randn(
            int(a.sigma_n_sample), self.r, generator=gen, dtype=DTYPE
        ) @ chol.T
        bw = getattr(a, "bandwidth", None)
        s = str(bw).strip().lower() if bw is not None else "median"
        if s == "median_mmd":
            med = math.sqrt(median_heuristic_sigma_sq(z))
            rule, scale = "mmd", 1.0
        else:
            med = math.sqrt(median_heuristic_h_sq(z))
            rule, scale = "svgd", 1.0
            if s not in ("", "median"):
                scale = float(s[1:]) if s.startswith("x") else float(s) / med
        sigma = scale * med
        print(
            f"[sigma] median heuristic/{rule} (r={self.r}, "
            f"N={a.sigma_n_sample}): median={med:.4f}  scale={scale:.3f}  "
            f"->  sigma={sigma:.4f}"
        )
        return float(sigma)

    def _build_target(self):
        """Project the posterior into the subspace -> the latent Gaussian rho_z.

        ``rho_z = N(m_z, S_z)``; only the SE-MMD-relevant ``S_z`` matters for
        the integration metric, so the representative ``y*`` mean is logged
        but immaterial. The exact SE-kernel closed forms (``mu_fn`` /
        ``c_rho``) come from :func:`latent_gaussian_target`.
        """
        a = self.args
        self.sigma = self._latent_sigma()
        # Representative held-out y* for the (immaterial) latent mean / meta.
        torch.manual_seed(int(a.base_seed) + 7)
        _, y_all = self.problem.sample_joint(2000)
        y_star = y_all[1500].double()
        mu_post, _ = self.problem.posterior(y_star)
        m_z = self.W.T @ mu_post.double()
        self.target = latent_gaussian_target(
            m_z, self.S_z, self.sigma, name="tomography_latent",
            meta={"r": self.r, "phi_deg": a.phi_deg, "n_img": a.n_img},
        )

    def _build_net(self) -> QuadratureAmortizer:
        a = self.args
        return QuadratureAmortizer(
            d=self.r,
            hidden=int(a.hidden),
            n_blocks=int(a.n_blocks),
            n_heads=int(a.n_heads),
            cond_hidden=int(a.cond_hidden),
            delta_scale=float(a.delta_scale),
        ).to(DTYPE)

    # ---- compute --------------------------------------------------------
    def train(self):
        """Train the DQ amortizer, eval warm/floor/oracle, build recon + UQ."""
        a = self.args
        print(
            f"[dq-tomo] M~Uniform{self.m_list}, batch={a.batch_size}, "
            f"steps={a.n_steps}, lr={a.lr}, sigma={self.sigma:.4f} | "
            f"net(hidden={a.hidden}, blocks={a.n_blocks}, heads={a.n_heads}, "
            f"delta_scale={a.delta_scale})"
        )

        # Metric cross-check before training: the explicit MMD^2 against
        # the Gaussian closed form, which catches a mu / c_rho / convention
        # bug.
        rel = cross_check_target(
            self.target, self.sigma, seed=int(a.base_seed) + 777
        )
        print(f"  metric cross-check (explicit vs closed form): rel={rel:.2e}")
        assert rel < 1e-8, f"metric cross-check failed: rel={rel:.2e}"

        torch.manual_seed(int(a.seed))
        net = self._build_net()
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  net params: {n_params}")
        opt = torch.optim.Adam(net.parameters(), lr=float(a.lr))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(a.base_seed))
        m_choices = torch.tensor(self.m_list)
        log_every = max(1, int(a.n_steps) // 20)
        curve = []
        ema = None
        t0 = time.time()
        net.train()
        for step in range(int(a.n_steps)):
            M = int(m_choices[torch.randint(len(m_choices), (1,), generator=gen)])
            z0 = _draw_batch(self.target, M, int(a.batch_size), gen)
            loss = _emit_mmd_batched(
                net, z0, self.target, self.sigma, a.jitter
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            v = float(loss.detach())
            ema = v if ema is None else 0.98 * ema + 0.02 * v
            if step % log_every == 0 or step == int(a.n_steps) - 1:
                curve.append((step, ema))
                print(
                    f"  step {step:>5d}/{a.n_steps}: M={M:>3d} "
                    f"MMD^2(mean)={v:.4e}  EMA={ema:.4e}"
                )
        train_secs = time.time() - t0
        print(f"  training wall-time: {train_secs:.1f} s")

        net.eval()
        eval_rows = self._evaluate(net)
        recon = self._reconstructions(net)

        verdict = self._verdict(eval_rows)
        results = {
            "r": self.r,
            "r_participation": int(self.r_participation),
            "dx": int(self.dx),
            "dy": int(self.dy),
            "n_img": int(self.n_img),
            "sigma": float(self.sigma),
            "m_list": self.m_list,
            "phi_deg": float(a.phi_deg),
            "metric_cross_check_rel": rel,
            "n_params": int(n_params),
            "n_steps": int(a.n_steps),
            "batch_size": int(a.batch_size),
            "lr": float(a.lr),
            "delta_scale": float(a.delta_scale),
            "eval_repeats": int(a.eval_repeats),
            "oracle_lr": float(a.oracle_lr),
            "oracle_iters": int(a.oracle_iters),
            "train_curve": curve,
            "train_secs": train_secs,
            "rows": eval_rows,
            "recon": recon,
            "verdict": verdict,
        }
        self._print_table(eval_rows, verdict)

        ckpt_path = os.path.join(checkpointsdir(a.experiment), "checkpoint.pth")
        torch.save(
            {"results": results, "state_dict": net.state_dict()}, ckpt_path
        )
        with open(
            os.path.join(checkpointsdir(a.experiment), "results.json"), "w"
        ) as fh:
            json.dump(results, fh, indent=2)
        print(f"Saved results to {ckpt_path}")

    def _evaluate(self, net: QuadratureAmortizer) -> list[dict]:
        """Held-out floor / reweight / warm-start / oracle latent MMD^2 per ``M``.

        The row is assembled by ``designed_quadrature.arms.eval_row``, the
        same builder the toy trainer uses, so the two cannot drift apart in
        what they record.
        """
        a = self.args
        rows = []
        for M in self.m_list:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(a.base_seed) + 50000 + 1000 * int(M))
            z0 = _draw_batch(self.target, M, int(a.eval_repeats), gen)
            R = z0.shape[0]
            mu0 = self.target.mu_fn(z0.reshape(R * M, self.r)).reshape(R, M)
            w_eq = torch.full((R, M), 1.0 / M, dtype=DTYPE)
            floor = torch.clamp(
                mmd_sq_batched(z0, w_eq, mu0, self.target.c_rho, self.sigma),
                min=0.0,
            )
            # The constrained solve at the untouched nodes.
            with torch.no_grad():
                w_rw = constrained_weights_batched(
                    z0, mu0, self.sigma, a.jitter
                )
                reweight = torch.clamp(
                    mmd_sq_batched(
                        z0, w_rw, mu0, self.target.c_rho, self.sigma
                    ),
                    min=0.0,
                )
            with torch.no_grad():
                warm = torch.clamp(
                    _emit_mmd_batched(
                        net, z0, self.target, self.sigma, a.jitter
                    ),
                    min=0.0,
                )
            _, oracle = move_descend_batched(
                z0, self.target.mu_fn, self.target.c_rho, self.sigma,
                float(a.oracle_lr), int(a.oracle_iters), a.jitter,
            )
            cond_seed = seed_gram_cond(z0, self.sigma)
            row = eval_row(M, floor, reweight, warm, oracle, cond_seed)
            rows.append(row)
            print(
                f"  [eval] M={M:>3d}: floor={row['floor'][0]:.3e} "
                f"rw={row['reweight'][0]:.3e} warm={row['warm'][0]:.3e} "
                f"oracle={row['oracle'][0]:.3e}  (warm/floor="
                f"{row['warm_over_floor']:.3f}, warm/oracle="
                f"{row['warm_over_oracle']:.3f}, move_gain="
                f"{row['move_gain']:.3f}, cond={cond_seed['median']:.2e}"
                f"{' LIMITED' if row['cond_limited'] else ''})"
            )
        return rows

    def reeval(self):
        """Re-score the saved net without training, for ``--phase reeval``.

        This target has one exact estimand, so a re-score is a no-op on the
        floor / warm / oracle columns. That is checked rather than assumed,
        which makes this a regression test on the whole eval path.
        """
        a = self.args
        net = self._build_net()
        net.load_state_dict(self.ckpt["state_dict"])
        net.eval()
        t1 = time.time()
        rows = self._evaluate(net)
        eval_secs = time.time() - t1
        verdict = self._verdict(rows)
        old = {int(r["M"]): r for r in self.results.get("rows", [])}
        drift = max(
            (
                abs(r[k][0] - old[r["M"]][k][0]) / max(old[r["M"]][k][0], 1e-300)
                for r in rows
                if r["M"] in old
                for k in ("floor", "warm", "oracle")
                if k in old[r["M"]]
            ),
            default=float("nan"),
        )
        print(f"[reeval] max relative drift of the stored arms: {drift:.3e}")
        if drift > 0.0:
            raise RuntimeError(
                f"[reeval] drift {drift:.3e} on a re-score that must be a "
                "no-op for floor/warm/oracle. The eval path changed; fix "
                "that before overwriting the record."
            )
        self.results.update(rows=rows, verdict=verdict, eval_secs=eval_secs)
        self._print_table(rows, verdict)
        # Atomic replace: checkpoint.pth is the only copy of the state_dict.
        ckpt_path = os.path.join(
            checkpointsdir(a.experiment), "checkpoint.pth"
        )
        payload = dict(self.ckpt)
        payload["results"] = self.results
        torch.save(payload, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        rj_path = os.path.join(checkpointsdir(a.experiment), "results.json")
        with open(rj_path + ".tmp", "w") as fh:
            json.dump(self.results, fh, indent=2)
        os.replace(rj_path + ".tmp", rj_path)
        print(f"Re-evaluated and saved to {ckpt_path}")

    def _reconstructions(self, net: QuadratureAmortizer) -> dict:
        """Image-space recon + per-pixel UQ for a few held-out ``y*``.

        For each held-out ``y*``: build the latent posterior ``N(m_z(y*),
        S_z)`` (re-centered to the ``y*`` mean; ``S_z`` is fixed), seed ``M``
        i.i.d. latent samples, emit the quadrature, and back-map the nodes
        ``x_j = U z_j``. The reconstruction is the weighted-quadrature mean
        ``sum_j w_j x_j`` (reshaped ``n x n``); the per-pixel UQ is the
        weighted spread ``sqrt(sum_j w_j (x_j - x_bar)^2)``. The ground truth
        is the joint sample ``x*`` (its informed projection ``P_H x*`` is the
        in-subspace ceiling). The relL2 of the quadrature mean to the true
        ``mu_post(y*)`` is reported.
        """
        a = self.args
        M = int(a.recon_M)
        n = self.n_img
        torch.manual_seed(int(a.base_seed) + 99)
        x_all, y_all = self.problem.sample_joint(4000)
        idx = [3990, 3991, 3992]
        out = {"M": M, "n_img": n, "cases": []}
        for k in idx:
            x_true = x_all[k].double()
            y_star = y_all[k].double()
            mu_post, _ = self.problem.posterior(y_star)
            mu_post = mu_post.double()
            m_z = self.W.T @ mu_post
            # Latent posterior centred at this y*'s mean (S_z fixed).
            tgt = latent_gaussian_target(m_z, self.S_z, self.sigma)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(a.base_seed) + 7000 + k)
            z0 = tgt.sampler(M, gen).unsqueeze(0)  # (1, M, r)
            with torch.no_grad():
                z = net(z0)
                mu = tgt.mu_fn(z.reshape(M, self.r))
                w = constrained_weights_batched(
                    z, mu.reshape(1, M), self.sigma, a.jitter
                )[0]
                z = z[0]
            # Back-map: x_j = U z_j (W^T U = I, the exact lift).
            x_nodes = z @ self.U.T                 # (M, dx)
            x_mean = (w.unsqueeze(1) * x_nodes).sum(dim=0)  # (dx,)
            diff = x_nodes - x_mean.unsqueeze(0)
            var = (w.unsqueeze(1) * diff**2).sum(dim=0).clamp_min(0)
            x_std = var.sqrt()
            # In-subspace ceiling + truncation floor diagnostics.
            p_true = self.U @ (self.W.T @ x_true)   # informed projection of x*
            rel_mu = float((x_mean - mu_post).norm() / mu_post.norm())
            rel_proj = float((x_mean - p_true).norm() / p_true.norm())
            out["cases"].append(
                {
                    "true": x_true.tolist(),
                    "recon": x_mean.tolist(),
                    "uq": x_std.tolist(),
                    "rel_mu_post": rel_mu,
                    "rel_informed_proj": rel_proj,
                }
            )
            print(
                f"  [recon] y*[{k}]: relL2(quad-mean vs mu_post)={rel_mu:.3f}  "
                f"relL2(quad-mean vs informed-proj of truth)={rel_proj:.3f}"
            )
        return out

    def _verdict(self, rows: list[dict]) -> dict:
        """Below-floor and above-oracle verdicts, excluding unreadable ``M``.

        A cell whose seed Gram is past ``COND_LIMIT`` is unreadable: the
        constrained solve every arm depends on is not trustworthy at float64
        there. Such a cell is marked and excluded rather than given a silent
        ridge. The lever split at the largest readable budget is summarised
        beside it.
        """
        lim = [bool(r.get("cond_limited")) for r in rows]
        readable = [r for r, x in zip(rows, lim) if not x]
        top = readable[-1] if readable else None
        return {
            "cond_limit": COND_LIMIT,
            "cond_limited_M": [r["M"] for r, x in zip(rows, lim) if x],
            "warm_below_floor_every_M": all(
                r["warm"][0] < r["floor"][0] for r in readable
            ),
            "warm_above_oracle_every_M": all(
                r["warm"][0] >= r["oracle"][0] - 1e-12 for r in readable
            ),
            "reweight_below_floor_every_M": all(
                r["reweight"][0] < r["floor"][0] for r in readable
            ),
            "top_readable_M": None if top is None else top["M"],
            "move_gain_at_top": None if top is None else top["move_gain"],
            "move_is_a_loss_at_top": bool(
                top is not None and top["move_gain"] < 1.0
            ),
            "phi_at_top": None if top is None else top["phi"],
            "verdicts_scope": (
                "all budgets" if not any(lim)
                else "budgets below the cond(K) limit only"
            ),
        }

    def _print_table(self, rows: list[dict], verdict: dict):
        print(
            f"\n=== tomography DQ warm start (latent r={self.r}, SE sigma="
            f"{self.sigma:.4f}, eval_repeats={self.args.eval_repeats}) ==="
        )
        print(
            f"{'M':>4} | {'floor':>11} | {'reweight':>11} | {'warm':>11} | "
            f"{'oracle':>11} | {'warm/fl':>8} | {'warm/orc':>8} | "
            f"{'move':>6} | {'cond':>9}"
        )
        for r in rows:
            mark = " *" if r.get("cond_limited") else ""
            print(
                f"{r['M']:>4} | {r['floor'][0]:>11.3e} | "
                f"{r['reweight'][0]:>11.3e} | {r['warm'][0]:>11.3e} "
                f"| {r['oracle'][0]:>11.3e} | {r['warm_over_floor']:>8.3f} | "
                f"{r['warm_over_oracle']:>8.3f} | {r['move_gain']:>6.3f} | "
                f"{r['cond_seed']['median']:>9.2e}{mark}"
            )
        if verdict.get("cond_limited_M"):
            print(
                f"  * cond(K) > {COND_LIMIT:g} -- MARKED, excluded from the "
                f"verdicts (M={verdict['cond_limited_M']})"
            )
        print(
            f"  WARM START < floor (median, every M): "
            f"{'YES' if verdict['warm_below_floor_every_M'] else 'NO'}"
        )
        print(
            f"  WARM START >= oracle (median, every M): "
            f"{'YES' if verdict['warm_above_oracle_every_M'] else 'no'}"
        )
        mg = verdict.get("move_gain_at_top")
        if mg is not None:
            print(
                f"  MOVE lever at M={verdict['top_readable_M']}: "
                f"reweight/warm={mg:.3f}"
                f"{'  -- A NET LOSS' if mg < 1.0 else ''}   "
                f"(phi={verdict['phi_at_top']:.4f})"
            )

    # ---- restore + render ----------------------------------------------
    def load_checkpoint(self):
        ckpt_path = os.path.join(
            checkpointsdir(self.args.experiment), "checkpoint.pth"
        )
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint does not exist: {ckpt_path}")
        self.ckpt = torch.load(ckpt_path, weights_only=False)
        self.results = self.ckpt["results"]

    def visualize(self):
        """Render the MMD-vs-M panel + the recon/UQ triplet (vector PDF + png)."""
        a = self.args
        out = plotsdir(a.experiment)
        res = self.results
        rows = res["rows"]
        self._print_table(rows, res["verdict"])

        with plt.rc_context(
            {"pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
             "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9}
        ):
            # ---- (1) MMD vs M (warm / floor / oracle) -------------------
            fig, axm = plt.subplots(1, 1, figsize=(4.6, 3.6))
            m_vals = [r["M"] for r in rows]
            for arm in ("floor", "warm", "oracle"):
                med = [r[arm][0] for r in rows]
                lo = [r[arm][1] for r in rows]
                hi = [r[arm][2] for r in rows]
                yerr = [
                    [m - l for m, l in zip(med, lo)],
                    [h - m for m, h in zip(med, hi)],
                ]
                axm.errorbar(
                    m_vals, med, yerr=yerr, color=_PALETTE[arm], marker="o",
                    markersize=4, linewidth=1.5, capsize=2.5, label=_LABELS[arm],
                )
            axm.set_xscale("log", base=2)
            axm.set_yscale("log")
            axm.set_xticks(m_vals)
            axm.set_xticklabels([str(m) for m in m_vals])
            axm.set_xlabel(r"node budget $M$")
            axm.set_ylabel(r"$\mathrm{MMD}^2(Q, \rho)$")
            axm.spines["top"].set_visible(False)
            axm.spines["right"].set_visible(False)
            axm.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                path = os.path.join(out, f"dq_tomography_mmd_vs_M.{ext}")
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(f"Saved to {path}")
            plt.close(fig)

            # ---- (2) recon + UQ triplet (true / quad-mean / per-pixel UQ)
            # The true and quadrature-mean panels share one symmetric
            # diverging (RdBu_r) range so the recon is verifiable against the
            # truth at a glance; the per-pixel UQ panel carries its own
            # viridis scale. Both scales are shown as labelled colorbars.
            recon = res["recon"]
            n = recon["n_img"]
            cases = recon["cases"]
            ncase = len(cases)
            import numpy as np
            trues = [np.asarray(c["true"]).reshape(n, n) for c in cases]
            recs = [np.asarray(c["recon"]).reshape(n, n) for c in cases]
            uqs = [np.asarray(c["uq"]).reshape(n, n) for c in cases]
            # Shared field scale across every true/mean panel; shared UQ scale.
            field_vmax = float(max(
                max(np.abs(t).max() for t in trues),
                max(np.abs(r).max() for r in recs),
            ))
            uq_vmax = float(max(u.max() for u in uqs))
            fig, axes = plt.subplots(
                ncase, 3, figsize=(6.2, 1.95 * ncase), squeeze=False
            )
            col_titles = ["true", "quadrature mean", "per-pixel UQ"]
            im_field = im_uq = None
            for i in range(ncase):
                im_field = axes[i, 0].imshow(
                    trues[i], cmap="RdBu_r", vmin=-field_vmax, vmax=field_vmax,
                    interpolation="nearest",
                )
                axes[i, 1].imshow(
                    recs[i], cmap="RdBu_r", vmin=-field_vmax, vmax=field_vmax,
                    interpolation="nearest",
                )
                im_uq = axes[i, 2].imshow(
                    uqs[i], cmap="viridis", vmin=0.0, vmax=uq_vmax,
                    interpolation="nearest",
                )
                for j in range(3):
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                    for sp in axes[i, j].spines.values():
                        sp.set_visible(False)
                    if i == 0:
                        axes[i, j].set_title(col_titles[j], fontsize=9,
                                             color="0.25", pad=4)
            # One shared field colorbar (true + quadrature mean) and one UQ
            # colorbar, spanning the full height of their respective columns.
            cb_field = fig.colorbar(
                im_field, ax=axes[:, :2].ravel().tolist(),
                fraction=0.046, pad=0.02, aspect=24,
            )
            cb_field.set_label("field value", fontsize=8)
            cb_field.ax.tick_params(labelsize=7)
            cb_uq = fig.colorbar(
                im_uq, ax=axes[:, 2].ravel().tolist(),
                fraction=0.046, pad=0.02, aspect=24,
            )
            cb_uq.set_label("posterior std", fontsize=8)
            cb_uq.ax.tick_params(labelsize=7)
            for ext in ("pdf", "png"):
                path = os.path.join(out, f"dq_tomography_recon.{ext}")
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(f"Saved to {path}")
            plt.close(fig)


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload"],
        sequence_args_and_types=[("m_list", int)],
    )

    experiment = DQTomography(args)
    if args.phase == "train":
        experiment.train()

    experiment.load_checkpoint()
    if args.phase == "reeval":
        experiment.reeval()
    experiment.visualize()

    if args.upload:
        upload_to_cloud(args)
