"""Amortized designed-quadrature warm start on a Gaussian reference (CPU).

Trains the set-equivariant architecture
(:class:`qfield.designed_quadrature.net.QuadratureAmortizer`) so that, given
``M`` i.i.d. samples of a single Gaussian ``rho``, it emits in ONE forward
pass an ``M``-node weighted quadrature -- the WARM START -- whose exact
SE-MMD^2 to ``rho`` sits BELOW the i.i.d.-``rho`` floor, for ANY ``M`` from one
trained net. At deploy time the only computation is that forward pass plus the
closed-form unit-sum weight solve.

The objective (single Gaussian ``rho``):

    min_theta  E_{z0 ~ rho, M ~ Uniform(m_list)}
               MMD^2( z0 + Delta z_theta(z0),  w_c(z0 + Delta z_theta(z0)) )

with the EXACT closed-form ``mu_fn`` / ``c_rho`` of the Gaussian target
(``target.gaussian_target``), the unit-sum-constrained weights
(``weights.constrained_weights_batched``), and the batched MMD^2
(``mmd.mmd_sq_batched``). Each Adam step uses a FRESH batch of M-sample sets;
a few thousand steps suffice in 2-D.

Evaluation, on held-out samples: for each ``M`` compare three numbers on the
same held-out sample sets --
  (i)   floor       -- (z0, 1/M): the i.i.d.-MC error of equal weights;
  (ii)  warm start  -- the trained net's emission (one forward + weight solve);
  (iii) oracle      -- per-instance ``move_descend`` on that sample set (the
                       achievable target).
The warm start should sit below the floor and approach the oracle, every ``M``.

Train:
    CUDA_VISIBLE_DEVICES="" python \
        scripts/designed_quadrature_gaussian_amortized.py \
        --experiment_name designed_quadrature_gaussian_amortized
Visualize:
    python scripts/designed_quadrature_gaussian_amortized.py \
        --experiment_name designed_quadrature_gaussian_amortized \
        --phase visualization

The ``target`` config option selects the reference ``rho`` via
``target.build_target``: ``"gaussian"`` (the default Gaussian posterior),
``"gmm"`` (the 5-mode anisotropic GMM), ``"gmm_cncv"`` (CNCV's 2-component
well-separated GMM at any ``d``), or ``"banana"``. The architecture and the
mmd / weights math are identical across targets; only ``mu_fn`` /
``c_rho`` / ``sampler`` differ.

Reference:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (the MMD-optimal designed
    quadrature; the per-instance oracle this net amortizes).
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

# CPU only: pin before importing torch. A config ``gpu_id`` other than -1 is
# rejected below.
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

from qfield.designed_quadrature import (  # noqa: E402
    build_target,
    constrained_weights_batched,
    cross_check_target,
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
from qfield.designed_quadrature.target import (  # noqa: E402
    whitened_gaussian_target,
    whitened_gmm_cncv_target,
)
from qfield.kernels import (  # noqa: E402
    median_heuristic_h_sq,
    median_heuristic_sigma_sq,
)

CONFIG_FILE = "designed_quadrature_gaussian_amortized.json"

DTYPE = torch.float64

# Cap the BLAS / intra-op pool: this is many tiny batched solves, where more
# threads only thrash.
torch.set_num_threads(4)

# Palette: i.i.d. floor grey, warm start blue, per-instance oracle red.
_PALETTE = {"floor": "#7f7f7f", "warm": "#1f77b4", "oracle": "#d62728"}
_LABELS = {
    "floor": "i.i.d. floor",
    "warm": "warm start (1 fwd pass)",
    "oracle": "per-instance optimum (from the i.i.d. seed)",
}


def _median_heuristic_sigma(
    target, n_sample: int, seed: int, *, rule: str = "svgd"
) -> float:
    """Per-target, per-``d`` median-heuristic SE bandwidth ``sigma``.

    Samples ``n_sample`` i.i.d. points from the target and returns the
    median-heuristic bandwidth under one of two rules:

    - ``rule="svgd"`` -- ``sqrt(median_heuristic_h_sq(sample))``, Liu &
      Wang's SVGD rule (median squared distance over ``2 log(N + 1)``).
      The divisor balances SVGD's repulsion against its drift; as an MMD
      ruler it makes ``sigma`` 3--4x too small AND dependent on ``N``.
      Selected by ``bandwidth = "median"``.
    - ``rule="mmd"`` -- ``sqrt(median_heuristic_sigma_sq(sample))``, the
      true median heuristic (no ``log`` divisor): a property of the target
      alone, stable in ``N``. Selected by ``bandwidth = "median_mmd"``.

    The estimate is deterministic per (target, ``d``) given ``n_sample``
    and ``seed``.

    Args:
        target: A target (its ``sampler``).
        n_sample: Reference-sample size for the median estimate.
        seed: Seed for the local generator (deterministic sigma).
        rule: ``"svgd"`` or ``"mmd"`` (the metric rule).

    Returns:
        The median-heuristic SE bandwidth ``sigma > 0``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z = target.sampler(int(n_sample), gen)
    if rule == "mmd":
        return math.sqrt(median_heuristic_sigma_sq(z))
    if rule == "svgd":
        return math.sqrt(median_heuristic_h_sq(z))
    raise ValueError(f"rule must be 'svgd' or 'mmd'; got {rule!r}")


def _ambient_gaussian_moments(target_name: str, d: int):
    """Ambient ``(mu_post, Sigma_post)`` for the Gaussian whitening metric.

    Rebuilds the EXACT Gaussian posterior ``(mu_post, Sigma_post)`` whose
    inverse covariance is the Mahalanobis whitening metric, matching
    ``target.gaussian_target`` (the LG posterior at ``d == 2``, the CNCV
    dimension-scaling posterior otherwise). The GMM uses its own
    pooled-whitening builder (``whitened_gmm_cncv_target``), so only the
    ``"gaussian"`` target is handled here.

    Args:
        target_name: Must be ``"gaussian"``.
        d: Ambient dimension.

    Returns:
        ``(mu, cov)`` float64 tensors of shapes ``(d,)`` / ``(d, d)``.
    """
    from qfield.designed_quadrature.target import (
        _cncv_gaussian_posterior,
    )
    from qfield.dataset.linear_gaussian import LinearGaussian

    name = target_name.lower()
    if name != "gaussian":
        raise ValueError(
            f"_ambient_gaussian_moments handles 'gaussian'; got {target_name!r}"
        )
    if d == 2:
        prior_mean = torch.zeros(2, dtype=DTYPE)
        prior_cov = torch.tensor([[1.5, 0.2], [0.2, 1.0]], dtype=DTYPE)
        forward = torch.tensor([[1.0, 0.3], [-0.1, 0.9]], dtype=DTYPE)
        obs_cov = torch.tensor([[0.15, 0.0], [0.0, 0.2]], dtype=DTYPE)
        problem = LinearGaussian(
            prior_mean, prior_cov, forward, obs_cov, dtype=DTYPE
        )
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(20260612)
            _, y_all = problem.sample_joint(8000)
        finally:
            torch.random.set_rng_state(rng_state)
        y_star = y_all[7500].to(DTYPE)
        mu_post, sigma_post = problem.posterior(y_star)
        return mu_post.to(DTYPE), sigma_post.to(DTYPE)
    mu_post, sigma_post, _ = _cncv_gaussian_posterior(d, 0.3, 20260612, DTYPE)
    return mu_post, sigma_post


def _draw_batch(
    target,
    M: int,
    B: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """A FRESH batch of ``B`` i.i.d. ``M``-sample sets from the target.

    Args:
        target: The Gaussian target (its ``sampler``).
        M: Node budget for this batch.
        B: Number of independent sets.
        gen: Generator (threaded for reproducibility / no global seed).

    Returns:
        Stacked seed sets, shape ``(B, M, d)``.
    """
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
    """Per-set exact SE-MMD^2 of the net's emitted quadrature.

    Forward pass -> designed nodes ``z = z0 + Delta z``; closed-form unit-sum
    weights at those nodes; exact batched MMD^2 with the Gaussian's ``mu_fn`` /
    ``c_rho``. Differentiable in the net params (autograd flows through both the
    emission and the weight solve).

    Args:
        net: The amortizer.
        z0: Seed sets, shape ``(B, M, d)``.
        target: The Gaussian target (``mu_fn`` / ``c_rho``).
        sigma: SE bandwidth.
        jitter: Tiny ridge for the weight solve.

    Returns:
        Per-set MMD^2, shape ``(B,)`` (un-clamped; the training loss is the
        mean).
    """
    B, M, d = z0.shape
    z = net(z0)  # (B, M, d)
    mu = target.mu_fn(z.reshape(B * M, d)).reshape(B, M)  # row-independent
    w = constrained_weights_batched(z, mu, sigma, jitter)
    return mmd_sq_batched(z, w, mu, target.c_rho, sigma)


class DesignedQuadratureAmortized:
    """One amortized run on the single Gaussian target."""

    def __init__(self, args):
        self.args = args
        if args.gpu_id != -1:
            raise ValueError(
                "designed_quadrature_gaussian_amortized is CPU-only (2-D toy, "
                f"tiny net); set gpu_id=-1, got {args.gpu_id}"
            )
        torch.set_default_dtype(DTYPE)
        self.d = int(args.d)
        self.m_list = list(args.m_list)

        # Resolve the SE bandwidth. The default is the per-target, per-``d``
        # MEDIAN HEURISTIC (``sigma_mode == "median"``); the fixed ``sigma``
        # from the config is an explicit override (``sigma_mode == "fixed"``).
        # The median is estimated on a sample from the target BEFORE the
        # target's ``mu`` / ``c_rho`` are built (they depend on ``sigma``), so
        # a throwaway target is built at the config ``sigma`` purely for its
        # sampler, the median sigma is picked, and the target is then rebuilt
        # at that sigma.
        self.n_ref = int(getattr(args, "n_ref", 20000))
        # L_train: the small bank the per-step MC estimand mu_fn integrates
        # over (banana only; closed-form targets ignore it). Smaller than
        # n_ref (L_eval, used by c_rho / the cross-check) so each training
        # step is O(M n_mu), not O(M n_ref).
        self.n_mu = int(getattr(args, "n_mu", 4000))
        # The ``bandwidth`` option is an alias onto the ``sigma_mode`` /
        # ``sigma`` mechanism: ``"median"`` -> median heuristic; a float ->
        # that fixed sigma. Absent, ``sigma_mode`` is used directly.
        # whiten=True builds the POSTERIOR-WHITENED (Mahalanobis) SE-kernel
        # target -- the isotropic SE kernel in the coordinate that whitens the
        # posterior -- with a d-aware bandwidth that holds the typical Gram
        # entry O(1) as d grows. Without it the isotropic median heuristic
        # collapses the Gram to K -> I in high d and node motion buys nothing.
        # The whitened target is N(R mu, I) with R the posterior whitening map;
        # the pipeline trains on the whitened nodes unchanged.
        self.whiten = bool(getattr(args, "whiten", False))
        if self.whiten:
            self._build_whitened_target(args)
            return

        bandwidth = getattr(args, "bandwidth", None)
        if bandwidth is not None and str(bandwidth).strip() != "":
            if str(bandwidth).lower() in ("median", "median_mmd"):
                args.sigma_mode = str(bandwidth).lower()
            else:
                args.sigma = float(bandwidth)
                args.sigma_mode = "fixed"
        sigma_mode = str(getattr(args, "sigma_mode", "median")).lower()
        if sigma_mode in ("median", "median_mmd"):
            n_sample = int(getattr(args, "sigma_n_sample", 4000))
            probe = build_target(
                args.target,
                args.sigma,
                d=self.d,
                dtype=DTYPE,
                n_ref=self.n_ref,
                n_mu=self.n_mu,
            )
            rule = "mmd" if sigma_mode == "median_mmd" else "svgd"
            sigma = _median_heuristic_sigma(
                probe, n_sample, int(args.base_seed) + 1234, rule=rule
            )
            print(
                f"[sigma] median heuristic/{rule} ({args.target}, d={self.d}, "
                f"N={n_sample}): sigma={sigma:.4f}  (config fixed={args.sigma})"
            )
            args.sigma = float(sigma)
        elif sigma_mode == "fixed":
            print(f"[sigma] fixed (config): sigma={args.sigma}")
        else:
            raise ValueError(
                "sigma_mode must be 'median', 'median_mmd' or 'fixed'; "
                f"got {sigma_mode!r}"
            )
        self.sigma = float(args.sigma)
        self.target = build_target(
            args.target,
            args.sigma,
            d=self.d,
            dtype=DTYPE,
            n_ref=self.n_ref,
            n_mu=self.n_mu,
        )

    def _build_whitened_target(self, args):
        """Build the posterior-whitened (Mahalanobis) SE-kernel target.

        Extracts the ambient ``(mu_post, Sigma_post)`` (the whitening metric
        ``Sigma_post^{-1}``) for the requested target, picks the WHITENED-space
        bandwidth by ``whiten_bandwidth``, and builds the whitened Gaussian
        target ``N(R mu, I)`` (the iso SE kernel in whitened coords). The
        bandwidth rules (whitened median squared distance is ``~2 d``):

          * ``"sqrt_d"`` (default) -- ``sigma = sqrt(d)``: off-diag Gram entry
            held at ``exp(-1)`` for any ``d`` (the principled constant-Gram
            rule);
          * a float ``c`` -- ``sigma = c * sqrt(d)`` (off-diag ``exp(-1/c^2)``);
          * ``"median"`` -- the standard median heuristic computed IN whitened
            coordinates; it still walls via the ``log(M+1)`` divisor.
        """
        rule = str(getattr(args, "whiten_bandwidth", "sqrt_d")).lower()
        if rule == "sqrt_d":
            sigma = math.sqrt(self.d)
        elif rule == "median":
            # Median heuristic in whitened coordinates (N(0, I)).
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(args.base_seed) + 1234)
            u = torch.randn(4000, self.d, generator=gen, dtype=DTYPE)
            sigma = math.sqrt(median_heuristic_h_sq(u))
        else:
            sigma = float(rule) * math.sqrt(self.d)
        print(
            f"[sigma] WHITENED (Mahalanobis pooled/posterior inv) target "
            f"({args.target}, d={self.d}, rule={rule}): sigma={sigma:.4f}"
        )
        args.sigma = float(sigma)
        self.sigma = float(sigma)
        name = args.target.lower()
        if name == "gaussian":
            mu_post, sigma_post = _ambient_gaussian_moments(name, self.d)
            self.target = whitened_gaussian_target(
                mu_post,
                sigma_post,
                sigma,
                name="gaussian_whitened",
                meta={"ambient_target": name},
                dtype=DTYPE,
            )
        elif name == "gmm_cncv":
            self.target = whitened_gmm_cncv_target(
                sigma, d=self.d, dtype=DTYPE
            )
        else:
            raise ValueError(
                f"whiten mode supports target 'gaussian' or 'gmm_cncv'; got "
                f"{args.target!r}"
            )

    def _verdict(self, rows: list[dict]) -> dict:
        """Below-floor / above-oracle verdicts with a bank-aware tolerance.

        For a sampled-bank target the eval metric resolves the reference
        only to its bank resolution ``(1 - c_rho)/n_ref``, and the
        eval-solved weights track the bank below that scale, so the
        warm-vs-oracle ordering there is noise rather than signal; the
        tolerance says so. Exact targets keep the strict ``1e-12``.

        A cell is also unreadable when its own seed Gram is past
        ``COND_LIMIT``: the constrained solve every arm depends on is then
        not trustworthy at float64, so the cell is MARKED and EXCLUDED from
        the verdicts rather than silently ridged.

        ``move_gain`` at the largest readable budget says what the learned
        displacement is worth once the free reweighting solve has run.
        """
        tol = 1e-12
        bank_res = None
        if getattr(self.target, "mu_fn_eval", None) is not None:
            n_ref = int(self.target.meta.get("n_ref", 1))
            bank_res = (1.0 - float(self.target.c_rho)) / n_ref
            tol = max(tol, bank_res)
        # A cell whose arms sit at or below the bank's own resolution is
        # graded against bank noise, not integration error, so it carries
        # neither verdict.
        bank_lim = [
            bank_res is not None and min(r["warm"][0], r["floor"][0]) <= bank_res
            for r in rows
        ]
        cond_lim = [bool(r.get("cond_limited")) for r in rows]
        limited = [b or c for b, c in zip(bank_lim, cond_lim)]
        readable = [r for r, lim in zip(rows, limited) if not lim]
        why = [w for w, on in
               (("the bank resolution", any(bank_lim)),
                ("cond(K)", any(cond_lim))) if on]
        top = readable[-1] if readable else None
        return {
            "bank_resolution": bank_res,
            "bank_limited_M": [r["M"] for r, lim in zip(rows, bank_lim) if lim],
            "cond_limit": COND_LIMIT,
            "cond_limited_M": [r["M"] for r, lim in zip(rows, cond_lim) if lim],
            "warm_below_floor_every_M": all(
                r["warm"][0] < r["floor"][0] for r in readable
            ),
            "warm_above_oracle_every_M": all(
                r["warm"][0] >= r["oracle"][0] - tol for r in readable
            ),
            "reweight_below_floor_every_M": all(
                r["reweight"][0] < r["floor"][0] for r in readable
            ),
            # the two-lever split, at the largest READABLE budget.
            "top_readable_M": None if top is None else top["M"],
            "move_gain_at_top": None if top is None else top["move_gain"],
            "move_is_a_loss_at_top": bool(
                top is not None and top["move_gain"] < 1.0
            ),
            "phi_at_top": None if top is None else top["phi"],
            "verdicts_scope": (
                "all budgets" if not any(limited)
                else "budgets above " + " and ".join(why) + " only"
            ),
        }

    def _build_net(self) -> QuadratureAmortizer:
        a = self.args
        net = QuadratureAmortizer(
            d=self.d,
            hidden=int(a.hidden),
            n_blocks=int(a.n_blocks),
            n_heads=int(a.n_heads),
            cond_hidden=int(a.cond_hidden),
            delta_scale=float(a.delta_scale),
        ).to(DTYPE)
        return net

    # -- compute ----------------------------------------------------------
    def train(self):
        """Train the amortizer + evaluate held-out warm-start vs floor vs oracle.

        Cross-checks the metric BEFORE training (a hard assert), trains on a
        fresh batch of ``M``-sample sets each step (``M`` uniform over the
        sweep), then evaluates on HELD-OUT fresh sample sets. Saves a JSON
        checkpoint the render reads back.
        """
        a = self.args
        print(
            f"[amortized] target={a.target}, M~Uniform{self.m_list}, "
            f"batch={a.batch_size}, steps={a.n_steps}, lr={a.lr}, "
            f"sigma={a.sigma} | net(hidden={a.hidden}, blocks={a.n_blocks}, "
            f"heads={a.n_heads}, delta_scale={a.delta_scale})"
        )

        # Metric cross-check BEFORE training (explicit MMD^2 against the
        # closed form): catches a mu / c_rho / convention bug up front.
        rel = cross_check_target(self.target, a.sigma, seed=int(a.base_seed) + 777)
        print(f"  metric cross-check (explicit vs closed form): rel={rel:.2e}")
        assert rel < 1e-8, f"metric cross-check failed: rel={rel:.2e}"

        torch.manual_seed(int(a.seed))
        net = self._build_net()
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  net params: {n_params}")
        opt = torch.optim.Adam(net.parameters(), lr=float(a.lr))

        # Training generator (never the global RNG); every step uses a FRESH
        # batch.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(a.base_seed))

        m_choices = torch.tensor(self.m_list)
        log_every = max(1, int(a.n_steps) // 20)
        curve = []  # (step, train MMD^2 EMA) for the convergence panel
        ema = None
        t0 = time.time()
        net.train()
        for step in range(int(a.n_steps)):
            # Uniform M over the sweep (one M per step keeps the batch a clean
            # tensor; the FiLM conditioning makes the net M-aware).
            M = int(m_choices[torch.randint(len(m_choices), (1,), generator=gen)])
            z0 = _draw_batch(self.target, M, int(a.batch_size), gen)
            loss = _emit_mmd_batched(
                net, z0, self.target, a.sigma, a.jitter
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

        # -- held-out evaluation ---------------------------------------------
        net.eval()
        t1 = time.time()
        eval_rows = self._evaluate(net)
        eval_secs = time.time() - t1
        print(f"  evaluation wall-time: {eval_secs:.1f} s")

        verdict = self._verdict(eval_rows)

        results = {
            "target": a.target,
            "d": self.d,
            "target_meta": self.target.meta,
            "sigma": a.sigma,
            "m_list": self.m_list,
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
            "eval_secs": eval_secs,
            "rows": eval_rows,
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
        """Held-out floor / reweight / warm-start / oracle MMD^2 per ``M``.

        For each ``M`` takes ``eval_repeats`` HELD-OUT i.i.d. sets (a generator
        seeded in a disjoint range from training: ``base_seed + 50000 + ...``),
        then on the SAME sets computes:
          floor      -- equal weights;
          reweight   -- the constrained solve at the UNTOUCHED i.i.d. nodes;
          warm start -- the net's one-forward emission + closed-form weights;
          oracle     -- per-instance ``move_descend`` (the achievable target).

        The reweight arm is the free lever: it costs one extra batched solve
        and no training, and it is the object ``phi = 1 - reweight/floor``
        profiles. Without it the split between the two levers cannot be read
        off the record. ``move_gain = reweight/warm`` is the displacement's
        OWN factor; below 1 it is a net loss.

        MEDIANS AND MEANS ARE BOTH RECORDED. The reported statistic is the
        median, but the floor's identity ``E MMD^2 = (1 - c_rho)/M`` is a
        statement about the mean. The floor is a degenerate V-statistic, so
        its distribution is a right-skewed weighted chi-square whose median
        sits strictly below its mean by a target-specific, ``M``-independent
        factor: ``M * floor[0]`` therefore MUST NOT reproduce ``1 - c_rho``.
        ``floor_mean`` is the column that identity applies to.

        Every arm is SCORED on the target's eval estimand where it has one
        (``mu_fn_eval`` -- the banana's kernel mean over the FULL reference
        bank, the same-bank companion of ``c_rho``, so the score carries no
        cross-bank additive offset), falling back to ``mu_fn`` for closed-form
        targets where the two coincide. The eval-time weights are solved from
        that same estimand (what a deployment holding this reference would
        solve from). The oracle DESCENDS on the training estimand ``mu_fn``,
        the objective the amortizer was trained against, and its endpoint is
        scored like every other arm.

        Args:
            net: The trained amortizer (eval mode).

        Returns:
            List of per-``M`` dicts.
        """
        a = self.args
        mu_eval = getattr(self.target, "mu_fn_eval", None) or self.target.mu_fn
        rows = []
        for M in self.m_list:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(a.base_seed) + 50000 + 1000 * int(M))
            z0 = _draw_batch(self.target, M, int(a.eval_repeats), gen)  # (R,M,d)
            R = z0.shape[0]

            # floor: equal weights on the SAME held-out nodes.
            mu0 = mu_eval(z0.reshape(R * M, self.d)).reshape(R, M)
            w_eq = torch.full((R, M), 1.0 / M, dtype=DTYPE)
            floor = torch.clamp(
                mmd_sq_batched(z0, w_eq, mu0, self.target.c_rho, a.sigma), min=0.0
            )

            # reweight only: the constrained solve at the UNTOUCHED nodes --
            # the free lever, no net, no descent. Same nodes, same estimand.
            with torch.no_grad():
                w_rw = constrained_weights_batched(z0, mu0, a.sigma, a.jitter)
                reweight = torch.clamp(
                    mmd_sq_batched(z0, w_rw, mu0, self.target.c_rho, a.sigma),
                    min=0.0,
                )

            # Conditioning measured on the EVAL clouds, so it describes the
            # Grams the arms above solved against rather than a fresh
            # look-alike sample.
            cond_seed = seed_gram_cond(z0, a.sigma)

            # warm start: one forward pass; weights solved from, and the rule
            # scored on, the eval estimand (no grad).
            with torch.no_grad():
                z = net(z0)
                mu_z = mu_eval(z.reshape(R * M, self.d)).reshape(R, M)
                w_z = constrained_weights_batched(z, mu_z, a.sigma, a.jitter)
                warm = torch.clamp(
                    mmd_sq_batched(z, w_z, mu_z, self.target.c_rho, a.sigma),
                    min=0.0,
                )

            # oracle: per-instance move_descend on the SAME held-out nodes
            # (descent on the training estimand), endpoint scored as above.
            zd, _ = move_descend_batched(
                z0,
                self.target.mu_fn,
                self.target.c_rho,
                a.sigma,
                float(a.oracle_lr),
                int(a.oracle_iters),
                a.jitter,
            )
            with torch.no_grad():
                zd = zd.detach()
                mu_d = mu_eval(zd.reshape(R * M, self.d)).reshape(R, M)
                w_d = constrained_weights_batched(zd, mu_d, a.sigma, a.jitter)
                oracle = torch.clamp(
                    mmd_sq_batched(zd, w_d, mu_d, self.target.c_rho, a.sigma),
                    min=0.0,
                )

            row = eval_row(M, floor, reweight, warm, oracle, cond_seed)
            rows.append(row)
            print(
                f"  [eval] M={M:>3d}: floor={row['floor'][0]:.3e} "
                f"rw={row['reweight'][0]:.3e} warm={row['warm'][0]:.3e} "
                f"oracle={row['oracle'][0]:.3e}  "
                f"(warm/floor={row['warm_over_floor']:.3f}, "
                f"warm/oracle={row['warm_over_oracle']:.3f}, "
                f"move_gain={row['move_gain']:.3f}, "
                f"cond={cond_seed['median']:.2e}"
                f"{' LIMITED' if row['cond_limited'] else ''})"
            )
        return rows

    def _print_table(self, rows: list[dict], verdict: dict):
        a = self.args
        print(
            f"\n=== {a.target} amortized warm start  (SE kernel, "
            f"sigma={a.sigma}, eval_repeats={a.eval_repeats}) ==="
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
        print(f"  verdict scope: {verdict['verdicts_scope']}")

    # -- restore ----------------------------------------------------------
    def load_checkpoint(self):
        ckpt_path = os.path.join(
            checkpointsdir(self.args.experiment), "checkpoint.pth"
        )
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint does not exist: {ckpt_path}")
        self.ckpt = torch.load(ckpt_path, weights_only=False)
        self.results = self.ckpt["results"]

    def reeval(self):
        """Re-score the SAVED net under the current eval estimand (no training).

        Rewrites ``rows`` / ``verdict`` / the eval timing in checkpoint.pth
        AND results.json; the training artifacts (curve, sigma, wall time) are
        kept verbatim. Invoke with ``--phase reeval``.

        Exact targets pass through here too, because the reweight-only arm,
        the means and the conditioning mark cannot be recovered from a stored
        row. That the old columns then reproduce is a regression test on the
        whole eval path: for an exact target the drift must be zero, and any
        non-zero value means the eval path moved under a run it was supposed
        to leave alone.
        """
        a = self.args
        exact = getattr(self.target, "mu_fn_eval", None) is None
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
        print(
            f"[reeval] max relative drift of the stored arms: {drift:.3e}"
            + ("  (exact target -- must be 0)" if exact else "")
        )
        if exact and drift > 0.0:
            raise RuntimeError(
                f"[reeval] exact target {a.target} drifted by {drift:.3e} on "
                "a re-score that must be a no-op for floor/warm/oracle. The "
                "eval path changed; fix that before overwriting the record."
            )
        self.results.update(
            rows=rows,
            verdict=verdict,
            eval_secs=eval_secs,
            target_meta=self.target.meta,
            estimand_repair="same-bank eval estimand (2026-08-06)",
        )
        self._print_table(rows, verdict)
        # Atomic replace: checkpoint.pth is the ONLY copy of the trained
        # state_dict, so never write onto it in place.
        ckpt_path = os.path.join(
            checkpointsdir(a.experiment), "checkpoint.pth"
        )
        torch.save(
            {"results": self.results, "state_dict": self.ckpt["state_dict"]},
            ckpt_path + ".tmp",
        )
        os.replace(ckpt_path + ".tmp", ckpt_path)
        rj_path = os.path.join(checkpointsdir(a.experiment), "results.json")
        with open(rj_path + ".tmp", "w") as fh:
            json.dump(self.results, fh, indent=2)
        os.replace(rj_path + ".tmp", rj_path)
        print(f"Re-evaluated and saved to {ckpt_path}")

    # -- render -----------------------------------------------------------
    def visualize(self):
        """Render the convergence panel + the warm-vs-floor-vs-oracle figure."""
        a = self.args
        rows = self.results["rows"]
        out = plotsdir(a.experiment)
        self._print_table(rows, self.results["verdict"])

        with plt.rc_context(
            {"pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False}
        ):
            fig, (axc, axm) = plt.subplots(1, 2, figsize=(9.4, 3.6))

            # (a) training convergence.
            curve = self.results["train_curve"]
            steps = [c[0] for c in curve]
            ema = [c[1] for c in curve]
            axc.plot(steps, ema, color="#1f77b4", linewidth=1.6)
            axc.set_yscale("log")
            axc.set_xlabel("training step")
            axc.set_ylabel(r"train $\mathrm{MMD}^2$ (EMA)")
            axc.set_title("convergence", fontsize=10)
            axc.spines["top"].set_visible(False)
            axc.spines["right"].set_visible(False)

            # (b) warm start vs floor vs oracle across M.
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
                    m_vals,
                    med,
                    yerr=yerr,
                    color=_PALETTE[arm],
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    capsize=2.5,
                    label=_LABELS[arm],
                )
            axm.set_xscale("log", base=2)
            axm.set_yscale("log")
            axm.set_xticks(m_vals)
            axm.set_xticklabels([str(m) for m in m_vals])
            axm.set_xlabel(r"node budget $M$")
            axm.set_ylabel(r"$\mathrm{MMD}^2(Q, \rho)$")
            axm.set_title("held-out warm start", fontsize=10)
            axm.spines["top"].set_visible(False)
            axm.spines["right"].set_visible(False)
            axm.legend(frameon=False, fontsize=9)

            fig.tight_layout()
            for ext in ("pdf", "png"):
                path = os.path.join(
                    out, f"designed_quadrature_amortized_{a.target}.{ext}"
                )
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(f"Saved to {path}")
            plt.close(fig)


def _median_iqr(vals: list[float]) -> tuple[float, float, float]:
    """``(median, q25, q75)`` over the finite entries of ``vals``."""
    t = torch.tensor(vals, dtype=DTYPE)
    t = t[torch.isfinite(t)]
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload"],
        sequence_args_and_types=[("m_list", int)],
    )

    experiment = DesignedQuadratureAmortized(args)
    if args.phase == "train":
        experiment.train()

    experiment.load_checkpoint()
    if args.phase == "reeval":
        experiment.reeval()
    experiment.visualize()

    if args.upload:
        upload_to_cloud(args)
