"""Gates that test whether a candidate reference is the posterior.

Two statistics whose null distributions are known EXACTLY, together with the
null simulators that turn them into gates.

Both rest on one fact. Truths are sampled ``xi ~ N(0, I_d)`` and observations
``y = G(xi) + sigma_y eps``, so a held-out pair is a sample of the JOINT. If a
candidate ``q(. | y)`` equals the true posterior, then ``xi' ~ q(. | y)``
paired with the same ``y`` is distributed exactly as the original pair:
``(xi', y) =_d (xi, y)``. Everything below is a consequence.

  * **Calibration** (:func:`calibration`). For any FIXED direction ``v``, the
    rank of ``v . xi_true`` among ``L`` samples of ``q(. | y)`` is uniform on
    ``{0, ..., L}``, so every central credible band covers at its nominal rate.
    "Fixed" is the load-bearing word: ``v`` may depend on the operator, which is
    what makes the operator's own singular directions a legitimate choice, but
    it must NOT be selected from the sample being tested.
  * **Misfit** (:func:`chi2_misfit`). ``G(xi') - y`` is distributed exactly
    ``N(0, sigma_y^2 I_m)``, so ``chi^2 / m`` is exactly ``chi^2_m / m``.
    MARGINALLY OVER ``y``, and only marginally -- conditional on one fixed
    observation it is not, and the spread across observations is where nearly
    all of this statistic's variance lives.

Two consequences of "marginally" decide the interface. First, the statistic
must be pooled over MANY OBSERVATIONS, and extra samples within one
observation buy almost nothing -- they share that observation's single noise
realization. So :func:`chi2_misfit` takes ONE sample per observation by
default, which also makes the null exact: the per-observation values are then
i.i.d. ``chi^2_m / m`` and the null distribution of their median can be
simulated without touching the forward operator at all
(:func:`chi2_null_band`). Second, a FIXED acceptance band is not a gate. Its
power does not improve with more data and can fall, because the statistic
concentrates inside the interval as the sample grows. The band must be derived
from the null at the sample size actually used, which is what
:func:`chi2_null_band` returns.

Both gates read one-dimensional projections, so a candidate with correct
per-direction marginals and wrong correlations passes. Testing every direction
of an orthonormal basis (the default here) narrows the gap but does not close
it: these certify the marginals along a chosen basis and the data misfit, not
the full joint.

Reference:
  - Talts, Betancourt, Simpson, Vehtari & Gelman, "Validating Bayesian inference
    algorithms with simulation-based calibration" (2018) -- the rank statistic.
  - Beskos, Girolami, Lan, Farrell & Stuart (2017), sec. 4.2 -- the benchmark.
"""

from __future__ import annotations

import numpy as np
import torch


def calibration(draws: torch.Tensor, xi_true: torch.Tensor,
                dirs: np.ndarray) -> dict:
    """Simulation-based calibration along fixed directions.

    Args:
        draws: Candidate posterior draws, shape ``(n_obs, L, d)``.
        xi_true: The truths those observations were generated from, shape
            ``(n_obs, d)``.
        dirs: Directions to test, shape ``(d, n_dirs)`` (columns). MUST be
            fixed independently of ``draws``: the operator's singular vectors
            qualify, the sample covariance's eigenvectors do not.

    Returns:
        ``cov50`` / ``cov90`` (mean coverage over all directions),
        ``max_dev50`` / ``max_dev90`` (the WORST single direction's absolute
        deviation from nominal -- the mean hides a single broken direction
        among many), ``ks`` (pooled rank distance from uniform), the exact
        discrete nominal levels, and the per-direction coverage vectors.
    """
    v = torch.as_tensor(dirs, dtype=draws.dtype, device=draws.device)
    proj = draws @ v                       # (n_obs, L, n_dirs)
    truth = (xi_true @ v).unsqueeze(1)     # (n_obs, 1, n_dirs)
    lt = (proj < truth).sum(dim=1)
    eq = (proj == truth).sum(dim=1)
    # Mid-rank on ties. Float ties are rare between distinct draws, but a
    # COLLAPSED candidate produces them systematically, and the strict form
    # would bias its ranks downward -- reading as displacement rather than as
    # the collapse it is.
    lvl = int(draws.shape[1])
    rank = (lt.double() + 0.5 * eq.double()) / lvl   # (n_obs, n_dirs)

    # The rank takes L + 1 discrete values, so the EXACT nominal coverage of a
    # central band is not the continuous level. At L = 512 the correction is
    # ~0.003; at small L it is a large fraction of any sensible tolerance.
    def _nominal(level: float) -> float:
        lo, hi = (1.0 - level) / 2.0, 1.0 + (level - 1.0) / 2.0
        grid = np.arange(lvl + 1) / lvl
        return float(np.mean((grid > lo) & (grid < hi)))

    out: dict = {}
    per: dict = {}
    for level, key in ((0.50, "50"), (0.90, "90")):
        lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
        inside = ((rank > lo) & (rank < hi)).double()
        cov_j = inside.mean(dim=0).cpu().numpy()      # (n_dirs,)
        nominal = _nominal(level)
        out[f"cov{key}"] = float(cov_j.mean())
        out[f"nominal{key}"] = nominal
        out[f"max_dev{key}"] = float(np.max(np.abs(cov_j - nominal)))
        per[f"cov{key}_per_dir"] = cov_j.tolist()

    r = np.sort(rank.flatten().cpu().numpy())
    n = len(r)
    i = np.arange(1, n + 1)
    out["ks"] = float(max(np.max(i / n - r), np.max(r - (i - 1) / n)))
    out["n_pairs"] = int(n)
    out["n_dirs"] = int(v.shape[1])
    out["per_direction"] = per
    return out


def calibration_null(n_obs: int, n_dirs: int, lvl: int, level: float,
                     n_rep: int, seed: int) -> dict:
    """Null distribution of the coverage statistics, by direct simulation.

    Under the null the ranks are i.i.d. uniform on ``{0, ..., lvl}``, so no
    model, no operator and no forward solve is needed: the null is simulated
    from the rank law itself. Returns the mean coverage's standard deviation
    and the 99.5th percentile of the WORST-direction deviation, which is the
    right tolerance for :func:`calibration`'s ``max_dev`` gate.
    """
    rng = np.random.default_rng(int(seed))
    lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    mean_cov, max_dev = np.empty(n_rep), np.empty(n_rep)
    for t in range(int(n_rep)):
        r = rng.integers(0, lvl + 1, size=(n_obs, n_dirs)) / lvl
        cov_j = np.mean((r > lo) & (r < hi), axis=0)
        mean_cov[t] = cov_j.mean()
        max_dev[t] = np.max(np.abs(cov_j - cov_j.mean()))
    return {
        "mean_cov_sd": float(mean_cov.std()),
        "max_dev_p995": float(np.quantile(max_dev, 0.995)),
    }


def chi2_misfit(draws: torch.Tensor, y_obs: np.ndarray, kl, fwd, sensors,
                sigma_y: float, n_per_obs: int = 1) -> dict:
    """Median ``chi^2 / m`` of candidate draws against their observations.

    ONE sample per observation by default: the statistic is exact only
    marginally over ``y``, so its variance is dominated by the spread across
    observations and extra samples within one observation are nearly free of
    information. One sample each also makes the per-observation values i.i.d.,
    which is what lets :func:`chi2_null_band` give an EXACT null with no
    forward solves.

    Costs ``n_obs * n_per_obs`` elliptic solves.

    Args:
        draws: Candidate draws, shape ``(n_obs, L, d)``.
        y_obs: The observations, shape ``(n_obs, m)``, UNSTANDARDIZED.
        kl, fwd, sensors: the KL prior, Darcy forward and sensor indices.
        sigma_y: Observation noise standard deviation.
        n_per_obs: Draws used per observation (the first ``n_per_obs``).

    Returns:
        ``{"median", "mean", "n_obs", "n_per_obs", "m", "values"}``.
    """
    m = int(y_obs.shape[1])
    vals = []
    for o in range(int(draws.shape[0])):
        for j in range(int(n_per_obs)):
            coeff = draws[o, j].double().cpu().numpy()
            r = fwd.observe(fwd.solve(kl.reconstruct(coeff)), sensors) - y_obs[o]
            vals.append(float(r @ r) / (m * sigma_y ** 2))
    v = np.asarray(vals)
    return {
        "median": float(np.median(v)), "mean": float(v.mean()),
        "n_obs": int(draws.shape[0]), "n_per_obs": int(n_per_obs),
        "m": m, "values": v.tolist(),
    }


def chi2_null_band(n_obs: int, m: int, alpha: float = 0.01,
                   n_rep: int = 200_000, seed: int = 0) -> dict:
    """Exact null band for the median of ``n_obs`` i.i.d. ``chi^2_m / m``.

    The gate is sized to the sample. Under the null the per-observation
    misfits are i.i.d. ``chi^2_m / m`` -- exactly, by the joint-exchangeability
    argument in the module docstring -- so the null distribution of their
    median needs no model, no operator and no PDE solve, only samples of
    ``chi^2_m``. The band therefore TIGHTENS as observations are added, which
    a fixed interval does not: a fixed band's power against a too-broad
    posterior can actually FALL with more data, because the statistic
    concentrates inside it.

    Args:
        n_obs: Number of observations the statistic will be pooled over.
        m: Number of sensors.
        alpha: Two-sided false-failure rate under the null.
        n_rep: Monte-Carlo replicates for the quantiles.
        seed: RNG seed (the band is a fixed property of ``(n_obs, m, alpha)``).

    Returns:
        ``{"lo", "hi", "target", "sd"}`` -- the acceptance interval, the exact
        null median ``median(chi^2_m) / m``, and the null standard deviation.
    """
    rng = np.random.default_rng(int(seed))
    med = np.median(
        rng.chisquare(m, size=(int(n_rep), int(n_obs))) / m, axis=1
    )
    return {
        "lo": float(np.quantile(med, alpha / 2.0)),
        "hi": float(np.quantile(med, 1.0 - alpha / 2.0)),
        "target": float(np.median(rng.chisquare(m, size=2_000_000)) / m),
        "sd": float(med.std()),
    }


def information_split(draws: torch.Tensor, dirs: np.ndarray) -> list[float]:
    """Per-direction posterior standard deviation along ``dirs``.

    There is no assumption-free target for these values: they can only be
    judged against a linearized or a sampled posterior. What the profile shows
    is the qualitative signature: contraction where the operator sees, the
    prior where it does not. Each observation is centred on ITS OWN sample
    mean, so this is posterior SPREAD and not the across-observation variation
    of the means.
    """
    c = (draws - draws.mean(dim=1, keepdim=True)).reshape(-1, draws.shape[-1])
    c = c.double().cpu().numpy()
    return np.std(c @ dirs, axis=0).tolist()


def invertibility(net, cond: torch.Tensor, n: int, seed: int) -> float:
    """Max ``|xi - inverse(forward(xi))|`` for the flow, at this conditioner.

    A coupling flow's inverse divides by the coupling scale at every tree level,
    so a drifted or badly-scaled net can emit garbage from ``sample`` while its
    ``log_prob`` still looks healthy: the density is evaluated in the forward
    direction and never exercises the division. Since the reference is consumed
    by sampling, the round trip is the property that matters.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    xi = torch.randn(int(n), net.n_in, generator=g).to(
        cond.device, cond.dtype
    )
    with torch.no_grad():
        z, _ = net.forward(xi, cond[: int(n)])
        back = net.inverse(z, cond[: int(n)])
    return float((xi - back).abs().max())
