"""The per-emission certification slack of the selection certificate.

WHAT THIS IS FOR.  :func:`~qfield.designed_quadrature.safeguard.select_emission`
carries a PATHWISE guarantee only where the reference's kernel mean is exact.
Where the reference is merely samplable the criterion is a plug-in estimate,
the selection can pick the wrong arm, and "never worse than the seeds"
survives only up to a slack, on an explicit high-probability event and against
a CERTIFICATION SPLIT taken independently of everything used to fit the arms.
This module computes that slack.

THE OBJECT.  For two candidate rules ``a = (z_a, w_a)`` and ``b = (z_b, w_b)``,
the witness is their difference of kernel embeddings,

    g(x) = w_a^T k(z_a, x) - w_b^T k(z_b, x),

whose scale is the discrepancy between the two rules,

    D_ab = MMD(Q_a, Q_b),
    D^2  = w_a^T K(z_a) w_a - 2 w_a^T K(z_a, z_b) w_b + w_b^T K(z_b) w_b,

and whose empirical variance over the ``L2`` certification samples is
``V_hat``.  The empirical-Bernstein slack is

    gamma_hat(delta) = 2 sqrt( 2 V_hat ln(4/delta) / L2 )
                     + (28/3) D ln(4/delta) / (L2 - 1).

For a menu of more than two arms a union bound over the pairs replaces
``ln(4/delta)`` by ``ln(12/delta)`` and reports the largest pairwise slack;
:func:`menu_slack` does exactly that.

THE HYPOTHESIS IS A CODE OBLIGATION, NOT A FORMALITY.  ``X`` must be
independent of every quantity used to fit or to score the arms.  Where the
seeds are themselves bank rows, passing that bank here would silently violate
the premise and return a number that certifies nothing.
:func:`pairwise_slack` cannot detect this; it is the caller's contract.

FREE CORRECTNESS CHECK.  At a zero-initialized head the moved arm and the
reweighted arm are bitwise identical, so ``g == 0`` pointwise, hence
``D = V_hat = gamma_hat = 0`` exactly.  Any implementation that does not
reproduce that is wrong.

Reference:
  - Maurer and Pontil, "Empirical Bernstein bounds and sample-variance
    penalization", COLT 2009 -- the inequality the slack instantiates.
"""

from __future__ import annotations

import math

import torch

from qfield.designed_quadrature.mmd import gram

__all__ = ["witness_evaluations", "pair_scale", "slack_value",
           "slack_from_witness", "pairwise_slack", "menu_slack",
           "selection_verdict", "COINCIDENCE_D"]

# Below this cross-arm RKHS distance the two rules are indistinguishable and a
# SIGN verdict would be read off round-off. :func:`selection_verdict` returns
# COINCIDENT there, and the collapsed slack is itself the certificate.
COINCIDENCE_D = 1e-4


def _cross_gram(z_a: torch.Tensor, z_b: torch.Tensor,
                sigma: float) -> torch.Tensor:
    """``K_{jl} = exp(-||z_a[j] - z_b[l]||^2 / (2 sigma^2))``, shape (m, n)."""
    d2 = torch.cdist(z_a, z_b) ** 2
    return torch.exp(-d2 / (2.0 * sigma**2))


def witness_evaluations(
    z_a: torch.Tensor, w_a: torch.Tensor,
    z_b: torch.Tensor, w_b: torch.Tensor,
    x: torch.Tensor, sigma: float,
) -> torch.Tensor:
    """The witness ``g_i = w_a^T k(z_a, X_i) - w_b^T k(z_b, X_i)``.

    Args:
        z_a: Nodes of the first rule, ``(m_a, d)``.
        w_a: Its weights, ``(m_a,)``.
        z_b: Nodes of the second rule, ``(m_b, d)``.
        w_b: Its weights, ``(m_b,)``.
        x: Certification samples, ``(L2, d)``. MUST be independent of every
            quantity used to fit or score the two rules -- see the module
            docstring.
        sigma: SE bandwidth, shared with the criterion the arms were scored on.

    Returns:
        ``(L2,)`` witness evaluations.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (L2, d), got {tuple(x.shape)}")
    if z_a.shape[-1] != x.shape[-1] or z_b.shape[-1] != x.shape[-1]:
        raise ValueError(
            "nodes and certification draws must share the ambient dimension: "
            f"z_a {tuple(z_a.shape)}, z_b {tuple(z_b.shape)}, "
            f"x {tuple(x.shape)}"
        )
    ka = _cross_gram(z_a, x, sigma)          # (m_a, L2)
    kb = _cross_gram(z_b, x, sigma)          # (m_b, L2)
    return w_a @ ka - w_b @ kb


def pair_scale(
    z_a: torch.Tensor, w_a: torch.Tensor,
    z_b: torch.Tensor, w_b: torch.Tensor, sigma: float,
) -> float:
    """``D = MMD(Q_a, Q_b)``, the witness's scale, in closed form.

    This is the ``D`` of the slack. It is a property of the two rules alone,
    with no certification sample entering, which is what lets the range term
    of the slack be computed before any sampling.
    """
    Ka = gram(z_a, sigma)
    Kb = gram(z_b, sigma)
    Kab = _cross_gram(z_a, z_b, sigma)
    d2 = (w_a @ Ka @ w_a) - 2.0 * (w_a @ Kab @ w_b) + (w_b @ Kb @ w_b)
    return math.sqrt(max(float(d2), 0.0))


def slack_value(
    v_hat: float, D: float, L2: int,
    delta: float = 0.05, log_numerator: float = 4.0,
) -> float:
    """The empirical-Bernstein slack, from its three sufficient statistics.

        gamma_hat(delta) = 2 sqrt( 2 V_hat ln(N/delta) / L2 )
                         + (28/3) D ln(N/delta) / (L2 - 1).

    The innermost entry point. :func:`slack_from_witness` and
    :func:`pairwise_slack` are this function reached from a witness and from
    the nodes respectively, so the formula exists ONCE. A caller that has
    stored ``(V_hat, D, L2)``, which is everything the slack depends on, can
    re-read a stored certificate at another confidence level through here,
    with no re-run and no second copy of the arithmetic.

    Args:
        v_hat: The witness's unbiased empirical variance.
        D: The pair's scale ``MMD(Q_a, Q_b)``.
        L2: Certification samples.
        delta: Failure probability.
        log_numerator: See :func:`pairwise_slack`.

    Returns:
        The slack.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta}")
    if D < 0.0:
        raise ValueError(f"D = MMD(Q_a, Q_b) must be >= 0, got {D}")
    if v_hat < 0.0:
        raise ValueError(f"the empirical variance must be >= 0, got {v_hat}")
    n = int(L2)
    if n < 2:
        raise ValueError(
            f"the empirical variance needs L2 >= 2 draws, got {n}"
        )
    ln = math.log(log_numerator / delta)
    return (2.0 * math.sqrt(2.0 * float(v_hat) * ln / n)
            + (28.0 / 3.0) * float(D) * ln / (n - 1))


def slack_from_witness(
    g: torch.Tensor, D: float,
    delta: float = 0.05, log_numerator: float = 4.0,
) -> dict:
    """The slack from a witness ALREADY evaluated on the split.

    The arithmetic of :func:`pairwise_slack`, entered one step later. A sweep
    that scores a whole menu against one split already holds the
    ``k(z_j, X_i)`` tables, and the witness is then an ``O(M L_2)``
    contraction of them with ZERO further kernel evaluations, whereas
    re-entering at the nodes would recompute the dominant cross-Gram once per
    pair instead of once per node set. Splitting the formula out keeps ONE
    implementation of the slack rather than letting such a caller fork it.

    Args:
        g: Witness evaluations ``g_i = h_{Q_a}(X_i) - h_{Q_b}(X_i)``,
            shape ``(L2,)``, over samples independent of the fit.
        D: The pair's scale ``MMD(Q_a, Q_b)`` from :func:`pair_scale`. It is a
            property of the two rules alone, so it is NOT recoverable from
            ``g`` and must be passed.
        delta: Failure probability.
        log_numerator: See :func:`pairwise_slack`.

    Returns:
        ``{gamma_hat, D, V_hat, g_bar, L2, delta}``.
    """
    n = int(g.numel())
    if n < 2:
        raise ValueError(
            f"the empirical variance needs L2 >= 2 draws, got {n}"
        )
    v_hat = float(g.var(unbiased=True))
    gamma = slack_value(v_hat, D, n, delta=delta,
                        log_numerator=log_numerator)
    return {"gamma_hat": gamma, "D": float(D), "V_hat": v_hat,
            "g_bar": float(g.mean()), "L2": n, "delta": delta}


def pairwise_slack(
    z_a: torch.Tensor, w_a: torch.Tensor,
    z_b: torch.Tensor, w_b: torch.Tensor,
    x: torch.Tensor, sigma: float,
    delta: float = 0.05, log_numerator: float = 4.0,
) -> dict:
    """The empirical-Bernstein slack for one pair.

    Args:
        z_a, w_a, z_b, w_b: The two candidate rules.
        x: Certification samples ``(L2, d)``, independent of the fit.
        sigma: SE bandwidth.
        delta: Failure probability.
        log_numerator: ``4``, the numerator of the slack's logarithm. The menu
            union bound's ``ln(12/delta)`` is this numerator with ``delta``
            replaced by ``delta/3``: the split and the numerator are the SAME
            union bound written two ways, so a caller passes one or the other,
            never both.
            :func:`menu_slack` sets this for you -- do not hand-tune it.

    Returns:
        ``{gamma_hat, D, V_hat, g_bar, L2, delta}``. ``g_bar`` is the empirical
        criterion difference the selection acted on, so a caller can compare
        the observed gap against the slack that governs it.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta}")
    g = witness_evaluations(z_a, w_a, z_b, w_b, x, sigma)
    return slack_from_witness(
        g, pair_scale(z_a, w_a, z_b, w_b, sigma),
        delta=delta, log_numerator=log_numerator,
    )


def menu_slack(
    candidates, x: torch.Tensor, sigma: float, delta: float = 0.05,
) -> dict:
    """The union-bounded slack over a menu.

    Every unordered pair is certified at ``delta / n_pairs`` with the
    ``ln(12/delta)`` numerator, and the reported slack is the largest, which
    is the quantity the certificate's conclusion is stated against.

    Args:
        candidates: Sequence of ``(z, w)`` pairs, in the same order the
            criterion scored them.
        x: Certification samples, independent of the fit.
        sigma: SE bandwidth.
        delta: Total failure probability across all pairs.

    Returns:
        ``{gamma_hat, pairs, n_pairs, delta}`` where ``pairs`` maps
        ``(i, j)`` to that pair's full record.
    """
    cands = list(candidates)
    if len(cands) < 2:
        raise ValueError("a menu needs at least two candidates")
    pairs, worst = {}, 0.0
    n_pairs = len(cands) * (len(cands) - 1) // 2
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            (za, wa), (zb, wb) = cands[i], cands[j]
            # Each pair at level delta/n_pairs under the slack's own
            # numerator. The three-arm case ln(12/delta) IS ln(4/(delta/3)),
            # so passing the split AND a 12 counts the union bound twice.
            rec = pairwise_slack(
                za, wa, zb, wb, x, sigma,
                delta=delta / n_pairs,
            )
            pairs[(i, j)] = rec
            worst = max(worst, rec["gamma_hat"])
    return {"gamma_hat": worst, "pairs": pairs,
            "n_pairs": n_pairs, "delta": delta}


def selection_verdict(
    delta_hat: float, gamma_hat: float, D: float,
    coincidence_D: float = COINCIDENCE_D,
) -> dict:
    """The printed verdict for one ordered pair.

    THE VERDICT IS DIRECTIONAL. With ``delta_hat = Jhat(Q_a) - Jhat(Q_b)`` and
    lower ``J`` better, the sign carries the whole claim:

        delta_hat < -gamma_hat  ->  CERTIFIED-BETTER  (a beats b on rho)
        delta_hat > +gamma_hat  ->  CERTIFIED-WORSE   (a LOSES to b on rho)
        otherwise               ->  WITHIN-SLACK

    The theorem gives ``sign(Delta) = sign(Delta-hat)`` on the event, so both
    outer branches are certifications, of OPPOSITE claims. An undirected
    ``|delta_hat| > gamma_hat`` test would report a rule certified strictly
    worse than its comparator as a certification success, which is the failure
    mode this signature exists to make impossible.

    The COINCIDENCE GUARD overrides both: at ``D < coincidence_D`` the two
    rules are the same rule to within round-off, the slack collapses to zero,
    and that collapsed slack is itself the certificate. Such a cell prints
    COINCIDENT and must be excluded from every rate rather than counted as a
    certification, since a sign read off round-off is not a sign.

    Args:
        delta_hat: ``Jhat(Q_a) - Jhat(Q_b)`` on the certification split.
        gamma_hat: The pair's slack from :func:`slack_from_witness`.
        D: The pair's scale ``MMD(Q_a, Q_b)``.
        coincidence_D: Guard threshold; default :data:`COINCIDENCE_D`.

    Returns:
        ``{verdict, certified_better, certified_worse, coincident, ratio}``.
        ``ratio = |delta_hat| / gamma_hat`` is ``nan`` where the slack
        collapses to zero (the bitwise-tie case), never ``inf``.
    """
    coincident = bool(D < coincidence_D)
    better = worse = False
    if coincident:
        verdict = "COINCIDENT"
    elif delta_hat < -gamma_hat:
        verdict, better = "CERTIFIED-BETTER", True
    elif delta_hat > gamma_hat:
        verdict, worse = "CERTIFIED-WORSE", True
    else:
        verdict = "WITHIN-SLACK"
    return {
        "verdict": verdict, "certified_better": better,
        "certified_worse": worse, "coincident": coincident,
        "ratio": (abs(delta_hat) / gamma_hat
                  if gamma_hat > 0 else float("nan")),
    }
