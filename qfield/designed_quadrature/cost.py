"""Measured cost ledgers: what a construction spends, counted rather than derived.

Every construction under test is charged in three separately-reported ledgers,
because the three are different resources:

  * ``target`` -- evaluations of the target's density and score, counted as a
    pair at one point (one unit buys both). This is the currency the
    per-observation constructions spend and the amortized map spends none of.
  * ``mu`` -- evaluations of the kernel mean ``z -> E_rho[k(., z)]``, the only
    reference hook the returned quadrature reads. Exact and free for a
    closed-form reference; an exact finite sum over atoms for a
    Nadaraya--Watson conditional; a finite-bank estimate for a sampled
    reference.
  * ``draw`` -- reference samples from ``rho``.

A reference sample and a target evaluation are different physical operations,
so they are never silently added. :func:`target_equivalents` combines them at
an explicit exchange rate ``draw_rate = (cost of one reference sample) / (cost
of one target evaluation)``, which the caller sweeps rather than fixes.

An amortized map is paid for twice: once, in advance, to fit it, and then per
query. :class:`FitCost` records that one-off spend in the same three ledgers,
so it is added to the per-query spend at the same explicit exchange rates and
never as an unpriced constant. :func:`break_even_queries` then reports how many
observations a family must be integrated over before fitting once has cost less
than solving each one, as the smallest ``n`` with

    fit + n * (amortized per query)  <  n * (per-observation solve),

that is ``n > fit / (incumbent - amortized)`` when that margin is positive, and
no break-even at all when it is not. Both rates are required arguments there,
because the answer moves by orders of magnitude with them.

Counting is by points, not by calls: a hook invoked once on ``(R, M, d)`` is
charged ``R * M``, since that is the number of places the target was
interrogated. Two call patterns in this package are worth knowing before a
ledger is read:

  * :func:`~qfield.designed_quadrature.move.move_descend` calls its
    ``mu_fn`` ``2 * n_iters + RAY_KMAX + 3`` times on ``(M, d)``: once at the
    seed, once for the frozen-weight gradient and once per ray probe of the
    certificate pool, then once in the forward pass and once in the post-step
    best-iterate evaluation of every Adam step.
  * :func:`~qfield.designed_quadrature.safeguard.select_emission`
    memoizes ``mu_fn`` by ``id(z)``, so a wrapped hook there reports the
    deduplicated count, which is not comparable to an unmemoized baseline.

This generalizes ``qfield.annealed.metrics.CountingFn``, which counts
``x.shape[0]``: on a batched ``(R, M, d)`` call that is ``R``, the number of
node sets, where the number of points interrogated is ``R * M``, so it
undercounts by a factor of ``M``, the node budget. The
``(n_calls, n_points, reset)`` contract is deliberately identical so the two
read the same way.

Reference:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142, which prices its construction
    in target density and gradient evaluations per particle, the currency
    reproduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


def n_points(x: torch.Tensor) -> int:
    """Number of points a hook was interrogated at, for input ``x``.

    The trailing axis is the feature dimension, so the point count is the
    product of every leading axis: ``(M, d) -> M``, ``(R, M, d) -> R * M``,
    ``(d,) -> 1``.

    Args:
        x: Query tensor with the feature dimension last.

    Returns:
        Number of evaluation points.

    Raises:
        ValueError: If ``x`` has no dimensions at all.
    """
    if x.ndim == 0:
        raise ValueError("n_points needs at least a feature axis; got a scalar")
    if x.ndim == 1:
        return 1
    count = 1
    for s in x.shape[:-1]:
        count *= int(s)
    return count


class CountingHook:
    """Wrap a point-evaluated callable and count calls and points.

    The wrapper is transparent: it forwards ``*args`` and ``**kwargs`` and
    returns whatever the wrapped callable returns, so it drops in wherever a
    ``mu_fn``, a score, or an ``(log_v0, sigma^2 grad_log_v0)`` embedding is
    expected. Only the FIRST positional argument is measured, which is the
    query set in every hook this package defines.

    Attributes:
        n_calls: Number of invocations since construction or :meth:`reset`.
        n_points: Total points interrogated over those invocations.
    """

    def __init__(self, fn: Callable, name: str = "hook"):
        """Wrap ``fn`` under the label ``name`` (used only in ``repr``)."""
        self.fn = fn
        self.name = str(name)
        self.n_calls = 0
        self.n_points = 0

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        self.n_calls += 1
        self.n_points += n_points(x)
        return self.fn(x, *args, **kwargs)

    def reset(self) -> None:
        """Zero both counters."""
        self.n_calls = 0
        self.n_points = 0

    def __repr__(self) -> str:
        return (
            f"CountingHook({self.name}: {self.n_calls} calls, "
            f"{self.n_points} points)"
        )


class CountingSampler:
    """Wrap a ``(n, generator) -> (n, d)`` reference sampler and count draws.

    Separate from :class:`CountingHook` because a sampler is charged by the
    number of samples it is asked for, which is its first argument rather than
    the shape of a query set.

    Attributes:
        n_calls: Number of invocations since construction or :meth:`reset`.
        n_points: Total samples requested over those invocations.
    """

    def __init__(self, sampler: Callable, name: str = "sampler"):
        """Wrap ``sampler`` under the label ``name`` (used only in ``repr``)."""
        self.sampler = sampler
        self.name = str(name)
        self.n_calls = 0
        self.n_points = 0

    def __call__(self, n: int, generator: torch.Generator, *args, **kwargs):
        if int(n) < 0:
            raise ValueError(f"a sampler cannot draw {n!r} points")
        self.n_calls += 1
        self.n_points += int(n)
        return self.sampler(int(n), generator, *args, **kwargs)

    def reset(self) -> None:
        """Zero both counters."""
        self.n_calls = 0
        self.n_points = 0

    def __repr__(self) -> str:
        return (
            f"CountingSampler({self.name}: {self.n_calls} calls, "
            f"{self.n_points} draws)"
        )


@dataclass
class CostLedger:
    """The three ledgers one arm spends, kept and reported separately.

    Constructed empty and populated by wrapping whichever hooks the arm
    actually reads; an arm that never touches the density leaves ``target`` at
    zero.
    """

    target: CountingHook | None = None
    mu: CountingHook | None = None
    draw: CountingSampler | None = None
    label: str = ""
    notes: dict = field(default_factory=dict)

    def reset(self) -> None:
        """Zero every populated ledger."""
        for hook in (self.target, self.mu, self.draw):
            if hook is not None:
                hook.reset()

    def totals(self) -> dict:
        """Point totals per ledger, with ``0`` for an unpopulated one.

        Returns:
            ``{n_target, n_mu, n_draw, calls_target, calls_mu, calls_draw}``.
        """

        def _pts(hook) -> int:
            return 0 if hook is None else int(hook.n_points)

        def _calls(hook) -> int:
            return 0 if hook is None else int(hook.n_calls)

        return {
            "n_target": _pts(self.target),
            "n_mu": _pts(self.mu),
            "n_draw": _pts(self.draw),
            "calls_target": _calls(self.target),
            "calls_mu": _calls(self.mu),
            "calls_draw": _calls(self.draw),
        }


@dataclass
class FitCost:
    """The one-off spend of fitting an amortized map, in the same three ledgers.

    Kept apart from :class:`CostLedger` because it is charged once rather than
    per query, and because its numbers usually come from a finished run rather
    than from a live counter: a fit that has already happened cannot be
    re-instrumented without re-running it. Every count therefore carries a
    :attr:`provenance` string saying where it was read from.

    Attributes:
        n_target: Density-and-score evaluations spent fitting. Zero for every
            construction in this package.
        n_mu: Kernel-mean evaluation points spent fitting.
        n_draw: Reference samples consumed building the fitting banks and the
            per-step seed sets.
        label: Human name for the fit.
        provenance: ``field -> where the number came from``. Populate it for
            every non-zero field; :meth:`check_provenance` enforces that.
        notes: Anything else the record should carry (step counts, budget
            grids, the bank size behind one kernel-mean point).
    """

    n_target: int = 0
    n_mu: int = 0
    n_draw: int = 0
    label: str = ""
    provenance: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def totals(self) -> dict:
        """Point totals per ledger, in the shape :func:`target_equivalents` reads."""
        return {
            "n_target": int(self.n_target),
            "n_mu": int(self.n_mu),
            "n_draw": int(self.n_draw),
        }

    def check_provenance(self) -> None:
        """Raise unless every non-zero count says where it came from.

        A count without a source reads as a measurement but is an assumption.
        The check is cheap and is called by :func:`break_even_queries` before
        anything is combined.

        Raises:
            ValueError: If a non-zero ledger has no provenance entry.
        """
        missing = [
            k for k in ("n_target", "n_mu", "n_draw")
            if int(getattr(self, k)) != 0 and not self.provenance.get(k)
        ]
        if missing:
            raise ValueError(
                f"FitCost({self.label!r}) has non-zero {missing} with no "
                "provenance; say where each number came from (a config field, "
                "a run's results.json, a replayed generator, a live counter)"
            )

    @classmethod
    def from_ledger(
        cls,
        ledger: CostLedger,
        label: str = "",
        provenance: dict | None = None,
        notes: dict | None = None,
    ) -> "FitCost":
        """Freeze a live :class:`CostLedger` into a fit cost.

        Wrap the trainer's hooks, run the fit, and hand the ledger here.
        Provenance defaults to naming the counter.
        """
        totals = ledger.totals()
        src = {
            k: f"measured by {type(ledger).__name__}({ledger.label or label!r})"
            for k in ("n_target", "n_mu", "n_draw")
            if totals[k] != 0
        }
        src.update(provenance or {})
        return cls(
            n_target=totals["n_target"],
            n_mu=totals["n_mu"],
            n_draw=totals["n_draw"],
            label=label or ledger.label,
            provenance=src,
            notes=dict(notes or {}),
        )


def break_even_queries(
    fit: FitCost,
    amortized: CostLedger | dict,
    incumbent: CostLedger | dict,
    draw_rate: float,
    mu_rate: float,
) -> dict:
    """Queries before fitting once beats solving each one, at stated rates.

    Cumulative cost after ``n`` observations is ``fit + n * a`` for the
    amortized map and ``n * b`` for the per-observation construction, so the
    break-even is the smallest integer ``n`` with ``fit + n a < n b``, that is
    ``n = floor(fit / (b - a)) + 1`` whenever ``b > a``. When ``b <= a`` there
    is no break-even at any ``n``, and the function says so rather than
    returning a large number.

    ``draw_rate`` and ``mu_rate`` have no defaults: the three ledgers are
    different physical resources and the answer moves by orders of magnitude
    with the conversion, so the caller states it.

    Args:
        fit: The one-off fit cost, with its provenance.
        amortized: Per-query ledger of the amortized construction.
        incumbent: Per-query ledger of the per-observation construction.
        draw_rate: Cost of one reference sample, in target-evaluation units.
        mu_rate: Cost of one kernel-mean evaluation POINT, in the same units.
            On a sampled reference one such point costs a pass over the
            estimand bank, so this is where that bank is priced.

    Returns:
        ``{fit, amortized_per_query, incumbent_per_query, margin_per_query,
        n_break_even, rates, reason}``. ``n_break_even`` is ``None`` exactly
        when ``reason`` is set.

    Raises:
        ValueError: If a rate is negative, or the fit cost carries an
            unsourced non-zero count.
    """
    fit.check_provenance()
    f = target_equivalents(fit, draw_rate=draw_rate, mu_rate=mu_rate)
    a = target_equivalents(amortized, draw_rate=draw_rate, mu_rate=mu_rate)
    b = target_equivalents(incumbent, draw_rate=draw_rate, mu_rate=mu_rate)
    margin = b - a
    out = {
        "fit": f,
        "amortized_per_query": a,
        "incumbent_per_query": b,
        "margin_per_query": margin,
        "rates": {"draw_rate": float(draw_rate), "mu_rate": float(mu_rate)},
        "n_break_even": None,
        "reason": None,
    }
    if margin <= 0.0:
        out["reason"] = (
            "the amortized construction costs at least as much per query as "
            "the per-observation one at these rates, so the fit is never "
            "repaid"
        )
        return out
    out["n_break_even"] = int(f // margin) + 1
    return out


def break_even_curve(
    fit: FitCost,
    amortized: CostLedger | dict,
    incumbent: CostLedger | dict,
    rates: list[tuple[float, float]],
) -> list[dict]:
    """:func:`break_even_queries` over an explicit list of ``(draw, mu)`` rates.

    Each row carries its own rates, so a record can be read without the
    caller's conventions in hand.
    """
    return [
        break_even_queries(fit, amortized, incumbent, dr, mr)
        for dr, mr in rates
    ]


def target_equivalents(
    ledger: CostLedger | FitCost | dict,
    draw_rate: float = 1.0,
    mu_rate: float = 0.0,
) -> float:
    """Total cost in target-evaluation equivalents, at an explicit exchange rate.

    A reference sample and a density-plus-score evaluation are different
    operations, so the conversion is never implicit. ``draw_rate = 1`` charges
    one reference sample as one target evaluation, which is generous to any
    construction that spends the density: wherever a density is available,
    taking an independent sample costs at least one density evaluation and
    usually many.

    ``mu_rate`` defaults to zero because on a closed-form reference the kernel
    mean is exact and consumes nothing; set it only when the reference is a
    finite bank and the caller wants the bank charged.

    Args:
        ledger: A :class:`CostLedger` or the dict its :meth:`CostLedger.totals`
            returns.
        draw_rate: Cost of one reference sample, in target-evaluation units.
        mu_rate: Cost of one kernel-mean evaluation, in the same units.

    Returns:
        The combined cost as a float.

    Raises:
        ValueError: If either rate is negative.
    """
    if draw_rate < 0.0 or mu_rate < 0.0:
        raise ValueError(
            f"exchange rates must be non-negative; got draw_rate={draw_rate!r}, "
            f"mu_rate={mu_rate!r}"
        )
    totals = (
        ledger.totals()
        if isinstance(ledger, (CostLedger, FitCost))
        else ledger
    )
    return (
        float(totals["n_target"])
        + float(draw_rate) * float(totals["n_draw"])
        + float(mu_rate) * float(totals["n_mu"])
    )
