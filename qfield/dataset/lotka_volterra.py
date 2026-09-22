"""Lotka--Volterra predator--prey inverse problem (Riabiz et al., 2022).

The benchmark the MCMC-compression literature reports on. The forward model
is the two-species system

    du1/dt = th1 u1 - th2 u1 u2 ,     du2/dt = th4 u1 u2 - th3 u2 ,

with u(0) = (1, 1), integrated on [0, 25] and observed at ``N_T`` equally
spaced times in both species under additive Gaussian noise of standard
deviation ``NOISE_SD``. The unknown is the log rate vector x = log th in
R^4 under a standard Gaussian prior, so th > 0 holds by construction. This
is the parameterization and the noise level of

  Riabiz, Chen, Cockayne, Swietach, Niederer, Mackey, Oates, "Optimal
  thinning of MCMC output", JRSS-B 84(4), 2022 (arXiv:2005.03952), Sec. 4.2
  and Appendix S4.3, whose own dataset is th* = (0.67, 1.33, 1, 1).

Two choices here are load-bearing.

1. **The state is integrated in logarithms.** Lotka--Volterra is
   conservative, so a trajectory cannot legitimately blow up; but the prey
   can undershoot through zero in floating point, after which the system
   diverges, sometimes silently: the solver reports success and returns a
   finite trajectory at an amplitude of 1e119. Integrating v = log u is
   positivity-preserving and cannot reach those states.

2. **Acceptance is an explicit predicate, not the solver's return code.**
   ``accept`` is what defines the family, so it must be applied identically
   in the simulator and inside any MCMC likelihood scored against it --
   otherwise the two target different measures and every comparison between
   them is void.

``laplace_posterior`` supplies the linearized posterior through forward
sensitivities. It also serves as a fixed likelihood-informed summary of the
observation, the device ``conditional.py`` exposes as ``y_feature_map`` for
conditioning on a high-dimensional y.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

T_END: float = 25.0
N_T: int = 2400
NOISE_SD: float = 0.2
D_X: int = 4
U0: tuple[float, float] = (1.0, 1.0)
RIABIZ_THETA: tuple[float, float, float, float] = (0.67, 1.33, 1.0, 1.0)
AMP_CAP: float = 1.0e4

_T_EVAL = np.linspace(0.0, T_END, N_T)


def _rhs_log(t: float, v: np.ndarray, th: np.ndarray) -> list[float]:
    """Right-hand side in log coordinates v = log u."""
    return [th[0] - th[1] * np.exp(v[1]), th[3] * np.exp(v[0]) - th[2]]


def _rhs_log_sens(t: float, s: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Log state augmented with forward sensitivities S = dv/dx.

    With th = exp(x), d th_j / d x_k = th_j delta_{jk}, so differentiating
    the log-state field gives, for each k,

        dS_{1k}/dt = th1 d_{1k} - th2 d_{2k} e^{v2} - th2 e^{v2} S_{2k} ,
        dS_{2k}/dt = th4 d_{4k} e^{v1} + th4 e^{v1} S_{1k} - th3 d_{3k} .
    """
    v, S = s[:2], s[2:].reshape(2, D_X)
    e1, e2 = np.exp(v[0]), np.exp(v[1])
    dv = np.array([th[0] - th[1] * e2, th[3] * e1 - th[2]])
    dS = np.empty((2, D_X))
    dS[0] = -th[1] * e2 * S[1]
    dS[1] = th[3] * e1 * S[0]
    dS[0, 0] += th[0]
    dS[0, 1] += -th[1] * e2
    dS[1, 3] += th[3] * e1
    dS[1, 2] += -th[2]
    return np.concatenate([dv, dS.ravel()])


def simulate(
    x: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    sensitivities: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Solve the system at log rates ``x``, in log coordinates.

    Args:
        x: Log rate vector, shape ``(4,)``.
        rtol: Relative solver tolerance.
        atol: Absolute solver tolerance.
        sensitivities: If True, also return ``du/dx``.

    Returns:
        ``(u, J)`` with ``u`` of shape ``(2, N_T)`` and ``J`` of shape
        ``(2, N_T, 4)`` or None. ``u`` is all-NaN if the solve failed.
    """
    th = np.exp(np.asarray(x, dtype=float))
    v0 = np.log(np.asarray(U0, dtype=float))
    if not sensitivities:
        sol = solve_ivp(_rhs_log, (0.0, T_END), v0, t_eval=_T_EVAL,
                        args=(th,), method="LSODA", rtol=rtol, atol=atol)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            return np.full((2, N_T), np.nan), None
        return np.exp(sol.y), None
    s0 = np.concatenate([v0, np.zeros(2 * D_X)])
    sol = solve_ivp(_rhs_log_sens, (0.0, T_END), s0, t_eval=_T_EVAL,
                    args=(th,), method="LSODA", rtol=rtol, atol=atol)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return np.full((2, N_T), np.nan), None
    u = np.exp(sol.y[:2])
    S = sol.y[2:].reshape(2, D_X, -1)
    return u, np.einsum("st,skt->stk", u, S)


def accept(u: np.ndarray, *, amp_cap: float = AMP_CAP) -> bool:
    """The family's membership predicate.

    A sample is in the family iff its trajectory is finite, positive and
    bounded by ``amp_cap``. THIS PREDICATE DEFINES THE TARGET MEASURE: any
    chain scored against a bank built with it must apply it too, or the two
    approximate different things.
    """
    return bool(np.all(np.isfinite(u)) and np.all(u > 0.0)
                and np.max(np.abs(u)) <= amp_cap)


def observe(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add the benchmark's observation noise; returns shape ``(2, N_T)``."""
    return u + NOISE_SD * rng.standard_normal(u.shape)


def laplace_posterior(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Gauss--Newton (linearized) posterior at ``x`` for observation ``y``.

    Prior N(0, I) and noise covariance ``NOISE_SD**2 I`` give precision
    ``H = J^T J / NOISE_SD**2 + I`` and mean ``x + H^{-1} J^T r /
    NOISE_SD**2`` with residual ``r = y - u``. Returns None if the solve
    or the acceptance predicate fails.
    """
    u, J = simulate(x, sensitivities=True)
    if J is None or not accept(u):
        return None
    Jf = J.reshape(-1, D_X)
    r = (np.asarray(y) - u).ravel()
    H = Jf.T @ Jf / NOISE_SD**2 + np.eye(D_X)
    cov = np.linalg.inv(H)
    return np.asarray(x, float) + cov @ (Jf.T @ r) / NOISE_SD**2, cov



SUMMARY_DIM: int = 14


def summary(y: np.ndarray) -> np.ndarray:
    """The fixed likelihood-informed summary of an observation.

    A deterministic, training-free function of ``y`` alone, so it is
    available at test time. Fourteen numbers stand in for the
    4800-dimensional trajectory: per species the log level, its spread, an
    upper and a lower quantile, the dominant Fourier mode and its share of
    the spectrum; then the cross-correlation between species and their lag.
    These are the quantities that identify a Lotka--Volterra orbit, namely
    amplitude, period and phase offset.

    Args:
        y: Observation, shape ``(2, N_T)``.

    Returns:
        Summary vector, shape ``(SUMMARY_DIM,)``.
    """
    f: list[float] = []
    for sp in range(2):
        v = np.clip(y[sp], 1e-3, None)
        lg = np.log(v)
        f += [float(lg.mean()), float(lg.std()),
              float(np.log(np.clip(y[sp].max(), 1e-3, None))),
              float(np.log(np.clip(np.percentile(y[sp], 2), 1e-3, None)))]
        z = v - v.mean()
        F = np.abs(np.fft.rfft(z))
        k = int(np.argmax(F[1:]) + 1)
        f += [float(np.log(k)), float(F[k] / (np.abs(F[1:]).sum() + 1e-9))]
    z0, z1 = y[0] - y[0].mean(), y[1] - y[1].mean()
    lag = int(np.argmax(np.correlate(z0, z1, "full")) - (len(z0) - 1))
    f += [float(np.corrcoef(y[0], y[1])[0, 1]), float(lag / len(z0))]
    return np.asarray(f)
