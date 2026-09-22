"""Preconditioned Crank-Nicolson posterior sampling in a normalizing flow's latent.

THE IDEA. The learned prior is a normalizing flow over the whitened KL
coefficients, so ``xi = flow.inverse(w)`` pushes a standard Gaussian latent
``w ~ N(0, I)`` onto the prior. pCN

    w' = sqrt(1 - beta^2) w + beta xi,     xi ~ N(0, I),

leaves ``N(0, I)`` invariant EXACTLY, at any step size, so the prior cancels
from the Metropolis ratio and the acceptance probability is the likelihood
ratio alone, ``min(1, exp(Phi(w) - Phi(w')))``. The proposal is
dimension-robust (acceptance does not degrade as ``d`` grows) and the LEARNED
prior is respected by construction rather than being re-evaluated. Reference:
Cotter, Roberts, Stuart & White, "MCMC methods for functions" (Statist. Sci.
2013); Beskos et al. (2016) sec. 4.2 for the groundwater instance and the
``beta`` used here.

THE MISFIT IS A PDE SOLVE. ``Phi(w) = ||S p(exp(u(xi))) - y||^2 / (2 sigma_y^2)``
costs one elliptic solve per chain step per observation, so the chain length is
the whole cost of the method and is not to be chosen by habit --
:mod:`scripts.darcy_stuart_pcn_acf` measures the autocorrelation time and sets
the thinning from it.
"""

from __future__ import annotations

import numpy as np
import torch

from qfield.models.hint_flow import HINTFlow


def load_flow(path: str, device: str = "cpu"):
    """Load a trained HINT prior; returns ``(flow, norm_mean, norm_std)``.

    The checkpoint stores the flow in NORMALIZED coordinates, so a latent draw
    becomes a KL coefficient by ``xi = flow.inverse(w) * (std + 1e-5) + mean``.
    The epsilon is the trainer's own and is kept so the two agree bit for bit.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    flow = HINTFlow(
        ck["n_in"], n_cond=0, n_hidden=ck["n_hidden"],
        n_flow_layers=ck["n_layers"], depth=ck.get("depth"),
    )
    flow.load_state_dict(ck["model"])
    flow.to(device).eval()
    return flow, ck["norm_mean"].to(device), ck["norm_std"].to(device)


def pcn_chain(
    flow, mean, std, kl, fwd, sensors, y_obs, sigma_y, *,
    n_steps: int, n_burn: int, beta: float, thin: int, d: int | None = None,
    device: str = "cpu", seed: int | None = None, progress: bool = False,
    w0: torch.Tensor | None = None,
):
    """One pCN chain for ONE observation. Returns ``(xi (n_kept, d), acceptance)``.

    PASS ``flow=None`` FOR THE EXACT GAUSSIAN PRIOR wherever the reference's
    training truths are themselves ``xi ~ N(0, I)``: its prior IS a standard
    Gaussian by construction, and a trained flow can only approximate the
    identity there. Bypassing the flow makes the reference the genuine
    posterior of the benchmark rather than the posterior under a fitted
    stand-in, removes a checkpoint dependency, drops the per-step flow
    inverse, and recovers the original algorithm exactly, since pCN was
    formulated on a Gaussian prior. The flow path is kept for a reference
    whose prior is genuinely learned.

    This is the single-observation form. One held-out survey at a time is what
    the reference bank is built from, and batching would tie the chains'
    lengths together for no gain.

    ``mean = std = None`` MAKES THAT PRIOR EXACT. The normalization
    ``xi = base * (std + 1e-5) + mean`` exists to undo the flow trainer's own
    standardization, and its ``1e-5`` is that trainer's epsilon; with
    ``flow=None`` there is no standardization to undo, so carrying the epsilon
    anyway would sample ``N(0, (1 + 1e-5)^2 I)`` rather than ``N(0, I)``.
    Passing ``mean = std = None`` (with the dimension in ``d``) applies the
    IDENTITY instead, so the deployed prior is the benchmark's own to machine
    precision. The deviation it removes is 1e-5 relative, far below the
    Monte-Carlo error of any chain.

    Args:
        flow: the trained HINT prior (unconditional).
        mean, std: the checkpoint's normalization, applied on the way out;
            ``None`` (both) applies the identity, which requires ``d``.
        kl: the KL prior (supplies ``reconstruct``).
        fwd: the Darcy forward (supplies ``solve`` / ``observe``).
        sensors: ``(ix, iy)`` grid indices.
        y_obs: the noisy survey, shape ``(n_sensors,)``.
        sigma_y: observation noise std.
        n_steps: total proposals, INCLUDING burn-in.
        n_burn: proposals discarded before storing.
        beta: pCN step size; acceptance is insensitive to ``d`` but not to this.
        thin: store every ``thin``-th post-burn state.
        d: latent dimension; required only when ``flow`` and ``mean`` are both
            ``None`` (nothing else then carries it).
        seed: seeds BOTH the torch proposal stream and the numpy accept stream,
            so a chain is reproducible end to end.
        progress: print an acceptance running mean every 10%.
        w0: optional explicit initial latent state, shape ``(d,)`` or
            ``(1, d)``. ``None``, the default, starts from a prior sample off
            the chain's own seeded stream. A multi-chain convergence audit
            passes deliberately OVERDISPERSED starts here, for example prior
            samples scaled up, so that chain agreement after burn-in is
            evidence of mixing rather than of a shared initialization. The
            proposal and accept streams are seeded identically either way.
    """
    identity = mean is None and std is None
    if (mean is None) != (std is None):
        raise ValueError(
            "mean and std must be given together (both for a normalized flow "
            "checkpoint, neither for the exact prior); got "
            f"mean={'None' if mean is None else 'tensor'}, "
            f"std={'None' if std is None else 'tensor'}"
        )
    if flow is not None:
        d = flow.n_in
    elif not identity:
        d = int(mean.shape[-1])
    elif d is None:
        raise ValueError(
            "d is required when flow, mean and std are all None -- nothing "
            "else carries the latent dimension"
        )
    d = int(d)
    g = torch.Generator(device=device)
    rng = np.random.default_rng(seed)
    if seed is not None:
        g.manual_seed(int(seed))

    def misfit(w):
        with torch.no_grad():
            # flow=None: the prior is exactly N(0, I), so the latent IS the
            # KL coefficient and the map is the identity. The pCN proposal
            # already preserves N(0, I), so nothing else changes. ``identity``
            # additionally drops the flow trainer's (std + 1e-5) rescaling,
            # which has nothing to undo here -- see the docstring.
            base = flow.inverse(w) if flow is not None else w
            out = base if identity else base * (std + 1e-5) + mean
            xi = out.cpu().numpy()[0]
        r = fwd.observe(fwd.solve(kl.reconstruct(xi)), sensors) - y_obs
        return 0.5 * float(r @ r) / sigma_y ** 2, xi

    if w0 is None:
        w = torch.randn(1, d, device=device, generator=g)
    else:
        w = torch.as_tensor(w0, dtype=torch.get_default_dtype(),
                            device=device).reshape(1, d).clone()
    phi, xi = misfit(w)
    kept, n_acc = [], 0
    for step in range(n_steps):
        wp = (np.sqrt(1.0 - beta ** 2) * w
              + beta * torch.randn(1, d, device=device, generator=g))
        phip, xip = misfit(wp)
        # REJECT a non-finite proposal explicitly, and take the uniform FIRST
        # so the random stream is unchanged for a chain that never sees one.
        # Without the guard: ``min(0.0, nan)`` returns 0.0 in Python, so the
        # ratio is ``exp(0) = 1`` and the NaN state is accepted with
        # probability one, after which every later ratio is NaN too, the chain
        # accepts unconditionally, and it degenerates into a pure prior sample
        # that still looks like a chain. A proposal whose misfit does not
        # evaluate carries no information, and the state must still be COUNTED
        # and stored below.
        u = rng.random()
        if np.isfinite(phip) and u < np.exp(min(0.0, phi - phip)):
            w, phi, xi = wp, phip, xip
            n_acc += 1
        if step >= n_burn and (step - n_burn) % thin == 0:
            kept.append(xi.copy())
        if progress and (step + 1) % max(1, n_steps // 10) == 0:
            print(f"    pCN {step + 1}/{n_steps}  acc={n_acc / (step + 1):.3f}",
                  flush=True)
    return np.stack(kept), n_acc / n_steps
