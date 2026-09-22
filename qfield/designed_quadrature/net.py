"""The set-equivariant, arbitrary-``M`` warm-start amortizer.

One network that, given a set of ``M`` independent samples ``z0 ~ rho``, emits
in a single forward pass an ``M``-node displacement ``Delta z``, so the
designed nodes are ``z = z0 + Delta z`` and their unit-sum-constrained weighted
quadrature ``Q = (z, w_c(z))`` integrates ``rho`` below the independent-sample
floor. A closed-form ``move_descend`` on the same set is the per-instance
optimum; the net learns to land near it for any ``M`` from one set of weights,
with no per-instance optimization.

The design choices that make it ``M``-free:

  * Permutation-equivariant body. Self-attention among the ``M`` particles
    (no positional encoding) is the only inter-particle coupling, followed by
    per-particle MLPs. The output ``Delta z_j`` is equivariant: permuting the
    input set permutes the displacements identically, so the emitted
    quadrature, being a set, is permutation-invariant.

  * Arbitrary ``M`` through FiLM and mean-attention. No layer's shape depends
    on ``M``: attention pools over the set, and a single global conditioning
    vector (an embedding of ``log M`` and the seed-set spread) FiLM-modulates
    every block. One trained net therefore handles every ``M``, and the
    seed-set spread feature lets it adapt the displacement scale to how tight
    the sample set is.

  * Displacement off the floor, so that doing nothing is the floor. The final
    linear layer is zero-initialized and the displacement is scaled by a small
    ``delta_scale``, so an untrained net emits ``Delta z`` near zero and the
    quadrature is essentially the constrained reweighting of the seeds. The
    net learns the correction, not the nodes.

  * Weights are not learned. The optimal unit-sum weights are the closed-form,
    differentiable
    :func:`qfield.designed_quadrature.weights.constrained_weights`
    solved at the emitted nodes (autograd flows through the solve), so the net
    only has to learn where to put the nodes; the weights are always
    MMD-optimal for those nodes. ``sum_j w_j = 1`` exactly, so constants
    integrate exactly.

For a single Gaussian ``rho`` the objective is the exact closed-form
squared-exponential MMD^2 of the emitted quadrature,
``E_{z0 ~ rho, M}[MMD^2(z0 + Delta z, w_c)]``, with ``mu_fn`` and ``c_rho`` the
target's exact kernel mean and self-affinity.

Reference:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (the MMD-optimal designed
    quadrature this net amortizes).
  - Zaheer et al., "Deep Sets", NeurIPS 2017; Lee et al., "Set Transformer",
    ICML 2019 (permutation-equivariant set architectures).
  - Perez et al., "FiLM", AAAI 2018 (feature-wise conditioning).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qfield.designed_quadrature.weights import (
    DEFAULT_JITTER,
    constrained_weights,
    constrained_weights_batched,
)


def _set_features(z0: torch.Tensor) -> torch.Tensor:
    """Global conditioning features of a batch of seed sets.

    Builds, per set, ``[log M, log mean-spread]``: the two quantities the net
    needs to pick a displacement scale that is right for any ``M`` and any
    sample-set tightness:

      * ``log M`` lets the net adapt to the node budget (a larger ``M`` wants
        finer, smaller moves);
      * the mean per-coordinate std of the set (its spread) lets it scale the
        displacement to the local geometry of the draw.

    Both are log-compressed and (cheaply) centered so the FiLM embedding sees
    well-scaled inputs. Permutation-invariant by construction (mean / std over
    the set axis).

    Args:
        z0: Seed sets, shape ``(B, M, d)``.

    Returns:
        Global features, shape ``(B, 2)``.
    """
    B, M, _ = z0.shape
    log_m = torch.full(
        (B, 1), float(torch.log(torch.tensor(float(M)))), dtype=z0.dtype, device=z0.device
    )
    # Use the biased (population) std so the spread feature is finite even at
    # M == 1, where the unbiased std divides by M - 1 and is NaN. For M >= 2
    # the value differs only by the M/(M-1) Bessel factor, which the FiLM
    # embedding absorbs.
    spread = z0.std(dim=1, unbiased=False).mean(dim=1, keepdim=True)  # (B, 1)
    log_spread = torch.log(spread.clamp_min(1e-6))
    return torch.cat([log_m, log_spread], dim=-1)  # (B, 2)


class _FiLM(nn.Module):
    """A FiLM modulator: maps a global vector to per-channel ``(gamma, beta)``.

    Produces a feature-wise affine modulation ``h -> (1 + gamma) * h + beta``
    of a per-particle hidden state from the set-global conditioning vector,
    broadcast across the ``M`` particles. ``gamma`` is added to one so the
    untrained modulation is the identity (clean "do nothing" default).
    """

    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * hidden)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.hidden = hidden

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply ``(1 + gamma) * h + beta`` with ``cond``-derived params.

        Args:
            h: Per-particle hidden state, shape ``(B, M, hidden)``.
            cond: Global conditioning vector, shape ``(B, cond_dim)``.

        Returns:
            Modulated hidden state, shape ``(B, M, hidden)``.
        """
        gamma_beta = self.lin(cond).unsqueeze(1)  # (B, 1, 2*hidden)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class _SetBlock(nn.Module):
    """One permutation-equivariant block: self-attention + FiLM-MLP.

    Multi-head self-attention couples the ``M`` particles (the only
    inter-particle interaction; no positional encoding => equivariant), then a
    FiLM-conditioned per-particle MLP. Both sublayers are residual with
    pre-norm (LayerNorm over the channel axis only, so it is per-particle and
    permutation-equivariant).
    """

    def __init__(self, hidden: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
        )
        self.film = _FiLM(cond_dim, hidden)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Residual attention + FiLM-MLP over the set.

        Args:
            h: Per-particle hidden state, shape ``(B, M, hidden)``.
            cond: Global conditioning vector, shape ``(B, cond_dim)``.

        Returns:
            Updated hidden state, shape ``(B, M, hidden)``.
        """
        a = self.norm1(h)
        attn_out, _ = self.attn(a, a, a, need_weights=False)
        h = h + attn_out
        m = self.norm2(h)
        h = h + self.film(self.mlp(m), cond)
        return h


class QuadratureAmortizer(nn.Module):
    """Set-equivariant, arbitrary-``M`` warm-start displacement network.

    Maps a batch of seed sets ``z0`` of shape ``(B, M, d)`` to per-particle
    displacements ``Delta z`` of the same shape; the designed nodes are
    ``z = z0 + Delta z``. The body is :class:`_SetBlock` stacked, FiLM-
    conditioned on a small embedding of ``[log M, log spread]``; the head is
    zero-initialized so an untrained net emits ``Delta z ~ 0`` (the floor).

    The weights are NOT a network output: callers obtain the closed-form
    unit-sum-constrained optimal weights at the emitted nodes via
    :meth:`weights` / :meth:`weights_batched` (a differentiable linear solve),
    so the only thing trained is node placement.
    """

    def __init__(
        self,
        d: int = 2,
        hidden: int = 64,
        n_blocks: int = 3,
        n_heads: int = 4,
        cond_hidden: int = 32,
        delta_scale: float = 0.5,
        budget_input: str | float | None = "log_m",
    ):
        """
        Args:
            d: Particle dimension.
            hidden: Per-particle hidden width.
            n_blocks: Number of equivariant attention blocks.
            n_heads: Attention heads (must divide ``hidden``).
            cond_hidden: Width of the FiLM conditioning embedding.
            delta_scale: Multiplier on the emitted displacement; the head is
                zero-initialized so the *initial* displacement is zero
                regardless, and this only bounds the early learning scale.
            budget_input: What the FiLM conditioning's budget column carries.
                ``"log_m"`` (the default) is the true ``log M`` of the seed
                set. ``None`` zeroes the column, removing the explicit budget
                input while keeping the spread feature; the column feeds a
                linear layer first, so a constant zero input receives exactly
                zero gradient and the column's weights stay at their
                initialization. A float clamps the column to that constant at
                every ``M``. The flag is not a parameter, so state dicts are
                interchangeable across modes.
        """
        super().__init__()
        # bool is an int subclass, so reject it explicitly: a stray
        # budget_input=True would otherwise clamp the column to 1.0.
        if isinstance(budget_input, bool) or (
            budget_input is not None
            and not isinstance(budget_input, (int, float))
            and budget_input != "log_m"
        ):
            raise ValueError(
                f"budget_input must be 'log_m', None, or a float; got {budget_input!r}"
            )
        self.budget_input = budget_input
        self.d = d
        self.delta_scale = delta_scale
        self.embed = nn.Linear(d, hidden)
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, cond_hidden),
        )
        self.blocks = nn.ModuleList(
            [_SetBlock(hidden, n_heads, cond_hidden) for _ in range(n_blocks)]
        )
        self.norm_out = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, d)
        # A zero-initialized head gives Delta z = 0 at initialization, so the
        # net starts exactly at the constrained-reweighting floor.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def displacement(self, z0: torch.Tensor) -> torch.Tensor:
        """Per-particle displacement ``Delta z`` for a batch of seed sets.

        Args:
            z0: Seed sets, shape ``(B, M, d)``.

        Returns:
            Displacements, shape ``(B, M, d)`` (zero at init).
        """
        feats = _set_features(z0)  # (B, 2): [log M, log spread]
        if self.budget_input is None:
            # The explicit budget column is zeroed and the spread kept. A zero
            # input to the first linear layer gives exactly zero gradient on
            # its column, so the signal is dead by construction.
            feats = torch.cat(
                [torch.zeros_like(feats[:, :1]), feats[:, 1:]], dim=-1
            )
        elif self.budget_input != "log_m":
            # The budget column is clamped to a constant.
            feats = torch.cat(
                [
                    torch.full_like(feats[:, :1], float(self.budget_input)),
                    feats[:, 1:],
                ],
                dim=-1,
            )
        cond = self.cond_mlp(feats)  # (B, cond_hidden)
        h = self.embed(z0)  # (B, M, hidden)
        for block in self.blocks:
            h = block(h, cond)
        return self.delta_scale * self.head(self.norm_out(h))

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        """Designed nodes ``z = z0 + Delta z`` for a batch of seed sets.

        Args:
            z0: Seed sets, shape ``(B, M, d)``.

        Returns:
            Designed node sets, shape ``(B, M, d)``.
        """
        return z0 + self.displacement(z0)

    # -- closed-form weights at the emitted nodes (not learned) -----------
    @staticmethod
    def weights(
        z: torch.Tensor,
        mu: torch.Tensor,
        sigma: float,
        jitter: float = DEFAULT_JITTER,
    ) -> torch.Tensor:
        """Closed-form unit-sum-constrained optimal weights at one node set.

        A thin pass-through to
        :func:`qfield.designed_quadrature.weights.constrained_weights`, so the
        net and the weights live behind one object.

        Args:
            z: Designed nodes, shape ``(M, d)``.
            mu: Exact kernel mean at the nodes, shape ``(M,)``.
            sigma: SE bandwidth.
            jitter: Tiny ridge for the solve.

        Returns:
            Constrained weights, shape ``(M,)``; ``sum = 1``.
        """
        return constrained_weights(z, mu, sigma, jitter)

    @staticmethod
    def weights_batched(
        z: torch.Tensor,
        mu: torch.Tensor,
        sigma: float,
        jitter: float = DEFAULT_JITTER,
    ) -> torch.Tensor:
        """Batched closed-form weights for ``(B, M, d)`` node sets.

        Pass-through to
        :func:`qfield.designed_quadrature.weights.constrained_weights_batched`.

        Args:
            z: Designed nodes, shape ``(B, M, d)``.
            mu: Exact kernel means at the nodes, shape ``(B, M)``.
            sigma: SE bandwidth.
            jitter: Tiny ridge for the solves.

        Returns:
            Constrained weights, shape ``(B, M)``; each row sums to one.
        """
        return constrained_weights_batched(z, mu, sigma, jitter)
