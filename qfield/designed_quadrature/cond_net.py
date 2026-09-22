"""Observation-conditioned, set-equivariant, arbitrary-``M`` amortizer.

The y-varying twin of
:class:`qfield.designed_quadrature.net.QuadratureAmortizer`. For a genuinely
conditional posterior, such as the per-observation posterior of a trained
conditional flow ``q_phi(. | c)`` whose shape moves with the observation, the
warm start must depend on the observation and not only on the seed set. This
network keeps the set-equivariant body and the closed-form unit-sum weights
unchanged and adds one pathway: a small CNN encoder of the (image-grid)
conditioner ``c`` (for example the FBP image) emits a set of conditioner
tokens, and each particle cross-attends to those tokens, so the emitted
displacement, and hence the designed quadrature, tracks the y-varying
posterior shape.

Design (the y-varying additions; everything else is the unconditional net):

  * **Conditioner encoder + tokens.** A shallow conv stack maps the
    single-channel ``S x S`` conditioner to a small spatial feature map, which
    is flattened to a set of ``S' * S'`` tokens (channels = ``cond_token_dim``).
    The tokens carry the observation's spatial information without collapsing
    it to one vector.
  * **Per-block particle->conditioner cross-attention.** After the
    self-attention + FiLM-MLP of each block, particles cross-attend (queries =
    particles, keys/values = conditioner tokens) through a residual sublayer.
    This is the only observation-dependent coupling; with no conditioner
    tokens it reduces to the unconditional body. The cross-attention output
    projection is zero-initialized, so an untrained net ignores the
    conditioner and reproduces the floor.
  * **Set-equivariance preserved.** Cross-attention is per-particle (each
    particle attends to the same token set), so permuting the particles
    permutes the displacements identically and the emitted quadrature is
    still permutation-invariant. The conditioner pathway is shared across
    particles, so it does not break the symmetry.
  * **Arbitrary ``M`` preserved.** No layer's shape depends on ``M``
    (attention pools over the set; FiLM uses the same
    ``[log M, log spread]`` global feature), so one trained net handles every
    ``M``.

The weights are not learned: callers obtain the closed-form
unit-sum-constrained optimal weights at the emitted nodes via
:func:`qfield.designed_quadrature.weights.constrained_weights_batched`
(a differentiable solve), exactly as for the unconditional net.

Reference:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (the MMD-optimal designed
    quadrature; the per-instance optimum this net amortizes).
  - Lee et al., "Set Transformer", ICML 2019 (set + cross-attention).
  - Perez et al., "FiLM", AAAI 2018 (feature-wise conditioning).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qfield.designed_quadrature.net import _FiLM, _SetBlock, _set_features
from qfield.designed_quadrature.weights import (
    DEFAULT_JITTER,
    constrained_weights,
    constrained_weights_batched,
)


class _CondTokenEncoder(nn.Module):
    """CNN encoder ``c -> conditioner tokens``.

    Maps a single-channel ``[B, 1, S, S]`` conditioner to ``[B, S'*S', dim]``
    tokens via a strided conv stack (down to ``S'`` per side) followed by a
    1x1 projection. ``S'`` is set by ``n_down`` halvings (``S' = S / 2**n_down``)
    so the token count ``S'*S'`` is bounded regardless of ``S``.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_ch: int = 32,
        n_down: int = 3,
        cond_in_ch: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = cond_in_ch
        for _ in range(n_down):
            layers += [
                nn.Conv2d(in_ch, hidden_ch, 3, stride=2, padding=1),
                nn.GELU(),
            ]
            in_ch = hidden_ch
        layers += [nn.Conv2d(hidden_ch, token_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """``[B, 1, S, S] -> [B, S'*S', token_dim]``."""
        feat = self.net(c)  # (B, token_dim, S', S')
        B, C, H, W = feat.shape
        return feat.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)


class _VectorTokenEncoder(nn.Module):
    """MLP encoder ``y -> conditioner tokens`` for a vector observation.

    The Darcy reference conditions on a 33-sensor reading, not an image, so
    the CNN encoder's spatial prior is wrong for it. This maps ``[B, dy]``
    through a two-layer MLP to ``n_tokens`` tokens of ``token_dim`` channels.
    Several tokens rather than one vector, so the per-particle
    cross-attention, which is where the zero-init "ignore the conditioner at
    init" property lives, is reused unchanged.
    """

    def __init__(self, in_dim: int, token_dim: int, n_tokens: int = 8,
                 hidden: int = 128):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_tokens * self.token_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """``[B, dy] -> [B, n_tokens, token_dim]``."""
        return self.net(c).reshape(-1, self.n_tokens, self.token_dim)


class _CrossBlock(nn.Module):
    """A :class:`_SetBlock` plus a residual particle->token cross-attention.

    The self-attention + FiLM-MLP body is the unchanged :class:`_SetBlock`;
    this adds one residual cross-attention sublayer (queries = particles,
    keys / values = conditioner tokens) whose output projection is
    zero-initialized, so the conditioner is ignored at init and the floor
    default is preserved.
    """

    def __init__(
        self, hidden: int, n_heads: int, cond_dim: int, token_dim: int
    ):
        super().__init__()
        self.block = _SetBlock(hidden, n_heads, cond_dim)
        self.norm_q = nn.LayerNorm(hidden)
        self.tok_proj = nn.Linear(token_dim, hidden)
        self.cross = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.out = nn.Linear(hidden, hidden)
        # Zero-init the cross-attention output, so an untrained net ignores
        # c and reproduces the floor.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.film = _FiLM(cond_dim, hidden)

    def forward(
        self,
        h: torch.Tensor,
        cond: torch.Tensor,
        tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Set-block, then, if tokens are given, residual cross-attention.

        Args:
            h: Per-particle hidden state, shape ``(B, M, hidden)``.
            cond: Global ``[log M, log spread]`` embedding, shape
                ``(B, cond_dim)``.
            tokens: Conditioner tokens ``(B, T, token_dim)``, or ``None``
                (then the block is the unconditional body).

        Returns:
            Updated hidden state, shape ``(B, M, hidden)``.
        """
        h = self.block(h, cond)
        if tokens is None:
            return h
        q = self.norm_q(h)
        kv = self.tok_proj(tokens)  # (B, T, hidden)
        attn, _ = self.cross(q, kv, kv, need_weights=False)
        return h + self.film(self.out(attn), cond)


class ConditionalQuadratureAmortizer(nn.Module):
    """Observation-conditioned, set-equivariant, arbitrary-``M`` amortizer.

    Maps a batch of seed sets ``z0`` of shape ``(B, M, d)`` and a batch of
    conditioners ``c`` of shape ``(B, 1, S, S)`` to per-particle displacements
    ``Delta z`` of shape ``(B, M, d)``; the designed nodes are ``z = z0 +
    Delta z``. The body is :class:`_CrossBlock` stacked (self-attention +
    FiLM-MLP, then particle->token cross-attention), FiLM-conditioned on a
    small embedding of ``[log M, log spread]``; the head is zero-initialized
    so an untrained net emits ``Delta z ~ 0`` (the floor).

    The weights are not a network output: callers obtain the closed-form
    unit-sum-constrained optimal weights at the emitted nodes via
    :meth:`weights` / :meth:`weights_batched`, so the only thing trained is node
    placement.

    Args:
        d: Particle (latent) dimension.
        hidden: Per-particle hidden width.
        n_blocks: Number of equivariant cross-attention blocks.
        n_heads: Attention heads (must divide ``hidden``).
        cond_hidden: Width of the FiLM ``[log M, log spread]`` embedding.
        delta_scale: Multiplier on the emitted displacement (the head is
            zero-initialized so the initial displacement is zero regardless).
        spatial_size: Side length ``S`` of the conditioner image grid.
        cond_token_dim: Channels of each conditioner token.
        cond_hidden_ch: Hidden conv channels in the conditioner encoder.
        cond_n_down: Strided-conv halvings in the encoder (token grid ``S' =
            S / 2**cond_n_down``).
        cond_in_channels: Conditioner channels (1 for an FBP image).
    """

    def __init__(
        self,
        d: int = 7,
        hidden: int = 128,
        n_blocks: int = 4,
        n_heads: int = 8,
        cond_hidden: int = 64,
        delta_scale: float = 0.5,
        spatial_size: int = 64,
        cond_token_dim: int = 32,
        cond_hidden_ch: int = 32,
        cond_n_down: int = 3,
        cond_in_channels: int = 1,
        cond_kind: str = "image",
        cond_vec_dim: int | None = None,
        cond_n_tokens: int = 8,
    ):
        super().__init__()
        if cond_kind not in ("image", "vector"):
            raise ValueError(f"cond_kind must be image|vector, got {cond_kind!r}")
        if cond_kind == "vector" and cond_vec_dim is None:
            raise ValueError("cond_kind='vector' requires cond_vec_dim")
        self.d = d
        self.delta_scale = delta_scale
        self.spatial_size = spatial_size
        self.cond_kind = cond_kind
        self.embed = nn.Linear(d, hidden)
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, cond_hidden),
        )
        # The encoder is the only thing that changes with the conditioner's
        # type; the cross-attention pathway, whose zero-init output makes an
        # untrained net reproduce the floor, is shared.
        if cond_kind == "image":
            self.cond_encoder = _CondTokenEncoder(
                cond_token_dim, cond_hidden_ch, cond_n_down, cond_in_channels
            )
        else:
            self.cond_encoder = _VectorTokenEncoder(
                int(cond_vec_dim), cond_token_dim, cond_n_tokens,
                hidden=cond_hidden_ch * 4,
            )
        self.blocks = nn.ModuleList(
            [
                _CrossBlock(hidden, n_heads, cond_hidden, cond_token_dim)
                for _ in range(n_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, d)
        # Zero-init the head, so Delta z = 0 at init and the net starts at
        # the i.i.d. (reweight) floor.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def displacement(
        self, z0: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-particle displacement ``Delta z`` for a batch of seed sets.

        Args:
            z0: Seed sets, shape ``(B, M, d)``.
            c: Conditioners, shape ``(B, 1, S, S)``, or ``None``, in which
                case the net runs unconditional.

        Returns:
            Displacements, shape ``(B, M, d)`` (zero at init).
        """
        cond = self.cond_mlp(_set_features(z0))  # (B, cond_hidden)
        tokens = None if c is None else self.cond_encoder(c)  # (B, T, token_dim)
        h = self.embed(z0)  # (B, M, hidden)
        for block in self.blocks:
            h = block(h, cond, tokens)
        return self.delta_scale * self.head(self.norm_out(h))

    def forward(
        self, z0: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Designed nodes ``z = z0 + Delta z`` for a batch of seed sets.

        Args:
            z0: Seed sets, shape ``(B, M, d)``.
            c: Conditioners, shape ``(B, 1, S, S)``, or ``None``.

        Returns:
            Designed node sets, shape ``(B, M, d)``.
        """
        return z0 + self.displacement(z0, c)

    # -- closed-form weights at the emitted nodes (not learned) -----------
    @staticmethod
    def weights(
        z: torch.Tensor,
        mu: torch.Tensor,
        sigma: float,
        jitter: float = DEFAULT_JITTER,
    ) -> torch.Tensor:
        """Closed-form unit-sum-constrained optimal weights at one node set.

        Pass-through to
        :func:`qfield.designed_quadrature.weights.constrained_weights`.
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
        """
        return constrained_weights_batched(z, mu, sigma, jitter)
