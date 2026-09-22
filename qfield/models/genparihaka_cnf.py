"""Conditional convolutional HINT normalizing flow ``q_phi(x | c)``.

A HINT (Hierarchical Invertible Neural Transport) flow, adapted from an
unconditional density ``p(x)`` to a conditional posterior density ``p(x | c)``
over an image or field ``x`` conditioned on a same-resolution conditioner
``c`` (for example a filtered-back-projection image, a warm-start estimate, or
any measurement summary rendered onto the image grid).

Flow family. Each HINT *tree* squeezes the input
(``[B,C,H,W] -> [B,4C,H/2,W/2]``), splits the channels in half,
processes the upper half recursively, uses it to predict the affine
coupling parameters ``(S, T)`` for the lower half
(``S = exp(clamp(s, -5, 5))``, ``x_lower <- S * x_lower + T``), then
processes the coupling-transformed lower half recursively. The recursion
bottoms out at 1x1 spatial resolution. ``n_flow_layers`` such trees are
stacked with fixed random pixel permutations between them. This is a
Glow / RealNVP-style affine-coupling flow with a hierarchical multiscale
(squeeze-recursion) tree structure -- the "HINT" of Kruse et al. and the
INN/cINN line of Ardizzone et al.

Base distribution: a standard normal in pixel space, so

    log p(x | c) = -0.5 * ||z||^2 - 0.5 * d * log(2 pi) + log|det J_phi(x; c)|,

with ``d = C * H * W`` and ``log|det J|`` the sum of ``log_S`` over all
spatial / channel locations accumulated through every tree (the pixel
permutations and the squeeze / unsqueeze are volume-preserving, so they
contribute zero log-determinant).

Conditioning mechanism. The flow is made conditional in the standard
cINN / NPE way: the conditioner ``c`` is rendered at the working
resolution and fed into every coupling subnet as extra input channels.
A small conditioner encoder produces a fixed feature map ``E(c)`` at the
full resolution; at each tree node the encoder feature is squeezed to the
node's spatial size and concatenated to the upper-half channels before the
``(S, T)`` regression. Because ``c`` enters only the coupling subnets (and
the coupling is affine in ``x`` regardless of how ``(S, T)`` depend on
``c``), the map remains exactly invertible in ``x`` for every fixed ``c``
and the log-determinant is unchanged in form: conditioning shifts the
node placement, never the tractability of the density.

Interface (inert module; an external trainer drives it):

    forward(x, c)      -> (z, log_det)                # encode  x |-> z
    inverse(z, c)      -> x                           # decode  z |-> x
    sample(c, n)       -> x                           # n samples from p(. | c)
    log_prob(x, c)     -> log p(x | c)

Architecture and hyperparameters: ``hidden_ch``, ``n_flow_layers``, the
three-conv coupling subnet with ``SiLU`` activations and a near-zero-initialized
final layer, the ``[-5, 5]`` log-scale clamp, the MLP base case at 1x1, and the
per-layer fixed random pixel permutation. The conditioner encoder and the
channel concatenation inside each subnet are what make the flow conditional.

References:
  - Kruse, Detommaso, Koethe, Scheichl, "HINT: Hierarchical Invertible
    Neural Transport for density estimation and Bayesian inference",
    AAAI 2021.
  - Ardizzone, Lueth, Kruse, Rother, Koethe, "Guided Image Generation
    with Conditional Invertible Neural Networks", arXiv:1907.02392
    (the cINN conditioning used here).
  - Dinh, Sohl-Dickstein, Bengio, "Density estimation using Real NVP"
    (affine coupling); Kingma, Dhariwal, "Glow" (squeeze multiscale).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _squeeze(x: torch.Tensor) -> torch.Tensor:
    """Volume-preserving squeeze ``[B, C, H, W] -> [B, 4C, H/2, W/2]``."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, 4 * C, H // 2, W // 2)
    return x


def _unsqueeze(x: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
    """Inverse squeeze ``[B, 4C, H/2, W/2] -> [B, C, H, W]``."""
    B, C4, Hh, Wh = x.shape
    C = C4 // 4
    x = x.reshape(B, C, 2, 2, Hh, Wh)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C, target_H, target_W)
    return x


class _ConvNet(nn.Module):
    """Three-conv coupling subnet emitting ``(log_S, T)`` for the lower half.

    Input is the upper-half channels concatenated with ``cond_ch`` channels
    of the (squeezed) conditioner feature map; output is ``2 * out_ch``
    channels split into the (raw) log-scale and the shift.
    """

    def __init__(
        self, in_ch: int, out_ch: int, cond_ch: int, hidden_ch: int = 32
    ) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch + cond_ch, hidden_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_ch, 2 * out_ch, 3, padding=1)
        nn.init.normal_(self.conv3.weight, std=0.01)
        nn.init.zeros_(self.conv3.bias)

    def forward(
        self, x_upper: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x_upper, cond], dim=1)
        h = F.silu(self.conv1(h))
        h = F.silu(self.conv2(h))
        out = self.conv3(h)
        return out[:, : self.out_ch], out[:, self.out_ch :]


class _ConvNetMLP(nn.Module):
    """MLP coupling subnet for the ``1x1`` spatial base case.

    Flattens the upper-half channels together with the (``1x1``) squeezed
    conditioner channels and regresses ``(log_S, T)`` reshaped to ``1x1``
    maps.
    """

    def __init__(
        self, in_ch: int, out_ch: int, cond_ch: int, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch + cond_ch, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_dim, 2 * out_ch)
        nn.init.normal_(self.final.weight, std=0.01)
        nn.init.zeros_(self.final.bias)

    def forward(
        self, x_upper: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x_upper.shape[0]
        h = torch.cat([x_upper.reshape(B, -1), cond.reshape(B, -1)], dim=1)
        h = self.net(h)
        out = self.final(h)
        S = out[:, : self.out_ch].reshape(B, self.out_ch, 1, 1)
        T = out[:, self.out_ch :].reshape(B, self.out_ch, 1, 1)
        return S, T


class _CondEncoder(nn.Module):
    """Conditioner encoder ``E(c)``: ``[B, C_c, H, W] -> [B, cond_ch, H, W]``.

    A shallow same-resolution conv stack producing a fixed feature map of
    the conditioner. The full-resolution feature is squeezed on the fly to
    each tree node's spatial size, so a single encoder feeds the whole
    hierarchy. Kept deliberately small (the heavy lifting is in the
    coupling subnets) and resolution-preserving so the squeeze bookkeeping
    matches the flow exactly.
    """

    def __init__(self, cond_in_ch: int, cond_ch: int, hidden_ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cond_in_ch, hidden_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, cond_ch, 3, padding=1),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


def _squeeze_to(feat: torch.Tensor, spatial_size: int) -> torch.Tensor:
    """Squeeze a full-resolution feature map down to ``spatial_size``.

    Repeated volume-preserving squeezes; channels grow by 4 per halving.
    Used to bring the encoder feature ``E(c)`` to a tree node's resolution.
    """
    while feat.shape[-1] > spatial_size:
        feat = _squeeze(feat)
    return feat


class _CondHINTTree(nn.Module):
    """Recursive squeeze-coupling tree (a single conditional HINT block)."""

    def __init__(
        self,
        in_ch: int,
        spatial_size: int,
        cond_feat_ch: int,
        full_size: int,
        hidden_ch: int = 32,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.spatial_size = spatial_size

        if spatial_size < 2 or spatial_size % 2 != 0:
            self.is_leaf = True
            return
        self.is_leaf = False

        squeezed_ch = 4 * in_ch
        squeezed_spatial = spatial_size // 2
        self.upper_ch = squeezed_ch // 2
        self.lower_ch = squeezed_ch - self.upper_ch

        # The encoder feature E(c) (cond_feat_ch at full_size) squeezed down
        # to this node's POST-squeeze spatial size gains 4x per halving.
        self.cond_node_size = squeezed_spatial
        n_halvings = int(round(math.log2(full_size / squeezed_spatial)))
        cond_ch_here = cond_feat_ch * (4 ** n_halvings)

        if squeezed_spatial <= 1:
            self.coupling = _ConvNetMLP(
                self.upper_ch, self.lower_ch, cond_ch_here, hidden_dim=hidden_ch * 2,
            )
        else:
            self.coupling = _ConvNet(
                self.upper_ch, self.lower_ch, cond_ch_here, hidden_ch=hidden_ch,
            )

        self.upper_tree = _CondHINTTree(
            self.upper_ch, squeezed_spatial, cond_feat_ch, full_size, hidden_ch,
        )
        self.lower_tree = _CondHINTTree(
            self.lower_ch, squeezed_spatial, cond_feat_ch, full_size, hidden_ch,
        )

    def _coupling_params(
        self, z_upper: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S_raw, T = self.coupling(z_upper, cond)
        log_S = torch.clamp(S_raw, -5.0, 5.0)
        return torch.exp(log_S), T, log_S

    def forward(
        self, x: torch.Tensor, cond_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_leaf:
            return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        _, _, H, W = x.shape
        x_sq = _squeeze(x)
        x_upper = x_sq[:, : self.upper_ch]
        x_lower = x_sq[:, self.upper_ch :]

        cond = _squeeze_to(cond_feat, self.cond_node_size)

        z_upper, ld_upper = self.upper_tree(x_upper, cond_feat)
        S, T, log_S = self._coupling_params(z_upper, cond)
        x_lower_coupled = S * x_lower + T
        z_lower, ld_lower = self.lower_tree(x_lower_coupled, cond_feat)

        z_sq = torch.cat([z_upper, z_lower], dim=1)
        z = _unsqueeze(z_sq, H, W)
        log_det = ld_upper + log_S.sum(dim=(1, 2, 3)) + ld_lower
        return z, log_det

    def inverse(self, z: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        if self.is_leaf:
            return z

        _, _, H, W = z.shape
        z_sq = _squeeze(z)
        z_upper = z_sq[:, : self.upper_ch]
        z_lower = z_sq[:, self.upper_ch :]

        cond = _squeeze_to(cond_feat, self.cond_node_size)

        x_upper = self.upper_tree.inverse(z_upper, cond_feat)
        S, T, _ = self._coupling_params(z_upper, cond)
        x_lower_coupled = self.lower_tree.inverse(z_lower, cond_feat)
        x_lower = (x_lower_coupled - T) / S

        x_sq = torch.cat([x_upper, x_lower], dim=1)
        return _unsqueeze(x_sq, H, W)


class ConditionalConvHINT(nn.Module):
    """Conditional HINT flow ``q_phi(x | c)`` over a square image/field.

    A stack of ``n_flow_layers`` :class:`_CondHINTTree`s with fixed random
    pixel permutations between them, all coupling subnets conditioned on a
    shared encoded conditioner feature ``E(c)``. The base distribution is a
    standard normal in pixel space.

    Args:
        in_channels: Number of channels of the modeled field ``x`` (1 for a
            single-channel image / field).
        cond_in_channels: Number of channels of the conditioner ``c`` (e.g.
            1 for an FBP image, more if several summaries are stacked).
        spatial_size: Spatial resolution of both ``x`` and ``c``. Must be a
            power of two so the squeeze recursion reaches ``1x1``.
        hidden_ch: Hidden channels inside each conv coupling subnet
            (default 32).
        n_flow_layers: Number of stacked trees (default 8).
        cond_feat_ch: Channels of the full-resolution conditioner feature
            map ``E(c)`` fed (squeezed) into every coupling subnet.

    The conditioner is assumed to live on the same spatial grid as ``x``;
    callers holding a low-dimensional measurement summary should render it
    onto the grid (broadcast / learned upsampling) before passing it in.
    """

    def __init__(
        self,
        in_channels: int = 1,
        cond_in_channels: int = 1,
        spatial_size: int = 128,
        hidden_ch: int = 32,
        n_flow_layers: int = 8,
        cond_feat_ch: int = 32,
    ) -> None:
        super().__init__()
        if (spatial_size & (spatial_size - 1)) != 0:
            raise ValueError(
                f"spatial_size={spatial_size} must be a power of two for HINT."
            )
        self.in_channels = in_channels
        self.cond_in_channels = cond_in_channels
        self.spatial_size = spatial_size
        self.n_pixels = spatial_size * spatial_size

        self.cond_encoder = _CondEncoder(cond_in_channels, cond_feat_ch, hidden_ch)

        self.trees = nn.ModuleList()
        perms, inv_perms = [], []
        for _ in range(n_flow_layers):
            self.trees.append(
                _CondHINTTree(
                    in_channels, spatial_size, cond_feat_ch, spatial_size, hidden_ch,
                )
            )
            perm = torch.randperm(self.n_pixels)
            perms.append(perm)
            inv_perms.append(torch.argsort(perm))
        self.register_buffer("perms", torch.stack(perms))
        self.register_buffer("inv_perms", torch.stack(inv_perms))

    def _permute(self, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W)[:, :, perm].reshape(B, C, H, W)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``x`` to latent ``z`` given ``c``; return ``(z, log_det)``.

        ``x``: ``[B, in_channels, S, S]``; ``c``: ``[B, cond_in_channels, S, S]``.
        ``z``: same shape as ``x``; ``log_det``: ``[B]`` (= ``log|det J_phi(x; c)|``).
        """
        cond_feat = self.cond_encoder(c)
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for k, tree in enumerate(self.trees):
            z = self._permute(z, self.perms[k])
            z, ld = tree(z, cond_feat)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode latent ``z`` back to ``x`` given ``c`` (exact inverse)."""
        cond_feat = self.cond_encoder(c)
        x = z
        for k in reversed(range(len(self.trees))):
            x = self.trees[k].inverse(x, cond_feat)
            x = self._permute(x, self.inv_perms[k])
        return x

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Exact conditional log-density ``log p(x | c)``, shape ``[B]``."""
        z, log_det = self.forward(x, c)
        d = self.in_channels * self.n_pixels
        log_pz = -0.5 * z.pow(2).sum(dim=(1, 2, 3)) - 0.5 * d * math.log(2 * math.pi)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, c: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Return ``num_samples`` posterior samples per conditioner in ``c``.

        ``c``: ``[B, cond_in_channels, S, S]``. Returns
        ``[B, num_samples, in_channels, S, S]`` (one block of samples per
        conditioner). For a single conditioner pass ``B = 1``.
        """
        self.eval()
        B = c.shape[0]
        device = c.device
        z = torch.randn(
            B * num_samples, self.in_channels, self.spatial_size, self.spatial_size,
            device=device,
        )
        c_rep = c.repeat_interleave(num_samples, dim=0)
        x = self.inverse(z, c_rep)
        return x.reshape(B, num_samples, self.in_channels,
                         self.spatial_size, self.spatial_size)
