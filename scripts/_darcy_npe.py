"""Load the trained Darcy NPE and sample from it.

A sibling module (not a package), imported the way ``_ckpt.py`` and
``_figstyle.py`` are: ``sys.path.insert(0, dirname(abspath(__file__)))`` then
``from _darcy_npe import ...``.

The reference measure of the Darcy experiment is a trained flow together with
the ``y`` standardization it was fitted under. The two must travel together or
the reference is silently wrong: a flow conditioned on ``y`` scaled by a
different ``(mean, std)`` than it was trained with raises no error, emits
plausible samples, and stops being the posterior. The checkpoint's own copies
are therefore the single source of truth, and :func:`load_npe` is the only way
in.

The architecture is likewise rebuilt from the stored config rather than from
loose flat fields, so a config change cannot leave a caller constructing a net
that merely happens to match.
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import os

import torch

from qfield.models.hint_flow import HINTFlow

NPE_ROOT = os.path.join(datadir("checkpoints"), "darcy_stuart_npe")

_ARCH_KEYS = ("n_hidden", "n_flow_layers", "depth", "n_mlp_layers")


def load_npe(path: str, device: str = "cpu"):
    """Return ``(net, y_mean, y_std, cfg)`` for a trained Darcy NPE.

    The net is in ``eval`` mode on ``device``; ``y_mean`` / ``y_std`` are the
    exact per-sensor standardization the flow was trained under, taken from
    the checkpoint and never recomputed.

    Args:
        path: Checkpoint file, or a run directory containing ``best.pth``.
        device: Torch device string.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "best.pth")
    if not os.path.isfile(path):
        raise ValueError(f"Checkpoint does not exist: {path}")
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    missing = [k for k in _ARCH_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"{path} predates the current checkpoint contract: its config is "
            f"missing {missing}. Retrain rather than guessing the "
            "architecture -- a wrong guess loads without error and samples a "
            "different distribution."
        )
    net = HINTFlow(
        int(ck["n_in"]), n_cond=int(ck["n_cond"]),
        **{k: int(cfg[k]) for k in _ARCH_KEYS},
    ).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck["y_mean"].to(device), ck["y_std"].to(device), cfg


def standardize(y_raw: torch.Tensor, y_mean: torch.Tensor,
                y_std: torch.Tensor) -> torch.Tensor:
    """Put raw observations into the conditioner scale the flow expects."""
    return (y_raw - y_mean) / y_std


def sample_batched(net, cond: torch.Tensor, n_draw: int, *,
                   chunk_obs: int = 8,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """``(n_obs, dy)`` conditioners -> ``(n_obs, n_draw, d)`` samples.

    Batched over observations, because the per-observation loop is
    launch-bound on a GPU: 0.455 s against 0.327 s batched for 256
    observations at 512 samples each. Chunking bounds the working set but not
    the total: 256 observations at 8192 samples runs in 2.1 s and peaks at
    1.88 GB even at ``chunk_obs = 8``, because the returned tensor itself is
    839 MB and chunking cannot shrink it. A caller wanting many observations
    at a large ``n_draw`` should therefore stream, calling this per
    observation group and consuming each result, rather than rely on
    ``chunk_obs`` to keep the footprint down.

    Chunking does not change the values: the latent is sampled row-major from
    one generator stream, so any ``chunk_obs`` gives bit-identical output.

    The latent is sampled through an explicit ``generator`` rather than
    ``HINTFlow.sample``'s global RNG, so a caller's reproducibility does not
    depend on how much randomness was consumed earlier in the process.

    Args:
        net: A trained :class:`HINTFlow` in eval mode.
        cond: Standardized conditioners, shape ``(n_obs, dy)``.
        n_draw: Samples per observation.
        chunk_obs: Observations per forward call.
        generator: CPU generator for the latent; ``None`` uses the global RNG.

    Returns:
        Samples, shape ``(n_obs, n_draw, d)``, on ``cond``'s device.
    """
    n_obs, out = int(cond.shape[0]), []
    with torch.no_grad():
        for i in range(0, n_obs, int(chunk_obs)):
            c = cond[i : i + int(chunk_obs)]
            b = c.shape[0]
            # Sample on CPU so the generator is device-independent, then
            # move.
            z = torch.randn(
                b * int(n_draw), net.n_in, generator=generator,
                dtype=cond.dtype,
            ).to(cond.device)
            # repeat_interleave, not repeat: row-major (obs, draw) ordering.
            c_rep = c.repeat_interleave(int(n_draw), dim=0)
            out.append(net.inverse(z, c_rep).reshape(b, int(n_draw), net.n_in))
    return torch.cat(out, dim=0)
