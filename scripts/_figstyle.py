"""Shared figure-style helpers -- matplotlib + numpy only.

A small, dependency-light module that centralizes the plotting conventions
so the figure-render scripts stay consistent:

  * ``PALETTE`` / colour constants -- the semantic method palette plus the
    ours / solve / gold accents.
  * ``apply_house_style`` -- vector-PDF rc (type-42 fonts, ASCII minus)
    plus sane default font sizes.
  * ``scatter_nodes`` -- a clean UNIFORM-size node scatter. Overlap reads
    as concentration through alpha buildup, never through marker size.
  * ``mode_images`` -- a row of n x n informed-mode heatmaps on a SHARED
    symmetric diverging scale (ticks/spines off).
  * ``mode_stars`` -- gold mode-anchor stars.
  * ``despine`` -- hide the top + right spines.

These helpers deliberately avoid any heavy dependency so they can be
imported by any render script without side effects beyond ``rcParams``.
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Colour constants.
# --------------------------------------------------------------------------
# Semantic method palette.
PALETTE = {
    "SVGD": "#2E86AB",           # blue
    "MSIP-Fredholm": "#D7263D",  # red
    "MSIP-GF": "#E0A800",        # gold
    "CBS": "#6A4C93",            # purple
    "DataDriven": "#1B998B",     # teal
}

C_OURS = "#ff7f0e"   # orange -- "ours" accent.
C_SOLVE = "#7f7f7f"  # grey -- "solve / reference" accent.
GOLD = "#E0A800"     # gold -- mode-anchor stars (matches MSIP-GF).


def apply_house_style() -> None:
    """Set ``plt.rcParams`` for editor-friendly vector PDFs + sane sizes.

    Embeds TrueType fonts (type 42) so the PDF/EPS are vector art an
    editor can open, renders the unicode minus as ASCII, and sets modest
    default font sizes.
    """
    plt.rcParams.update(
        {
            # Vector-PDF: type-42 (TrueType) fonts + ASCII minus.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            # Sane default font sizes.
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def despine(ax) -> None:
    """Hide the top and right spines of ``ax``."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def scatter_nodes(
    ax,
    xy,
    color,
    *,
    marker="o",
    s=22.0,
    alpha=0.65,
    edge="#222222",
    edge_lw=0.45,
    zorder=3,
    weights=None,
    alpha_floor=0.35,
    alpha_ceil=0.9,
):
    """Scatter quadrature nodes with a UNIFORM marker size.

    Marker AREA is the SAME (``s``) for every node -- overlap reads as
    concentration through alpha buildup, never through size.

    If ``weights`` is given, only the per-node ALPHA is modulated across a
    clamped band ``[alpha_floor, alpha_ceil]`` by the normalized |weight|;
    the size stays uniform. If ``weights is None``, a single uniform
    ``alpha`` is used for all nodes.

    Parameters
    ----------
    ax : matplotlib Axes
    xy : array-like, shape (N, 2)
        Node coordinates.
    color : colour
        Single face colour for all nodes.
    marker : str
        Marker shape (uniform across nodes).
    s : float
        UNIFORM marker area (points^2) for every node.
    alpha : float
        Uniform alpha used when ``weights is None``.
    edge, edge_lw : colour, float
        Marker edge colour + linewidth. Defaults to a thin near-black
        (``#222222``) edge, which separates the coloured markers from both
        bright (viridis-green) and dark (purple) backdrops.
    zorder : int
    weights : array-like or None, shape (N,)
        If given, |weight| modulates per-node alpha only (size stays
        uniform).
    alpha_floor, alpha_ceil : float
        Clamped band for the weight-modulated alpha.

    Returns
    -------
    matplotlib.collections.PathCollection
    """
    xy = np.asarray(xy, dtype=float)

    if weights is None:
        a = alpha
    else:
        w = np.abs(np.asarray(weights, dtype=float))
        wmax = float(w.max()) if w.size else 0.0
        if wmax > 0.0:
            wn = w / wmax  # normalized |weight| in [0, 1].
        else:
            wn = np.zeros_like(w)
        a = alpha_floor + (alpha_ceil - alpha_floor) * wn

    return ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=s,                 # UNIFORM size -- never size-by-weight.
        c=color,
        marker=marker,
        alpha=a,
        edgecolors=edge,
        linewidths=edge_lw,
        zorder=zorder,
    )


def mode_images(
    axes,
    modes,
    n,
    *,
    cmap="RdBu_r",
    symmetric=True,
    vmax=None,
    titles=None,
):
    """Render flattened length-(n*n) mode vectors as clean n x n heatmaps.

    Each row of ``modes`` is reshaped to an ``n x n`` image and drawn into the
    corresponding axis of ``axes`` with a diverging colormap, NO ticks, and all
    spines hidden. The whole row SHARES one symmetric color scale so two
    visually identical rows of informed modes read as identical (a per-image
    rescale would hide a sign or magnitude mismatch). By default the shared
    scale is the per-row ``vmax = max |mode|`` (``vmin = -vmax``); pass an
    explicit ``vmax`` to impose ONE global scale across several calls (e.g. a
    multi-row grid where a near-zero residual row must read as flat against the
    same scale as the mode rows). Modes are defined up to sign, so the caller
    should sign-align before passing them in.

    Args:
        axes: A 1-D array / list of matplotlib Axes, one per mode (its length
            must be >= ``len(modes)``).
        modes: Array-like of shape ``(k, n*n)``; each row a flattened mode.
        n: Grid side length (each mode reshapes to ``n x n``).
        cmap: Diverging colormap name (default ``"RdBu_r"``).
        symmetric: If True (default) use a shared symmetric scale; if False,
            autoscale each image independently.
        vmax: Optional explicit symmetric half-range. When given (and
            ``symmetric``), every image uses ``vmin=-vmax, vmax=+vmax`` rather
            than the per-row ``max |mode|`` -- the way to share ONE color scale
            across multiple ``mode_images`` calls. ``None`` (default) keeps the
            per-row behavior.
        titles: Optional per-column titles (length ``k``), placed above each
            image. ``None`` for no titles.

    Returns:
        list of the per-image ``AxesImage`` handles.
    """
    arr = np.asarray(modes, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != n * n:
        raise ValueError(
            f"modes must have shape (k, n*n)=(k, {n * n}); got {arr.shape}."
        )
    k = arr.shape[0]
    if len(axes) < k:
        raise ValueError(
            f"need at least {k} axes for {k} modes; got {len(axes)}."
        )
    if not symmetric:
        vmax = None
    elif vmax is None:
        vmax = float(np.abs(arr).max()) if arr.size else None
    else:
        vmax = float(vmax)
    vmin = -vmax if vmax is not None else None

    images = []
    for i in range(k):
        ax = axes[i]
        kw = {} if not symmetric else {"vmin": vmin, "vmax": vmax}
        im = ax.imshow(
            arr[i].reshape(n, n), cmap=cmap, interpolation="nearest", **kw
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if titles is not None:
            ax.set_title(titles[i], fontsize=8, color="0.25", pad=3)
        images.append(im)
    return images


def mode_stars(
    ax,
    mu,
    *,
    s=130.0,
    color=GOLD,
    edge="black",
    edge_lw=0.5,
    zorder=4,
):
    """Draw gold mode-anchor stars at the mode locations ``mu``.

    Parameters
    ----------
    ax : matplotlib Axes
    mu : array-like, shape (K, 2)
        Mode-anchor coordinates.

    Returns
    -------
    matplotlib.collections.PathCollection
    """
    mu = np.asarray(mu, dtype=float)
    return ax.scatter(
        mu[:, 0],
        mu[:, 1],
        marker="*",
        s=s,
        c=color,
        edgecolors=edge,
        linewidths=edge_lw,
        zorder=zorder,
    )
