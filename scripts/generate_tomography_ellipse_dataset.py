"""Generate the 64x64 limited-angle tomography training dataset (random ellipses).

Uses the forward operator and FBP of
``qfield.dataset.tomography.parallel_beam_radon_operator``.

Geometry:
  * Image: 64x64 (dx = 4096).
  * Forward: 2D parallel-beam Radon (sparse projector).
  * Limited-angle with a 60-degree missing wedge: visible angles [-60, +60] deg
    (phi_deg = 60), so [60, 120] deg is unobserved.
  * 60 projection angles, 90 detector bins (dy = 60 * 90 = 5400).
  * Additive Gaussian noise at 40 dB SNR, measured on the clean sinogram, with
    sigma fixed per sample so each sample achieves 40 dB.

Phantom: a brain-like image of a random number of overlapping ellipses (count
in [5, 15]; randomized centers, semi-axes, rotation, sign and intensity), plus
an optional skull-like bright outer rim. Pixel values clipped to [0, ~1].

Per sample we store:
  * x      : the 64x64 image (the unknown).
  * y      : the noisy limited-angle sinogram, FLAT (dy,) and also reshaped on
             read as (n_angles, n_detectors).
  * c      : c = FBP(y), the filtered-back-projection reconstruction (64x64),
             the CNF conditioner.

Crash-safety: HDF5 written in batches; each batch opens in append mode, grows
the resizable datasets, writes, bumps ``n_train_written`` / ``n_test_written``,
and closes. At any instant between batches the file is closed and
self-consistent, and the run is resumable (restart with the same --out
continues from what is already written). The forward operator ``A`` is stored
once and shared.

Run (CPU, light):
    python scripts/generate_tomography_ellipse_dataset.py
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import argparse
import math
import os
import sys
import time

import numpy as np
import torch


import h5py  # noqa: E402

from qfield.dataset.tomography import (  # noqa: E402
    parallel_beam_radon_operator,
)

# --------------------------------------------------------------------------
# Setup parameters.
# --------------------------------------------------------------------------
N = 64                 # image side; dx = N*N = 4096
N_ANGLES = 60          # projection angles across the visible fan
PHI_DEG = 60.0         # half-range -> visible [-60, 60] deg, wedge [60, 120]
N_DETECTORS = 90       # detector bins per angle
SNR_DB = 40.0          # additive-Gaussian SNR on the clean sinogram


# --------------------------------------------------------------------------
# Random ellipse phantom.
# --------------------------------------------------------------------------
def random_ellipse_phantom(
    n: int, rng: np.random.Generator, skull: bool = True
) -> np.ndarray:
    """A brain-like image of a random number of overlapping ellipses.

    On the ``[-1, 1]^2`` square: an optional bright skull-like outer rim, a
    softer brain background, then a random number (``[5, 15]``) of overlapping
    ellipses with randomized centers, semi-axes, rotation, and signed
    intensities, summed in order. Values are clipped to ``[0, ~1.1]``.

    Args:
        n: Grid side (the image is ``n x n``).
        rng: NumPy random generator (drives all randomness).
        skull: If True, add a bright outer rim + softer interior (brain-like).

    Returns:
        float32 image, shape (n, n), values in [0, ~1.1].
    """
    coords = np.linspace(-1.0, 1.0, n)
    gx, gy = np.meshgrid(coords, coords)  # gx: col (x), gy: row (y)
    img = np.zeros((n, n), dtype=np.float64)

    def add_ellipse(val, cx, cy, sa, sb, ang_deg):
        t = math.radians(ang_deg)
        xr = (gx - cx) * math.cos(t) + (gy - cy) * math.sin(t)
        yr = -(gx - cx) * math.sin(t) + (gy - cy) * math.cos(t)
        mask = (xr / sa) ** 2 + (yr / sb) ** 2 <= 1.0
        img[mask] += val

    # Optional skull-like rim + brain interior (randomized but always brain-ish).
    if skull:
        # Outer bright rim: a large bright ellipse with a slightly smaller dark
        # one carved out of it, leaving a rim; then a soft brain background.
        rx = rng.uniform(0.62, 0.74)
        ry = rng.uniform(0.80, 0.92)
        rot = rng.uniform(-12.0, 12.0)
        add_ellipse(1.0, 0.0, 0.0, rx, ry, rot)                 # skull (bright)
        add_ellipse(-0.55, 0.0, 0.0, rx * 0.92, ry * 0.93, rot)  # brain (dark)

    # Random number of interior ellipses.
    n_ell = int(rng.integers(5, 16))  # 5..15 inclusive
    for _ in range(n_ell):
        val = rng.uniform(-0.45, 0.55)
        cx = rng.uniform(-0.50, 0.50)
        cy = rng.uniform(-0.55, 0.55)
        sa = rng.uniform(0.06, 0.30)
        sb = rng.uniform(0.06, 0.30)
        ang = rng.uniform(0.0, 180.0)
        add_ellipse(val, cx, cy, sa, sb, ang)

    return np.clip(img, 0.0, 1.1).astype(np.float32)


# --------------------------------------------------------------------------
# Filtered back-projection (Ram-Lak).
# --------------------------------------------------------------------------
def fbp(
    y_flat: np.ndarray,
    a_t: torch.Tensor,
    n_angles: int,
    n_detectors: int,
) -> np.ndarray:
    """FBP: ramp-filter the sinogram, then adjoint A^T. Returns flat (dx,).

    ``a_t`` is the transpose A^T (dx, dy), passed in to avoid retransposing per
    call. Standard Ram-Lak FBP up to a discretization constant.
    """
    proj = y_flat.reshape(n_angles, n_detectors)
    freqs = np.fft.fftfreq(n_detectors)
    ramp = np.abs(freqs)
    filt = np.fft.ifft(np.fft.fft(proj, axis=1) * ramp[None, :], axis=1).real
    yf = torch.from_numpy(filt.reshape(-1).astype(np.float64))
    return (a_t @ yf).numpy()


# --------------------------------------------------------------------------
# Crash-safe HDF5.
# --------------------------------------------------------------------------
def init_h5(path, a_np, angles_np, meta, batch):
    d_y, d_x = a_np.shape
    with h5py.File(path, "w") as f:
        # shared, write-once: the forward operator + angle grid.
        f.create_dataset("A", data=a_np, compression="gzip", compression_opts=4)
        f.create_dataset("angles_rad", data=angles_np)
        # per-sample, resizable, append-only (train + test splits).
        for split in ("train", "test"):
            f.create_dataset(
                f"{split}/x", shape=(0, N, N), maxshape=(None, N, N),
                chunks=(min(batch, 32), N, N), dtype="f4",
                compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                f"{split}/y", shape=(0, d_y), maxshape=(None, d_y),
                chunks=(min(batch, 64), d_y), dtype="f4",
                compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                f"{split}/c", shape=(0, N, N), maxshape=(None, N, N),
                chunks=(min(batch, 32), N, N), dtype="f4",
                compression="gzip", compression_opts=4,
            )
        f.attrs["n_train_written"] = 0
        f.attrs["n_test_written"] = 0
        for k, v in meta.items():
            f.attrs[k] = v


def append_h5(path, split, x, y, c):
    with h5py.File(path, "a") as f:
        key = f"n_{split}_written"
        n0 = int(f.attrs[key])
        n1 = n0 + len(x)
        for name, arr in ((f"{split}/x", x), (f"{split}/y", y), (f"{split}/c", c)):
            ds = f[name]
            ds.resize(n1, axis=0)
            ds[n0:n1] = arr
        f.attrs[key] = n1
        f.flush()
    return n1


def read_counts(path):
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        return (
            int(f.attrs.get("n_train_written", 0)),
            int(f.attrs.get("n_test_written", 0)),
        )


# --------------------------------------------------------------------------
# Generate one split (resumable).
# --------------------------------------------------------------------------
def generate_split(path, split, n_target, n_done, a, a_t, rng, batch, t0):
    """Generate samples [n_done, n_target) for one split, appending in batches.

    The ``rng`` must already be advanced past the first ``n_done`` samples
    (the same per-sample variates are burned on resume) so resumed runs are
    bit-identical to a single uninterrupted run.
    """
    d_y, d_x = a.shape
    a_d = a.double()  # for A @ x
    idx = n_done
    while idx < n_target:
        b1 = min(idx + batch, n_target)
        xb, yb, cb = [], [], []
        for _ in range(idx, b1):
            x_img = random_ellipse_phantom(N, rng, skull=True)
            x = torch.from_numpy(x_img.reshape(-1).astype(np.float64))
            y_clean = a_d @ x
            signal_pow = float((y_clean ** 2).mean())
            sigma = math.sqrt(signal_pow / (10.0 ** (SNR_DB / 10.0)))
            noise = torch.from_numpy(rng.standard_normal(d_y) * sigma)
            y = (y_clean + noise).numpy()
            c = fbp(y, a_t, N_ANGLES, N_DETECTORS).reshape(N, N)
            xb.append(x_img)
            yb.append(y.astype(np.float32))
            cb.append(c.astype(np.float32))
        n_now = append_h5(
            path, split,
            np.stack(xb).astype(np.float32),
            np.stack(yb).astype(np.float32),
            np.stack(cb).astype(np.float32),
        )
        idx = b1
        el = time.time() - t0
        rate = (idx - n_done) / max(el, 1e-9)
        eta = (n_target - idx) / max(rate, 1e-9)
        print(f"  [{split}] {n_now}/{n_target}  {rate:.1f} samp/s  "
              f"ETA {eta/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.join(datadir("tomography_ellipse"),
                             "tomography_ellipse.h5"),
    )
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Forward operator (built once, shared). float64 for accuracy; stored f4.
    print("[setup] building limited-angle Radon operator ...", flush=True)
    a, angles = parallel_beam_radon_operator(
        N, n_angles=N_ANGLES, phi_deg=PHI_DEG, n_detectors=N_DETECTORS,
        dtype=torch.float64,
    )
    a_t = a.T.contiguous()
    d_y, d_x = a.shape
    print(f"[setup] A shape = {tuple(a.shape)}  (dy={d_y}, dx={d_x})", flush=True)

    meta = dict(
        N=N, n_angles=N_ANGLES, phi_deg=PHI_DEG, n_detectors=N_DETECTORS,
        snr_db=SNR_DB, dx=d_x, dy=d_y, seed=args.seed,
        n_train_target=args.n_train, n_test_target=args.n_test,
        forward="parallel_beam_radon_limited_angle",
        phantom="random_ellipses_5to15_brainlike",
        conditioner="c = FBP(y) (Ram-Lak filtered backprojection)",
    )

    counts = read_counts(args.out)
    if counts is None:
        init_h5(args.out, a.numpy().astype(np.float32),
                angles.numpy().astype(np.float64), meta, args.batch)
        n_tr_done, n_te_done = 0, 0
        print(f"[init] fresh dataset at {args.out}", flush=True)
    else:
        n_tr_done, n_te_done = counts
        print(f"[resume] {args.out} has train={n_tr_done} test={n_te_done}; "
              f"continuing", flush=True)

    # Deterministic, resumable RNG: one generator drives train then test, in a
    # fixed per-sample order. On resume the variates of the samples already
    # written are re-burned so the continuation is bit-identical to one full
    # run. Each sample consumes a variable number of phantom variates plus d_y
    # noise variates. The variable phantom count means the stream cannot be
    # skipped cheaply; instead the finished samples are regenerated and
    # discarded (phantom and noise only, no FBP).
    rng = np.random.default_rng(args.seed)

    def burn(n):
        for _ in range(n):
            random_ellipse_phantom(N, rng, skull=True)
            rng.standard_normal(d_y)

    t0 = time.time()
    # --- train split ---
    if n_tr_done < args.n_train:
        burn(n_tr_done)  # advance rng past already-written train samples
        generate_split(args.out, "train", args.n_train, n_tr_done,
                       a, a_t, rng, args.batch, t0)
    else:
        burn(args.n_train)  # advance to the test stream start
    # --- test split --- (rng now positioned right after the full train stream)
    if n_te_done < args.n_test:
        burn(n_te_done)
        generate_split(args.out, "test", args.n_test, n_te_done,
                       a, a_t, rng, args.batch, t0)

    n_tr, n_te = read_counts(args.out)
    print(f"[finished] train={n_tr} test={n_te} in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[path] {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
