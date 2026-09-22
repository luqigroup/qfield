"""Designed-quadrature warm start on a trained conditional-flow posterior.

The reference ``rho`` is the sampled posterior of a trained conditional flow
``q_phi(x | c)`` (``ConditionalConvHINT`` on limited-angle tomography), whose
shape moves with the observation ``c = FBP(y)``. The amortizer is the
observation-conditioned set-equivariant network, whose particles cross-attend
to a CNN encoding of ``c``.

Pipeline:

  1. Operating space is a rank-``r`` informed latent. The 4096-dimensional
     image walls the squared-exponential kernel, so the quadrature lives in a
     rank-``r`` subspace: either the informed subspace of the exact linear
     forward (``H = A^T Sigma_obs^{-1} A``, via
     :class:`qfield.subspace.InformedSubspace`, with the exact lift
     ``W^T U = I``) or the leading principal directions of the flow samples.
     ``r`` is set by the spectrum of the flow-posterior samples projected into
     that subspace (cumulative energy ``r_energy``, capped at ``rank_cap``).
  2. The reference is the sampled flow posterior. For each observation, sample
     ``cnf_n`` posterior points ``x ~ q_phi(. | c)``, project to the latent
     ``z = W^T x``, and build a per-observation sampled-bank target
     (:func:`qfield.designed_quadrature.sampled_bank_target`). No density and
     no score are used.
  3. The conditioned amortizer
     (:class:`qfield.designed_quadrature.cond_net.ConditionalQuadratureAmortizer`)
     is MMD-regressed onto the per-observation latent reference, with ``M``
     uniform over the config's ``m_list`` and a median-heuristic bandwidth.
  4. Evaluation on held-out observations. Per ``M``: the independent-sample
     latent floor (equal weights), the warm start (one forward pass), and the
     per-instance optimum (``move_descend`` on each set), reported as median
     and interquartile range across the held-out observations.
  5. Reconstruction and uncertainty. Back-map the emitted nodes to image space
     (``x_j = U z_j``), then form the weighted-quadrature mean
     ``sum_j w_j x_j`` and the per-pixel spread
     ``sqrt(sum_j w_j (x_j - x_bar)^2)``.

The GPU is shared, so flow sampling reads the free GPU memory and falls back to
CPU when it is below a threshold or the device is unavailable; the rest of the
pipeline (subspace, amortizer training, MMD) is CPU-only.

Train (sample the flow bank, build the subspace, train, evaluate, reconstruct):
    python scripts/dq_cnf_tomography.py --experiment_name dq_cnf_tomography
Visualize (re-render figures from the saved artifact):
    python scripts/dq_cnf_tomography.py --experiment_name dq_cnf_tomography \
        --phase visualization

References:
  - Belhadji, Sharp, Marzouk, arXiv:2605.14142 (MMD-optimal designed
    quadrature).
  - Spantini et al. 2015; Zahm et al. 2022 (the likelihood-informed subspace).
  - Kruse et al. 2021 (HINT).
"""

from __future__ import annotations
from projorg import gitdir, logsdir, plotsdir  # noqa: E402

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

# Limit fragmentation on the tight shared GPU before torch imports CUDA.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from projorg import (  # noqa: E402
    datadir,
    checkpointsdir,
    plotsdir,
    setup_environment,
    upload_to_cloud,
)

import qfield.subspace as sub  # noqa: E402
from qfield.designed_quadrature import (  # noqa: E402
    constrained_weights_batched,
    mmd_sq_batched,
    move_descend_batched,
    sampled_bank_target,
)
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)
from qfield.dataset.tomography import limited_angle_tomography  # noqa: E402
from qfield.kernels import median_heuristic_h_sq  # noqa: E402
from qfield.models.genparihaka_cnf import ConditionalConvHINT  # noqa: E402

# setup_environment reads a fixed CONFIG_FILE, so DQ_CNF_CONFIG is the hook for
# pointing a run at a different config (for example a small smoke config).
CONFIG_FILE = os.environ.get("DQ_CNF_CONFIG", "dq_cnf_tomography.json")
DATA_H5 = os.path.join(datadir("tomography_ellipse"), "tomography_ellipse.h5")
CNF_CKPT = os.path.join(checkpointsdir("tomography_cnf"), "best.pt")

DTYPE = torch.float64

# Cap the BLAS / intra-op pool: many tiny batched solves on a CPU-contended box.
torch.set_num_threads(4)

_PALETTE = {"floor": "#3a7ca5", "warm": "#ff7f0e", "opt": "#7f7f7f"}
_LABELS = {
    "floor": "i.i.d. floor",
    "warm": "warm start (1 fwd pass)",
    "opt": "per-instance optimum",
}


# --------------------------------------------------------------------------- #
# GPU safety: pick a device only if there is enough free memory. Sampling is the
# only CUDA work; everything else is CPU.
# --------------------------------------------------------------------------- #
def _pick_cnf_device(min_free_gb: float) -> str:
    """``"cuda"`` only if free GPU memory exceeds ``min_free_gb``; else ``"cpu"``.

    Reads the device's free memory without allocating; if CUDA is unavailable or
    the free budget is below the threshold, the flow runs on CPU.
    """
    if not torch.cuda.is_available():
        print("[gpu] CUDA unavailable -> CNF sampling on CPU", flush=True)
        return "cpu"
    try:
        free, total = torch.cuda.mem_get_info()  # bytes, current device
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[gpu] mem_get_info failed ({exc}) -> CPU", flush=True)
        return "cpu"
    free_gb = free / 1e9
    print(
        f"[gpu] free={free_gb:.2f} GB / total={total / 1e9:.2f} GB "
        f"(need {min_free_gb:.2f} GB)",
        flush=True,
    )
    if free_gb >= min_free_gb:
        return "cuda"
    print("[gpu] insufficient free GPU memory -> CNF sampling on CPU", flush=True)
    return "cpu"


def _oom_retry(fn, *, tries: int = 100, wait: float = 4.0, label: str = ""):
    """Run ``fn`` retrying on CUDA OOM with a fixed backoff (shared GPU)."""
    last = None
    for att in range(tries):
        try:
            return fn()
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            last = exc
            torch.cuda.empty_cache()
            if att % 10 == 0:
                print(
                    f"  [oom-retry {label}] attempt {att}, waiting {wait}s ...",
                    flush=True,
                )
            time.sleep(wait)
    raise RuntimeError(f"persistent OOM in {label}") from last


def _median_iqr(vals: list[float]) -> tuple[float, float, float]:
    """``(median, q25, q75)`` over the finite entries of ``vals``."""
    t = torch.tensor(vals, dtype=DTYPE)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE))
    return float(q[1]), float(q[0]), float(q[2])


class DQCNFTomography:
    """One warm-start run on the sampled conditional-flow tomography posterior."""

    def __init__(self, args):
        self.args = args
        torch.set_default_dtype(DTYPE)
        self.r_cap = int(args.rank_cap)
        self.m_list = list(args.m_list)
        self.n_img = int(args.n_img)  # CNF grid (64)
        self.subspace_mode = str(getattr(args, "subspace", "pca")).lower()
        # At r >= 16 the isotropic squared-exponential kernel walls in the
        # rank-r latent exactly as it does in the ambient space (the Gram tends
        # to I under the median heuristic). The remedy is the Mahalanobis
        # (posterior-whitened) kernel with the dimension-aware sigma = sqrt(r);
        # for a sampled reference the whitening metric is the latent sample
        # covariance Sigma_z, so M = Sigma_z^{-1}.
        self.whiten = bool(int(getattr(args, "whiten", 0)))
        self.whiten_scope = str(getattr(args, "whiten_scope", "per_obs")).lower()
        self.whiten_jitter = float(getattr(args, "whiten_jitter", 1e-6))
        self.dx = self.n_img * self.n_img
        self.x_mean = torch.zeros(self.dx, dtype=DTYPE)  # PCA centering (0 for H)
        if self.subspace_mode == "h":
            self._build_H_subspace()

    # ---- informed subspace (the projector + exact lift) -----------------
    def _build_H_subspace(self):
        """Rank-``r_cap`` H-informed subspace from the exact stored forward.

        Built from the dataset's own forward operator ``A`` (read from the
        HDF5) at the dataset geometry, a representative isotropic noise level,
        and a smoothing field prior. ``W`` projects (``z = x @ W``), ``U``
        lifts (``x = U z``, ``W^T U = I``). The dataset has no Gaussian prior
        on ``x`` (random-ellipse phantoms), so the prior enters here only as
        the whitening metric; the data-driven ``"pca"`` subspace avoids that
        modeling choice.
        """
        a = self.args
        with h5py.File(DATA_H5, "r") as f:
            a64 = torch.from_numpy(f["A"][:]).double()  # (dy, dx) exact forward
        dy = a64.shape[0]
        sobs64 = (float(a.sigma_obs) ** 2) * torch.eye(dy, dtype=torch.float64)
        # Smoothing field prior as the whitening metric only (Laplacian field).
        _, meta = limited_angle_tomography(
            n=a.n_img, n_angles=a.n_angles, phi_deg=a.phi_deg,
            sigma_obs=a.sigma_obs, n_detectors=a.n_detectors,
            prior="laplacian", prior_delta=a.prior_delta,
            prior_alpha=a.prior_alpha, prior_tau=a.prior_tau,
            dtype=torch.float32,
        )
        spr64 = meta["Sigma_pr"]
        self.phi_full = sub.InformedSubspace.from_H(
            a64, sobs64, spr64, r=self.r_cap
        )
        self.W_full = self.phi_full._w.double()          # (dx, r_cap): z = x @ W
        self.U_full = self.phi_full.directions.double()  # (dx, r_cap): x = U z
        print(
            f"[subspace] H-informed | dx={self.dx} | rank_cap={self.r_cap} | "
            f"H-spectrum top-5="
            f"{[f'{v:.2e}' for v in self.phi_full.eigvals[:5].tolist()]}",
            flush=True,
        )

    def _build_pca_subspace(self):
        """Data-driven rank-``r_cap`` subspace: leading PCs of the flow bank.

        The dataset has no Gaussian prior (random-ellipse phantoms), so the
        latent is the subspace the flow posterior occupies: the leading
        principal directions of the pooled, mean-removed held-in flow posterior
        samples. The columns of ``U`` are orthonormal, so the projector is
        ``W = U`` (``W^T U = I``, the exact lift) and ``z = (x - x_mean) @ U``,
        ``x = x_mean + z @ U^T``. ``r`` is chosen from this same spectrum in
        :meth:`_choose_rank`.
        """
        x = self.x_train_img  # (n_obs * n_per, dx) pooled flow image samples
        self.x_mean = x.mean(dim=0)
        xc = x - self.x_mean.unsqueeze(0)
        # Economy SVD: right singular vectors are the PCs; sing^2/(N-1) the var.
        _, s, vh = torch.linalg.svd(xc, full_matrices=False)
        k = min(self.r_cap, vh.shape[0])
        self.U_full = vh[:k].T.contiguous().double()    # (dx, r_cap)
        self.W_full = self.U_full                        # orthonormal: W = U
        self.pca_eig_full = ((s[:k] ** 2) / max(1, xc.shape[0] - 1)).double()
        print(
            f"[subspace] data-driven PCA | dx={self.dx} | rank_cap={k} | "
            f"PCA-var top-5="
            f"{[f'{v:.2e}' for v in self.pca_eig_full[:5].tolist()]}",
            flush=True,
        )

    # ---- CNF bank sampling (GPU-safe) -----------------------------------
    def _load_cnf(self, device: str) -> tuple[ConditionalConvHINT, float, float]:
        """Build the trained CNF on ``device`` and return ``(model, c_mean, c_std)``."""
        ck = torch.load(CNF_CKPT, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        torch.manual_seed(int(cfg["model_seed"]))
        model = ConditionalConvHINT(
            in_channels=1, cond_in_channels=1, spatial_size=self.n_img,
            hidden_ch=cfg["hidden_ch"], n_flow_layers=cfg["n_flow_layers"],
            cond_feat_ch=cfg["cond_feat_ch"],
        )
        model.load_state_dict(ck["model"])
        # The CNF was trained in float32; pin it to float32 (independent of this
        # script's float64 default) so sampling matches its native precision and
        # the conditioner / latent dtypes line up.
        model = model.to(device=device, dtype=torch.float32).eval()
        print(
            f"[cnf] loaded step={ck['step']} val={ck['best_val']:.4f} "
            f"on {device}",
            flush=True,
        )
        return model, float(cfg["c_mean"]), float(cfg["c_std"])

    def _sample_cnf_images(
        self, idxs: list[int], split: str, n_per: int, device: str,
        c_mean: float, c_std: float, model: ConditionalConvHINT,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Sample the flow posterior images for the given observations.

        For each observation index, collect ``n_per`` finite, in-range
        posterior samples ``x ~ q_phi(. | c)`` (``c`` standardized by the
        statistics the flow was trained with), flattened to ``(n_per, dx)``.
        Projection to the latent happens centrally after the subspace is built.
        Sampling is wrapped in an OOM retry so a transient memory spike pauses
        rather than kills the run.

        Robust filtering. The trained ``ConditionalConvHINT`` occasionally
        sends a tail sample to a non-finite value, or to a finite but
        astronomical one (about 1e38, the float32 ceiling) that is still
        ``isfinite`` yet would dominate any covariance, SVD or kernel. Both are
        dropped: a sample is kept only if every entry is finite and
        ``|x| <= sample_abs_max``. Each observation is topped up by resampling
        on fresh seeds until it has ``n_per`` good samples. If an observation
        cannot reach ``n_keep_min`` good samples after ``max_resample_rounds``
        rounds, that whole observation is dropped from the bank, its
        conditioner being pathological rather than the source of a one-off tail
        blow-up. The dropped fraction is logged and the raw non-finite rate is
        required to stay below a bound.

        Returns:
            ``(x_bank, c_imgs, x_true, c_raw, kept)`` with
            ``x_bank`` ``(n_kept, n_per, dx)`` flattened finite image samples,
            ``c_imgs`` ``(n_kept, 1, S, S)`` standardized conditioners (the
            amortizer input), ``x_true`` ``(n_kept, S, S)`` ground truth,
            ``c_raw`` ``(n_kept, S, S)`` raw FBP (for figures), all on CPU, and
            ``kept`` the list of positions in ``idxs`` that survived filtering.
        """
        a = self.args
        n = self.n_img
        with h5py.File(DATA_H5, "r") as f:
            x_true = f[f"{split}/x"][idxs].astype("float32")  # (n_obs, S, S)
            c_raw = f[f"{split}/c"][idxs].astype("float32")
        c_std_img = (c_raw - c_mean) / c_std
        # float32 conditioner for the float32 flow; a float64 copy is stored
        # for the float64 amortizer.
        c_f32 = torch.from_numpy(c_std_img)[:, None].to(
            device=device, dtype=torch.float32
        )  # (n_obs, 1, S, S)
        chunk = int(a.cnf_sample_chunk)
        abs_max = float(getattr(a, "sample_abs_max", 1e4))
        max_rounds = int(getattr(a, "max_resample_rounds", 12))
        # Minimum finite count an obs must reach to be kept (else dropped).
        n_keep_min = int(getattr(a, "n_keep_min", 0)) or max(
            8, min(n_per, n_per // 2)
        )
        # Ceiling on the tolerated non-finite / out-of-range rate; above it the
        # flow is pervasively unstable and sampling raises rather than masks.
        max_bad_frac = float(getattr(a, "max_bad_frac", 0.10))

        x_rows, kept = [], []
        n_drawn_total = 0  # raw samples drawn (incl. re-draws)
        n_bad_total = 0    # raw samples rejected (non-finite or out-of-range)
        dropped_obs = []

        def _good_mask(flat: torch.Tensor) -> torch.Tensor:
            """Per-sample keep mask: finite everywhere AND within ``abs_max``."""
            finite = torch.isfinite(flat).all(dim=1)
            bounded = flat.abs().amax(dim=1) <= abs_max
            return finite & bounded

        # The flow samples in float32, its native precision; set the default
        # dtype to float32 around sampling so its internal randn matches.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            for i in range(len(idxs)):
                ci = c_f32[i : i + 1]
                good = []  # list of (m_i, dx) finite CPU chunks
                n_good = 0
                rnd = 0
                # Keep sampling (topping up) until n_per good samples or the
                # round budget is exhausted.
                while n_good < n_per and rnd < max_rounds:
                    # Fresh seed per (obs, round) so a top-up gives new samples.
                    torch.manual_seed(
                        int(a.base_seed) + 90000 + 1000 * int(idxs[i]) + rnd
                    )
                    need = n_per - n_good
                    drawn = 0
                    while drawn < need:
                        m = min(chunk, need - drawn)
                        smp = _oom_retry(
                            lambda ci=ci, m=m: model.sample(ci, m),
                            label=f"cnf_sample[{split}:{idxs[i]}:r{rnd}]",
                        )
                        flat = smp[0, :, 0].reshape(m, n * n).double().cpu()
                        mask = _good_mask(flat)
                        n_drawn_total += m
                        n_bad_total += int((~mask).sum())
                        if mask.any():
                            good.append(flat[mask])
                            n_good += int(mask.sum())
                        drawn += m
                    rnd += 1
                if n_good >= n_keep_min:
                    bank = torch.cat(good, dim=0) if good else flat[:0]
                    x_rows.append(bank[:n_per])  # exactly n_per (>= n_keep_min)
                    kept.append(i)
                else:
                    dropped_obs.append((int(idxs[i]), n_good))

        finally:
            torch.set_default_dtype(prev_dtype)

        if not x_rows:
            raise RuntimeError(
                f"[bank:{split}] EVERY observation was dropped (CNF pervasively "
                f"non-finite); refuse to proceed."
            )
        # Pad short-but-kept banks to n_per by tiling (only obs with n_keep_min
        # <= n_good < n_per; keeps the (n_kept, n_per, dx) shape rectangular).
        for j, bank in enumerate(x_rows):
            if bank.shape[0] < n_per:
                reps = (n_per + bank.shape[0] - 1) // bank.shape[0]
                x_rows[j] = bank.repeat(reps, 1)[:n_per]
        x_bank = torch.stack(x_rows, dim=0)  # (n_kept, n_per, dx)

        bad_frac = n_bad_total / max(1, n_drawn_total)
        print(
            f"[bank:{split}] filter | raw non-finite/out-of-range: "
            f"{n_bad_total}/{n_drawn_total} = {100 * bad_frac:.3f}% | "
            f"obs kept={len(kept)}/{len(idxs)} dropped={len(dropped_obs)}"
            + (f" {dropped_obs}" if dropped_obs else ""),
            flush=True,
        )
        # If the flow is pervasively bad, surface it rather than mask it.
        if bad_frac > max_bad_frac:
            raise RuntimeError(
                f"[bank:{split}] non-finite/out-of-range rate {100 * bad_frac:.2f}% "
                f"exceeds {100 * max_bad_frac:.1f}% -- the CNF looks pervasively "
                f"unstable, not a few tail blow-ups; refusing to silently mask."
            )

        c_keep = c_f32[kept].cpu().double()
        x_true_keep = torch.from_numpy(x_true[kept]).double()
        c_raw_keep = torch.from_numpy(c_raw[kept]).double()
        return x_bank, c_keep, x_true_keep, c_raw_keep, kept

    def _project(self, x_bank: torch.Tensor) -> torch.Tensor:
        """Project an image bank ``(n_obs, n_per, dx)`` to the latent ``(..., r)``.

        ``z = (x - x_mean) @ W`` (``x_mean`` is zero in the H subspace, the PCA
        mean otherwise); restricted to the chosen rank ``r``.
        """
        xc = x_bank - self.x_mean.reshape(1, 1, -1)
        return xc @ self.W  # (n_obs, n_per, r)

    def _build_banks(self):
        """Sample the held-in and held-out flow banks once.

        The heavy sampling is done once and reused for the whole run. After
        sampling, the subspace (PCA on the held-in images, or the pre-built H
        subspace) and rank are fixed and every bank is projected to the chosen
        latent.
        """
        a = self.args
        device = _pick_cnf_device(float(a.gpu_min_free_gb))
        model, c_mean, c_std = self._load_cnf(device)
        self.c_mean, self.c_std = c_mean, c_std

        train_idx = list(range(int(a.n_train_obs)))
        test_idx = list(range(int(a.n_eval_obs)))
        recon_idx = list(
            range(int(a.n_eval_obs), int(a.n_eval_obs) + int(a.n_recon_obs))
        )

        t0 = time.time()
        print(
            f"[bank] sampling CNF: train_obs={len(train_idx)} x "
            f"{a.cnf_n_train}, eval_obs={len(test_idx)} x {a.cnf_n_eval}, "
            f"recon_obs={len(recon_idx)} x {a.cnf_n_eval}",
            flush=True,
        )
        xb_train, self.c_train, self.x_train, _, _ = self._sample_cnf_images(
            train_idx, "train", int(a.cnf_n_train), device, c_mean, c_std, model,
        )
        xb_eval, self.c_eval, self.x_eval, self.craw_eval, _ = (
            self._sample_cnf_images(
                test_idx, "test", int(a.cnf_n_eval), device, c_mean, c_std, model,
            )
        )
        xb_recon, self.c_recon, self.x_recon, self.craw_recon, _ = (
            self._sample_cnf_images(
                recon_idx, "test", int(a.cnf_n_eval), device, c_mean, c_std,
                model,
            )
        )
        if device == "cuda":
            del model
            torch.cuda.empty_cache()
        print(f"[bank] CNF sampling wall-time: {time.time() - t0:.1f} s", flush=True)

        # Build the subspace (PCA needs the held-in images), choose r, project.
        if self.subspace_mode == "pca":
            self.x_train_img = xb_train.reshape(-1, self.dx)
            self._build_pca_subspace()
        self._choose_rank()
        self.z_train = self._project(xb_train)
        self.z_eval = self._project(xb_eval)
        self.z_recon = self._project(xb_recon)
        if self.whiten:
            self._whiten_banks()

    # ---- Mahalanobis whitening in the latent (metric Sigma_z) -----------
    def _whiten_map(self, z_bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Whitening for one latent bank ``(n_per, r)`` from its sample covariance.

        Estimates ``Sigma_z = cov(z)`` (the latent sample covariance,
        ``O(n_per r^2)``), takes the Cholesky ``Sigma_z = L L^T``, and returns
        the whitening map ``R = L^{-1}`` (``u = R (z - z_mean)``), its inverse
        ``L`` (the back-map ``z = z_mean + L u``), and the latent mean
        ``z_mean``. A small relative ridge keeps ``Sigma_z`` positive definite
        for small banks. The Mahalanobis squared-exponential kernel with metric
        ``M = Sigma_z^{-1}`` is exactly the isotropic kernel in ``u``, since
        ``(z - z')^T Sigma_z^{-1} (z - z') = ||u - u'||^2``, so whitening the
        bank and running the isotropic ``sampled_bank_target`` at
        ``sigma = sqrt(r)`` realizes the whitened kernel unchanged.
        """
        z_mean = z_bank.mean(dim=0)
        zc = z_bank - z_mean.unsqueeze(0)
        cov = (zc.T @ zc) / max(1, zc.shape[0] - 1)
        cov = 0.5 * (cov + cov.T)
        # Relative ridge for SPD on small banks (jitter * mean eigenvalue).
        ridge = self.whiten_jitter * float(torch.diagonal(cov).mean().clamp_min(1e-30))
        cov = cov + ridge * torch.eye(self.r, dtype=DTYPE)
        chol = torch.linalg.cholesky(cov)            # L
        r_map = torch.linalg.inv(chol)               # R = L^{-1}
        return r_map, chol, z_mean

    def _whiten_banks(self):
        """Whiten every latent bank by the sampled flow-posterior metric.

        ``per_obs`` (default): each observation's bank is whitened by its own
        ``Sigma_z``, so the metric tracks the observation-varying posterior
        shape; the per-observation maps are stored for the reconstruction
        back-map. ``pooled``: one ``Sigma_z`` from the pooled held-in latent
        samples whitens every bank. The whitened banks replace ``z_train`` /
        ``z_eval`` / ``z_recon``; everything downstream runs unchanged on the
        whitened coordinates ``u``.
        """
        if self.whiten_scope == "pooled":
            zp = self.z_train.reshape(-1, self.r)
            r_map, chol, z_mean = self._whiten_map(zp)
            self._whiten_train = [(r_map, chol, z_mean)] * self.z_train.shape[0]
            self._whiten_eval = [(r_map, chol, z_mean)] * self.z_eval.shape[0]
            self._whiten_recon = [(r_map, chol, z_mean)] * self.z_recon.shape[0]
        elif self.whiten_scope == "per_obs":
            self._whiten_train = [self._whiten_map(self.z_train[o])
                                  for o in range(self.z_train.shape[0])]
            self._whiten_eval = [self._whiten_map(self.z_eval[o])
                                 for o in range(self.z_eval.shape[0])]
            self._whiten_recon = [self._whiten_map(self.z_recon[o])
                                  for o in range(self.z_recon.shape[0])]
        else:
            raise ValueError(
                f"whiten_scope must be 'per_obs' or 'pooled'; got "
                f"{self.whiten_scope!r}"
            )

        def _apply(bank, maps):
            out = torch.empty_like(bank)
            for o in range(bank.shape[0]):
                r_map, _, z_mean = maps[o]
                out[o] = (bank[o] - z_mean.unsqueeze(0)) @ r_map.T
            return out

        self.z_train = _apply(self.z_train, self._whiten_train)
        self.z_eval = _apply(self.z_eval, self._whiten_eval)
        self.z_recon = _apply(self.z_recon, self._whiten_recon)
        # The pooled whitened sample covariance should be near I, which is what
        # makes the kernel isotropic in u.
        up = self.z_train.reshape(-1, self.r)
        upc = up - up.mean(dim=0, keepdim=True)
        cov_u = (upc.T @ upc) / max(1, upc.shape[0] - 1)
        off = (cov_u - torch.eye(self.r, dtype=DTYPE)).abs()
        print(
            f"[whiten] scope={self.whiten_scope} | latent r={self.r} | "
            f"Sigma_z^-1 metric | whitened-cov dev from I: "
            f"max={float(off.max()):.2e} mean={float(off.mean()):.2e}",
            flush=True,
        )

    def _latent_unwhiten(
        self, u: torch.Tensor, maps: list, k: int
    ) -> torch.Tensor:
        """Map whitened coords ``u`` (..., r) back to the rank-r latent.

        Inverse of the whitening ``u = (z - z_mean) @ R^T``: ``z = z_mean +
        u @ L^T`` (``L = R^{-1}``). Used only for the recon back-map (the MMD
        arms run entirely in ``u``-coords and need no inverse).
        """
        _, chol, z_mean = maps[k]
        return z_mean.unsqueeze(0) + u @ chol.T

    # ---- rank from the CNF-sample spectrum ------------------------------
    def _choose_rank(self):
        """Pick ``r`` from the cumulative energy of the flow-sample spectrum.

        Reads the spectrum the subspace exposes (PCA variances, or the
        H-informed projection of the held-in flow samples' covariance), and
        sets ``r`` to the smallest rank whose cumulative energy reaches
        ``r_energy``, capped at ``rank_cap`` and floored at ``r_min``.
        """
        a = self.args
        if self.subspace_mode == "pca":
            eig = self.pca_eig_full.clone()
        else:
            # H subspace: spectrum of the held-in flow samples projected into it.
            z = (
                (self.x_train_img - self.x_mean.unsqueeze(0)) @ self.W_full
                if hasattr(self, "x_train_img")
                else None
            )
            if z is None:
                eig = self.phi_full.eigvals[: self.r_cap].clone()
            else:
                zc = z - z.mean(dim=0, keepdim=True)
                cov = (zc.T @ zc) / max(1, zc.shape[0] - 1)
                eig = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).clamp_min(0.0)
                eig = torch.sort(eig, descending=True).values
        energy = torch.cumsum(eig, dim=0) / eig.sum().clamp_min(1e-30)
        r_e = float(a.r_energy)
        r = int(torch.searchsorted(energy, torch.tensor(r_e)).item()) + 1
        r = max(int(a.r_min), min(r, self.r_cap, eig.numel()))
        self.r = r
        self.cnf_spectrum = eig.tolist()
        self.cnf_energy_at_r = float(energy[r - 1])
        print(
            f"[rank] spectrum top-5={[f'{v:.2e}' for v in eig[:5].tolist()]} "
            f"| r={r} (>= {r_e:.2f} energy, cap {self.r_cap}) | "
            f"energy@r={self.cnf_energy_at_r:.3f}",
            flush=True,
        )
        # Restrict the projector / lift to the chosen rank.
        self.W = self.W_full[:, :r]
        self.U = self.U_full[:, :r]

    # ---- bandwidth ------------------------------------------------------
    def _latent_sigma(self) -> float:
        """Squared-exponential bandwidth in the (possibly whitened) latent.

        ``bandwidth`` config:
          * ``"sqrt_r"`` -- the dimension-aware constant-Gram bandwidth
            ``sigma = sqrt(r)``: in the whitened latent the typical
            off-diagonal Gram entry is held at ``exp(-1)``, since the whitened
            pairwise squared distance is about ``2r``. It may be followed by a
            scale (for example ``"sqrt_r:x0.8"``) for a more conservative rule.
          * ``"median"`` -- the median heuristic on the pooled latent samples.
          * ``"x<f>"`` -- median heuristic scaled by ``f``.
          * a float -- sigma so the median is scaled to it.

        ``sqrt_r`` is the pairing that goes with the whitened kernel; the
        median heuristic still walls even in whitened coordinates, because its
        ``log(M+1)`` divisor makes sigma too small.
        """
        a = self.args
        bw = getattr(a, "bandwidth", None)
        s = "" if bw is None else str(bw).strip().lower()
        if s.startswith("sqrt_r"):
            scale = 1.0
            rest = s[len("sqrt_r"):].lstrip(":").strip()
            if rest:
                scale = float(rest[1:]) if rest.startswith("x") else float(rest)
            sigma = scale * math.sqrt(self.r)
            print(
                f"[sigma] constant-Gram sqrt(r) (r={self.r}, whiten="
                f"{self.whiten}): scale={scale:.3f} -> sigma={sigma:.4f}",
                flush=True,
            )
            return float(sigma)
        z = self.z_train.reshape(-1, self.r)
        # Subsample for the median heuristic (O(N^2) pairwise).
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(a.base_seed) + 1234)
        n = min(int(a.sigma_n_sample), z.shape[0])
        idx = torch.randperm(z.shape[0], generator=gen)[:n]
        med = math.sqrt(median_heuristic_h_sq(z[idx]))
        scale = 1.0
        if s != "" and s != "median":
            scale = float(s[1:]) if s.startswith("x") else float(s) / med
        sigma = scale * med
        print(
            f"[sigma] median heuristic (r={self.r}, N={n}): median={med:.4f} "
            f"scale={scale:.3f} -> sigma={sigma:.4f}",
            flush=True,
        )
        return float(sigma)

    # ---- per-observation targets ----------------------------------------
    def _obs_target(self, z_bank_row: torch.Tensor, n_mu: int):
        """A per-observation sampled-bank target from one observation's latent bank."""
        return sampled_bank_target(
            z_bank_row, self.sigma, n_mu=n_mu, name="cnf_latent",
            meta={"r": self.r},
        )

    def _build_net(self) -> ConditionalQuadratureAmortizer:
        a = self.args
        return ConditionalQuadratureAmortizer(
            d=self.r,
            hidden=int(a.hidden),
            n_blocks=int(a.n_blocks),
            n_heads=int(a.n_heads),
            cond_hidden=int(a.cond_hidden),
            delta_scale=float(a.delta_scale),
            spatial_size=self.n_img,
            cond_token_dim=int(a.cond_token_dim),
            cond_hidden_ch=int(a.cond_hidden_ch),
            cond_n_down=int(a.cond_n_down),
            cond_in_channels=1,
        ).to(DTYPE)

    # ---- training -------------------------------------------------------
    def train(self):
        """Sample the flow bank, choose ``r``, train the conditioned amortizer."""
        a = self.args
        self._build_banks()  # samples, builds the subspace, chooses r, projects
        self.sigma = self._latent_sigma()

        # Per-observation targets + the per-step mu sub-bank size (defaults to
        # the full per-observation train bank).
        n_mu = int(getattr(a, "n_mu", 0)) or int(a.cnf_n_train)
        targets = [
            self._obs_target(self.z_train[o], n_mu)
            for o in range(self.z_train.shape[0])
        ]
        c_train = self.c_train  # (n_obs, 1, S, S) standardized conditioner

        print(
            f"[dq-cnf] M~Uniform{self.m_list}, batch_obs={a.batch_obs}, "
            f"steps={a.n_steps}, lr={a.lr}, sigma={self.sigma:.4f} | "
            f"net(hidden={a.hidden}, blocks={a.n_blocks}, heads={a.n_heads}, "
            f"delta_scale={a.delta_scale})",
            flush=True,
        )

        torch.manual_seed(int(a.seed))
        net = self._build_net()
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  net params: {n_params:,}", flush=True)
        opt = torch.optim.Adam(net.parameters(), lr=float(a.lr))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(a.base_seed))
        m_choices = torch.tensor(self.m_list)
        n_obs = len(targets)
        log_every = max(1, int(a.n_steps) // 20)
        curve, ema = [], None
        t0 = time.time()
        net.train()
        for step in range(int(a.n_steps)):
            M = int(m_choices[torch.randint(len(m_choices), (1,), generator=gen)])
            # A minibatch of observations; each contributes one M-sample set.
            obs = torch.randperm(n_obs, generator=gen)[: int(a.batch_obs)]
            z0 = torch.stack(
                [targets[int(o)].sampler(M, gen) for o in obs], dim=0
            )  # (B, M, r)
            c_b = c_train[obs]  # (B, 1, S, S)
            B, _, d = z0.shape
            z = net(z0, c_b)  # (B, M, r)
            # Per-observation mu and c_rho: the target shape moves with the
            # observation.
            mu = torch.stack(
                [
                    targets[int(o)].mu_fn(z[i])
                    for i, o in enumerate(obs)
                ],
                dim=0,
            )  # (B, M)
            c_rho = torch.stack(
                [targets[int(o)].c_rho for o in obs], dim=0
            )  # (B,)
            w = constrained_weights_batched(z, mu, self.sigma, a.jitter)
            # Batched MMD with a per-row c_rho, the observation-varying
            # self-affinity.
            K = torch.exp(-(torch.cdist(z, z) ** 2) / (2.0 * self.sigma**2))
            quad = torch.einsum("rj,rjl,rl->r", w, K, w)
            cross = (w * mu).sum(dim=-1)
            loss = (quad - 2.0 * cross + c_rho).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            v = float(loss.detach())
            ema = v if ema is None else 0.98 * ema + 0.02 * v
            if step % log_every == 0 or step == int(a.n_steps) - 1:
                curve.append((step, ema))
                print(
                    f"  step {step:>5d}/{a.n_steps}: M={M:>3d} "
                    f"MMD^2(mean)={v:.4e}  EMA={ema:.4e}",
                    flush=True,
                )
        train_secs = time.time() - t0
        print(f"  training wall-time: {train_secs:.1f} s", flush=True)

        net.eval()
        eval_rows = self._evaluate(net)
        recon = self._reconstructions(net)

        verdict = {
            "warm_below_floor_every_M": all(
                r["warm"][0] < r["floor"][0] for r in eval_rows
            ),
            "warm_above_opt_every_M": all(
                r["warm"][0] >= r["opt"][0] - 1e-12 for r in eval_rows
            ),
        }
        results = {
            "r": self.r,
            "rank_cap": self.r_cap,
            "whiten": bool(self.whiten),
            "whiten_scope": self.whiten_scope if self.whiten else None,
            "bandwidth": str(getattr(a, "bandwidth", "")),
            "cnf_energy_at_r": self.cnf_energy_at_r,
            "cnf_spectrum": self.cnf_spectrum,
            "dx": int(self.dx),
            "n_img": self.n_img,
            "sigma": float(self.sigma),
            "m_list": self.m_list,
            "phi_deg": float(a.phi_deg),
            "n_params": int(n_params),
            "n_steps": int(a.n_steps),
            "batch_obs": int(a.batch_obs),
            "lr": float(a.lr),
            "delta_scale": float(a.delta_scale),
            "n_train_obs": int(a.n_train_obs),
            "n_eval_obs": int(a.n_eval_obs),
            "cnf_n_train": int(a.cnf_n_train),
            "cnf_n_eval": int(a.cnf_n_eval),
            "opt_lr": float(a.opt_lr),
            "opt_iters": int(a.opt_iters),
            "train_curve": curve,
            "train_secs": train_secs,
            "rows": eval_rows,
            "recon": recon,
            "verdict": verdict,
        }
        self._print_table(eval_rows, verdict)

        ckpt_path = os.path.join(checkpointsdir(a.experiment), "checkpoint.pth")
        torch.save(
            {"results": results, "state_dict": net.state_dict()}, ckpt_path
        )
        with open(
            os.path.join(checkpointsdir(a.experiment), "results.json"), "w"
        ) as fh:
            json.dump(results, fh, indent=2)
        print(f"Saved results to {ckpt_path}", flush=True)

    # ---- evaluation (per held-out observation) --------------------------
    def _evaluate(self, net: ConditionalQuadratureAmortizer) -> list[dict]:
        """Held-out floor, warm and per-instance-optimum latent MMD^2 per ``M``.

        For each ``M`` and each held-out observation, take one ``M``-sample set
        from that observation's latent bank, then compute the three arms; the
        metric varies across observations. Reported as median and interquartile
        range over the held-out observations.
        """
        a = self.args
        n_mu_eval = int(a.cnf_n_eval)  # eval mu uses the full eval bank
        targets = [
            self._obs_target(self.z_eval[o], n_mu_eval)
            for o in range(self.z_eval.shape[0])
        ]
        c_eval = self.c_eval
        rows = []
        for M in self.m_list:
            floor_vals, warm_vals, opt_vals = [], [], []
            for o, tgt in enumerate(targets):
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(a.base_seed) + 50000 + 1000 * int(M) + o)
                z0 = tgt.sampler(M, gen).unsqueeze(0)  # (1, M, r)
                mu0 = tgt.mu_fn(z0[0]).reshape(1, M)
                w_eq = torch.full((1, M), 1.0 / M, dtype=DTYPE)
                floor = float(
                    torch.clamp(
                        mmd_sq_batched(z0, w_eq, mu0, tgt.c_rho, self.sigma),
                        min=0.0,
                    )[0]
                )
                with torch.no_grad():
                    z = net(z0, c_eval[o : o + 1])
                    mu = tgt.mu_fn(z[0]).reshape(1, M)
                    w = constrained_weights_batched(z, mu, self.sigma, a.jitter)
                    warm = float(
                        torch.clamp(
                            mmd_sq_batched(z, w, mu, tgt.c_rho, self.sigma),
                            min=0.0,
                        )[0]
                    )
                _, opt = move_descend_batched(
                    z0, tgt.mu_fn, tgt.c_rho, self.sigma,
                    float(a.opt_lr), int(a.opt_iters), a.jitter,
                )
                floor_vals.append(floor)
                warm_vals.append(warm)
                opt_vals.append(float(opt[0]))
            fr = _median_iqr(floor_vals)
            wr = _median_iqr(warm_vals)
            orc = _median_iqr(opt_vals)
            rows.append(
                {
                    "M": int(M), "floor": fr, "warm": wr, "opt": orc,
                    "warm_over_floor": wr[0] / fr[0] if fr[0] > 0 else float("inf"),
                    "warm_over_opt": wr[0] / orc[0] if orc[0] > 0 else float("inf"),
                }
            )
            print(
                f"  [eval] M={M:>3d}: floor={fr[0]:.3e} warm={wr[0]:.3e} "
                f"opt={orc[0]:.3e}  (warm/floor={rows[-1]['warm_over_floor']:.3f}, "
                f"warm/opt={rows[-1]['warm_over_opt']:.3f})",
                flush=True,
            )
        return rows

    # ---- recon + UQ -----------------------------------------------------
    def _reconstructions(self, net: ConditionalQuadratureAmortizer) -> dict:
        """Image-space reconstruction and per-pixel spread, held-out observations.

        For each reconstruction observation: seed ``M`` independent latent
        samples, emit the quadrature conditioned on that observation's ``c``,
        back-map the nodes ``x_j = U z_j``, and form the weighted-quadrature
        mean ``sum_j w_j x_j`` and the per-pixel spread. The flow posterior
        mean (the equal-weight bank mean, lifted) is the reference the
        quadrature is compared against.
        """
        a = self.args
        M = int(a.recon_M)
        n = self.n_img
        out = {"M": M, "n_img": n, "cases": []}
        for k in range(self.z_recon.shape[0]):
            tgt = self._obs_target(self.z_recon[k], int(a.cnf_n_eval))
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(a.base_seed) + 7000 + k)
            # Pool ``K`` independent quadratures, each at the trained budget
            # ``M``, into one weighted measure, so the per-pixel spread map is
            # a low-noise estimate rather than the grainy spread of a single
            # ``M``-node set. ``K = 1`` gives the single-quadrature behavior,
            # and no node count exceeds the trained budget.
            K = int(getattr(a, "recon_pool", 1))
            xn_list, w_list = [], []
            with torch.no_grad():
                for _ in range(K):
                    z0 = tgt.sampler(M, gen).unsqueeze(0)  # (1, M, r)
                    zk = net(z0, self.c_recon[k : k + 1])
                    muk = tgt.mu_fn(zk[0]).reshape(1, M)
                    wk = constrained_weights_batched(zk, muk, self.sigma, a.jitter)[0]
                    zk = zk[0]
                    # Un-whiten the emitted nodes back to the rank-r latent
                    # before lifting (identity when whitening is off).
                    zk_lat = self._latent_unwhiten(zk, self._whiten_recon, k) \
                        if self.whiten else zk
                    # Back-map: x_j = x_mean + U z_j (the exact lift).
                    xn_list.append(self.x_mean.unsqueeze(0) + zk_lat @ self.U.T)
                    w_list.append(wk / K)                  # pool: total mass 1
            x_nodes = torch.cat(xn_list, dim=0)            # (K*M, dx)
            w = torch.cat(w_list, dim=0)                   # (K*M,)
            x_mean = (w.unsqueeze(1) * x_nodes).sum(dim=0)
            diff = x_nodes - x_mean.unsqueeze(0)
            var = (w.unsqueeze(1) * diff**2).sum(dim=0).clamp_min(0)
            x_std = var.sqrt()
            # Flow posterior mean (full latent bank, lifted): the reference.
            z_full = self.z_recon[k]                            # (n_per, r), u-coords
            z_full_lat = self._latent_unwhiten(z_full, self._whiten_recon, k) \
                if self.whiten else z_full
            cnf_mean = self.x_mean + (z_full_lat.mean(dim=0)) @ self.U.T  # (dx,)
            x_true = self.x_recon[k].reshape(-1)
            rel_cnf = float((x_mean - cnf_mean).norm() / cnf_mean.norm())
            # Informed projection of the truth: x_mean + U W^T (x_true - x_mean).
            p_true = self.x_mean + self.U @ (self.W.T @ (x_true - self.x_mean))
            rel_proj = float((x_mean - p_true).norm() / p_true.norm())
            out["cases"].append(
                {
                    "true": self.x_recon[k].tolist(),
                    "fbp": self.craw_recon[k].tolist(),
                    "recon": x_mean.tolist(),
                    "cnf_mean": cnf_mean.tolist(),
                    "uq": x_std.tolist(),
                    "rel_cnf_mean": rel_cnf,
                    "rel_informed_proj": rel_proj,
                }
            )
            print(
                f"  [recon] obs[{k}]: relL2(quad-mean vs CNF-mean)={rel_cnf:.3f}  "
                f"relL2(quad-mean vs informed-proj of truth)={rel_proj:.3f}",
                flush=True,
            )
        return out

    # ---- reporting ------------------------------------------------------
    def _print_table(self, rows: list[dict], verdict: dict):
        print(
            f"\n=== CNF tomography DQ warm start (latent r={self.r}, SE sigma="
            f"{self.sigma:.4f}) ===",
            flush=True,
        )
        print(
            f"{'M':>4} | {'floor':>11} | {'warm':>11} | {'per-inst':>11} | "
            f"{'warm/fl':>8} | {'warm/opt':>8}"
        )
        for r in rows:
            print(
                f"{r['M']:>4} | {r['floor'][0]:>11.3e} | {r['warm'][0]:>11.3e} "
                f"| {r['opt'][0]:>11.3e} | {r['warm_over_floor']:>8.3f} | "
                f"{r['warm_over_opt']:>8.3f}"
            )
        print(
            f"  WARM START < floor (median, every M): "
            f"{'YES' if verdict['warm_below_floor_every_M'] else 'NO'}"
        )
        print(
            f"  WARM START >= per-instance optimum (median, every M): "
            f"{'YES' if verdict['warm_above_opt_every_M'] else 'no'}"
        )

    # ---- restore + render ----------------------------------------------
    def load_checkpoint(self):
        ckpt_path = os.path.join(
            checkpointsdir(self.args.experiment), "checkpoint.pth"
        )
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint does not exist: {ckpt_path}")
        self.ckpt = torch.load(ckpt_path, weights_only=False)
        self.results = self.ckpt["results"]

    def visualize(self):
        """Render the MMD-vs-M panel and the reconstruction figure (PDF and PNG)."""
        a = self.args
        out = plotsdir(a.experiment)
        res = self.results
        rows = res["rows"]
        self.r = res["r"]
        self.sigma = res["sigma"]
        self._print_table(rows, res["verdict"])

        with plt.rc_context(
            {"pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
             "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9}
        ):
            # ---- (1) MMD vs M (floor / warm / per-instance optimum) -----
            fig, axm = plt.subplots(1, 1, figsize=(4.6, 3.6))
            m_vals = [r["M"] for r in rows]
            for arm in ("floor", "warm", "opt"):
                med = [r[arm][0] for r in rows]
                lo = [r[arm][1] for r in rows]
                hi = [r[arm][2] for r in rows]
                yerr = [
                    [m - l for m, l in zip(med, lo)],
                    [h - m for m, h in zip(med, hi)],
                ]
                axm.errorbar(
                    m_vals, med, yerr=yerr, color=_PALETTE[arm], marker="o",
                    markersize=4, linewidth=1.5, capsize=2.5, label=_LABELS[arm],
                )
            axm.set_xscale("log", base=2)
            axm.set_yscale("log")
            axm.set_xticks(m_vals)
            axm.set_xticklabels([str(m) for m in m_vals])
            axm.set_xlabel(r"node budget $M$")
            axm.set_ylabel(r"$\mathrm{MMD}^2(Q, \rho)$")
            axm.spines["top"].set_visible(False)
            axm.spines["right"].set_visible(False)
            axm.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                path = os.path.join(out, f"dq_cnf_tomography_mmd_vs_M.{ext}")
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(f"Saved to {path}", flush=True)
            plt.close(fig)

            # ---- (2) true / flow mean / quadrature mean / per-pixel spread
            # The true, flow-mean and quadrature-mean panels share one
            # symmetric diverging field range and a single labelled colorbar,
            # so the quadrature mean can be read against the truth and the flow
            # mean directly; the spread panel carries its own scale and
            # colorbar.
            import numpy as np
            recon = res["recon"]
            n = recon["n_img"]
            cases = recon["cases"]
            ncase = len(cases)
            trues = [np.asarray(c["true"]).reshape(n, n) for c in cases]
            cnfms = [np.asarray(c["cnf_mean"]).reshape(n, n) for c in cases]
            recs = [np.asarray(c["recon"]).reshape(n, n) for c in cases]
            uqs = [np.asarray(c["uq"]).reshape(n, n) for c in cases]
            # Shared symmetric field scale across true / CNF mean / quad mean;
            # shared UQ scale.
            field_vmax = float(max(
                max(np.abs(t).max() for t in trues),
                max(np.abs(m).max() for m in cnfms),
                max(np.abs(r).max() for r in recs),
            ))
            uq_vmax = float(max(u.max() for u in uqs))
            fig, axes = plt.subplots(
                ncase, 4, figsize=(7.8, 1.95 * ncase), squeeze=False
            )
            col_titles = [
                "true", "CNF mean", "quadrature mean", "per-pixel UQ",
            ]
            im_field = im_uq = None
            for i in range(ncase):
                im_field = axes[i, 0].imshow(
                    trues[i], cmap="RdBu_r", vmin=-field_vmax, vmax=field_vmax,
                    interpolation="nearest",
                )
                axes[i, 1].imshow(
                    cnfms[i], cmap="RdBu_r", vmin=-field_vmax, vmax=field_vmax,
                    interpolation="nearest",
                )
                axes[i, 2].imshow(
                    recs[i], cmap="RdBu_r", vmin=-field_vmax, vmax=field_vmax,
                    interpolation="nearest",
                )
                im_uq = axes[i, 3].imshow(
                    uqs[i], cmap="viridis", vmin=0.0, vmax=uq_vmax,
                    interpolation="nearest",
                )
                for j in range(4):
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                    for sp in axes[i, j].spines.values():
                        sp.set_visible(False)
                    if i == 0:
                        axes[i, j].set_title(
                            col_titles[j], fontsize=9, color="0.25", pad=4
                        )
            # One shared field colorbar (true + CNF mean + quadrature mean) and
            # one UQ colorbar, spanning their respective columns.
            cb_field = fig.colorbar(
                im_field, ax=axes[:, :3].ravel().tolist(),
                fraction=0.046, pad=0.02, aspect=24,
            )
            cb_field.set_label("field value", fontsize=8)
            cb_field.ax.tick_params(labelsize=7)
            cb_uq = fig.colorbar(
                im_uq, ax=axes[:, 3].ravel().tolist(),
                fraction=0.046, pad=0.02, aspect=24,
            )
            cb_uq.set_label("posterior std", fontsize=8)
            cb_uq.ax.tick_params(labelsize=7)
            for ext in ("pdf", "png"):
                path = os.path.join(out, f"dq_cnf_tomography_recon.{ext}")
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(f"Saved to {path}", flush=True)
            plt.close(fig)


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload"],
        sequence_args_and_types=[("m_list", int)],
    )

    experiment = DQCNFTomography(args)
    if args.phase == "train":
        experiment.train()

    experiment.load_checkpoint()
    experiment.visualize()

    if args.upload:
        upload_to_cloud(args)
