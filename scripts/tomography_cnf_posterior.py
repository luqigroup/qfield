"""Train a conditional CNF as the tomography posterior.

Learns ``rho = p(x | c)`` for limited-angle brain-phantom tomography, where
``x`` is a ``64x64`` phantom and ``c = FBP(y)`` is the Ram-Lak
filtered-backprojection conditioner rendered on the same grid. The model is
:class:`qfield.models.genparihaka_cnf.ConditionalConvHINT` (a conditional
convolutional HINT flow) with single-channel ``64x64`` ``x`` conditioned on
single-channel ``64x64`` ``c``.

Training is maximum likelihood (minimize ``NLL = -log p(x | c)``) with Adam +
cosine schedule, batches STREAMED from the HDF5 file (the 763 MB dataset is
never fully resident). Checkpoints (model + optimizer + step + best-val-NLL +
RNG) are written FREQUENTLY to ``data/checkpoints/tomography_cnf/`` so the run
survives a parent-process death and resumes from the latest checkpoint.

Every CUDA allocation (model build, train step, sampling) is wrapped in an
OOM-retry-with-backoff, so a transient memory spike from another process on the
same GPU pauses the run rather than killing it.

The visualization step samples ``n_samples`` posterior samples for a few
held-out test cases given ``c = FBP(y)``, forms the posterior mean and
per-pixel standard deviation, reports correlation / PSNR of the mean against
the true ``x`` and against the FBP baseline, reports how far the per-pixel
standard deviation tracks edges and the error, and renders
``tomography_cnf_posterior.png`` (rows = test cases, columns =
[true x | FBP c | CNF mean | CNF std]).

Run (train; resumes if a checkpoint exists):
    python scripts/tomography_cnf_posterior.py
Visualize only (from the best checkpoint):
    python scripts/tomography_cnf_posterior.py --phase visualization
"""

from __future__ import annotations
from projorg import datadir, plotsdir  # noqa: E402

import argparse
import json
import math
import os
import time

import numpy as np

DATA_H5 = os.path.join(datadir("tomography_ellipse"), "tomography_ellipse.h5")
CKPT_DIR = os.path.join(datadir("checkpoints"), "tomography_cnf")
FIG_DIR = plotsdir("paper")

# Limit allocator fragmentation; must be set before torch imports CUDA.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import h5py  # noqa: E402
import torch  # noqa: E402


# --------------------------------------------------------------------------- #
# OOM-resilient helpers (another process on the GPU can spike transiently).
# --------------------------------------------------------------------------- #
def oom_retry(fn, *, tries: int = 200, wait: float = 4.0, label: str = ""):
    """Run ``fn``, retrying on CUDA OOM with a fixed backoff."""
    last = None
    for att in range(tries):
        try:
            return fn()
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            last = e
            torch.cuda.empty_cache()
            if att % 10 == 0:
                print(f"  [oom-retry {label}] attempt {att}, waiting {wait}s ...",
                      flush=True)
            time.sleep(wait)
    raise RuntimeError(f"persistent OOM in {label}") from last


# --------------------------------------------------------------------------- #
# Conditioner normalization. FBP ``c`` has a tiny native scale (std ~6e-3), so
# it is standardized with FIXED train-set statistics to give the coupling
# subnets an O(1) signal (the statistics are saved in the checkpoint so eval
# and resume stay consistent).
# --------------------------------------------------------------------------- #
def compute_cond_stats(h5_path: str, n: int = 4000) -> tuple[float, float]:
    with h5py.File(h5_path, "r") as f:
        c = f["train/c"][:n].astype("float32")
    return float(c.mean()), float(c.std() + 1e-12)


def make_loader(h5_path: str, split: str, batch_size: int, c_mean: float,
                c_std: float, *, shuffle: bool, seed: int = 0):
    """Yield ``(x, c)`` minibatches streamed from HDF5 (chunked reads).

    ``x`` is the phantom (native [0, 1.1]); ``c`` is FBP standardized by the
    fixed train stats. Both returned as ``[B, 1, 64, 64]`` float32 tensors.
    """
    f = h5py.File(h5_path, "r")
    xset, cset = f[f"{split}/x"], f[f"{split}/c"]
    n = xset.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        idx = np.sort(order[i:i + batch_size])  # sorted for fast h5 fancy read
        x = xset[idx].astype("float32")
        c = (cset[idx].astype("float32") - c_mean) / c_std
        x = torch.from_numpy(x)[:, None]
        c = torch.from_numpy(c)[:, None]
        yield x, c
    f.close()


def build_model(cfg, device):
    from qfield.models.genparihaka_cnf import ConditionalConvHINT

    def _b():
        torch.manual_seed(cfg["model_seed"])
        m = ConditionalConvHINT(
            in_channels=1,
            cond_in_channels=1,
            spatial_size=64,
            hidden_ch=cfg["hidden_ch"],
            n_flow_layers=cfg["n_flow_layers"],
            cond_feat_ch=cfg["cond_feat_ch"],
        ).to(device)
        return m

    return oom_retry(_b, label="build_model")


# --------------------------------------------------------------------------- #
# Train.
# --------------------------------------------------------------------------- #
def save_ckpt(path, model, opt, sched, step, epoch, best_val, cfg, rng_state):
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict() if sched is not None else None,
            "step": step,
            "epoch": epoch,
            "best_val": best_val,
            "cfg": cfg,
            "torch_rng": rng_state,
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic: a crash mid-write never corrupts the ckpt


@torch.no_grad()
def eval_nll(model, h5_path, cfg, device, max_batches: int):
    model.eval()
    tot, cnt = 0.0, 0
    d = 64 * 64
    for bi, (x, c) in enumerate(make_loader(
            h5_path, "test", cfg["batch_size"], cfg["c_mean"], cfg["c_std"],
            shuffle=False)):
        if bi >= max_batches:
            break
        x, c = x.to(device), c.to(device)
        x = x + cfg["dequant_sigma"] * torch.randn_like(x)  # same target as train
        lp = oom_retry(lambda: model.log_prob(x, c), label="eval")
        # Robust: ignore any non-finite per-sample logp in the running mean so a
        # single pathological sample cannot NaN the whole validation number.
        lp = lp[torch.isfinite(lp)]
        tot += float((-lp).sum().item())
        cnt += int(lp.numel())
    model.train()
    return tot / max(cnt, 1) / d  # NLL per dim (nats/pixel) for readability


def train(cfg):
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    latest = os.path.join(CKPT_DIR, "latest.pt")
    best = os.path.join(CKPT_DIR, "best.pt")

    # Conditioner stats (fixed; persisted in cfg via ckpt).
    if "c_mean" not in cfg:
        cm, cs = compute_cond_stats(DATA_H5)
        cfg["c_mean"], cfg["c_std"] = cm, cs
        print(f"cond stats: mean={cm:.5g} std={cs:.5g}", flush=True)

    model = build_model(cfg, device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparams:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * cfg["steps_per_epoch"]
    warmup = cfg["warmup_steps"]

    def lr_lambda(s):
        if s < warmup:
            return s / max(warmup, 1)
        prog = (s - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    step, epoch, best_val = 0, 0, float("inf")
    # Resume: prefer latest.pt, but FALL BACK to best.pt if latest is
    # non-finite (a prior run may have diverged to NaN after saving best).
    resume_from = None
    for cand in (latest, best):
        if os.path.exists(cand):
            ckc = torch.load(cand, map_location="cpu")
            ok = all(torch.isfinite(v).all().item()
                     for v in ckc["model"].values()
                     if v.dtype.is_floating_point)
            if ok:
                resume_from = cand
                break
            print(f"  {cand} is NON-FINITE, skipping for resume", flush=True)
    if resume_from is not None:
        ck = torch.load(resume_from, map_location=device)
        model.load_state_dict(ck["model"])
        # Only restore optimizer/sched/step if resuming from the SAME run state
        # (latest). When falling back to best, restart the optimizer fresh from
        # best's weights (the latest opt state is poisoned) but keep the step
        # so the LR schedule continues sensibly.
        if resume_from == latest:
            opt.load_state_dict(ck["opt"])
            if ck.get("sched") is not None:
                sched.load_state_dict(ck["sched"])
            if ck.get("torch_rng") is not None:
                torch.set_rng_state(ck["torch_rng"].cpu())
        step, epoch, best_val = ck["step"], ck["epoch"], ck["best_val"]
        if resume_from != latest:
            for _ in range(step):
                sched.step()  # advance schedule to match the resumed step
        cfg["c_mean"], cfg["c_std"] = ck["cfg"]["c_mean"], ck["cfg"]["c_std"]
        print(f"RESUMED from {resume_from}: step={step} epoch={epoch} "
              f"best_val={best_val:.4f}", flush=True)

    model.train()
    d = 64 * 64
    ema = None
    n_skip = 0
    t_log = time.time()
    while epoch < cfg["epochs"]:
        seen = 0
        for x, c in make_loader(DATA_H5, "train", cfg["batch_size"],
                                cfg["c_mean"], cfg["c_std"], shuffle=True,
                                seed=cfg["base_seed"] + epoch):
            x, c = x.to(device), c.to(device)
            # DEQUANTIZATION: the phantom x is heavily QUANTIZED (~18 levels;
            # 56% of pixels exactly 0). A continuous flow tries to place a delta
            # (infinite density) on those discrete values -> log-density and
            # log_det blow up to +-inf -> NaN. Adding small Gaussian noise turns
            # the discrete target into a smooth one the flow can fit stably.
            x = x + cfg["dequant_sigma"] * torch.randn_like(x)

            def _step():
                # Returns (loss_value, applied?). Affine-coupling flows can
                # spike to non-finite loss/grad on a bad batch (the exp scale,
                # clamp [-5,5] -> S up to e^5); applying that update poisons the
                # weights to NaN forever. So we backprop, then SKIP opt.step if
                # the loss OR the global grad-norm is non-finite -- the bad
                # batch is dropped, the weights stay clean.
                opt.zero_grad(set_to_none=True)
                lp = model.log_prob(x, c)
                loss = -(lp.mean()) / d  # nats/pixel
                if not torch.isfinite(loss):
                    return float("nan"), False
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["grad_clip"])
                if not torch.isfinite(gnorm):
                    opt.zero_grad(set_to_none=True)
                    return float(loss.item()), False
                opt.step()
                return float(loss.item()), True

            lv, applied = oom_retry(_step, label="train_step")
            if applied:
                sched.step()
            else:
                n_skip += 1
                if n_skip <= 20 or n_skip % 50 == 0:
                    print(f"  [skip] non-finite step at epoch {epoch} step "
                          f"{step} (total skips {n_skip})", flush=True)
            step += 1
            seen += x.shape[0]
            if math.isfinite(lv):
                ema = lv if ema is None else 0.98 * ema + 0.02 * lv

            if step % cfg["log_every"] == 0:
                dt = time.time() - t_log
                print(f"epoch {epoch} step {step} nll/pix {lv:.4f} "
                      f"(ema {ema:.4f}) lr {sched.get_last_lr()[0]:.2e} "
                      f"{cfg['log_every'] / dt:.1f} it/s", flush=True)
                t_log = time.time()

            if step % cfg["ckpt_every"] == 0:
                save_ckpt(latest, model, opt, sched, step, epoch, best_val,
                          cfg, torch.get_rng_state())

            if step % cfg["steps_per_epoch"] == 0:
                break

        epoch += 1
        # End-of-epoch validation + best tracking.
        val = eval_nll(model, DATA_H5, cfg, device, cfg["val_batches"])
        print(f"== epoch {epoch} done: VAL nll/pix {val:.4f} "
              f"(best {best_val:.4f}) ==", flush=True)
        save_ckpt(latest, model, opt, sched, step, epoch, best_val, cfg,
                  torch.get_rng_state())
        if val < best_val:
            best_val = val
            save_ckpt(best, model, opt, sched, step, epoch, best_val, cfg,
                      torch.get_rng_state())
            print(f"  new BEST val {best_val:.4f} -> {best}", flush=True)

    print(f"TRAINING DONE: {cfg['epochs']} epochs, best val nll/pix "
          f"{best_val:.4f}", flush=True)
    return best if os.path.exists(best) else latest


# --------------------------------------------------------------------------- #
# Verify + figure.
# --------------------------------------------------------------------------- #
def _corr(a, b):
    a = a.ravel().astype("float64")
    b = b.ravel().astype("float64")
    return float(np.corrcoef(a, b)[0, 1])


def _psnr(true, est):
    mse = float(np.mean((true - est) ** 2))
    peak = float(true.max() - true.min()) or 1.0
    return 10 * math.log10(peak ** 2 / (mse + 1e-12))


def visualize(cfg, ckpt_path, n_cases: int = 4, n_samples: int = 64):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device)
    cfg = {**cfg, **{k: ck["cfg"][k] for k in ("c_mean", "c_std", "hidden_ch",
            "n_flow_layers", "cond_feat_ch", "model_seed")}}
    model = build_model(cfg, device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded ckpt step={ck['step']} val={ck['best_val']:.4f}", flush=True)

    # First n_cases held-out test items.
    with h5py.File(DATA_H5, "r") as f:
        x_true = f["test/x"][:n_cases].astype("float32")
        c_raw = f["test/c"][:n_cases].astype("float32")
    c_std_img = (c_raw - cfg["c_mean"]) / cfg["c_std"]
    c_t = torch.from_numpy(c_std_img)[:, None].to(device)

    means, stds, corr_m, corr_f, psnr_m, psnr_f = [], [], [], [], [], []
    for i in range(n_cases):
        ci = c_t[i:i + 1]
        smp = oom_retry(lambda: model.sample(ci, n_samples), label="sample")
        smp = smp[0, :, 0].cpu().numpy()  # [n_samples, 64, 64]
        m = smp.mean(0)
        s = smp.std(0)
        means.append(m)
        stds.append(s)
        corr_m.append(_corr(x_true[i], m))
        corr_f.append(_corr(x_true[i], c_raw[i]))
        psnr_m.append(_psnr(x_true[i], m))
        psnr_f.append(_psnr(x_true[i], c_raw[i]))

    print("per-case  corr(mean) / corr(FBP) | psnr(mean) / psnr(FBP)",
          flush=True)
    for i in range(n_cases):
        print(f"  case {i}: {corr_m[i]:.3f} / {corr_f[i]:.3f} | "
              f"{psnr_m[i]:.2f} / {psnr_f[i]:.2f}", flush=True)
    print(f"MEAN corr(CNF)={np.mean(corr_m):.3f}  corr(FBP)={np.mean(corr_f):.3f}"
          f"  psnr(CNF)={np.mean(psnr_m):.2f}  psnr(FBP)={np.mean(psnr_f):.2f}",
          flush=True)

    # UQ sanity: correlation of per-pixel std with the |edge| map of the truth.
    edge_corrs = []
    for i in range(n_cases):
        gy, gx = np.gradient(x_true[i].astype("float64"))
        edges = np.hypot(gy, gx)
        edge_corrs.append(_corr(edges, stds[i]))
    print(f"UQ sanity: corr(per-pixel std, |grad x_true|) mean="
          f"{np.mean(edge_corrs):.3f} (positive => std tracks edges)",
          flush=True)

    # Per-pixel error and the calibration check: does the std predict the error?
    errs = [np.abs(means[i] - x_true[i]) for i in range(n_cases)]
    cal_corrs = [_corr(stds[i], errs[i]) for i in range(n_cases)]
    print(f"calibration: corr(per-pixel std, |mean - x_true|) mean="
          f"{np.mean(cal_corrs):.3f} (positive => std predicts error)",
          flush=True)

    # Figure: rows = cases, cols = [true | FBP | CNF mean | |error| | CNF std].
    # |error| and CNF std share ONE magma scale (vmin=0, common vmax across all
    # cases) so their spatial correlation -- the calibration -- is read off
    # directly, and the std colour range is consistent across rows.
    os.makedirs(FIG_DIR, exist_ok=True)
    # Robust shared scale: the 99th percentile of the pooled std + |error|
    # values, so a single high-error case does not wash out the structure of
    # the rest while the colour range stays consistent across all rows.
    _pooled = np.concatenate([s.ravel() for s in stds] + [e.ravel() for e in errs])
    unc_vmax = float(np.percentile(_pooled, 99.0))
    fig, axes = plt.subplots(n_cases, 5, figsize=(11.2, 2.3 * n_cases))
    if n_cases == 1:
        axes = axes[None]
    col_titles = ["true $x$", "FBP $c$", "CNF mean", "$|$error$|$", "CNF std"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11)
    for i in range(n_cases):
        vmax = float(x_true[i].max())
        im0 = axes[i, 0].imshow(x_true[i], cmap="gray", vmin=0, vmax=vmax)
        im1 = axes[i, 1].imshow(c_raw[i], cmap="gray")
        im2 = axes[i, 2].imshow(means[i], cmap="gray", vmin=0, vmax=vmax)
        im3 = axes[i, 3].imshow(errs[i], cmap="magma", vmin=0, vmax=unc_vmax)
        im4 = axes[i, 4].imshow(stds[i], cmap="magma", vmin=0, vmax=unc_vmax)
        for ax, im in zip(axes[i], (im0, im1, im2, im3, im4)):
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    png = os.path.join(FIG_DIR, "tomography_cnf_posterior.png")
    pdf = os.path.join(FIG_DIR, "tomography_cnf_posterior.pdf")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"FIGURE: {png}", flush=True)

    # --- Individual panels -------------------------------------------------
    # Each cell is dumped image-only (no axes, no colorbar) so it can be laid
    # out in a grid elsewhere; true/mean share one gray range, |error|/std
    # share the magma range, and the two colorbars are standalone.
    import matplotlib as mpl
    panel_dir = os.path.join(FIG_DIR, "cnf_panels")
    os.makedirs(panel_dir, exist_ok=True)
    gray_vmax = float(max(x_true.max(), max(m.max() for m in means)))

    def _panel(arr, name, cmap, vmin, vmax):
        f, a = plt.subplots(figsize=(2.0, 2.0))
        a.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        a.axis("off")
        for ext in ("pdf", "png"):
            f.savefig(os.path.join(panel_dir, f"{name}.{ext}"),
                      dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(f)

    def _cbar(name, cmap, vmin, vmax):
        f, a = plt.subplots(figsize=(0.32, 2.0))
        mpl.colorbar.ColorbarBase(
            a, cmap=plt.get_cmap(cmap),
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        f.savefig(os.path.join(panel_dir, f"{name}.pdf"),
                  bbox_inches="tight", pad_inches=0.02)
        plt.close(f)

    for i in range(n_cases):
        _panel(x_true[i], f"case{i}_true", "gray", 0, gray_vmax)
        _panel(c_raw[i], f"case{i}_fbp", "gray",
               float(c_raw[i].min()), float(c_raw[i].max()))
        _panel(means[i], f"case{i}_mean", "gray", 0, gray_vmax)
        _panel(errs[i], f"case{i}_err", "magma", 0, unc_vmax)
        _panel(stds[i], f"case{i}_std", "magma", 0, unc_vmax)
    _cbar("cbar_gray", "gray", 0, gray_vmax)
    _cbar("cbar_magma", "magma", 0, unc_vmax)
    print(f"PANELS: {panel_dir} ({n_cases} cases x 5 panels + 2 colorbars)",
          flush=True)

    return {
        "corr_cnf": float(np.mean(corr_m)),
        "corr_fbp": float(np.mean(corr_f)),
        "psnr_cnf": float(np.mean(psnr_m)),
        "psnr_fbp": float(np.mean(psnr_f)),
        "uq_edge_corr": float(np.mean(edge_corrs)),
        "figure": png,
    }


def default_cfg():
    return {
        "experiment_name": "tomography_cnf",
        # architecture. cond_feat_ch is 1 because at spatial_size=64 the
        # conditioner-channel squeeze blows up 4x per halving -> 4^6=4096x at
        # the 1x1 leaves: cond_feat_ch=32 gives a 14 GB / 3.5e9-param model.
        # cond_feat_ch=1 is 122M params and ~2.6 GB peak.
        "hidden_ch": 32,
        "n_flow_layers": 8,
        "cond_feat_ch": 1,
        "model_seed": 20260612,
        # optimization
        "batch_size": 16,
        "lr": 1.5e-4,
        "weight_decay": 0.0,
        "grad_clip": 5.0,
        # dequantization noise sigma on x (phantom quantized to ~18 levels,
        # spacing ~0.06; sigma ~1/3 of that smooths the delta without blurring).
        "dequant_sigma": 0.02,
        "epochs": 40,
        "steps_per_epoch": 1250,  # 20000/16
        "warmup_steps": 500,
        # bookkeeping
        "log_every": 50,
        "ckpt_every": 250,
        "val_batches": 30,
        "base_seed": 20260612,
        "phase": "train",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="train",
                    choices=["train", "visualization"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    cfg = default_cfg()
    if args.config and os.path.exists(args.config):
        cfg.update(json.load(open(args.config)))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
        cfg["steps_per_epoch"] = 20000 // args.batch_size

    if args.phase == "train":
        ckpt = train(cfg)
        visualize(cfg, ckpt)
    else:
        best = os.path.join(CKPT_DIR, "best.pt")
        latest = os.path.join(CKPT_DIR, "latest.pt")
        ckpt = best if os.path.exists(best) else latest
        visualize(cfg, ckpt)


if __name__ == "__main__":
    main()
