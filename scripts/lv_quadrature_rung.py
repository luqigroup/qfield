"""Train the quadrature field on the Lotka--Volterra family.

One set-equivariant map takes an observation's summary and ``M`` independent
samples of that observation's reference posterior, and returns an ``M``-node
signed-weight quadrature, at any budget, in one forward pass, across the whole
family.

Everything happens in the Laplace-whitened frame, which is where the criterion
measures a distribution rather than a mean. The conditioner is the same
29-number summary the flow was given, fed to the amortizer's existing vector
token encoder.

Weights are solved in closed form at the emitted nodes, never learned, and
autograd flows through the solve.

    python scripts/lv_quadrature_rung.py
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402


from qfield.dataset import lotka_volterra as lv  # noqa: E402
from qfield.designed_quadrature.cond_net import (  # noqa: E402
    ConditionalQuadratureAmortizer,
)
from qfield.designed_quadrature.weights import (  # noqa: E402
    constrained_weights_batched,
)
from qfield.kernels import median_heuristic_sigma_sq  # noqa: E402
from qfield.models.hint_flow import HINTFlow  # noqa: E402

torch.set_num_threads(3)

OUT = datadir("lotka_volterra")
DTYPE = torch.float64
D = lv.D_X
BASE_SEED = 20260612
M_LIST = [32, 64, 128, 256, 512]
# Swept through the environment, so this arm can be tuned like the peers it is
# compared against.
N_MU = int(os.environ.get("LV_N_MU", 2048))
BATCH_OBS = int(os.environ.get("LV_BATCH_OBS", 8))
N_STEPS = int(os.environ.get("LV_N_STEPS", 8000))
LR = float(os.environ.get("LV_LR", 3e-4))
TAG = os.environ.get("LV_TAG", "")
JITTER = 1e-8
N_VAL_OBS = 64
VAL_EVERY = int(os.environ.get("LV_VAL_EVERY", 200))
PATIENCE = int(os.environ.get("LV_PATIENCE", 8))
# Subsample the training family; None (the default) uses every observation.
N_TRAIN_OBS = os.environ.get("LV_N_TRAIN_OBS")
# Validation budgets.
VAL_M = [int(x) for x in os.environ.get("LV_VAL_M", "32,64,128").split(",")]
HIDDEN = int(os.environ.get("LV_HIDDEN", 64))
N_BLOCKS = int(os.environ.get("LV_N_BLOCKS", 3))
N_HEADS = int(os.environ.get("LV_N_HEADS", 4))
DELTA_SCALE = float(os.environ.get("LV_DELTA_SCALE", 0.5))


class Flow:
    """The reference: samples in the whitened frame, per observation."""

    def __init__(self) -> None:
        ck = torch.load(os.path.join(
            OUT, f"flow{os.environ.get('LV_FLOW_TAG', '')}.pth"),
            weights_only=False)
        self.net = HINTFlow(n_in=D, n_cond=ck["n_cond"],
                            n_hidden=ck["n_hidden"],
                            n_flow_layers=ck["n_layers"]).to(DTYPE)
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.cm, self.cs, self.zs = ck["cond_mean"], ck["cond_std"], ck["z_std"]

    def cond(self, feat: np.ndarray, refine: np.ndarray) -> torch.Tensor:
        return torch.tensor((np.hstack([feat, refine]) - self.cm) / self.cs,
                            dtype=DTYPE)

    def draw(self, c: torch.Tensor, n: int) -> torch.Tensor:
        """``(B, n, d)`` samples of the reference, in the whitened frame."""
        B = c.shape[0]
        rep = c.repeat_interleave(n, 0)
        with torch.no_grad():
            z = self.net.sample(B * n, rep, dtype=DTYPE)
        return z.reshape(B, n, D)


def objective(net, z0, bank, sigma, cond, M):
    """``MMD^2`` up to the constant ``c_rho``, at the emitted nodes."""
    inv2s = 1.0 / (2.0 * sigma ** 2)
    z = net(z0, cond)
    mu = torch.exp(-(torch.cdist(z, bank) ** 2) * inv2s).mean(-1)
    w = constrained_weights_batched(z, mu, sigma, JITTER)
    G = torch.exp(-(torch.cdist(z, z) ** 2) * inv2s)
    return (torch.einsum("bi,bij,bj->b", w, G, w)
            - 2 * torch.einsum("bi,bi->b", w, mu))


def _save(state, sigma, cond_dim, curve) -> None:
    """Write the best-so-far checkpoint.

    Called on every improvement, not only at the end, so a long run killed
    partway does not lose everything. Written to a temporary file and renamed,
    so a kill mid-write cannot leave a truncated checkpoint behind.
    """
    dst = os.path.join(OUT, f"quadfield{TAG}.pth")
    tmp = dst + ".tmp"
    torch.save({"state": state, "sigma": sigma, "cond_dim": cond_dim,
                "hidden": HIDDEN, "n_blocks": N_BLOCKS, "n_heads": N_HEADS,
                "delta_scale": DELTA_SCALE, "m_list": M_LIST,
                "val_m": VAL_M, "n_steps": N_STEPS,
                "n_train_obs": int(N_TRAIN_OBS) if N_TRAIN_OBS else None,
                "curve": np.array(curve)}, tmp)
    os.replace(tmp, dst)


def main() -> None:
    t0 = time.time()
    flow = Flow()
    # Under LV_FRAME=new the conditioner carries the flow-moment frame too,
    # since that frame is what defines the coordinates the reference lives in.
    if os.environ.get("LV_FRAME") == "new":
        rf = np.load(os.path.join(OUT, "train_reframed.npz"))
        keep = rf["finite"]
        F = rf["feat"][keep]
        R = np.hstack([rf["refine"][keep], rf["frame"][keep]])
    else:
        d = np.load(os.path.join(OUT, "train_refined.npz"))
        ok = d["converged"]
        F, R = d["feat"][ok], d["refine"][ok]
    C_all = flow.cond(F, R)
    n_val = N_VAL_OBS
    C_val, C_tr = C_all[-n_val:], C_all[:-n_val]
    if N_TRAIN_OBS:
        C_tr = C_tr[:int(N_TRAIN_OBS)]
    print(f"[data] {C_tr.shape[0]} training observations, {n_val} held out "
          f"as a validation FAMILY (whole observations, never trained on)",
          flush=True)

    g = torch.Generator().manual_seed(BASE_SEED + 7)
    # Fix the bandwidth across a sweep. Recomputing it per run from a fresh
    # probe gives each configuration its own kernel, so the runs would not be
    # comparable.
    _pin = os.environ.get("LV_SIGMA")
    if _pin:
        sigma = float(_pin)
    else:
        probe = flow.draw(C_tr[:64], 100).reshape(-1, D)
        sigma = float(np.sqrt(median_heuristic_sigma_sq(probe)))
    print(f"[sigma] pooled median heuristic over flow draws: {sigma:.4f}",
          flush=True)

    torch.manual_seed(BASE_SEED)
    net = ConditionalQuadratureAmortizer(
        d=D, hidden=HIDDEN, n_blocks=N_BLOCKS, n_heads=N_HEADS,
        delta_scale=DELTA_SCALE, cond_kind="vector",
        cond_vec_dim=C_all.shape[1]).to(DTYPE)
    print(f"[net] vector-conditioned amortizer, "
          f"{sum(p.numel() for p in net.parameters()):,} parameters", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # The validation sets are sampled once and reused at every call (common
    # random numbers). Resampling them per call makes the signal carry a fresh
    # Monte-Carlo error of the same size as the achievable improvement, which
    # turns the validation curve into noise and early stopping into a coin
    # toss.
    torch.manual_seed(BASE_SEED + 55)
    _MV = max(VAL_M)
    with torch.no_grad():
        _VAL_DRAWS = [flow.draw(C_val[j:j + 8], _MV + N_MU)
                      for j in range(0, n_val, 8)]

    def _val() -> float:
        """Mean over the validation budgets of the mean per-budget objective.

        Averaging across budgets rather than validating at a single one, which
        the scoring bank may resolve on very few members of the family.
        """
        tot = 0.0
        with torch.no_grad():
            for Mv in VAL_M:
                for j, a in zip(range(0, n_val, 8), _VAL_DRAWS):
                    cb = C_val[j:j + 8]
                    tot += float(objective(net, a[:, :Mv],
                                           a[:, _MV:], sigma, cb, Mv).sum())
        return tot / (n_val * len(VAL_M))

    best, stale, best_sd = float("inf"), 0, None
    curve: list[tuple[int, float, float]] = []
    net.train()
    for step in range(N_STEPS):
        M = M_LIST[int(torch.randint(len(M_LIST), (1,), generator=g))]
        idx = torch.randperm(C_tr.shape[0], generator=g)[:BATCH_OBS]
        cb = C_tr[idx]
        a = flow.draw(cb, M + N_MU)
        loss = objective(net, a[:, :M], a[:, M:], sigma, cb, M).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if (step + 1) % VAL_EVERY == 0:
            net.eval()
            v = _val()
            net.train()
            curve.append((step + 1, float(loss), v))
            if v < best - 1e-10:
                best, stale = v, 0
                best_sd = {k: t.clone() for k, t in net.state_dict().items()}
                _save(best_sd, sigma, C_all.shape[1], curve)
            else:
                stale += 1
            print(f"  step {step+1:5d}  M={M:4d}  train {float(loss):+.6e}  "
                  f"val {v:+.6e}  best {best:+.6e}", flush=True)
            if stale >= PATIENCE:
                print(f"  early stop at step {step+1}", flush=True)
                break

    net.load_state_dict(best_sd)
    _save(best_sd, sigma, C_all.shape[1], curve)
    print(f"[done] tag={TAG!r} delta_scale={DELTA_SCALE} n_mu={N_MU} "
          f"hidden={HIDDEN} blocks={N_BLOCKS} lr={LR}  "
          f"best val {best:+.6e}  in {time.time()-t0:.0f} s", flush=True)


def _unused_old_save(best_sd, sigma, cond_dim, curve):
    torch.save({"state": best_sd, "sigma": sigma, "cond_dim": cond_dim,
                "hidden": HIDDEN, "n_blocks": N_BLOCKS, "n_heads": N_HEADS,
                "delta_scale": DELTA_SCALE, "m_list": M_LIST,
                "val_m": VAL_M, "n_steps": N_STEPS,
                "n_train_obs": int(N_TRAIN_OBS) if N_TRAIN_OBS else None,
                "curve": np.array(curve)},
               os.path.join(OUT, f"quadfield{TAG}.pth"))


if __name__ == "__main__":
    main()
