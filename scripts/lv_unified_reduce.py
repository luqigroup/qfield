"""Merge the sharded unified-budget runs into one record.

One shard per ``(observation, budget)`` cell, as the array job writes them.
Pooling is over the RAW per-cell values the shards carry, so the median is the
median over every (observation, replicate) actually run, not a median of
medians.

    python scripts/lv_unified_reduce.py <shard_dir>
"""
from __future__ import annotations
from projorg import datadir  # noqa: E402

import glob
import json
import os
import sys

import numpy as np

ARMS = ["floor", "rw", "mv", "ours", "thinning", "stein",
        "herding", "sbq", "recombine"]


def main() -> None:
    shard_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(datadir("lotka_volterra"), "unified_shards")
    files = sorted(glob.glob(os.path.join(shard_dir, "*.json")))
    if not files:
        raise SystemExit(f"no shards under {shard_dir}")

    raw: dict[int, dict[str, list]] = {}
    res: dict[int, list] = {}
    secs: dict[int, dict[str, list]] = {}
    meta, obs_seen = None, set()
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        meta = meta or {k: d[k] for k in
                        ("M", "sigma", "checkpoint", "score_L", "what")}
        obs_seen.update(d.get("shard", {}).get("observations", []))
        for n_s, cell in d["raw"].items():
            n = int(n_s)
            raw.setdefault(n, {a: [] for a in ARMS})
            for a in ARMS:
                raw[n][a].extend(cell.get(a, []))
            res.setdefault(n, []).extend(d["raw_resolution"][n_s])
            # The shard divides its accumulated seconds by the declared
            # observation count while running only the observations in its own
            # slice, so it under-reports by exactly that ratio. Undo it here.
            n_decl = int(d.get("n_obs", 1)) or 1
            n_ran = max(1, len(d.get("shard", {}).get("observations", [1])))
            fix = n_decl / n_ran
            sc = d["cells"].get(n_s, {}).get("secs_per_cell", {})
            secs.setdefault(n, {})
            for a, v in sc.items():
                secs[n].setdefault(a, []).append(v * fix)

    out = dict(meta or {})
    out["n_shards"] = len(files)
    out["n_observations_pooled"] = len(obs_seen)
    out["n_grid"] = sorted(raw)
    out["cells"] = {}
    for n in sorted(raw):
        row = {}
        for a in ARMS:
            v = np.array(raw[n][a], dtype=float)
            v = v[np.isfinite(v)]
            row[a] = float(np.median(v)) if v.size else None
            row[f"{a}_n"] = int(v.size)
        row["resolution"] = float(np.median(res[n]))
        row["secs_per_cell"] = {a: float(np.mean(x))
                                for a, x in secs.get(n, {}).items()}
        out["cells"][str(n)] = row

    print(f"pooled {len(files)} shards, {len(obs_seen)} observations\n")
    hdr = [a for a in ARMS if a != "floor"]
    print(f"{'n':>7s} " + "".join(f"{a:>11s}" for a in hdr))
    for n in sorted(raw):
        r = out["cells"][str(n)]
        line = f"{n:7d} "
        for a in hdr:
            line += (f"{r['floor'] / r[a]:10.1f}x"
                     if r.get(a) and r[a] > 0 else f"{'-':>11s}")
        print(line)
    print("\n(depth below the independent-sample floor; "
          "'ours' = move + solve, 'rw' = solve only, 'mv' = move only)")

    if any(out["cells"][str(n)].get("secs_per_cell") for n in sorted(raw)):
        print("\nwall seconds per observation (one emission, one budget)\n")
        print(f"{'n':>7s} " + "".join(f"{a:>11s}" for a in hdr))
        for n in sorted(raw):
            sc = out["cells"][str(n)].get("secs_per_cell", {})
            line = f"{n:7d} "
            for a in hdr:
                line += (f"{sc[a]:10.3f}s" if a in sc and sc[a] >= 0.0005
                         else (f"{'<1ms':>11s}" if a in sc else f"{'-':>11s}"))
            print(line)

    dst = os.path.join(datadir("records"), "lv_unified_budget.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[done] -> {dst}")


if __name__ == "__main__":
    main()
