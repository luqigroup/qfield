"""The two-lever split across every re-scored reference, in three currencies.

The split can be quoted three ways, and they sound like they disagree:

  phi        = 1 - reweight/floor   -- the SHARE OF THE FLOOR the constrained
                                       solve removes;
  move_gain  = reweight/warm        -- the LINEAR FACTOR the learned
                                       displacement then applies to what is
                                       left;
  move share = (log(floor/warm) - log(floor/reweight)) / log(floor/warm)
                                    -- the displacement's share of the total
                                       LOG DEPTH below the floor.

All three are the same measurement, and quoting one in another's language is
the easiest way to get a description wrong, so this harness prints them
together and never one alone.

The ``warm`` arm in these records is raw, not safeguarded: the trainers'
evaluation sites compute ``net(z0)`` and solve, and neither applies
``select_emission``. So ``move_gain < 1`` says the trained displacement alone
lost to the reweighting it was handed; it does not say the selected rule got
worse, because that rule selects from a menu containing the reweighted seeds
and would return those. Two consequences:

  * a net-loss cell is a statement about the map, never about what is returned;
  * the plotted ``warm`` curve is conservative at such a cell, showing a value
    slightly worse than what would actually be returned.

Evaluation-only and CPU-only: a pure read of the ``results.json`` the trainers
wrote.

Two anchors are printed, and the second is not optional. Each reference's top
readable budget -- the largest cell that is neither bank-limited nor past the
``cond(K)`` limit -- answers what that reference does at the deepest budget it
can be read at, since a rank-deficient weight solve carries no ordering
between arms. It is the wrong anchor for any claim about a trend in the
dimension, because the top readable budget itself varies with the reference,
so a dimension trend read off that column confounds dimension with budget. The
common anchor, the largest budget readable on every reference, is what a
dimension claim must be quoted at.

Run:
    CUDA_VISIBLE_DEVICES="" python scripts/dq_lever_summary.py
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import glob
import json
import math
import os

GRID = "*m_list-32-64-128-256-512*"
OUT = os.path.join(datadir("records"), "dq_lever_summary.json")


def _name(d: str, r: dict) -> str:
    if r.get("target") is None and r.get("r") is not None:
        return f"tomography r={int(r['r'])}"
    return f"{r['target']} d={int(r['d'])}"


def _split(row: dict) -> dict:
    """The three currencies for one cell."""
    fl, rw, wa = row["floor"][0], row["reweight"][0], row["warm"][0]
    d_rw = math.log10(fl / rw) if rw > 0 else float("inf")
    d_wa = math.log10(fl / wa) if wa > 0 else float("inf")
    return {
        "M": int(row["M"]),
        "phi": row["phi"],
        "move_gain": row["move_gain"],
        "move_log_share": (d_wa - d_rw) / d_wa if d_wa > 0 else float("nan"),
        "depth_reweight_dex": d_rw,
        "depth_warm_dex": d_wa,
        "move_is_a_loss": bool(row["move_gain"] < 1.0),
    }


def rows() -> tuple[list[dict], int | None]:
    """Per-reference splits, plus the largest budget readable on ALL of them."""
    out = []
    for rj in sorted(glob.glob(os.path.join(datadir("checkpoints"), GRID, "results.json"))):
        r = json.load(open(rj))
        rr = r.get("rows", [])
        if not rr or "reweight" not in rr[0]:
            continue  # not re-scored yet
        v = r.get("verdict", {})
        top = v.get("top_readable_M")
        row = next((x for x in rr if x["M"] == top), None)
        if row is None:
            continue
        bad = set(v.get("cond_limited_M", [])) | set(v.get("bank_limited_M", []))
        out.append({
            "reference": _name(rj, r),
            "top_readable_M": int(top),
            "readable_M": sorted(x["M"] for x in rr if x["M"] not in bad),
            "at_top": _split(row),
            "by_M": {str(x["M"]): _split(x) for x in rr if x["M"] not in bad},
            "cond_limited_M": v.get("cond_limited_M", []),
            "bank_limited_M": v.get("bank_limited_M", []),
        })
    out.sort(key=lambda x: x["reference"])
    common = None
    if out:
        shared = set(out[0]["readable_M"])
        for x in out[1:]:
            shared &= set(x["readable_M"])
        common = max(shared) if shared else None
    return out, common


def _table(rs: list[dict], key, title: str) -> None:
    print(f"\n{title}")
    print(f"{'reference':22s} {'M':>5} {'phi':>8} {'move x':>8} "
          f"{'move log-share':>15}   marked")
    print("-" * 78)
    for x in rs:
        s = key(x)
        if s is None:
            print(f"{x['reference']:22s} {'--':>5}   (not readable there)")
            continue
        mark = ",".join(str(m) for m in x["cond_limited_M"]) or "-"
        flag = "  <-- NET LOSS" if s["move_is_a_loss"] else ""
        print(f"{x['reference']:22s} {s['M']:>5} {s['phi']:>8.4f} "
              f"{s['move_gain']:>8.3f} {s['move_log_share']:>15.3f}   "
              f"{mark}{flag}")


def main() -> None:
    rs, common = rows()
    if not rs:
        print("No re-scored records yet (no row carries a reweight arm).")
        return
    _table(rs, lambda x: x["at_top"],
           "AT EACH REFERENCE'S TOP READABLE BUDGET "
           "(deepest reading; NOT comparable across references)")
    if common is not None:
        _table(rs, lambda x: x["by_M"].get(str(common)),
               f"AT THE COMMON BUDGET M={common} "
               "(the anchor any dimension claim must be quoted at)")
    else:
        print("\nNO COMMON READABLE BUDGET across these references -- no "
              "dimension claim can be quoted at a fixed budget.")
    print("-" * 78)
    print(f"{len(rs)} reference(s) re-scored. phi = share of the floor the FREE "
          "solve removes;\nmove x = the learned displacement's factor on what "
          "remains; log-share = its\nshare of the total depth below the floor. "
          "All three describe one measurement.")
    # Monotonicity of the decay is measured rather than assumed, and reported
    # per adjacent pair rather than endpoint to endpoint, which is what
    # distinguishes a monotone sequence from one that merely ends lower than
    # it starts.
    pairs = bad = 0
    for x in rs:
        seq = [x["by_M"][str(M)]["move_log_share"]
               for M in (32, 64, 128, 256, 512) if str(M) in x["by_M"]]
        for a, b in zip(seq, seq[1:]):
            pairs += 1
            bad += b > a
    if pairs:
        n_mono = sum(
            all(a >= b for a, b in zip(sq, sq[1:]))
            for sq in ([x["by_M"][str(M)]["move_log_share"]
                        for M in (32, 64, 128, 256, 512) if str(M) in x["by_M"]]
                       for x in rs)
        )
        print(f"\nMONOTONICITY of the displacement's log-share in M: "
              f"{n_mono}/{len(rs)} references monotone decreasing; "
              f"{pairs - bad}/{pairs} adjacent pairs fall."
              + ("" if bad else "  No exceptions."))

    losses = [(x["reference"], x["at_top"]["M"], x["at_top"]["move_gain"])
              for x in rs if x["at_top"]["move_is_a_loss"]]
    if losses:
        print("\nTHE TRAINED DISPLACEMENT IS A NET LOSS at the top readable "
              "budget on:")
        for ref, M, g in losses:
            print(f"    {ref} at M={M}: reweight/warm={g:.3f} "
                  f"({100 * (1 / g - 1):.1f}% worse than not moving)")
        print("These records' `warm` arm is RAW (no select_emission), so this "
              "is a statement\nabout the MAP, not about the emission: the "
              "shipped rule selects from a menu\ncontaining the reweighted "
              "seeds and would return those. The plotted curve is\nthus "
              "CONSERVATIVE at these cells. Never report a net loss as a gain, "
              "and never\nreport it as the emission getting worse.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"common_readable_M": common, "rows": rs}, fh, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
