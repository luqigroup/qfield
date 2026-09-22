"""Harvest every level the floor-composite figure plots into a JSON record.

Imports ``render_composite_floor``'s own loaders rather than re-deriving them,
so both read one table of runs, and writes what the figure draws to a record
file. It trains nothing, samples nothing and re-plots nothing: it copies
numbers the runs already computed.

The record carries three things:

  * every arm of all six panels, at every budget, as plotted;
  * the shuffled-conditioner control (``null_heldout``) and the
    constant-conditioner control (``const_cond_heldout``), which the figure
    does not draw;
  * the ``cond_limited`` flag per cell, so a reader can tell which cells the
    figure draws open.

Run:  CUDA_VISIBLE_DEVICES="" python scripts/harvest_floor_composite.py
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from render_composite_floor import (  # noqa: E402
    DARCY_GLOB,
    M_LIST,
    MMD_VS_M_ROWS,
    TOMO_GLOB,
    _load_canon,
    _load_cnf,
    _load_multiseed,
    _ms_lookup,
    _one_results,
)

_OUT = os.path.join(datadir("records"), "dq_floor_composite.json")


def _triple(row, key):
    """A record's (median, q25, q75) triple, or None where the arm is absent."""
    v = row.get(key)
    if v is None:
        return None
    return [float(v[0]), float(v[1]), float(v[2])]


def _closed_form_panel(res, ms_cfg):
    """One closed-form rung, with its multiseed band if it has one."""
    rows = res["rows"]
    out = {
        "m_list": [int(r["M"]) for r in rows],
        "cond_limited": [bool(r.get("cond_limited")) for r in rows],
        "floor": [_triple(r, "floor") for r in rows],
        "reweight": [_triple(r, "reweight") for r in rows],
        "oracle": [_triple(r, "oracle") for r in rows],
        "warm_single_seed": [_triple(r, "warm") for r in rows],
    }
    if ms_cfg is not None:
        # What the panel draws: the across-training-seed median with an
        # interquartile band.
        out["warm_plotted_multiseed"] = [
            [float(x) for x in ms_cfg["per_M"][str(M)]["warm_mmd2"]["median_iqr"]]
            for M in out["m_list"]
        ]
        out["warm_over_floor_multiseed"] = [
            [float(x) for x in
             ms_cfg["per_M"][str(M)]["warm_over_floor"]["median_iqr"]]
            for M in out["m_list"]
        ]
        out["n_training_seeds"] = len(ms_cfg.get("seeds", [])) or None
    return out


def main() -> None:
    canon = _load_canon()
    ms = _load_multiseed()

    panels = {}

    # ---- row 1: the three closed-form references -------------------------
    missing = [k for k in MMD_VS_M_ROWS if canon.get(k) is None]
    if missing:
        raise RuntimeError(
            f"composite harvest: no canonical run for {missing}. Harvesting a "
            "record with a hole would give the provenance file the same gap "
            "the figure would have had."
        )
    # (family, kernel, d) keys into the multiseed record, in MMD_VS_M_ROWS
    # order.
    ms_keys = (("gaussian", "iso", 2), ("gmm", "iso", 10), ("banana", "iso", 2))
    ms_cfgs = [_ms_lookup(ms, *k) for k in ms_keys]
    if any(c is None for c in ms_cfgs):
        absent = [k for k, c in zip(ms_keys, ms_cfgs) if c is None]
        raise RuntimeError(
            f"composite harvest: the multiseed record has no config for "
            f"{absent}, so the harvested record would band some panels and "
            "not others."
        )
    for key, cfg in zip(MMD_VS_M_ROWS, ms_cfgs):
        panels[key] = _closed_form_panel(canon[key], cfg)

    # ---- row 2, left: the exact conditional posterior --------------------
    tomo = _one_results(TOMO_GLOB, "tomography rung")
    panels["tomography"] = {
        "m_list": [int(r["M"]) for r in tomo["rows"]],
        "cond_limited": [bool(r.get("cond_limited")) for r in tomo["rows"]],
        "floor": [_triple(r, "floor") for r in tomo["rows"]],
        "reweight": [_triple(r, "reweight") for r in tomo["rows"]],
        "oracle": [_triple(r, "oracle") for r in tomo["rows"]],
        "warm": [_triple(r, "warm") for r in tomo["rows"]],
        "warm_over_floor": [float(r["warm_over_floor"]) for r in tomo["rows"]],
        "note": (
            "One representative held-out observation under an UNCONDITIONAL "
            "amortizer: this rung exercises the exact kernel mean, not the "
            "conditioning."
        ),
    }

    # ---- row 2, middle: the trained conditional flow ---------------------
    cnf = _load_cnf()
    panels["trained_flow"] = {
        "m_list": list(M_LIST),
        "n_pipeline_seeds": cnf["_n_seeds"],
        "band": "min-max across the per-seed medians",
        **{arm: {"median": [float(x) for x in cnf[arm][0]],
                 "lo": [float(x) for x in cnf[arm][1]],
                 "hi": [float(x) for x in cnf[arm][2]]}
           for arm in ("floor", "warm", "opt")},
        "note": (
            "This rung's records carry NO reweight arm, so the panel draws "
            "three of the figure's five curves."
        ),
    }

    # ---- row 2, right: the groundwater rung, plus both controls ----------
    darcy = _one_results(DARCY_GLOB, "groundwater rung")
    drows = darcy["rows"]
    panels["groundwater"] = {
        "m_list": [int(r["M"]) for r in drows],
        "cond_limited": [bool(r.get("cond_limited")) for r in drows],
        "floor_heldout": [_triple(r, "floor_heldout") for r in drows],
        "reweight_heldout": [_triple(r, "reweight_heldout") for r in drows],
        "ours_heldout": [_triple(r, "ours_heldout") for r in drows],
        "oracle": [_triple(r, "opt") for r in drows],
        "ours_over_floor_heldout": [
            float(r["ours_over_floor_heldout"]) for r in drows],
        "activation": [float(r["activation"]) for r in drows],
        # The two controls. The figure draws the shuffled one and not the
        # constant one.
        "null_heldout_shuffled_conditioner": [
            _triple(r, "null_heldout") for r in drows],
        "const_cond_heldout": [_triple(r, "const_cond_heldout")
                               for r in drows],
        "note": (
            "Read on the HELD-OUT bank B. Bank A's ours <= floor is a "
            "certificate, so plotting it would plot a theorem."
        ),
    }

    # ---- the two controls, as ratios --------------------------------------
    controls = []
    for r in drows:
        em = float(r["ours_heldout"][0])
        controls.append({
            "M": int(r["M"]),
            "shuffled_over_emission": float(r["null_heldout"][0]) / em,
            "constant_over_emission": float(r["const_cond_heldout"][0]) / em,
        })

    record = {
        "what": (
            "Every level fig:floor-composite plots, plus the two conditioner "
            "controls of Sec. 6.1, harvested from the runs' own results.json "
            "artifacts. Provenance for the paper's headline below-floor "
            "claim; no recomputation and no re-plotting."
        ),
        "figure": "paper/figures/dq_floor_composite.pdf (fig:floor-composite)",
        "renderer": "scripts/render_composite_floor.py",
        "m_list": list(M_LIST),
        "panels": panels,
        "conditioner_controls": controls,
        "conditioner_control_note": (
            "shuffled: the displacement alone is fed a mismatched "
            "observation, while the seeds and the solved weights still come "
            "from the true posterior, so it prices the displacement's use of "
            "the observation rather than conditioning as such. constant: the "
            "network is fed the mean of the evaluation observations, held "
            "fixed across targets -- an observation-shaped input it cannot "
            "use, not the absence of one. Both ratios exceed 1 at every "
            "budget, and the constant control sits nearer the emission than "
            "the shuffled one throughout."
        ),
        "cond_limited_note": (
            "Cells flagged cond_limited are drawn OPEN and are read as values "
            "rather than as an ordering between arms. They are the two "
            "low-dimensional closed-form panels above the smallest budget; "
            "the mixture panel and all three lower-row rungs are clear at "
            "every budget."
        ),
    }

    tmp = _OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2)
    os.replace(tmp, _OUT)

    print(f"Saved to {_OUT}")
    n_open = sum(sum(p.get("cond_limited", [])) for p in panels.values())
    print(f"  panels {len(panels)}, budgets {len(M_LIST)}, "
          f"conditioning-limited cells {n_open}")
    for c in controls:
        print(f"  M={c['M']:>3}: shuffled/emission "
              f"{c['shuffled_over_emission']:.4f}, constant/emission "
              f"{c['constant_over_emission']:.4f}")


if __name__ == "__main__":
    main()
