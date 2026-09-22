"""Train the banana reference on its own config.

A thin driver: the experiment class is
:class:`designed_quadrature_gaussian_amortized.DesignedQuadratureAmortized`
unchanged; only the config differs. The banana needs its own config file, and
therefore its own driver, because its reference banks are larger than every
other toy: ``n_ref`` 20k -> 160k and the per-step ``mu``-bank ``n_mu``
4k -> 32k (criterion ``n_ref > M/r``). ``n_mu`` is not a field of the shared
Gaussian config, and adding it there would rename every toy run on disk,
because the config is the run identity.

The config carries ``whiten`` / ``whiten_bandwidth`` fields it does not vary,
because consumers resolve the run by a ``*bandwidth-median_mmd_whiten-*``
glob and the experiment name only carries tokens for fields the config has.

Train:
    CUDA_VISIBLE_DEVICES="" python scripts/designed_quadrature_banana2_amortized.py \
        --experiment_name designed_quadrature_banana2_amortized
Render-only:
    ... --phase visualization
"""

from __future__ import annotations
from projorg import datadir  # noqa: E402

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from projorg import setup_environment, upload_to_cloud  # noqa: E402

from designed_quadrature_gaussian_amortized import (  # noqa: E402
    DesignedQuadratureAmortized,
)

CONFIG_FILE = "designed_quadrature_banana2_amortized.json"

if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload"],
        sequence_args_and_types=[("m_list", int)],
    )

    experiment = DesignedQuadratureAmortized(args)
    if args.phase == "train":
        experiment.train()

    experiment.load_checkpoint()
    if args.phase == "reeval":
        experiment.reeval()
    experiment.visualize()

    if args.upload:
        upload_to_cloud(args)
