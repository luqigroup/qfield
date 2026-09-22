"""Deterministic checkpoint resolution by projorg config-as-identity.

An experiment's checkpoint dir IS its config's identity: ``setup_environment``
names the run by ``make_experiment_name(args, ignore_arg_list=...)`` over the
parsed config. This module rebuilds that exact name straight from the config
file (no ``sys.argv``, so it is safe inside compute scripts with their own CLI)
and reads ``checkpointsdir(name, mkdir=False)`` directly -- no glob, no
stale-dir ambiguity, no mtime guessing. Selecting by
``sorted(glob.glob(...))[-1]`` or by mtime instead silently picks a stale
experiment dir after a re-run, since several runs of one experiment share a
name prefix and differ only in a few swept fields.
``make_experiment_name`` SHA-truncates over-long names via
``shorten_filename``, so the rebuilt name matches the one on disk exactly.

Resolving uses ``mkdir=False`` so it never creates an empty checkpoint dir as
a side effect of looking one up.

Rebuilding a name always succeeds -- every config yields *some* string -- so a
mismatch otherwise surfaces far downstream as ``torch.load``'s bare "no such
file", or as a render silently reading nothing. A compute script that passes an
extended ``ignore_arg_list`` (render-only knobs, say) rebuilds a name whose
readable prefix matches the real run's and whose SHA tail does not.
:func:`checkpoint_path` therefore checks the directory and raises, naming the
existing prefix twins and, when one exists, the exact ``ignore_extra`` that
resolves. Pass ``must_exist=False`` only where a missing dir is a legitimate
state (a write target, or a caller with its own fallback).
"""

from __future__ import annotations

import argparse
import itertools
import os

from projorg import checkpointsdir, configsdir, datadir, plotsdir
from projorg.config import make_experiment_name, read_config

# Fields ``setup_environment`` excludes from the experiment name (they are
# run-control, not identity); must match every compute script's call.
_IGNORE = ["experiment_name", "gpu_id", "phase", "upload"]

# Resolution-failure report: how many prefix twins to name, and the widest
# ``ignore_extra`` the hunt tries (compute scripts extend the shared list by
# one or two render-only knobs; three is already unheard of, and the search is
# combinatorial in the config's field count).
_MAX_TWINS = 6
_MAX_IGNORE_HUNT = 2


def _checkpoints_root() -> str:
    """``data/checkpoints`` without creating the run dir under it."""
    return datadir("checkpoints", mkdir=False)


def _prefix_twins(name: str, stem: str) -> list[str]:
    """Existing checkpoint dirs of the same experiment, closest name first.

    ``make_experiment_name`` SHA-truncates past 255 characters, so sibling runs
    that differ only in a late field share their whole readable prefix. Ranking
    by common-prefix length puts those SHA twins at the top -- exactly the runs
    a failed resolution is most likely to have meant.
    """
    root = _checkpoints_root()
    if not os.path.isdir(root):
        return []
    scored = [
        (len(os.path.commonprefix([entry, name])), entry)
        for entry in os.listdir(root)
        if entry.startswith(stem + "_")
        and os.path.isdir(os.path.join(root, entry))
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [entry for _, entry in scored]


def _hunt_ignore_extra(cfg: dict, ignore: list) -> tuple | None:
    """Find an ``ignore_extra`` that WOULD resolve, if a small one exists.

    A compute script's extended ``ignore_arg_list`` is unrecoverable from the
    directory name (the SHA cannot be inverted), but it is cheap to search
    forward: build the name for each small subset of the config's remaining
    fields and ask the filesystem. Excluding a field and the field being absent
    from the config produce the SAME string, so this also catches genuine
    schema drift.
    """
    root = _checkpoints_root()
    if not os.path.isdir(root):
        return None
    keys = [k for k in cfg if k not in ignore]
    for size in range(1, _MAX_IGNORE_HUNT + 1):
        for subset in itertools.combinations(keys, size):
            name = make_experiment_name(
                argparse.Namespace(**cfg),
                ignore_arg_list=list(ignore) + list(subset),
            )
            if os.path.isdir(os.path.join(root, name)):
                return subset
    return None


def _resolution_error(name: str, cfg: dict, ignore: list) -> FileNotFoundError:
    """The loud report for a name that rebuilt fine but is not on disk."""
    stem = str(cfg.get("experiment_name", name.split("_")[0]))
    twins = _prefix_twins(name, stem)
    lines = [
        f"No checkpoint dir for the rebuilt experiment name:\n  {name}\n",
        "The name rebuilt cleanly, so this is a config-identity MISMATCH, "
        "not a typo: some field differs from what the run was made with, or "
        "the compute script excluded a field from the name that this lookup "
        "did not.",
    ]
    hit = _hunt_ignore_extra(cfg, ignore)
    if hit is not None:
        lines.append(
            "\nFOUND IT -- an existing run resolves if you mirror the "
            "compute script's extended ignore_arg_list:\n"
            f"  ignore_extra={hit!r}\n"
            "Check that against the script's setup_environment call before "
            "trusting it."
        )
    if twins:
        lines.append(
            f"\nClosest existing runs ({len(twins)} share this experiment's "
            "prefix; SHA twins first):"
        )
        lines += [f"  {t}" for t in twins[:_MAX_TWINS]]
        if len(twins) > _MAX_TWINS:
            lines.append(f"  ... and {len(twins) - _MAX_TWINS} more")
    else:
        lines.append(
            "\nNo existing run shares this experiment's prefix -- it has "
            "probably never been trained on this box."
        )
    lines.append(
        "\nIf the run genuinely does not exist yet (a write target, or a "
        "caller with its own fallback), pass must_exist=False."
    )
    return FileNotFoundError("\n".join(lines))


def experiment_name(
    config_file: str, ignore_extra=None, **overrides
) -> str:
    """Rebuild a run's projorg experiment name from its config file.

    Reads ``config_file`` from ``configsdir()``, applies any train-time
    ``overrides`` (same values that were passed at compute time), and returns
    ``make_experiment_name`` over the result with the shared ``_IGNORE`` list --
    identical to what ``setup_environment`` produced for the run.

    Some compute scripts pass an EXTENDED ``ignore_arg_list`` to
    ``setup_environment``, typically to keep render-only knobs out of the run
    identity. Pass those EXTRA names via ``ignore_extra`` so the rebuilt
    name mirrors the SAME ignore list the run used -- the name still IS the
    config's identity, with the run-control fields the script excluded left
    out.

    Args:
        config_file: Config filename in ``configsdir()`` (e.g.
            ``"designed_quadrature_gaussian_amortized.json"``).
        ignore_extra: Optional iterable of additional field names the compute
            script excluded from the name (beyond the shared ``_IGNORE``).
        **overrides: Optional config fields overridden at compute time.

    Returns:
        The experiment name (the checkpoint directory's basename).
    """
    cfg = read_config(os.path.join(configsdir(), config_file))
    cfg.update(overrides)
    ignore = _IGNORE + list(ignore_extra or [])
    return make_experiment_name(
        argparse.Namespace(**cfg), ignore_arg_list=ignore
    )


def checkpoint_path(
    config_file: str,
    fname: str = "checkpoint.pth",
    ignore_extra=None,
    must_exist: bool = True,
    **overrides,
) -> str:
    """Absolute path to a run's checkpoint file, resolved by config identity.

    The resolved DIRECTORY is checked (not ``fname``): a run whose dir exists
    but which never wrote a particular file is the caller's business, while a
    dir that does not exist means the rebuilt name is not this run's identity
    and every path under it is fiction.

    Args:
        config_file: Config filename in ``configsdir()``.
        fname: File within the checkpoint dir (default ``"checkpoint.pth"``).
        ignore_extra: Optional iterable of additional field names the compute
            script excluded from the name (see :func:`experiment_name`).
        must_exist: Raise if the resolved directory is absent (default). Set
            ``False`` when the caller legitimately expects it to be missing --
            a write target, or a lookup with its own fallback.
        **overrides: Optional config fields overridden at compute time.

    Returns:
        ``<checkpoints>/<experiment_name>/<fname>`` (dir not created).

    Raises:
        FileNotFoundError: ``must_exist`` and no such directory. The message
            names the closest existing runs and, when one exists, the
            ``ignore_extra`` that resolves.
    """
    name = experiment_name(config_file, ignore_extra=ignore_extra, **overrides)
    ckpt_dir = checkpointsdir(name, mkdir=False)
    if must_exist and not os.path.isdir(ckpt_dir):
        cfg = read_config(os.path.join(configsdir(), config_file))
        cfg.update(overrides)
        raise _resolution_error(
            name, cfg, _IGNORE + list(ignore_extra or [])
        )
    return os.path.join(ckpt_dir, fname)


def plots_path(
    config_file: str, fname: str, ignore_extra=None, **overrides
) -> str:
    """Absolute path to a run's plots artifact, resolved by config identity.

    Sibling of :func:`checkpoint_path` for artifacts stored under
    ``plotsdir`` (e.g. metrics JSONs) rather than the checkpoint dir. The
    experiment name is rebuilt the same way (config-as-identity), so this
    points at the dir the run wrote under ``plotsdir`` -- no glob, no mtime.

    Deliberately does NOT check existence: callers report a missing artifact
    themselves, with the figure stem in hand, and a render phase may
    legitimately resolve a plots path before writing it.

    Args:
        config_file: Config filename in ``configsdir()``.
        fname: File within the plots dir (e.g. ``"metrics.json"``).
        ignore_extra: Optional iterable of additional field names the compute
            script excluded from the name (see :func:`experiment_name`).
        **overrides: Optional config fields overridden at compute time.

    Returns:
        ``<plots>/<experiment_name>/<fname>`` (dir not created).
    """
    name = experiment_name(config_file, ignore_extra=ignore_extra, **overrides)
    return os.path.join(plotsdir(name, mkdir=False), fname)
