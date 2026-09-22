"""Automatic download of the raw inputs the experiments train on.

Trained networks are deliberately not distributed. What is hosted is the raw
data each problem is trained from, so a fresh clone retrains rather than
restoring someone else's run. Every script calls :func:`ensure` on the files
it opens, and the call is a no-op once the file is on disk.

Paths are resolved by ``projorg``: a registry key is relative to the project's
``data/`` directory and nothing here builds a path by hand.

Two tiers, by problem:

``groundwater``      the 300,000 simulated permeability/pressure pairs the
                     reference flow is fitted to, and from which the field's
                     observations are drawn.
``lotka_volterra``   the family's training and held-out members, the
                     conditioner blocks, and the exact chains the reference is
                     audited against.

Limited-angle tomography needs no download. Its exact posterior is closed
form, and the samples its flow trains on are simulated by the script.

The URLs below are public direct-download links.
"""

from __future__ import annotations

import os
import sys
import urllib.request

from projorg import datadir

# repo-relative path under ``data/`` -> (tier, direct-download URL).
REGISTRY: dict[str, tuple[str, str]] = {
    "darcy_stuart/darcy_stuart_bank_300k.h5":
        ("groundwater", "https://www.dropbox.com/scl/fi/fqr34xy19drpeln3b6qpw/darcy_stuart_bank_300k.h5?rlkey=bzdp2ndtrxdljf9qury0e8mjo&dl=1"),
    "lotka_volterra/train.npz":
        ("lotka_volterra", "https://www.dropbox.com/scl/fi/dmqyv6ep2k13611he5djm/train.npz?rlkey=wwij8t6ooc68zw9h5j4l32bt6&dl=1"),
    "lotka_volterra/train_reframed.npz":
        ("lotka_volterra", "https://www.dropbox.com/scl/fi/ntkfuunr3869a4629rwmf/train_reframed.npz?rlkey=rxcr1v28kohd44b3kiadmacmw&dl=1"),
    "lotka_volterra/train_refined.npz":
        ("lotka_volterra", "https://www.dropbox.com/scl/fi/nw60v2dpzduyptcknbf8n/train_refined.npz?rlkey=7p94onirz6cie1t8uunw3hxb9&dl=1"),
    "lotka_volterra/test.npz":
        ("lotka_volterra", "https://www.dropbox.com/scl/fi/8jh7ol9kp7inpytumy5n6/test.npz?rlkey=ykkih13h75inqwfjpjoxvdgwn&dl=1"),
    "lotka_volterra/test_reframed.npz":
        ("lotka_volterra", "https://www.dropbox.com/scl/fi/op6u8xwh5ztev9xpnlyw9/test_reframed.npz?rlkey=u1oeotwpuqanazexj6pihn2t9&dl=1"),
    "lotka_volterra/test_refined.npz":
        ("lotka_volterra", "https://www.dropbox.com/scl/fi/220xttkhzk1i8ok8r5qob/test_refined.npz?rlkey=1yak3m9wd3w3hwew2wkfkdqdj&dl=1"),
    "lotka_volterra/ridge.npz":
        ("lotka_volterra", "https://www.dropbox.com/scl/fi/pn08fpzqeg9p8g9zrf4nk/ridge.npz?rlkey=ghixho4j4twgabmd32yre92vz&dl=1"),
    "lotka_volterra/chains.npz":
        ("lotka_volterra", "https://www.dropbox.com/scl/fi/0y3t734qu3qv3j6kfbzfq/chains.npz?rlkey=816zwnt06w592lyxp895fw0wj&dl=1"),
    "lotka_volterra/chains_newframe.npz":
        ("lotka_volterra", "https://www.dropbox.com/scl/fi/cvcew1dpc47r6ngzxz0v5/chains_newframe.npz?rlkey=p9s4i29b1gva04vdk29cr2fmy&dl=1"),
}

TIERS = sorted({tier for tier, _ in REGISTRY.values()})


def _report(key: str, done: int, total: int) -> None:
    if total <= 0:
        return
    pct = 100.0 * done / total
    sys.stdout.write(f"\r  {key}: {pct:5.1f}% of {total / 1e6:.0f} MB")
    sys.stdout.flush()


def path_for(key: str) -> str:
    """The absolute path a registry key resolves to, creating its directory."""
    if key not in REGISTRY:
        raise KeyError(f"{key} is not in the registry; see qfield/download.py")
    head, tail = os.path.split(key)
    return os.path.join(datadir(head), tail)


def ensure(key: str) -> str:
    """Return the local path to ``key``, downloading it on first use."""
    dest = path_for(key)
    if os.path.isfile(dest):
        return dest
    url = REGISTRY[key][1]
    print(f"downloading {key}")
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                _report(key, done, total)
        print()
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"could not fetch {key} from {url}: {exc}") from exc
    os.replace(tmp, dest)
    return dest


def ensure_tier(tier: str) -> list[str]:
    """Fetch every file of one tier, and return their paths."""
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier}; the tiers are {TIERS}")
    return [ensure(k) for k, (t, _) in REGISTRY.items() if t == tier]


def missing() -> list[str]:
    """The registry keys not yet on disk."""
    return [k for k in REGISTRY if not os.path.isfile(path_for(k))]
