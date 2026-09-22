"""The release itself: what imports, what the manifest promises, what Lean proves."""
import ast
import glob
import importlib
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "qfield")


def _modules():
    out = []
    for path in glob.glob(os.path.join(PKG, "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, ROOT)[: -len(".py")]
        out.append(rel.replace(os.sep, "."))
    return sorted(out)


@pytest.mark.parametrize("name", _modules())
def test_every_module_imports(name):
    importlib.import_module(name)


def test_importing_the_package_stays_small():
    """The package must not drag its whole implementation in on import."""
    code = "import qfield, sys; print(len([m for m in sys.modules if m.startswith('qfield')]))"
    n = int(subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True, cwd=ROOT).stdout.strip())
    assert n < len(_modules()), f"importing qfield loaded {n} modules"


def test_no_module_refers_to_the_research_package():
    hits = []
    needle = "amortized" + "_msip"
    for path in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        if ".git" in path or os.path.abspath(path) == os.path.abspath(__file__):
            continue
        with open(path, encoding="utf-8") as fh:
            if needle in fh.read():
                hits.append(os.path.relpath(path, ROOT))
    assert not hits, hits


def test_scripts_parse_and_build_no_paths_by_hand():
    """projorg owns the directories, so no script rebuilds the repository root."""
    offenders = []
    for path in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py"))):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        ast.parse(src)
        if re.search(r"os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)", src):
            offenders.append(os.path.basename(path))
    assert not offenders, offenders


def test_the_download_manifest_is_well_formed():
    from qfield import download

    assert download.REGISTRY, "the manifest is empty"
    for key, (tier, url) in download.REGISTRY.items():
        assert not key.startswith("/") and ".." not in key, key
        assert os.path.dirname(key), f"{key} names no problem directory"
        assert tier in download.TIERS, key
        assert url.startswith("https://"), key
        assert url.endswith("dl=1"), f"{key} is not a direct download"
    urls = [u for _, u in download.REGISTRY.values()]
    assert len(set(urls)) == len(urls), "two entries share a URL"
    assert "checkpoints" not in download.TIERS, "trained networks are not distributed"


def test_download_paths_go_through_projorg():
    from qfield import download

    key = next(iter(download.REGISTRY))
    assert os.path.isabs(download.path_for(key))
    with pytest.raises(KeyError):
        download.path_for("not/in/the/registry.npz")


def test_the_records_the_figures_read_are_present():
    records = glob.glob(os.path.join(ROOT, "data", "records", "*.json"))
    assert len(records) >= 15, f"only {len(records)} records shipped"


class TestFormal:
    """The Lean development ships complete."""

    def _lean_sources(self):
        return glob.glob(os.path.join(ROOT, "formal", "**", "*.lean"), recursive=True)

    def test_the_project_files_are_committed(self):
        for name in ("lean-toolchain", "lakefile.toml", "lake-manifest.json",
                     "Axioms.lean", "QuadratureField.lean"):
            assert os.path.isfile(os.path.join(ROOT, "formal", name)), name

    def test_nothing_is_admitted(self):
        bad = []
        for path in self._lean_sources():
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    code = line.split("--")[0]
                    if re.search(r"\b(sorry|admit|native_decide)\b", code):
                        bad.append(f"{os.path.basename(path)}:{i}")
        assert not bad, bad

    def test_the_audit_covers_the_theorems(self):
        with open(os.path.join(ROOT, "formal", "Axioms.lean"), encoding="utf-8") as fh:
            audited = [ln for ln in fh if ln.startswith("#print axioms")]
        assert len(audited) >= 25, f"only {len(audited)} theorems audited"

    def test_every_audited_name_exists_in_a_source(self):
        with open(os.path.join(ROOT, "formal", "Axioms.lean"), encoding="utf-8") as fh:
            names = [ln.split()[-1].split(".")[-1] for ln in fh
                     if ln.startswith("#print axioms")]
        declared = set()
        for path in self._lean_sources():
            with open(path, encoding="utf-8") as fh:
                declared |= set(re.findall(r"^(?:theorem|lemma)\s+([A-Za-z0-9_']+)",
                                           fh.read(), flags=re.M))
        assert not [n for n in names if n not in declared]
