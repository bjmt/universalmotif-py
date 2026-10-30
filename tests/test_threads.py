import inspect
import os
import subprocess
import sys

import pytest
import universalmotif as um
from universalmotif import _core


@pytest.fixture
def restore_nthreads():
    before = um.get_nthreads()
    yield
    um.set_nthreads(before)


def test_set_and_get(restore_nthreads):
    for n in (4, 0, 1):
        um.set_nthreads(n)
        assert um.get_nthreads() == n


@pytest.mark.parametrize("bad", [-1, 1.5, "2", True, None])
def test_invalid_values(restore_nthreads, bad):
    um.set_nthreads(3)
    with pytest.raises(ValueError, match="non-negative integer"):
        um.set_nthreads(bad)
    assert um.get_nthreads() == 3


def _imported_setting(value):
    env = {k: v for k, v in os.environ.items() if k != "UNIVERSALMOTIF_NTHREADS"}
    if value is not None:
        env["UNIVERSALMOTIF_NTHREADS"] = value
    code = "import universalmotif as um; print(um.get_nthreads())"
    out = subprocess.run([sys.executable, "-W", "default", "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    return int(out.stdout), out.stderr


def test_environment_variable():
    assert _imported_setting(None) == (1, "")
    assert _imported_setting("3") == (3, "")
    assert _imported_setting(" 0 ") == (0, "")
    for bad in ("-2", "two", "1.5"):
        n, err = _imported_setting(bad)
        assert n == 1 and "ignoring UNIVERSALMOTIF_NTHREADS" in err


def test_functions_use_the_setting(restore_nthreads, monkeypatch):
    seen = []
    real = _core._create_sequences

    def spy(*args):
        seen.append(args[-1])
        return real(*args)

    monkeypatch.setattr(_core, "_create_sequences", spy)
    um.create_sequences(seqnum=2, seqlen=10, seed=1)
    um.set_nthreads(3)
    um.create_sequences(seqnum=2, seqlen=10, seed=1)
    # A number given to the function still wins.
    um.create_sequences(seqnum=2, seqlen=10, seed=1, nthreads=2)
    um.create_sequences(seqnum=2, seqlen=10, seed=1, nthreads=0)
    assert seen == [1, 3, 2, 0]


def test_setting_passes_through_to_other_functions(restore_nthreads, monkeypatch):
    from universalmotif import _enrich

    seen = []
    real_scan, real_coocc = _enrich.scan_sequences, _core._coocc

    def scan(*args, **kwargs):
        seen.append(("scan", kwargs["nthreads"]))
        return real_scan(*args, **kwargs)

    def coocc(*args):
        seen.append(("coocc", args[-2]))
        return real_coocc(*args)

    monkeypatch.setattr(_enrich, "scan_sequences", scan)
    monkeypatch.setattr(_core, "_coocc", coocc)
    um.set_nthreads(2)
    motifs = [um.example_motifs()[i] for i in (3, 5)]
    um.motif_coocc(motifs, um.example_sequences("targets"))
    assert seen == [("scan", 2), ("coocc", 2)]


def _public_callables():
    for name in um.__all__:
        obj = getattr(um, name)
        if inspect.isclass(obj):
            for attr, member in vars(obj).items():
                if callable(member) and not attr.startswith("_"):
                    yield f"{name}.{attr}", member
        elif callable(obj):
            yield name, obj


def test_every_nthreads_defaults_to_the_setting():
    found = []
    for name, fn in _public_callables():
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if "nthreads" in params:
            found.append(name)
            assert params["nthreads"].default is None, name
    assert {"scan_sequences", "compare_motifs", "view_motifs", "Genome.extract"} <= set(found)
