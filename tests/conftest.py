"""Helpers for the reference values that the Rust crate exports from the R
package (see scripts/make_reference.R in ../universalmotif-rs)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from universalmotif import _core

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "universalmotif-rs" / "tests" / "reference"


def load_reference(name, key="cases"):
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"reference values not found at {path}")
    return json.loads(path.read_text())[key]


def reference_file(name):
    return REFERENCE_DIR / "files" / name


def args(v):
    """Function arguments recorded by R; an empty R list() is written as []."""
    return v if isinstance(v, dict) else {}


def num(x):
    if x is None:
        return None
    if isinstance(x, str):
        return {"NaN": math.nan, "NA": math.nan, "Inf": math.inf, "-Inf": -math.inf}[x]
    return float(x)


def nums(x):
    if x is None:
        return []
    if not isinstance(x, list):
        return [num(x)]
    return [num(v) for v in x]


def motif_from_record(r):
    gap = r["gap"]
    state = {
        "name": r["name"],
        "altname": r["altname"],
        "family": r["family"],
        "organism": r["organism"],
        "alphabet": r["alphabet"],
        "type": r["type"],
        # The port gives CWMs no total IC.
        "icscore": math.nan if r["type"] == "CWM" else num(r["icscore"]),
        "nsites": num(r["nsites"]),
        "pseudocount": num(r["pseudocount"]),
        "bkg": nums(r["bkg"]),
        "bkgsites": num(r["bkgsites"]),
        "consensus": r["consensus"],
        "strand": r["strand"],
        "pval": num(r["pval"]),
        "qval": num(r["qval"]),
        "eval": num(r["eval"]),
        "extrainfo": [tuple(p) for p in r["extrainfo"]],
        "gap": {
            "isgapped": gap["isgapped"],
            "gaploc": [int(v) for v in nums(gap["gaploc"])],
            "mingap": [int(v) for v in nums(gap["mingap"])],
            "maxgap": [int(v) for v in nums(gap["maxgap"])],
        },
        "matrix": [nums(c) for c in r["matrix"]],
    }
    return _core._motif_from_state(state)


def r_rounding_differs(pos, nsites):
    """Whether R's rounding of a column of probabilities to ``nsites`` counts
    overshoots by more than one count, in which case the port rounds
    differently."""
    n = 100.0 if nsites is None or math.isnan(nsites) or nsites <= 1 else nsites
    total = sum(math.floor(p * n + 0.5) for p in pos)
    return total - n > 1


def ic_rounds_differently(m):
    """Whether the total IC of a PPM motif goes through a column that R rounds
    differently."""
    return m.type == "PPM" and any(r_rounding_differs(c, m.nsites) for c in m.matrix.T)


def own_icscore(m):
    """The total IC computed afresh from a motif's own matrix and settings."""
    import universalmotif as um

    bkg = list(m.bkg.values())
    return sum(um.position_icscore(c, bkg=bkg, type=m.type, pseudocount=m.pseudocount,
                                   nsites=m.nsites or 0) for c in m.matrix.T)


def assert_motif_matches(got, record, ctx="", skip_nsites=False, skip_derived=False,
                         skip_matrix=False):
    """Compare a motif with a record from R. ``skip_derived`` skips the total IC
    and consensus, which the R package leaves as they were after some
    operations where the port recomputes them, and ``skip_matrix`` the matrix
    too, where R rounds counts differently."""
    want = motif_from_record(record)
    fields = ["name", "altname", "family", "organism", "alphabet", "type", "strand",
              "extrainfo", "isgapped", "gaploc", "mingap", "maxgap"]
    if not (skip_derived or skip_matrix):
        fields.append("consensus")
    for field in fields:
        assert getattr(got, field) == getattr(want, field), f"{ctx}: {field}"
    # Numbers read from files can differ from R in the last bit, because R's
    # parser is not always correctly rounded.
    for field in ("bkgsites", "pval", "qval", "eval") + (() if skip_nsites else ("nsites",)):
        g, w = getattr(got, field), getattr(want, field)
        if g is None or w is None:
            assert g == w, f"{ctx}: {field}"
        else:
            assert math.isclose(g, w, rel_tol=1e-12), f"{ctx}: {field}"
    if got.type == "CWM":
        assert math.isnan(got.icscore), ctx
    elif not (skip_derived or skip_matrix or ic_rounds_differently(want)):
        assert math.isclose(got.icscore, want.icscore, rel_tol=1e-12, abs_tol=1e-12), ctx
    assert math.isclose(got.pseudocount, want.pseudocount), ctx
    np.testing.assert_allclose(list(got.bkg.values()), list(want.bkg.values()),
                               rtol=1e-12, err_msg=ctx)
    if not skip_matrix:
        np.testing.assert_allclose(got.matrix, want.matrix, rtol=1e-12, atol=1e-12, err_msg=ctx)
