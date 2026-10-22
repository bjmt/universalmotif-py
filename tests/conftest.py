"""Helpers for the reference values that umotif exports from
universalmotif (see scripts/make_reference.R in the umotif repository,
currently checked out at ../motifkit)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pyumotif import _umotif

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "motifkit" / "tests" / "reference"


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
        "icscore": num(r["icscore"]),
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
    return _umotif._motif_from_state(state)


def assert_motif_matches(got, record, ctx="", skip_nsites=False):
    want = motif_from_record(record)
    for field in ("name", "altname", "family", "organism", "alphabet", "type",
                  "consensus", "strand", "extrainfo", "isgapped", "gaploc",
                  "mingap", "maxgap"):
        assert getattr(got, field) == getattr(want, field), f"{ctx}: {field}"
    # Numbers read from files can differ from R in the last bit, because R's
    # parser is not always correctly rounded.
    for field in ("bkgsites", "pval", "qval", "eval") + (() if skip_nsites else ("nsites",)):
        g, w = getattr(got, field), getattr(want, field)
        if g is None or w is None:
            assert g == w, f"{ctx}: {field}"
        else:
            assert math.isclose(g, w, rel_tol=1e-12), f"{ctx}: {field}"
    assert math.isclose(got.icscore, want.icscore, rel_tol=1e-12, abs_tol=1e-12), ctx
    assert math.isclose(got.pseudocount, want.pseudocount), ctx
    np.testing.assert_allclose(list(got.bkg.values()), list(want.bkg.values()),
                               rtol=1e-12, err_msg=ctx)
    np.testing.assert_allclose(got.matrix, want.matrix, rtol=1e-12, atol=1e-12, err_msg=ctx)
