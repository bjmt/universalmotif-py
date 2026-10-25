"""The Python API against reference values from the R package."""

import math

import numpy as np
import pytest

import universalmotif as um

from conftest import (
    assert_motif_matches,
    load_reference,
    motif_from_record,
    num,
    nums,
    own_icscore,
    r_rounding_differs,
)


def _kwargs(args):
    if isinstance(args, list):  # jsonlite writes an empty list() as []
        args = {}
    out = {}
    for k in ("type", "name", "alphabet"):
        if k in args:
            out[k] = args[k]
    if "pseudocount" in args:
        out["pseudocount"] = num(args["pseudocount"])
    if args.get("nsites") is not None:
        out["nsites"] = int(num(args["nsites"]))
    if args.get("bkg") is not None:
        out["bkg"] = nums(args["bkg"])
    return out


def test_create_motif():
    for i, case in enumerate(load_reference("create.json")):
        inp, args = case["input"], case["args"]
        ctx = f"case {i}: {inp} {args}"
        kwargs = _kwargs(args)
        if inp["kind"] == "matrix":
            cols = [nums(c) for c in inp["matrix"]]
            matrix = [list(row) for row in zip(*cols)]
            call = lambda: um.create_motif(matrix, letters=inp["rows"], **kwargs)
            if "type" not in kwargs and any(v < 0 for c in cols for v in c):
                # The R package reads such a matrix as a PWM and converts it
                # to a PPM; the port asks for the type instead.
                with pytest.raises(ValueError, match="negative values"):
                    call()
                continue
            if len(cols) == 1 and case.get("motif") is None:
                # The R package fails on single-column matrices with row
                # names.
                assert call().type == "PCM"
                continue
        else:
            strings = inp["strings"]
            value = strings[0] if len(strings) == 1 else strings
            call = lambda: um.create_motif(value, **kwargs)
            # A single consensus string becomes probabilities rounded to
            # nsites counts, which R does unevenly for some ambiguity letters.
            n = kwargs.get("nsites")
            if len(strings) == 1 and n is not None:
                to_ppm = um.consensus_to_ppm
                if not all(c in "ACGTURYSWKMBDHVN" for c in value):
                    to_ppm = um.consensus_to_ppm_aa
                try:
                    cols = [list(to_ppm(c)) for c in value]
                except ValueError:
                    cols = []
                if cols and any(r_rounding_differs(c, n) for c in cols):
                    got = call()
                    if case.get("motif") is None:
                        # R gives the largest count a negative value and fails.
                        assert "only positive values" in case["error"], ctx
                    else:
                        assert_motif_matches(got, case["motif"], ctx, skip_matrix=True)
                    for j, c in enumerate(cols):
                        want = [x / n for x in um.ppm_to_pcm(c, nsites=n)]
                        np.testing.assert_allclose(got.matrix[:, j], want, err_msg=ctx)
                    continue
        if case.get("motif") is None:
            with pytest.raises(ValueError):
                call()
        else:
            assert_motif_matches(call(), case["motif"], ctx)


def test_convert_type():
    for i, case in enumerate(load_reference("convert.json")):
        motif = motif_from_record(case["input"])
        args = case["args"]
        ctx = f"case {i}: {motif.name} {args}"
        call = lambda: um.convert_type(
            motif,
            args["type"],
            pseudocount=num(args.get("pseudocount")),
            relative_entropy=bool(args.get("relative_entropy", False)),
        )
        if case.get("motif") is None:
            with pytest.raises(ValueError):
                call()
        else:
            got = call()
            # A new pseudocount changes the total IC, and a CWM has none, so
            # the port recomputes it where the R package keeps the old one.
            from_cwm = motif.type == "CWM"
            new_pseudo = (num(args.get("pseudocount")) not in (None, motif.pseudocount)
                          and args["type"] != motif.type)
            assert_motif_matches(got, case["motif"], ctx, skip_derived=from_cwm or new_pseudo)
            if from_cwm and got.type != "CWM":
                assert math.isclose(got.icscore, own_icscore(got), rel_tol=1e-12), ctx
            elif new_pseudo:
                want = own_icscore(motif.replace(pseudocount=num(args["pseudocount"])))
                assert math.isclose(got.icscore, want, rel_tol=1e-12), ctx


def test_ops():
    for i, case in enumerate(load_reference("ops.json")):
        op, args = case["op"], case["args"]
        motif = motif_from_record(case["input"])
        ctx = f"case {i}: {op} on {motif.name} {args}"
        skip_nsites = False
        # These two change the matrix or pseudocount, and the port recomputes
        # the total IC and consensus where the R package keeps the old ones.
        derived = op in ("round_motif", "normalize")
        if op == "motif_rc":
            call = lambda: um.motif_rc(motif)
        elif op == "round_motif":
            call = lambda: um.round_motif(motif, num(args["pct_tolerance"]))
        elif op == "normalize":
            call = lambda: um.normalize(motif)
        elif op == "trim_motifs":
            # The R package fills in nsites = 100 here; this port does not.
            skip_nsites = motif.nsites is None
            call = lambda: um.trim_motifs(
                motif, min_ic=num(args["min_ic"]), trim_from=args["trim_from"]
            )
        elif op == "trim_cwm":
            call = lambda: um.trim_cwm(
                motif,
                trim_threshold=num(args["trim_threshold"]),
                abs_threshold=num(args.get("abs_threshold")),
                trim_from=args["trim_from"],
            )
        else:
            raise AssertionError(f"unknown op {op}")
        if case.get("motif") is None:
            with pytest.raises(ValueError, match="completely trimmed"):
                call()
        else:
            got = call()
            if op == "normalize" and motif.type == "PCM" and motif.pseudocount == 0:
                # The R package leaves a PCM's pseudocount at zero, so its
                # weights stay infinite; the port sets it to 1.
                assert got.pseudocount == 1, ctx
                got = got.replace(pseudocount=0)
            assert_motif_matches(got, case["motif"], ctx, skip_nsites=skip_nsites,
                                 skip_derived=derived)
            if derived:
                assert math.isclose(got.icscore, own_icscore(got), rel_tol=1e-12), ctx
            if skip_nsites:
                assert got.nsites is None, ctx
