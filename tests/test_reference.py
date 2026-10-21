"""The Python API against reference values from universalmotif."""

import pytest

import pyumotif as um

from conftest import assert_motif_matches, load_reference, motif_from_record, num, nums


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
            if len(cols) == 1 and case.get("motif") is None:
                # universalmotif fails on single-column matrices with row
                # names (see umotif's DIVERGENCES.md).
                assert call().type == "PCM"
                continue
        else:
            strings = inp["strings"]
            value = strings[0] if len(strings) == 1 else strings
            call = lambda: um.create_motif(value, **kwargs)
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
            assert_motif_matches(call(), case["motif"], ctx)


def test_ops():
    for i, case in enumerate(load_reference("ops.json")):
        op, args = case["op"], case["args"]
        motif = motif_from_record(case["input"])
        ctx = f"case {i}: {op} on {motif.name} {args}"
        skip_nsites = False
        if op == "motif_rc":
            call = lambda: um.motif_rc(motif)
        elif op == "round_motif":
            call = lambda: um.round_motif(motif, num(args["pct_tolerance"]))
        elif op == "normalize":
            call = lambda: um.normalize(motif)
        elif op == "trim_motifs":
            # universalmotif fills in nsites = 100 here; umotif does not.
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
            assert_motif_matches(got, case["motif"], ctx, skip_nsites=skip_nsites)
            if skip_nsites:
                assert got.nsites is None, ctx
