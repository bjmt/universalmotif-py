"""Packed sequences, and scanning FASTA files without reading them whole."""

import gzip

import numpy as np
import pytest

import universalmotif as um

from conftest import args, load_reference, motif_from_record, reference_file


def _scan_motifs():
    return [motif_from_record(r) for r in load_reference("scan.json", "motifs")]


def test_three_inputs_scan_alike():
    motifs = _scan_motifs()
    for i, case in enumerate(load_reference("scan.json")):
        a = args(case["args"])
        ctx = f"scan case {i}: {case['file']} {a}"
        path = reference_file(case["file"])
        opts = dict(
            pvalue=a.get("pvalue", 1e-4),
            rc=a.get("RC", True),
            no_overlaps=a.get("no.overlaps", False),
            no_overlaps_by=a.get("no.overlaps.by", "pvalue"),
            no_overlaps_by_strand=a.get("no.overlaps.by.strand", False),
            no_overlaps_by_motif=a.get("no.overlaps.by.motif", False),
        )
        want = um.scan_sequences(motifs, um.read_fasta(path), **opts)
        assert want.height > 0, ctx
        streamed = um.scan_sequences(motifs, path, nthreads=2, **opts)
        assert streamed.equals(want), ctx
        packed = um.read_fasta(path, packed=True)
        assert um.scan_sequences(motifs, packed, nthreads=2, **opts).equals(want), ctx


def test_other_alphabets_scan_alike(tmp_path):
    for prefix, alphabet in (("aa", "AA"), ("custom", "QRSTU")):
        motifs = [motif_from_record(r) for r in load_reference("alphabets.json", f"{prefix}_motifs")]
        seqs = load_reference("alphabets.json", f"{prefix}_sequences")
        named = {f"s{i}": s for i, s in enumerate(seqs)}
        want = um.scan_sequences(motifs, named, pvalue=0.01)
        assert want.height > 0
        um.write_fasta(named, tmp_path / f"{prefix}.fa", width=50)
        assert um.scan_sequences(motifs, tmp_path / f"{prefix}.fa", pvalue=0.01).equals(want)
        packed = um.pack_sequences(named, alphabet=alphabet)
        assert um.scan_sequences(motifs, packed, pvalue=0.01).equals(want)


def test_packed_mapping(tmp_path):
    p = um.pack_sequences({"a": "acgtNNACGTnry", "b": "", "c": "TTTT"})
    assert p.alphabet == "DNA"
    assert len(p) == 3 and list(p) == ["a", "b", "c"]
    # Upper case for the alphabet's letters, everything else as it was.
    assert p["a"] == "ACGTNNACGTnry"
    assert dict(p) == {"a": "ACGTNNACGTnry", "b": "", "c": "TTTT"}
    assert "a" in p and "z" not in p and 3 not in p
    with pytest.raises(KeyError):
        p["z"]
    assert p.lengths == {"a": 13, "b": 0, "c": 4}
    assert p.nbytes > 0 and "3 DNA sequences" in repr(p)
    assert um.pack_sequences(["ACGU"], alphabet="RNA")["seq1"] == "ACGU"
    assert um.pack_sequences(["xyXY"], alphabet="xy")["seq1"] == "xyXY"

    dup = _write(tmp_path / "dup.fa", ">same\nACGT\n>same\nGG\n")
    with pytest.raises(ValueError, match="more than once"):
        um.read_fasta(dup, packed=True)
    with pytest.raises(ValueError, match="encoded as DNA"):
        um.scan_sequences(um.create_motif("ACGU", alphabet="RNA"), p)

    path = _write(tmp_path / "x.fa", "header junk\n>one\nACGT\nacgt\n>two\nNNNN\n")
    assert dict(um.read_fasta(path, skip=1, packed=True)) == {"one": "ACGTACGT", "two": "NNNN"}
    with pytest.raises(ValueError):
        um.scan_sequences(um.create_motif("ACGT"), tmp_path / "missing.fa")


def test_packed_sequences_work_as_a_mapping():
    # Functions without a packed path decode the sequences as they go.
    seqs = um.read_fasta(reference_file("promoters.fa"))
    packed = um.read_fasta(reference_file("promoters.fa"), packed=True)
    upper = {k: v.upper() for k, v in seqs.items()}
    assert dict(packed) == upper
    motifs = _scan_motifs()[:2]
    a = um.enrich_motifs(motifs, upper, seed=1, qvalue=1)
    b = um.enrich_motifs(motifs, packed, seed=1, qvalue=1)
    assert a.equals(b)
    np.testing.assert_array_equal(um.get_bkg(packed, k=2)["count"], um.get_bkg(upper, k=2)["count"])


def _write(path, text):
    path.write_text(text)
    return path


def test_gzipped_fasta(tmp_path):
    plain = reference_file("promoters.fa")
    gz = tmp_path / "promoters.fa.gz"
    gz.write_bytes(gzip.compress(plain.read_bytes()))
    assert um.read_fasta(gz) == um.read_fasta(plain)
    assert dict(um.read_fasta(gz, packed=True)) == dict(um.read_fasta(plain, packed=True))
    motifs = _scan_motifs()
    assert um.scan_sequences(motifs, gz).equals(um.scan_sequences(motifs, plain))
