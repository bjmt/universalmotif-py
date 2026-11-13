from collections import Counter

import numpy as np
import pytest
import universalmotif as um


def _seqs(n=4, length=50, seed=1):
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(list("ACGT"), length)) for _ in range(n)]
    seqs[1] = seqs[1][:10] + "NNN" + seqs[1][13:40] + "N" + seqs[1][41:]
    return seqs


def _dinucleotides(s):
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def test_shuffles_keep_dinucleotides_and_ns():
    seqs = _seqs()
    refs = um.shuffled_references(um.one_hot(seqs), n=6)
    assert refs.shape == (4, 6, 4, 50) and refs.dtype == np.float32
    for seq, copies in zip(seqs, refs):
        shuffled = um.from_one_hot(copies)
        assert sum(s != seq for s in shuffled) >= 5
        for s in shuffled:
            assert _dinucleotides(s) == _dinucleotides(seq)
            assert [i for i, c in enumerate(s) if c == "N"] == [i for i, c in enumerate(seq) if c == "N"]
            # Each stretch between Ns keeps its first and last letters.
            ends = [0, len(seq) - 1] + ([9, 13, 39, 41] if "N" in seq else [])
            assert all(s[i] == seq[i] for i in ends)
    triples = um.from_one_hot(um.shuffled_references(um.one_hot(seqs[:1]), n=3, k=3)[0])
    assert all(Counter(t[i:i + 3] for i in range(48)) == Counter(seqs[0][i:i + 3] for i in range(48))
               for t in triples)


def test_layouts_dtypes_and_single_sequences():
    seqs = _seqs()
    x = um.one_hot(seqs)
    want = um.shuffled_references(x, n=3)
    last = um.shuffled_references(x.transpose(0, 2, 1).astype(np.int8), n=3)
    assert last.shape == (4, 3, 50, 4) and last.dtype == np.int8
    np.testing.assert_array_equal(last.transpose(0, 1, 3, 2), want)
    single = um.shuffled_references(x[2], n=3)
    assert single.shape == (3, 4, 50)
    np.testing.assert_array_equal(single, um.shuffled_references(x[2:3], n=3, offset=0)[0])
    assert um.shuffled_references(x.astype(bool), n=3).dtype == bool
    assert um.shuffled_references(x, n=0).shape == (4, 0, 4, 50)


def test_batches_and_seeds():
    x = um.one_hot(_seqs(n=6))
    whole = um.shuffled_references(x, n=4, seed=3, nthreads=2)
    batches = np.concatenate([um.shuffled_references(x[:2], n=4, seed=3),
                              um.shuffled_references(x[2:], n=4, seed=3, offset=2)])
    np.testing.assert_array_equal(batches, whole)
    np.testing.assert_array_equal(um.shuffled_references(x, n=6, seed=3)[:, :4], whole)
    assert not np.array_equal(um.shuffled_references(x, n=4, seed=4), whole)


def test_tensors_come_back_as_tensors():
    torch = pytest.importorskip("torch")
    x = um.one_hot(_seqs())
    out = um.shuffled_references(torch.from_numpy(x), n=2)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
    np.testing.assert_array_equal(out.numpy(), um.shuffled_references(x, n=2))


def test_errors():
    x = um.one_hot(_seqs())
    with pytest.raises(ValueError, match="one-hot"):
        um.shuffled_references(x * 0 + 0.25)
    with pytest.raises(ValueError, match="shape"):
        um.shuffled_references(np.zeros((2, 5, 7)))
    with pytest.raises(ValueError, match="shape"):
        um.shuffled_references(np.zeros(10))
    with pytest.raises(ValueError, match="n must"):
        um.shuffled_references(x, n=-1)
    with pytest.raises(ValueError, match="unknown shuffling method"):
        um.shuffled_references(x, method="other")
