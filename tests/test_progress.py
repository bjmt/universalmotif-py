import threading
import time

import pytest
import universalmotif as um
from universalmotif import _core, _progress


@pytest.fixture
def capture(monkeypatch, capsys):
    """Bars drawn at once; read them with capture.readouterr().err."""
    monkeypatch.setattr(_progress, "_DELAY", 0.0)
    monkeypatch.setattr(_progress, "_INTERVAL", 0.001)
    yield capsys
    um.set_progress(None)


@pytest.fixture(scope="module")
def seqs():
    return um.create_sequences(seqnum=300, seqlen=400, seed=1)


def test_off_by_default_and_quick_calls_show_nothing(capsys, seqs):
    um.shuffle_sequences(seqs, seed=1)
    um.shuffle_sequences(seqs, seed=1, progress=True)
    assert capsys.readouterr().err == ""


def test_bars_reach_the_end_and_change_nothing(capture, seqs):
    m = um.example_motifs()[3]
    plain = um.scan_sequences(m, seqs)
    shown = um.scan_sequences(m, seqs, progress=True)
    assert shown.equals(plain)

    # Kept running a moment after the scan, so that the bar is surely drawn.
    def scan_then_wait():
        hits = um.scan_sequences(m, seqs)
        time.sleep(0.05)
        return hits

    assert _progress.run(scan_then_wait).equals(plain)
    text = capture.readouterr().err
    assert text.startswith("\rscanning:") and "100.0%" in text and text.endswith("\n")
    targets, controls = um.example_sequences("targets"), um.example_sequences("controls")
    found = um.motif_finder(targets, controls, min_width=6, max_width=8, seed=1, progress=True)
    again = um.motif_finder(targets, controls, min_width=6, max_width=8, seed=1)
    assert found.height > 0 and found.drop("motif").equals(again.drop("motif"))
    assert "searching" in capture.readouterr().err


def test_the_setting_and_the_environment(capture, monkeypatch, seqs):
    assert not um.get_progress()
    monkeypatch.setenv("UNIVERSALMOTIF_PROGRESS", "1")
    assert um.get_progress()
    um.set_progress(False)
    assert not um.get_progress() and not _progress.wanted(None)
    assert _progress.wanted(True)
    um.set_progress(True)
    assert _progress.wanted(None) and not _progress.wanted(False)
    with pytest.raises(TypeError):
        um.set_progress(1)
    with pytest.raises(TypeError, match="progress must be"):
        um.shuffle_sequences(seqs[:5], progress="yes")


def test_errors_reach_the_caller(capture):
    with pytest.raises(ValueError, match="k must be at least 1"):
        um.shuffle_sequences(["ACGT"], k=0, progress=True)


def test_a_cancelled_handle_stops_the_call(seqs):
    handle = _core.Progress()
    handle.cancel()
    with pytest.raises(ValueError, match="cancelled"):
        _core._with_progress(handle, lambda: um.scan_sequences(um.example_motifs()[3], seqs))
    # Without the handle, the same call runs.
    assert um.scan_sequences(um.example_motifs()[3], seqs).height > 0


def test_ctrl_c_cancels_and_waits(monkeypatch, seqs):
    """A KeyboardInterrupt while the bar is drawn cancels the handle and is
    raised once the work has stopped."""
    monkeypatch.setattr(_progress, "_DELAY", 0.0)
    seen = {}
    real = _progress._state

    def interrupt(handle, o):
        seen["handle"] = handle
        raise KeyboardInterrupt

    monkeypatch.setattr(_progress, "_state", interrupt)
    started = threading.Event()

    def slow():
        started.set()
        return um.motif_finder(seqs, min_width=6, max_width=12, nmotifs=5, seed=1)

    with pytest.raises(KeyboardInterrupt):
        _progress.run(slow)
    assert started.is_set() and seen["handle"].stage in ("", "shuffling", "searching")
    monkeypatch.setattr(_progress, "_state", real)


def test_genome_scans_count_the_whole_genome(capture, tmp_path, monkeypatch):
    from universalmotif import _analysis

    fa = tmp_path / "g.fa"
    um.write_fasta({f"chr{i}": s for i, s in enumerate(um.create_sequences(seqnum=3, seqlen=5000, seed=2))}, fa)
    g = um.Genome(fa)
    plain = um.scan_sequences(um.example_motifs()[3], g)
    seen = []
    real = _analysis._core._scan_sequences

    def spy(*args):
        o = _progress._outers.get(threading.get_ident())
        seen.append((o.stage, o.total, o.part) if o else None)
        return real(*args)

    monkeypatch.setattr(_analysis._core, "_scan_sequences", spy)
    hits = um.scan_sequences(um.example_motifs()[3], g, progress=True)
    assert hits.equals(plain)
    # One batch, counted against the length of the whole genome.
    assert seen == [("scanning", 15000, 15000)]


def test_the_command_line_flag(capture, tmp_path):
    from universalmotif._cli import main

    fa = tmp_path / "t.fa"
    um.write_fasta(um.example_sequences("targets"), fa)
    um.write_meme(um.example_motifs()[:2], tmp_path / "m.meme")
    assert main(["scan", str(tmp_path / "m.meme"), "-s", str(fa), "--progress", "-o", str(tmp_path / "h.tsv")]) == 0
    assert main(["scan", str(tmp_path / "m.meme"), "-s", str(fa), "--no-progress", "-o", str(tmp_path / "h2.tsv")]) == 0
    # The setting is put back afterwards.
    assert not um.get_progress()
    assert (tmp_path / "h.tsv").read_text() == (tmp_path / "h2.tsv").read_text()
