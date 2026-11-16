"""Progress bars for long-running functions.

A function given ``progress=True`` runs on a worker thread. Meanwhile this
thread reads its progress handle and draws a bar on stderr. The bar starts a
second after the call, so quick calls show nothing. Ctrl-C cancels the call
and waits for it to stop before raising.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from . import _core

T = TypeVar("T")

_ENV = "UNIVERSALMOTIF_PROGRESS"
_setting: bool | None = None
_DELAY = 1.0
_INTERVAL = 0.1
# Python-level progress of the calls running with a bar, by worker thread.
_outers: dict[int, _Outer] = {}


def set_progress(enabled: bool | None) -> None:
    """Show progress bars for long-running functions by default.

    Functions that can take a while take a ``progress`` argument, and this
    setting decides when it is left out. Such functions include

    - [`scan_sequences()`][universalmotif.scan_sequences] on a genome,
    - [`motif_finder()`][universalmotif.motif_finder],
    - [`enrich_motifs()`][universalmotif.enrich_motifs].

    A bar appears on stderr once a call has run for a second, so quick calls
    show nothing. Pressing Ctrl-C stops the call. Setting the
    ``UNIVERSALMOTIF_PROGRESS`` environment variable to 1 turns bars on for
    every session.

    Parameters
    ----------
    enabled
        Whether to show progress by default; None goes back to the
        environment variable.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.set_progress(True)
    >>> um.get_progress()
    True
    >>> um.set_progress(None)
    """
    global _setting
    if enabled is not None and not isinstance(enabled, bool):
        raise TypeError("enabled must be True, False or None")
    _setting = enabled


def get_progress() -> bool:
    """Whether long-running functions show progress bars by default.

    See [`set_progress()`][universalmotif.set_progress].

    Examples
    --------
    >>> import universalmotif as um
    >>> um.get_progress()
    False
    """
    if _setting is not None:
        return _setting
    return os.environ.get(_ENV, "").strip().lower() in ("1", "true", "yes", "on")


class _Outer:
    """Progress kept in Python, for functions that loop over several calls
    to the extension. Of ``total`` units, ``done`` are finished and ``part``
    belong to the call under way."""

    def __init__(self, stage: str, total: int):
        self.stage, self.total, self.done, self.part = stage, total, 0, 0


class outer:
    """Count units of work in Python, within a call run with a bar.

    Use ``with outer("scanning", n) as o:``. Set ``o.part = k`` before a call
    to the extension that covers k units, and add k to ``o.done`` after it.
    The bar then shows the whole loop rather than each call. Outside a call
    with a bar, the counts go nowhere."""

    def __init__(self, stage: str, total: int):
        self._o = _Outer(stage, total)
        self._key = threading.get_ident()

    def __enter__(self) -> _Outer:
        _outers[self._key] = self._o
        return self._o

    def __exit__(self, *exc) -> None:
        _outers.pop(self._key, None)


def _fmt_count(n: float) -> str:
    for size, unit in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return f"{n:.0f}"


def _fmt_time(s: float) -> str:
    s = int(s)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


class _Bar:
    """A one-line bar on stderr, redrawn in place."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stderr
        self.width = 0
        self.shown = False

    def draw(self, label: str, fraction: float | None, count: float, elapsed: float) -> None:
        if fraction is None:
            text = f"{label}: {_fmt_count(count)} [{_fmt_time(elapsed)}]"
        else:
            fraction = min(max(fraction, 0.0), 1.0)
            filled = int(round(24 * fraction))
            left = elapsed * (1 - fraction) / fraction if fraction > 0 else 0
            text = (f"{label}: {100 * fraction:5.1f}% |{'#' * filled}{'-' * (24 - filled)}| "
                    f"[{_fmt_time(elapsed)}<{_fmt_time(left)}]")
        self.stream.write("\r" + text + " " * max(self.width - len(text), 0))
        self.stream.flush()
        self.width = len(text)
        self.shown = True

    def close(self) -> None:
        if self.shown:
            self.stream.write("\n")
            self.stream.flush()


def _state(handle, o: _Outer | None) -> tuple[str, float | None, float]:
    """The label, the fraction done (None when it is not known) and the
    count to show."""
    stage, done, total = handle.stage, handle.done, handle.total
    if o is not None and o.total > 0:
        inner = done / total if total > 0 else 0.0
        return o.stage, (o.done + inner * o.part) / o.total, o.done
    if total > 0:
        return stage or "working", done / total, done
    return stage or "working", None, done


def wanted(progress: bool | None) -> bool:
    """Whether a call given ``progress`` should show a bar.

    It should when ``progress`` is true, or when it is None and
    [`get_progress()`][universalmotif.get_progress] is true. A call that is
    already part of one showing a bar never gets its own."""
    if progress is None:
        progress = get_progress()
    elif not isinstance(progress, bool):
        raise TypeError("progress must be True, False or None")
    return progress and threading.get_ident() not in _running


def run(f: Callable[[], T], stream=None) -> T:
    """``f()`` on a worker thread, with a progress bar."""
    handle = _core.Progress()
    result: dict = {}

    def work() -> None:
        _running.add(threading.get_ident())
        try:
            result["value"] = _core._with_progress(handle, f)
        except BaseException as e:  # noqa: BLE001 - raised again by the caller
            result["error"] = e
        finally:
            _running.discard(threading.get_ident())

    thread = threading.Thread(target=work, name="universalmotif-progress", daemon=True)
    bar = _Bar(stream)
    start = time.monotonic()
    thread.start()
    try:
        while thread.is_alive():
            thread.join(_INTERVAL)
            elapsed = time.monotonic() - start
            o = _outers.get(thread.ident)
            # Nothing is drawn before the first stage has begun.
            if elapsed >= _DELAY and thread.is_alive() and (handle.stage or o is not None):
                bar.draw(*_state(handle, o), elapsed)
    except KeyboardInterrupt:
        handle.cancel()
        thread.join()
        bar.close()
        raise
    if bar.shown:
        label, fraction, count = _state(handle, None)
        bar.draw(label, None if fraction is None else 1.0, count, time.monotonic() - start)
    bar.close()
    if "error" in result:
        raise result["error"]
    return result["value"]


# Worker threads running a call with a bar, so that calls they make in turn
# report to it rather than starting bars of their own.
_running: set[int] = set()
