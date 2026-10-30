"""The package-wide number of threads."""

from __future__ import annotations

import os
import warnings

_ENV = "UNIVERSALMOTIF_NTHREADS"


def _check(n: object) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"nthreads must be a non-negative integer, not {n!r}")
    return n


def _from_env() -> int:
    value = os.environ.get(_ENV, "").strip()
    if not value:
        return 1
    try:
        return _check(int(value))
    except ValueError:
        warnings.warn(f"ignoring {_ENV}={value!r}, which is not a non-negative integer; "
                      "using 1 thread", stacklevel=2)
        return 1


_nthreads_setting = _from_env()


def set_nthreads(n: int) -> None:
    """Set the number of threads used by functions that take ``nthreads``.

    Every function with an ``nthreads`` argument uses this setting when the
    argument is left as ``None``, and an ``nthreads`` given to a function
    still wins. The setting starts at 1, or at the value of the
    ``UNIVERSALMOTIF_NTHREADS`` environment variable when the package is
    imported, and 0 means all available cores.

    Examples
    --------
    >>> import universalmotif as um
    >>> before = um.get_nthreads()
    >>> um.set_nthreads(4)
    >>> um.get_nthreads()
    4
    >>> um.set_nthreads(before)
    """
    global _nthreads_setting
    _nthreads_setting = _check(n)


def get_nthreads() -> int:
    """The number of threads set with
    [`set_nthreads()`][universalmotif.set_nthreads], where 0 means all
    available cores."""
    return _nthreads_setting


def _nthreads(n: int | None) -> int:
    """``n``, or the package setting if it is ``None``."""
    return _nthreads_setting if n is None else n


__all__ = ["get_nthreads", "set_nthreads"]
