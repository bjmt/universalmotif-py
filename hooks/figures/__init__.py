"""Example figures for the API reference.

Each module here defines ``EXAMPLES``, a dict from the name a reference page
documents with ``::: universalmotif.<name>`` to a list of ``Example``s. The
``plot_examples`` hook draws them when the docs are built and shows them
after that entry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Example:
    """One figure: ``draw`` returns a matplotlib Figure (or Axes), or an SVG
    string, and ``caption`` says what it shows and with which data."""

    caption: str
    draw: Callable[[], Any]
