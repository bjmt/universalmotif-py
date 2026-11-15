"""Arguments renamed for clearer names, which keep working for a while
under their old names."""

from __future__ import annotations

import warnings
from typing import Any


def renamed(func: str, old: str, value: Any, new: str, stacklevel: int = 3) -> Any:
    """Warn that ``old`` is an old name for ``new`` in ``func``, and give
    back its value."""
    warnings.warn(f"{func}(): {old} is now called {new}, and the old name will be removed",
                  FutureWarning, stacklevel=stacklevel)
    return value
