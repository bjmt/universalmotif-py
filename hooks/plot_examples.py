"""Draw the example figures of the plotting functions and show them in the
API reference, after the entry of each function.

The examples live in hooks/figures/. They are drawn once per change
to them or to the package, and kept in .cache/plot_examples between builds.
"""

from __future__ import annotations

import hashlib
import html
import importlib
import pkgutil
import re
import sys
from pathlib import Path

from mkdocs.structure.files import File

_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / ".cache" / "plot_examples"
_DEST = "reference/plots"
_ENTRY = re.compile(r"^::: universalmotif\.([\w.]+)[ \t]*$")
_drawn: dict[str, list[tuple[str, str]]] = {}


def _key() -> str:
    """A hash of the example code and the package, so figures are redrawn
    only when either changes."""
    h = hashlib.sha256()
    sources = sorted((_ROOT / "hooks" / "figures").glob("*.py"))
    sources += sorted((_ROOT / "python" / "universalmotif").glob("*.py"))
    for p in sources:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    for so in (_ROOT / "python" / "universalmotif").glob("_core*.so"):
        h.update(str(so.stat().st_mtime_ns).encode())
    return h.hexdigest()[:16]


def _examples() -> dict:
    # Drop modules imported by an earlier build, so that mkdocs serve picks
    # up edits to the examples.
    for name in [m for m in sys.modules if m == "figures" or m.startswith("figures.")]:
        del sys.modules[name]
    sys.path.insert(0, str(_ROOT / "hooks"))
    try:
        import figures

        found: dict = {}
        for info in pkgutil.iter_modules(figures.__path__):
            module = importlib.import_module(f"figures.{info.name}")
            for name, examples in module.EXAMPLES.items():
                if name in found:
                    raise ValueError(f"examples of {name} are defined twice")
                found[name] = examples
        return found
    finally:
        sys.path.remove(str(_ROOT / "hooks"))


def _draw(out: Path) -> dict[str, list[tuple[str, str]]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawn = {}
    for name, examples in sorted(_examples().items()):
        files = []
        for i, ex in enumerate(examples, 1):
            result = ex.draw()
            if isinstance(result, str):
                fname = f"{name}-{i}.svg"
                (out / fname).write_text(result)
            else:
                fig = getattr(result, "figure", result)
                fname = f"{name}-{i}.png"
                fig.savefig(out / fname, dpi=110, bbox_inches="tight", facecolor="white")
                plt.close(fig)
            files.append((fname, ex.caption))
        drawn[name] = files
    return drawn


def on_config(config):
    global _drawn
    out = _CACHE / _key()
    index = out / "index.txt"
    if index.exists():
        _drawn = {}
        for line in index.read_text().splitlines():
            name, fname, caption = line.split("\t", 2)
            _drawn.setdefault(name, []).append((fname, caption))
        return config
    for old in _CACHE.glob("*"):
        if old.is_dir():
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
    out.mkdir(parents=True)
    _drawn = _draw(out)
    index.write_text("".join(f"{n}\t{f}\t{c}\n" for n, fs in _drawn.items() for f, c in fs))
    return config


def on_files(files, config):
    out = _CACHE / _key()
    for figs in _drawn.values():
        for fname, _ in figs:
            files.append(File.generated(config, f"{_DEST}/{fname}", abs_src_path=str(out / fname)))
    return files


def _gallery(page_dir: str, figures: list[tuple[str, str]]) -> str:
    prefix = Path(_DEST).relative_to(page_dir) if page_dir else Path(_DEST)
    items = "\n".join(
        f'<figure markdown="span">\n  ![{html.escape(caption, quote=True)}]({prefix.as_posix()}/{fname})'
        f'{{ loading=lazy }}\n  <figcaption>{html.escape(caption)}</figcaption>\n</figure>'
        for fname, caption in figures
    )
    return f'\n<div class="plot-examples" markdown>\n{items}\n</div>\n'


def on_page_markdown(markdown, page, config, files):
    if not page.file.src_uri.startswith("reference/"):
        return markdown
    page_dir = str(Path(page.file.src_uri).parent)
    lines = markdown.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        m = _ENTRY.match(lines[i])
        i += 1
        if m and m.group(1) in _drawn:
            # The entry's options are indented under it.
            while i < len(lines) and lines[i].startswith((" ", "\t")):
                out.append(lines[i])
                i += 1
            out.append(_gallery(page_dir, _drawn[m.group(1)]))
    return "\n".join(out)
