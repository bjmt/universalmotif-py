"""Fill the command-line page with the help of every command, taken from
the parser itself so that it cannot fall out of date."""

import argparse

_MARKER = "<!-- command help -->"


def _help(parser: argparse.ArgumentParser) -> str:
    # A fixed width keeps the text the same on every machine.
    import os

    old = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = "88"
    try:
        return parser.format_help()
    finally:
        if old is None:
            del os.environ["COLUMNS"]
        else:
            os.environ["COLUMNS"] = old


def on_page_markdown(markdown, page, config, files):
    if _MARKER not in markdown:
        return markdown
    from universalmotif._cli import build_parser

    parser = build_parser("umotif")
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    parts = []
    for name, p in sub.choices.items():
        parts.append(f"### {name}\n\n```text\n{_help(p)}```\n")
        nested = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
        for action in nested:
            for inner_name, inner in action.choices.items():
                parts.append(f"#### {name} {inner_name}\n\n```text\n{_help(inner)}```\n")
    return markdown.replace(_MARKER, "\n".join(parts))
