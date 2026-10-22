"""The examples in the docstrings of _core.pyi run as doctests.

The compiled classes are documented in the stub file, which pytest does not
collect doctests from, so its docstrings are parsed and run here.
"""

import ast
import doctest
from pathlib import Path

import pytest

import universalmotif

STUB = Path(universalmotif.__file__).parent / "_core.pyi"


def _docstrings():
    tree = ast.parse(STUB.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            doc = ast.get_docstring(node, clean=True)
            if doc and ">>>" in doc:
                yield node.name, doc


DOCSTRINGS = dict(_docstrings())


@pytest.mark.parametrize("name", list(DOCSTRINGS))
def test_stub_docstring(name):
    test = doctest.DocTestParser().get_doctest(DOCSTRINGS[name], {}, name, str(STUB), 0)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    runner.run(test)
    assert runner.summarize(verbose=False).failed == 0
