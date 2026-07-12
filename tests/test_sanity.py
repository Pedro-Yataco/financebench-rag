"""Sanity checks for the project scaffold."""

import sys

import src


def test_python_is_312() -> None:
    assert sys.version_info[:2] == (3, 12)


def test_project_package_imports() -> None:
    assert src.__name__ == "src"
