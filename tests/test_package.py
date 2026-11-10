"""Smoke tests confirming the package and its modules import correctly."""

import importlib

import pytest

import neural_net

MODULES = ["layers", "losses", "model", "optimizers", "init", "train", "gradcheck", "data"]


def test_version_is_defined() -> None:
    assert isinstance(neural_net.__version__, str)
    assert neural_net.__version__.count(".") == 2


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    assert importlib.import_module(f"neural_net.{module}") is not None
