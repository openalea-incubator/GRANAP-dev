"""
The comparison folder (devplan_metafspm.md, Q6): every original test, replayed on the MPG-backed organs (one
test_<name>.py per original file, see _replay.py), and the entity-by-entity cross-checks (test_cross_check.py).

  pytest test/mpg_comparison                  everything (about 25 min with the cross-checks)
  pytest test/mpg_comparison -m "not slow"    the quick subset, for each conversion step
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.dirname(HERE)):                # _replay, and the original tests' helpers (needle_configs)
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: the heaviest replays and cross-checks (minutes each)")


@pytest.fixture(autouse=True)
def _mpg_backed_library(monkeypatch):
    """During each comparison test, the library code that builds organs builds the MPG-backed ones."""
    import importlib
    from openalea.granap import mpg
    from openalea.granap.organ_class import Organ
    from _replay import CONSTRUCTING
    for module_name, attribute in CONSTRUCTING:
        monkeypatch.setattr(importlib.import_module(module_name), attribute, getattr(mpg, attribute))
    monkeypatch.setattr(Organ, "create_from_input", staticmethod(mpg.create_from_input))
