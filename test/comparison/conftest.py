"""
The comparison folder: GRANAP (the main package, generating into metafspm's MPG) against its legacy source
(openalea.granap.legacy, the original, frozen), entity by entity — cells, GeoDataFrame, network, XML, the stages.

  pytest test/comparison                  everything (minutes)
  pytest test/comparison -m "not slow"    the quick subset
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.dirname(HERE)):                # the comparison helpers, and the tests' (needle_configs)
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: the heaviest cross-checks (minutes each)")



@pytest.fixture(autouse=True)
def _reviewed_cgroups():
    """The main package's added cgroups, replayed on the legacy's map (its only difference there)."""
    from openalea.granap.anatomy_writer import CGROUPS_ADDED
    from openalea.granap.legacy import anatomy_writer as legacy
    saved = dict(legacy.CGROUP_MAP)
    legacy.CGROUP_MAP.update(CGROUPS_ADDED)
    yield
    legacy.CGROUP_MAP.clear()
    legacy.CGROUP_MAP.update(saved)
