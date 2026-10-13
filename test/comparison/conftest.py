"""
The comparison folder: GRANAP (the main package, generating into metafspm's MPG) against its legacy source
(openalea.granap.legacy, the original, frozen), entity by entity — cells, GeoDataFrame, network, XML, the stages.

  pytest test/comparison                  everything (minutes)
  pytest test/comparison -m "not slow"    the quick subset
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.dirname(HERE)):                # the comparison helpers, and the tests' (needle_configs)
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: the heaviest cross-checks (minutes each)")
