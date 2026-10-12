"""test/test_vascular_regression.py, replayed on the MPG-backed organs (see _replay.py)."""
import pytest
pytestmark = pytest.mark.slow
from _replay import replay

globals().update(replay("test_vascular_regression"))
