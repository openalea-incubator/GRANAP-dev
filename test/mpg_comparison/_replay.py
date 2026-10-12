"""
Replaying an original GRANAP test file against the MPG-backed organs (devplan_metafspm.md, Q6): the file's source is
executed in a fresh module, then every organ class it bound is replaced by its MPG-backed counterpart, so that its
tests, fixtures and helpers build MPG-backed organs and keep their original assertions. The original file is not
copied: a change to it is replayed as is.
"""
import importlib.util
import os
import sys

from openalea.granap import mpg
from openalea.granap.cellset_organ import CellSetOrgan
from openalea.granap.leaf_class import DicotLeafAnatomy, LeafAnatomy, MonocotLeafAnatomy
from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.roi_organ import RoiOrgan
from openalea.granap.root_class import RootAnatomy
from openalea.granap.root_dicot_class import DicotRootAnatomy
from openalea.granap.root_monocot_class import MonocotRootAnatomy
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.stem_dicot_class import DicotStemAnatomy
from openalea.granap.stem_dicot_continuous_class import ContinuousDicotStemAnatomy
from openalea.granap.stem_monocot_class import MonocotStemAnatomy

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKED = {cls: getattr(mpg, cls.__name__) for cls in (
    RootAnatomy, MonocotRootAnatomy, DicotRootAnatomy, StemAnatomy, MonocotStemAnatomy, DicotStemAnatomy,
    ContinuousDicotStemAnatomy, LeafAnatomy, MonocotLeafAnatomy, DicotLeafAnatomy, NeedleAnatomy, CellSetOrgan,
    RoiOrgan)}
# The library modules that build organs themselves (the others only inherit from them)
CONSTRUCTING = (("openalea.granap.root_series", "RootAnatomy"),)


def replay(name: str) -> dict:
    """The namespace of the original test module test/<name>.py, its organ classes MPG-backed."""
    path = os.path.join(TESTS, name + ".py")
    spec = importlib.util.spec_from_file_location(f"mpg_replay_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for attribute, value in list(vars(module).items()):
        if isinstance(value, type) and value in BACKED:
            setattr(module, attribute, BACKED[value])
    return {k: v for k, v in vars(module).items() if not k.startswith("__")}
