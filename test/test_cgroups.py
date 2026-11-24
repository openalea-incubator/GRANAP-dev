"""
Every cell type GRANAP's presets generate has its MECHA cgroup (anatomy_writer.CGROUP_MAP): none falls
back to 4 (cortex) unlisted, and the shoots' phloem is MECHA's (sieve elements 11, companion cells 12).
"""
import pytest

from openalea.granap.anatomy_writer import CGROUP_MAP
from openalea.granap.input_data import OrganInputData
from openalea.granap.leaf_class import LeafAnatomy
from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.root_class import RootAnatomy
from openalea.granap.stem_class import StemAnatomy

PRESETS = {"for_root": RootAnatomy, "for_woody_dicot": RootAnatomy, "for_dicot_stem": StemAnatomy,
           "for_monocot_stem": StemAnatomy, "for_dicot_leaf": LeafAnatomy, "for_monocot_leaf": LeafAnatomy,
           "for_needle": NeedleAnatomy}


@pytest.mark.parametrize("preset", list(PRESETS))
def test_every_generated_type_has_its_cgroup(preset):
    organ = PRESETS[preset](getattr(OrganInputData, preset)(), seed=0)
    organ.generate_cells()
    types = {c.type for c in organ.all_cells.cells}
    assert types <= set(CGROUP_MAP), sorted(types - set(CGROUP_MAP))
    if "sieve element" in types:
        assert CGROUP_MAP["sieve element"] == 11 and CGROUP_MAP["companion cell"] == 12
