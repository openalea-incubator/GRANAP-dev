"""
The network stored at generation (built in the MPG from the section's cells and rings, openalea.granap.metafspm.network)
against the legacy exporter's (openalea.granap.legacy): node for node, edge for edge, every attribute, the counts and
the wall map, and the adjacency matrix read from the Connections. On every organ kind: roots, stems, leaves (protected
air spaces, air links), the needle, and the sections read from files (ROI: no junction shared by two rings; CellSet).
The legacy's network gets the main package's air rules first (pores as air nodes, every air space's membrane
wall_air, the pores' links to their chambers): those are the only differences allowed.
"""
import os

import pytest

from openalea.granap.cellset_organ import CellSetOrgan
from openalea.granap.input_data import OrganInputData
from openalea.granap.leaf_class import LeafAnatomy
from openalea.granap.legacy.cellset_organ import CellSetOrgan as LegacyCellSet
from openalea.granap.legacy.leaf_class import LeafAnatomy as LegacyLeaf
from openalea.granap.legacy.roi_organ import RoiOrgan as LegacyRoi
from openalea.granap.roi_organ import RoiOrgan

from test_cross_check import ORGANS, SEED, _apply_air_rules, _check_network, legacy

INPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inputs")
ROI = dict(folder_path=os.path.join(INPUTS, "Hypocotyl_RoiSet"), mm_per_pixel=0.001, smooth_factor=0.05)
CELLSET = os.path.join(INPUTS, "partial_CellSet_brasicaceae.xml")

CASES = {name: (lambda o=original, d=data: o(legacy(d()), seed=SEED), lambda b=backed, d=data: b(d(), seed=SEED))
         for name, (original, backed, data, _) in ORGANS.items()}
for name, preset in (("monocot_leaf", OrganInputData.for_monocot_leaf), ("dicot_leaf", OrganInputData.for_dicot_leaf)):
    CASES[name] = (lambda p=preset: LegacyLeaf(legacy(p()), seed=SEED), lambda p=preset: LeafAnatomy(p(), seed=SEED))
CASES["roi"] = (lambda: LegacyRoi(**ROI), lambda: RoiOrgan(**ROI))
CASES["cellset"] = (lambda: LegacyCellSet(CELLSET, seed=SEED), lambda: CellSetOrgan(CELLSET, seed=SEED))
QUICK = {"monocot_default", "dicot_leaf", "roi"}


@pytest.mark.parametrize("name", [pytest.param(n, marks=() if n in QUICK else pytest.mark.slow) for n in CASES])
def test_the_stored_network_is_the_legacy_exporters(name):
    build_old, build_new = CASES[name]
    old, new = build_old(), build_new()
    for organ in (old, new):
        organ.generate_cells()
    old.export_to_adjencymatrix(bridges=False)
    new.export_to_adjencymatrix(bridges=False)                 # the stored network, as stored
    _check_network(_apply_air_rules(old), new)                    # the legacy's, with the reviewed air rules
