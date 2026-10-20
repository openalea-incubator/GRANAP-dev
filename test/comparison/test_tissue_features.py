"""
The tissue branch's features that are not single sections (devplan_metafspm.md §62), GRANAP against its legacy
source: the monocot root series (apex → collar sections with tracked, fusing metaxylem vessels) section by section and
its track table, and the 3D extrusion of a section (every prism's type, vertices and faces) in each aerenchyma mode,
the organ's own 2D section left as it was.
"""
import numpy as np
import pytest

from openalea.granap.input_data import OrganInputData
from openalea.granap.legacy.input_data import OrganInputData as LegacyInputData
from openalea.granap.legacy.root_class import RootAnatomy as LegacyRoot
from openalea.granap.legacy.root_series import RootSeries as LegacyRootSeries
from openalea.granap.root_class import RootAnatomy
from openalea.granap.root_series import RootSeries

from test_cross_check import _check_cells

SERIES = dict(lengths=[0.0, 50.0, 100.0], n_fused=lambda length: 2 if length >= 50.0 else 1,
              vessel_radius=(0.03, 0.03), stele_radius=(0.12, 0.12))


def _legacy(data):
    return LegacyInputData.from_dict_list(data.to_dict_list())


def test_root_series():
    old = LegacyRootSeries(_legacy(OrganInputData.for_root()), seed=0, **SERIES).generate()
    new = RootSeries(OrganInputData.for_root(), seed=0, **SERIES).generate()
    assert old.lengths == new.lengths
    for a, b in zip(old.sections, new.sections):
        _check_cells(a["root"], b["root"])
        assert a["vessels"] == b["vessels"]
    assert old.track_rows == new.track_rows


def _small_root(cls, data_cls):
    """test_cells_3d's small root (large cells), with aerenchyma."""
    data = OrganInputData.for_root()
    for layer in ("epidermis", "exodermis", "cortex", "endodermis", "pericycle"):
        data.set_values(layer, cell_diameter=0.06, cell_width=0.06)
    data.set_values("stele", cell_diameter=0.03, cell_diameter_center=0.03)
    data.set_value("aerenchyma", "aerenchyma_proportion", 0.2)
    return cls(data if data_cls is OrganInputData else _legacy(data), seed=0)


@pytest.mark.parametrize("mode", ["before", "after", "none"])
def test_cells_3d(mode):
    old, new = _small_root(LegacyRoot, LegacyInputData), _small_root(RootAnatomy, OrganInputData)
    a = old.generate_cells_3d(n_axial_repeats=3.0, aerenchyma_mode=mode, seed=0)
    b = new.generate_cells_3d(n_axial_repeats=3.0, aerenchyma_mode=mode, seed=0)
    assert (a.z_min, a.z_max) == (b.z_min, b.z_max) and len(a.cells) == len(b.cells)
    for p, q in zip(a.cells, b.cells):
        assert p["type"] == q["type"] and p["faces"] == q["faces"] and p["volume"] == q["volume"]
        np.testing.assert_array_equal(p["vertices"], q["vertices"])
    old.generate_cells()                                     # the organ's own section afterwards: still the same
    new.generate_cells()
    _check_cells(old, new)
