"""Unit tests for the shared special-tissue vocabulary (special_tissues.py).

These cover the organ-agnostic structural primitives directly; the needle's
``add_canal`` / ``add_stomata`` (which now sit on top of ``place_resin_duct`` /
``place_stomata``) are guarded byte-for-byte by ``test_vascular_regression.py``.
"""

import os
import sys

import numpy as np
from shapely.geometry import Point, Polygon

sys.path.append(os.path.abspath(".."))

from openalea.granap.cell_manager import CellManager
from openalea.granap.cell_class import Cell
from openalea.granap.special_tissues import carve_and_insert, consider_as_cell


def _square(cx, cy, h=0.4):
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


# ---------------------------------------------------------------------------
# Cell.radial — the shared seeding idiom (angle/radius from a centre)
# ---------------------------------------------------------------------------

def test_cell_radial_polar_from_center():
    c = Cell.radial("xylem", 3.0, 4.0, 0.2, id_group=5, center=(0.0, 0.0))
    assert c.type == "xylem"
    assert c.id_group == 5 and c.id_cell == 5            # id_cell defaults to id_group
    assert np.isclose(c.radius, 5.0)                     # hypot(3, 4)
    assert np.isclose(c.angle, np.arctan2(4.0, 3.0))
    assert np.isclose(c.area, np.pi * 0.1 ** 2)          # disc of `diameter`


def test_cell_radial_center_offset_and_explicit_ids():
    c = Cell.radial("phloem", 1.0, 1.0, 0.4, id_group=2,
                    center=Point(1.0, 0.0), id_cell=9, id_layer=3)
    assert c.id_group == 2 and c.id_cell == 9 and c.id_layer == 3
    assert np.isclose(c.radius, 1.0)                     # (1,0) -> (1,1)
    assert np.isclose(c.angle, np.pi / 2)                # straight up
    # matches the hand-written idiom it replaces
    assert np.isclose(c.radius, np.sqrt((1.0 - 1.0) ** 2 + (1.0 - 0.0) ** 2))


def test_carve_and_insert_removes_overlap_and_adds():
    cm = CellManager()
    # three existing cells with polygons; the middle one overlaps the carve zone
    for i, x in enumerate((0.0, 5.0, 10.0)):
        cm.add_cell(Cell(x=x, y=0.0, diameter=0.8, type="cortex",
                         id_cell=i, id_group=i, polygon=_square(x, 0.0)))

    carve = _square(5.0, 0.0, h=0.2)            # overlaps only the middle cell
    new_cells = [Cell(x=5.0, y=0.0, diameter=0.1, type="duct", id_cell=0, id_group=0)]

    carve_and_insert(cm, [carve], new_cells)

    types = sorted(c.type for c in cm.cells)
    assert types == ["cortex", "cortex", "duct"]   # middle cortex carved out, duct added
    # recalculate_cell_properties renumbers id_cell to the list index
    assert [c.id_cell for c in cm.cells] == list(range(len(cm.cells)))


def test_carve_and_insert_buffer_widens_removal():
    cm = CellManager()
    cm.add_cell(Cell(x=1.0, y=0.0, diameter=0.8, type="cortex",
                     id_cell=0, id_group=0, polygon=_square(1.0, 0.0)))
    carve = _square(0.0, 0.0, h=0.2)            # does NOT touch the cell at x=1...

    carve_and_insert(cm, [carve], [], buffer=0.0)
    assert len(cm.cells) == 1                    # ...so nothing removed

    carve_and_insert(cm, [carve], [], buffer=0.6)   # buffered out it now reaches it
    assert cm.cells == []


def test_consider_as_cell_collapses_region():
    cm = CellManager()
    for i, x in enumerate((0.0, 0.3, 0.6)):
        cm.add_cell(Cell(x=x, y=0.0, diameter=0.1, type="phloem",
                         id_cell=i, id_group=i, polygon=_square(x, 0.0, h=0.05)))

    region = _square(0.3, 0.0, h=1.0)           # covers all three
    cell = consider_as_cell(cm, region, "sieve_plate")

    assert cell.type == "sieve_plate"
    assert cell.polygon is region
    # the three small cells were replaced by the single collapsed cell
    assert len(cm.cells) == 1
    assert cm.cells[0].type == "sieve_plate"


def test_consider_as_cell_keep_existing():
    cm = CellManager()
    cm.add_cell(Cell(x=0.0, y=0.0, diameter=0.1, type="phloem",
                     id_cell=0, id_group=0, polygon=_square(0.0, 0.0, h=0.05)))
    region = _square(0.0, 0.0, h=1.0)
    consider_as_cell(cm, region, "blob", replace=False)
    assert len(cm.cells) == 2                    # original kept, blob added
    assert sorted(c.type for c in cm.cells) == ["blob", "phloem"]
