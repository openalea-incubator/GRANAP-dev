"""Tests for the 3D hydraulic network (doc/NETWORK_3D_PLAN.md).

P1: ``build_wall_registry`` -- the 2D walls and junctions, shared by the 2D
network export and the 3D network.
"""

import pytest

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy
from openalea.granap.anatomy_writer import build_wall_registry


def _small_root():
    data = OrganInputData.for_root()
    for layer in ("epidermis", "exodermis", "cortex", "endodermis", "pericycle"):
        data.set_values(layer, cell_diameter=0.06, cell_width=0.06)
    data.set_values("stele", cell_diameter=0.03, cell_diameter_center=0.03)
    return RootAnatomy(data, seed=0)


@pytest.fixture(scope="module")
def section():
    return _small_root().generate_cells()


def test_registry_walls_and_junctions_are_consistent(section):
    reg = build_wall_registry(section)
    walls = list(reg.walls.values())

    assert [w["id"] for w in walls] == list(range(len(walls)))
    assert reg.junction_ids == {vk: i for i, vk in enumerate(reg.junctions)}
    for w in walls:
        assert 1 <= len(w["cells"]) <= 2
        assert set(w["cells"]) <= set(reg.cells_gdf.index)
        assert w["junc_start"] in reg.junction_ids and w["junc_end"] in reg.junction_ids
        assert w["length"] > 0


def test_registry_walls_cover_each_cell_boundary(section):
    """The walls of a cell add up to its perimeter (up to vertex snapping)."""
    reg = build_wall_registry(section)
    perimeter = {i: 0.0 for i in reg.cells_gdf.index}
    for w in reg.walls.values():
        for c in w["cells"]:
            perimeter[c] += w["length"]
    for i, p in perimeter.items():
        assert p == pytest.approx(reg.cells_gdf.loc[i, "geometry"].length, rel=1e-3)


def test_registry_wall_thickness(section):
    """Shared wall: t(a) + t(b); outer wall: t(a) + t('outerwall');
    a type missing from the dict takes 'default'."""
    t = {"cortex": 1.0, "epidermis": 3.0, "outerwall": 10.0, "default": 0.5}
    reg = build_wall_registry(section, cell_wall_thickness=t)
    types = reg.cells_gdf["type"]
    for w in reg.walls.values():
        expected = sum(t.get(types[c], t["default"]) for c in w["cells"])
        if len(w["cells"]) == 1:
            expected += t["outerwall"]
        assert w["wall_thickness"] == pytest.approx(expected)


def test_registry_of_the_3d_section_covers_every_3d_cell():
    """The 3D build's own section gives the walls of the 3D cells."""
    result = _small_root().generate_cells_3d(n_axial_repeats=2.0, seed=0)
    reg = build_wall_registry(result.section)
    assert {c["cell_2d"] for c in result.cells} <= set(reg.cells_gdf.index)
