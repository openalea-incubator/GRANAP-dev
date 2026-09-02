"""Smoke test for Organ.generate_cells_3d() / generate_cell_3d.py: literal-copy
2D-cell extrusion, no 3D Voronoi. Deliberately a small/coarse organ (large
cell_diameter -> few 2D cells) so this stays fast in CI.
"""

from collections import Counter

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy
from openalea.granap.generate_cell_3d import VESSEL_TYPES


def _watertight(faces) -> bool:
    """Every edge of a closed polyhedron must be shared by exactly 2 faces."""
    edge_count = Counter()
    for face in faces:
        n = len(face)
        for i in range(n):
            edge_count[tuple(sorted((face[i], face[(i + 1) % n])))] += 1
    return set(edge_count.values()) == {2}


def _small_root(aerenchyma_proportion=None):
    data = OrganInputData.for_root()
    for layer in ("epidermis", "exodermis", "cortex", "endodermis", "pericycle"):
        data.set_values(layer, cell_diameter=0.06, cell_width=0.06)
    data.set_values("stele", cell_diameter=0.03, cell_diameter_center=0.03)
    if aerenchyma_proportion is not None:
        data.set_value("aerenchyma", "aerenchyma_proportion", aerenchyma_proportion)
    return RootAnatomy(data, seed=0)


def _check_common(result):
    assert result.cells, "no cells produced"

    summary = result.summary()
    for vtype in ("metaxylem", "protoxylem", "phloem"):
        if vtype not in summary:
            continue
        # Vessels get exactly one extrusion each -- no repeated rows.
        assert summary[vtype] > 0

    for cell in result.cells:
        assert len(cell["faces"]) >= 3
        assert len(cell["vertices"]) >= 6  # a prism has >= 2 * 3 vertices
        assert _watertight(cell["faces"]), f"non-watertight {cell['type']} cell"

    # Every non-vessel cell's vertices fall within [z_min, z_max]; every
    # vessel spans the whole segment exactly (one extrusion, no phase).
    for cell in result.cells:
        zs = cell["vertices"][:, 2]
        assert zs.min() >= result.z_min - 1e-9
        assert zs.max() <= result.z_max + 1e-9
        if cell["type"] in VESSEL_TYPES:
            assert abs(zs.min() - result.z_min) < 1e-9
            assert abs(zs.max() - result.z_max) < 1e-9
    return summary


def test_generate_cells_3d_smoke():
    root = _small_root()
    result = root.generate_cells_3d(n_axial_repeats=4.0, seed=0)
    _check_common(result)


def test_ordinary_intercellular_space_is_disabled():
    # Aerenchyma off -> the only source of "air space" cells (ordinary
    # intercellular generation) is disabled at the 2D source, so none appear.
    root = _small_root(aerenchyma_proportion=0.0)
    result = root.generate_cells_3d(n_axial_repeats=4.0, seed=0)
    summary = _check_common(result)
    assert "air space" not in summary


def test_aerenchyma_is_extruded_like_a_normal_cell():
    # Aerenchyma on -> Organ.add_aerenchyma retypes some real tissue cells to
    # "air space" (it never creates new ones); those must still come through
    # the same extrusion path as everything else, not be skipped.
    root = _small_root(aerenchyma_proportion=0.2)
    result = root.generate_cells_3d(n_axial_repeats=4.0, seed=0)
    summary = _check_common(result)
    assert summary.get("air space", 0) > 0
