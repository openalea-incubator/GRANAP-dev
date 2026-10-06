"""Tests for Organ.generate_cells_3d() / generate_cell_3d.py: literal-copy
2D-cell extrusion, no 3D Voronoi.

The organ fixtures are deliberately small/coarse (large cell_diameter -> few 2D
cells, few axial repeats) so this stays fast in CI.
"""

from collections import Counter

import pytest

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy
from openalea.granap.stem_monocot_class import MonocotStemAnatomy
from openalea.granap.leaf_class import DicotLeafAnatomy
from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.generate_cell_3d import VESSEL_TYPES, AIR_SPACE_TYPE


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


# One organ per family, so the shared Organ-level contract stays exercised
# beyond the root the pipeline was first written against.
ORGAN_BUILDERS = {
    "root":   _small_root,
    "stem":   lambda: MonocotStemAnatomy(OrganInputData.for_monocot_stem(), seed=0),
    "leaf":   lambda: DicotLeafAnatomy(OrganInputData.for_dicot_leaf(), seed=0),
    "needle": lambda: NeedleAnatomy(OrganInputData.for_needle(), seed=0),
}


def _check_common(result):
    assert result.cells, "no cells produced"

    for cell in result.cells:
        assert len(cell["faces"]) >= 3
        assert len(cell["vertices"]) >= 6  # a prism has >= 2 * 3 vertices
        assert cell["volume"] > 0
        assert _watertight(cell["faces"]), f"non-watertight {cell['type']} cell"

    # Every cell's vertices fall within [z_min, z_max]; every vessel spans the
    # whole segment exactly (one extrusion, no phase).
    for cell in result.cells:
        zs = cell["vertices"][:, 2]
        assert zs.min() >= result.z_min - 1e-9
        assert zs.max() <= result.z_max + 1e-9
        if cell["type"] in VESSEL_TYPES:
            assert abs(zs.min() - result.z_min) < 1e-9
            assert abs(zs.max() - result.z_max) < 1e-9
    return result.summary()


@pytest.mark.parametrize("organ_name", sorted(ORGAN_BUILDERS))
def test_generate_cells_3d_every_organ_family(organ_name):
    """The pipeline only touches Organ-level API, so it must work for every
    organ family -- not just the root it was first written against."""
    organ = ORGAN_BUILDERS[organ_name]()
    result = organ.generate_cells_3d(n_axial_repeats=2.0, seed=0)
    summary = _check_common(result)
    assert summary, f"{organ_name} produced no typed cells"


def test_ordinary_intercellular_space_is_disabled():
    # Aerenchyma off and ordinary intercellular generation disabled at the 2D
    # source -> a root has no other route to "air space", so none appear.
    root = _small_root(aerenchyma_proportion=0.0)
    result = root.generate_cells_3d(n_axial_repeats=4.0, seed=0)
    summary = _check_common(result)
    assert AIR_SPACE_TYPE not in summary


# ---------------------------------------------------------------------------
# Aerenchyma timing
# ---------------------------------------------------------------------------

def test_aerenchyma_before_gives_continuous_axial_lacunae():
    """"before" is the rice-root case: the lacuna is one long channel, so each
    air-space cell must be a single full-span prism, not a stack of rows."""
    root = _small_root(aerenchyma_proportion=0.2)
    result = root.generate_cells_3d(n_axial_repeats=4.0, aerenchyma_mode="before", seed=0)
    summary = _check_common(result)

    assert summary.get(AIR_SPACE_TYPE, 0) > 0
    for cell in result.cells:
        if cell["type"] == AIR_SPACE_TYPE:
            zs = cell["vertices"][:, 2]
            assert abs(zs.min() - result.z_min) < 1e-9
            assert abs(zs.max() - result.z_max) < 1e-9


def test_aerenchyma_after_gives_distributed_voids():
    """"after" is the leaf case: aerenchyma is suppressed in 2D and recreated
    per-prism on the stack, so the voids are short rows, not full-span files."""
    root = _small_root(aerenchyma_proportion=0.2)
    result = root.generate_cells_3d(n_axial_repeats=4.0, aerenchyma_mode="after", seed=0)
    summary = _check_common(result)

    air = [c for c in result.cells if c["type"] == AIR_SPACE_TYPE]
    assert air, "'after' mode produced no air space"

    span = result.z_max - result.z_min
    heights = [c["vertices"][:, 2].max() - c["vertices"][:, 2].min() for c in air]
    assert all(h < span - 1e-9 for h in heights), \
        "'after' voids must be individual rows, not full-height channels"


def test_aerenchyma_none_produces_no_air():
    root = _small_root(aerenchyma_proportion=0.2)
    result = root.generate_cells_3d(n_axial_repeats=4.0, aerenchyma_mode="none", seed=0)
    summary = _check_common(result)
    assert summary.get(AIR_SPACE_TYPE, 0) == 0


def test_aerenchyma_mode_is_validated():
    root = _small_root()
    with pytest.raises(ValueError, match="aerenchyma_mode"):
        root.generate_cells_3d(n_axial_repeats=2.0, aerenchyma_mode="whenever", seed=0)


# ---------------------------------------------------------------------------
# The organ must survive its own 3D build
# ---------------------------------------------------------------------------

def test_generate_cells_3d_leaves_the_organ_unchanged():
    """generate_cells_3d overrides intercellular/aerenchyma params for its 2D
    pass; it must restore them, or the organ's own 2D section is silently
    different afterwards."""
    reference = _small_root(aerenchyma_proportion=0.2)
    reference.generate_cells()
    expected_cells = len(reference.all_cells.cells)
    expected_air = sum(1 for c in reference.all_cells.cells if c.type == AIR_SPACE_TYPE)

    organ = _small_root(aerenchyma_proportion=0.2)
    saved_ics = list(organ.intercellular_spaces_params)
    saved_prop = organ.aerenchyma_params.get("aerenchyma_proportion")

    organ.generate_cells_3d(n_axial_repeats=2.0, aerenchyma_mode="after", seed=0)

    assert organ.intercellular_spaces_params == saved_ics
    assert organ.aerenchyma_params.get("aerenchyma_proportion") == saved_prop

    organ.generate_cells()
    assert len(organ.all_cells.cells) == expected_cells
    assert sum(1 for c in organ.all_cells.cells if c.type == AIR_SPACE_TYPE) == expected_air
