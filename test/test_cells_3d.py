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


# ---------------------------------------------------------------------------
# Identity and per-type axial heights (the 3D network's inputs)
# ---------------------------------------------------------------------------

def _rows_by_cell(result):
    rows = {}
    for cell in result.cells:
        rows.setdefault(cell["cell_2d"], []).append(cell)
    return {i: sorted(r, key=lambda c: c["row"]) for i, r in rows.items()}


def _inner_heights(result, cell_type):
    """Heights of the rows of ``cell_type`` that are not clipped by the segment ends."""
    return {round(c["z1"] - c["z0"], 9) for c in result.cells
            if c["type"] == cell_type
            and c["z0"] > result.z_min + 1e-9 and c["z1"] < result.z_max - 1e-9}


def test_every_3d_cell_knows_its_2d_cell_and_rows_tile_the_segment():
    root = _small_root()
    result = root.generate_cells_3d(n_axial_repeats=3.0, seed=0)
    section = result.section
    assert section is not None and len(section) > 0

    for i, rows in _rows_by_cell(result).items():
        assert section.iloc[i]["type"] == rows[0]["type"]
        assert [c["row"] for c in rows] == list(range(len(rows)))
        assert abs(rows[0]["z0"] - result.z_min) < 1e-9
        assert abs(rows[-1]["z1"] - result.z_max) < 1e-9
        for below, above in zip(rows, rows[1:]):
            assert abs(below["z1"] - above["z0"]) < 1e-9, "rows of a file must touch"
        area = section.iloc[i]["geometry"].area
        for c in rows:
            assert c["volume"] == pytest.approx(area * (c["z1"] - c["z0"]))
            zs = c["vertices"][:, 2]
            assert zs.min() == pytest.approx(c["z0"]) and zs.max() == pytest.approx(c["z1"])
    # Every 2D cell with a polygon is extruded.
    assert len(_rows_by_cell(result)) == sum(
        1 for g in section["geometry"] if g is not None and not g.is_empty)


def test_axial_heights_differ_per_type_inside_one_layer():
    """stele parenchyma and protoxylem share the stele layer, yet each takes its
    own height; metaxylem given a height becomes vessel elements in series."""
    root = _small_root()
    heights = {"stele": 0.05, "protoxylem": 0.2, "metaxylem": 0.3}
    result = root.generate_cells_3d(axial_heights=heights, segment_length=1.0, seed=0)

    assert result.z_max - result.z_min == pytest.approx(1.0)
    for t, h in heights.items():
        assert result.height_sources[t] == "axial_heights"
        assert _inner_heights(result, t) == {h}, t
    meta = [rows for rows in _rows_by_cell(result).values() if rows[0]["type"] == "metaxylem"]
    assert meta and all(len(rows) >= 3 for rows in meta), "metaxylem must be elements in series"
    # A vessel type without a height stays one full-span tube.
    assert result.height_sources["phloem"] == "full span"
    assert all(len(rows) == 1 for rows in _rows_by_cell(result).values()
               if rows[0]["type"] == "phloem")


def test_axial_heights_win_over_the_layer_height():
    data = OrganInputData.for_root()
    for layer in ("epidermis", "exodermis", "cortex", "endodermis", "pericycle"):
        data.set_values(layer, cell_diameter=0.06, cell_width=0.06)
    data.set_values("stele", cell_diameter=0.03, cell_diameter_center=0.03)
    data.set_values("cortex", axial_height=0.25)

    layer_only = RootAnatomy(data, seed=0).generate_cells_3d(segment_length=1.0, seed=0)
    assert layer_only.height_sources["cortex"] == "layer"
    assert _inner_heights(layer_only, "cortex") == {0.25}

    both = RootAnatomy(data, seed=0).generate_cells_3d(
        axial_heights={"cortex": 0.1}, segment_length=1.0, seed=0)
    assert both.height_sources["cortex"] == "axial_heights"
    assert _inner_heights(both, "cortex") == {0.1}


def test_missing_heights_are_logged(caplog):
    root = _small_root()
    with caplog.at_level("INFO", logger="openalea.granap.generate_cell_3d"):
        result = root.generate_cells_3d(axial_heights={"cortex": 0.1}, n_axial_repeats=2.0, seed=0)
    assert result.height_sources["epidermis"] == "diameter ratio"
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "epidermis" in msg and "cortex" not in msg.split("using")[0]


def test_bad_heights_are_rejected():
    root = _small_root()
    with pytest.raises(ValueError, match="axial_heights"):
        root.generate_cells_3d(axial_heights={"cortex": 0.0}, seed=0)
    with pytest.raises(ValueError, match="segment_length"):
        root.generate_cells_3d(segment_length=-1.0, seed=0)
