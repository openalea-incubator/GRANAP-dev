"""CellSet ingestion + the GRANAP-stele graft (``CellSetOrgan``).

The fixture is a real CellSet digitization of an Arabidopsis root cross-section
(``test/inputs/partial_CellSet_brasicaceae.xml``, coordinates in micrometres) whose
whole stele is a single **annular** cell -- one outer ring with the two visible
vessels punched out as holes.  MECHA cannot consume a cell with holes, which is
what ``CellSetOrgan`` exists to fix.

The load-bearing test here is :func:`test_no_interior_border_walls`.  Everything
else can look right while the graft interface is silently disconnected.
"""

import math
import os

import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from openalea.granap.anatomy_writer import AnatomyWriter
from openalea.granap.cellset_organ import CellSetOrgan
from openalea.granap.cellset_reader import assemble_rings, read_cellset
from openalea.granap.special_tissues import weld_points_onto_rings

HERE = os.path.dirname(os.path.abspath(__file__))
CELLSET = os.path.join(HERE, "inputs", "partial_CellSet_brasicaceae.xml")
SEED = 0

# Measured from the file itself (areas in um^2).
SOURCE_CENSUS = {
    "epidermis": (21, 5163.7),
    "cortex": (8, 3422.1),
    "endodermis": (8, 1105.1),
    "stele": (1, 732.0),
    "xylem": (2, 59.3),
}
STELE_RING_AREA = 791.3   # the annular cell's outer ring, holes filled
VESSEL_AREAS = (32.6, 26.7)


@pytest.fixture(scope="module")
def section():
    return read_cellset(CELLSET)


@pytest.fixture(scope="module")
def organ():
    o = CellSetOrgan(CELLSET, seed=SEED)
    o.generate_cells()
    o.export_to_adjencymatrix()
    return o


# ---------------------------------------------------------------- reader


def test_assemble_rings_simple_square():
    """Four walls in arbitrary order and direction close into one ring."""
    rings = assemble_rings([
        [(0, 0), (1, 0)],
        [(1, 1), (1, 0)],       # reversed
        [(1, 1), (0, 1)],
        [(0, 0), (0, 1)],       # reversed
    ])
    assert len(rings) == 1
    assert rings[0][0] == rings[0][-1]
    assert Polygon(rings[0]).area == pytest.approx(1.0)


def test_assemble_rings_detects_annulus():
    """A cell whose walls close into two rings is annular, outer ring first."""
    outer = [[(0, 0), (4, 0)], [(4, 0), (4, 4)], [(4, 4), (0, 4)], [(0, 4), (0, 0)]]
    hole = [[(1, 1), (2, 1)], [(2, 1), (2, 2)], [(2, 2), (1, 2)], [(1, 2), (1, 1)]]
    rings = assemble_rings(outer + hole)
    assert len(rings) == 2
    assert Polygon(rings[0]).area == pytest.approx(16.0)
    assert Polygon(rings[1]).area == pytest.approx(1.0)


def test_reader_census_and_units(section):
    assert len(section.cells) == 40
    assert len(section.walls) == 115
    census = section.census()
    assert set(census) == set(SOURCE_CENSUS)
    for tag, (n, area_um2) in SOURCE_CENSUS.items():
        assert census[tag][0] == n
        # default scale=0.001 converts um -> mm, which is what MECHA's default
        # im_scale=1000 expects to read back.
        assert census[tag][1] * 1e6 == pytest.approx(area_um2, abs=0.1)
    xs = [p[0] for w in section.walls.values() for p in w]
    assert 0.09 < min(xs) < 0.11 and 0.21 < max(xs) < 0.23


def test_the_stele_is_annular(section):
    annular = section.annular_cells
    assert len(annular) == 1
    cell = annular[0]
    assert cell.tag == "stele"
    assert len(cell.rings) == 3          # outer + the two vessels
    assert cell.outer_ring_polygon.area * 1e6 == pytest.approx(STELE_RING_AREA, abs=0.1)
    holes = sorted(
        (Polygon(r).area * 1e6 for r in cell.rings[1:]), reverse=True
    )
    assert holes == pytest.approx(sorted(VESSEL_AREAS, reverse=True), abs=0.1)
    # the holes are exactly the two digitized xylem cells
    xylem = [c.polygon.area * 1e6 for c in section.cells if c.tag == "xylem"]
    assert sorted(xylem, reverse=True) == pytest.approx(holes, abs=0.05)


def test_reader_does_not_resample(section):
    """Outlines must stay exactly the wall points.

    ``RoiOrgan`` smooths every outline (a 200-point resample) and thereby
    destroys the vertex sharing that CellSet's shared-wall representation gives
    for free -- which is why its exported network has no cell-to-cell edges.
    """
    for cell in section.cells:
        own = {
            (round(p[0], 12), round(p[1], 12))
            for wid in cell.wall_ids
            for p in section.walls.get(wid, [])
        }
        for ring in cell.rings:
            for point in ring:
                assert (round(point[0], 12), round(point[1], 12)) in own


def test_source_cells_tile_the_section(section):
    """The digitized cells tile the section exactly -- no gaps, no overlaps.

    This is the baseline the graft has to preserve: any gap it introduces shows
    up as a one-cell wall (see test_no_interior_border_walls).
    """
    polys = [c.polygon for c in section.cells]
    union = unary_union(polys)
    assert union.geom_type == "Polygon"
    assert not union.interiors
    assert union.area == pytest.approx(sum(p.area for p in polys), rel=1e-12)


# ---------------------------------------------------------------- weld helper


def test_weld_inserts_only_mid_edge_points():
    from openalea.granap.cell_class import Cell

    cell = Cell(x=0.5, y=0.5, diameter=1.0, type="cortex", id_cell=0,
                polygon=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    before = cell.polygon.area
    n = weld_points_onto_rings(
        [cell],
        [
            (0.5, 0.0),   # mid-edge -> inserted
            (1.0, 0.0),   # existing vertex -> skipped
            (0.5, 0.5),   # interior -> skipped
            (2.0, 2.0),   # far away -> skipped
        ],
    )
    assert n == 1
    assert (0.5, 0.0) in list(cell.polygon.exterior.coords)
    # inserting a collinear vertex cannot change the geometry
    assert cell.polygon.area == pytest.approx(before, rel=1e-12)


# ---------------------------------------------------------------- the graft


def test_annulus_is_gone_and_region_is_tiled(organ):
    gdf = organ.generate_cells()
    assert not gdf.empty
    for geom in gdf["geometry"]:
        assert geom.geom_type == "Polygon"
        assert not geom.interiors
        assert geom.is_valid
    assert list(gdf.index) == list(range(len(gdf)))   # MECHA indexes arithmetically

    report = organ.graft_report
    assert len(report) == 1
    assert report[0]["tag"] == "stele"
    assert report[0]["n_inner_kept"] == 2
    assert report[0]["coverage"] == pytest.approx(1.0, abs=1e-6)
    assert report[0]["n_vertices_welded"] > 0

    # area is conserved: generated stele + the kept vessels == the annulus' ring
    generated = sum(
        c.polygon.area for c in organ.all_cells.cells
        if c.type in ("pericycle", "stele", "phloem") or (c.type == "xylem" and c.cgroup == 0)
    )
    vessels = sum(
        c.polygon.area for c in organ.all_cells.cells
        if c.type == "xylem" and c.cgroup == 13
    )
    assert (generated + vessels) * 1e6 == pytest.approx(STELE_RING_AREA, abs=0.5)


def test_real_tissues_are_untouched(organ, section):
    """Everything outside the regenerated region keeps its measured area."""
    for tag in ("epidermis", "cortex", "endodermis"):
        got = sum(c.polygon.area for c in organ.all_cells.cells if c.type == tag)
        assert got * 1e6 == pytest.approx(SOURCE_CENSUS[tag][1], abs=0.1)
    vessels = sorted(
        (c.polygon.area * 1e6 for c in organ.all_cells.cells
         if c.type == "xylem" and c.cgroup == 13),
        reverse=True,
    )
    assert vessels == pytest.approx(sorted(VESSEL_AREAS, reverse=True), abs=0.1)


def test_no_interior_border_walls(organ):
    """The graft interface must be *shared*, not merely adjacent.

    A wall with one flanking cell is worse than a missing connection: MECHA's
    ``NetworkBuilder.identify_border_walls_junctions`` files a single-reference
    wall whose owner is not epidermis under ``border_aerenchyma``, i.e. it
    becomes a gas-space boundary in the middle of the tissue.  So border walls
    must appear only on the section's real outer surface -- and the source file
    has exactly 22 single-reference walls, owned by 21 epidermis cells and one
    cortex cell.
    """
    rep = organ.topology_report()
    assert rep["interior_border_walls"] == {}
    assert rep["border_walls"] == {"epidermis": 21, "cortex": 1}
    assert rep["multi_cell_walls"] == 0


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_no_interior_border_walls_at_any_seed(seed):
    """No gas-space boundary mid-tissue, for *every* seed -- not just ``seed=0``.

    This used to hold only at 0: seeds 1-5 produced 22-49 interior border walls.
    Three separate causes, all of them silent:

    * ``absorb_residual`` was handed the crowd of zero-area slivers that any
      boolean ``difference`` returns (16-21 per seed, most exactly 0.0) because
      ``_graft`` never passed it a ``min_area``.  Merging one closes nothing but
      still inserts vertices into its host's ring.
    * ``_graft`` conformed only donor -> real, so the vertices those merges added
      along donor <-> donor walls were never shared, and each un-shared vertex
      splits one wall into three single-reference ones.
    * ``conform_boundaries`` spliced each cut point into just the *nearest* ring,
      leaving any other ring it lay on keying a longer wall.

    Seed 0 passed purely because it happened to need no absorbing at all, which
    is what kept all three invisible on the one seed anyone ran.

    ``multi_cell_walls`` is asserted only at seed 0 (see
    ``test_no_interior_border_walls``): seeds 2 and 4 still produce exactly one
    wall with three flanking cells, because ``carve_cells`` keeps only the
    largest part when a real vessel splits a donor cell in two, and the
    discarded part becomes a gap that ``absorb_residual`` hands to a neighbour.
    That is a real but far milder defect -- and, unlike the above, it is not
    silent: ``topology_report`` warns about it.
    """
    organ = CellSetOrgan(CELLSET, seed=seed)
    organ.generate_cells()
    organ.export_to_adjencymatrix()
    rep = organ.topology_report()
    assert rep["interior_border_walls"] == {}


def test_interface_is_symplastically_connected(organ):
    """Plasmodesmata must cross both halves of the graft boundary."""
    pairs = organ.topology_report()["plasmodesmata_pairs"]
    assert pairs.get(("endodermis", "pericycle"), 0) >= 8    # outward, one per endodermis cell
    assert pairs.get(("stele", "xylem"), 0) >= 4             # inward, around the real vessels
    assert pairs.get(("phloem", "stele"), 0) >= 1


def test_generated_stele_has_the_expected_tissues(organ):
    """Golden-ish census for the donor at seed 0.

    Expect to rebaseline this whenever the donor configuration in
    ``default_stele_input`` changes; review the render before doing so.
    """
    tags = {c.type for c in organ.all_cells.cells}
    assert {"pericycle", "stele", "phloem", "xylem"} <= tags
    assert "cambium" not in tags     # retagged to stele: CGROUP_MAP sends it to 12 = companion
    counts = {t: sum(1 for c in organ.all_cells.cells if c.type == t) for t in tags}
    assert counts["pericycle"] == 26
    assert counts["stele"] == 34
    assert counts["phloem"] == 2
    # The xylem is digitized data only: the donor's 7 vessels are retagged away,
    # so GRANAP never invents a vessel next to the measured ones.
    assert counts["xylem"] == 2
    assert organ.graft_report[0]["n_donor_xylem_removed"] == 7
    assert all(
        c.cgroup == 13 for c in organ.all_cells.cells if c.type == "xylem"
    )


def test_xylem_plate_follows_the_measured_vessel_axis(organ):
    """The donor's diarch plate is rotated onto the real vessels' axis."""
    vessels = [
        c.polygon.centroid for c in organ.all_cells.cells
        if c.type == "xylem" and c.cgroup == 13
    ]
    measured = math.degrees(
        math.atan2(vessels[1].y - vessels[0].y, vessels[1].x - vessels[0].x)
    )
    angle = organ.graft_report[0]["angle"]
    assert math.cos(math.radians(2 * (angle - measured))) == pytest.approx(1.0, abs=1e-6)


def test_a_broken_interface_is_not_silent(organ):
    """A single-reference interior wall must be *reported*, never swallowed.

    Asserted on injected data, not on a deliberately-misconfigured build, and
    that is the whole point.  This test used to pick an option set that happens
    to break (``recenter=False``, then ``overshoot=0``) and assert it warns —
    which couples the guarantee to how badly that option set breaks *on this
    platform*.  It does not survive contact with one: ``overshoot=0`` leaves
    only 3 interior border walls out of thousands on x86, and on macOS/arm64
    those 3 close up and nothing warns at all.  Chasing that with a third
    option set would just re-arm the same trap.

    So test the property directly: hand ``topology_report`` one wall with a
    single flanking cell whose tag is *not* on the outer surface, and it must
    warn and report it.  The default graft is clean, so the same organ also
    pins the converse — no false alarm on a sound network.
    """
    rep = organ.topology_report()               # clean: must not cry wolf
    assert rep["interior_border_walls"] == {}

    # A wall id that cannot collide with a real one, owned by a stele cell --
    # interior by definition, so it must not be excused as outer surface.
    idx = next(i for i, c in enumerate(organ.all_cells.cells) if c.type == "stele")
    node = idx + organ.n_walls + organ.n_junctions
    fake = max(organ._wall_to_cells) + 1
    organ._wall_to_cells[fake] = [node]
    try:
        with pytest.warns(UserWarning, match="not fully shared"):
            rep = organ.topology_report()
    finally:
        del organ._wall_to_cells[fake]          # leave the module fixture clean
    assert rep["interior_border_walls"] == {"stele": 1}


def test_overshoot_zero_leaves_real_rim_gaps():
    """``overshoot=0`` is a geometric failure, not a welding one.

    Without growing the donor past the region rim, the tessellation's outer ring
    is straightened into chords that cut the corners of the wiggly real outline.
    No amount of vertex welding can close that, which is why the donor is grown
    and clipped back.  Coverage is the robust signal (~0.994 on x86); the count
    of resulting border walls is not, so it is deliberately not asserted here.
    """
    organ = CellSetOrgan(CELLSET, seed=SEED, overshoot=0.0)
    organ.generate_cells()
    assert organ.graft_report[0]["coverage"] < 0.999


# ---------------------------------------------------------------- export


def test_write_to_xml_roundtrip(organ, tmp_path):
    """The file handed to MECHA must itself be annulus-free and fully grouped."""
    path = tmp_path / "hybrid.xml"
    AnatomyWriter(organ).write_to_xml(str(path))

    back = read_cellset(str(path), scale=1.0)   # already in mm
    assert len(back.cells) == len(organ.all_cells.cells)
    assert not back.annular_cells               # the whole point of the exercise

    from openalea.granap.anatomy_writer import CGROUP_MAP

    valid = set(CGROUP_MAP.values())
    for cell in back.cells:
        # group 0 is not a soft failure: MECHA does a bare CGROUP_TO_TYPE[cgroup]
        assert cell.group != 0
        assert cell.group in valid

    # the digitized cells keep their own CellSet group integers
    groups = sorted({c.group for c in back.cells})
    assert 2 in groups and 3 in groups and 4 in groups and 13 in groups
    assert 16 in groups                          # pericycle, from the graft
