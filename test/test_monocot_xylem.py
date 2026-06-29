"""Tests for monocot star-shaped xylem + pith feature.

Visual scenario gallery lives in ``example/monocot_xylem_gallery.py``.
"""

import os
import sys

from shapely.geometry import Point

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def make_star_root(**xylem_overrides) -> RootAnatomy:
    data = OrganInputData.for_root()
    data.set_value("xylem", "xylem_shape", "star")
    for field, value in xylem_overrides.items():
        data.set_value("xylem", field, value)
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()
    return root


def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


def test_star_mode_no_pith():
    """Star mode without pith: xylem vessels exist."""
    root = make_star_root()
    counts = cell_type_counts(root)
    assert "xylem" in counts, "Expected xylem cells in star mode"
    assert counts["xylem"] > 0, "Expected at least one xylem cell"


def test_star_mode_with_pith():
    """Star mode with pith_radius=0.05: no xylem cells inside the pith circle,
    and stele/pith cells are present inside it."""
    pith_r = 0.05
    root = make_star_root(pith_radius=pith_r)
    counts = cell_type_counts(root)

    pith_circle = Point(0.0, 0.0).buffer(pith_r)

    # No xylem vessels should be placed inside the pith circle
    xylem_in_pith = [
        c for c in root.all_cells.cells
        if c.type == "xylem" and pith_circle.contains(Point(c.x, c.y))
    ]
    assert len(xylem_in_pith) == 0, (
        f"Found {len(xylem_in_pith)} xylem cells inside the pith circle — expected 0"
    )

    # Stele/pith cells should exist inside the pith circle
    stele_in_pith = [
        c for c in root.all_cells.cells
        if c.type == "stele" and pith_circle.contains(Point(c.x, c.y))
    ]
    assert len(stele_in_pith) > 0, "Expected stele (pith) cells inside the pith circle"

    assert "xylem" in counts, "Expected xylem cells outside the pith"


def test_star_vs_default_both_produce_cells():
    """Both modes produce a reasonable number of cells."""
    root_default = RootAnatomy(OrganInputData.for_root())
    root_default.generate_cells()
    counts_default = cell_type_counts(root_default)

    counts_star = cell_type_counts(make_star_root())

    assert sum(counts_default.values()) > 10, "Default mode produced too few cells"
    assert sum(counts_star.values()) > 10, "Star mode produced too few cells"


if __name__ == "__main__":
    test_star_mode_no_pith()
    test_star_mode_with_pith()
    test_star_vs_default_both_produce_cells()
    print("ALL MONOCOT XYLEM TESTS PASSED")
