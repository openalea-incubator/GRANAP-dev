"""Tests for monocot arch-mode xylem (metaxylem ring + protoxylem) + pith, and
star-mode xylem (star-shaped vessel region + phloem in the valleys).

Visual scenario gallery lives in ``example/monocot_iris.py`` (arch) and
``example/monocot_xylem_gallery.py`` (all modes).
"""

import os
import sys

from shapely.geometry import Point

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def make_arch_root(**xylem_overrides) -> RootAnatomy:
    data = OrganInputData.for_root()
    data.set_value("xylem", "xylem_shape", "arch")
    for field, value in xylem_overrides.items():
        data.set_value("xylem", field, value)
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()
    return root


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


def test_arch_mode_no_pith():
    """Arch mode without pith: metaxylem vessels exist."""
    root = make_arch_root()
    counts = cell_type_counts(root)
    assert "metaxylem" in counts, "Expected metaxylem cells in arch mode"
    assert counts["metaxylem"] > 0, "Expected at least one metaxylem cell"


def test_arch_mode_with_pith():
    """Arch mode with a pith: no vessels inside the pith circle, but stele
    (pith parenchyma) cells are present there."""
    pith_r = 0.05
    root = make_arch_root(pith_radius=pith_r)

    pith_circle = Point(0.0, 0.0).buffer(pith_r)

    vessels_in_pith = [
        c for c in root.all_cells.cells
        if c.type in ("metaxylem", "protoxylem") and pith_circle.contains(Point(c.x, c.y))
    ]
    assert len(vessels_in_pith) == 0, (
        f"Found {len(vessels_in_pith)} vessels inside the pith circle — expected 0"
    )

    stele_in_pith = [
        c for c in root.all_cells.cells
        if c.type == "stele" and pith_circle.contains(Point(c.x, c.y))
    ]
    assert len(stele_in_pith) > 0, "Expected stele (pith) cells inside the pith circle"


def test_arch_exact_metaxylem_count():
    """n_metaxylem places exactly that many metaxylem vessels (Voronoi groups)."""
    root = make_arch_root(n_vascular_peak=19, n_metaxylem=15, vessel_diameter=0.05)
    groups = {c.id_group for c in root.all_cells.cells if c.type == "metaxylem"}
    assert len(groups) == 15, f"Expected 15 metaxylem, got {len(groups)}"


def test_arch_vs_default_both_produce_cells():
    """Both modes produce a reasonable number of cells."""
    root_default = RootAnatomy(OrganInputData.for_root())
    root_default.generate_cells()
    counts_default = cell_type_counts(root_default)

    counts_arch = cell_type_counts(make_arch_root())

    assert sum(counts_default.values()) > 10, "Default mode produced too few cells"
    assert sum(counts_arch.values()) > 10, "Arch mode produced too few cells"


def test_star_mode_produces_xylem_and_phloem():
    """Star mode packs xylem vessels into the star and phloem into the valleys."""
    root = make_star_root()
    counts = cell_type_counts(root)
    assert counts.get("xylem", 0) > 0, "Expected star xylem vessels"
    assert counts.get("phloem", 0) > 0, "Expected phloem strands in the valleys"


def test_star_mode_with_pith():
    """Star mode with a pith: no xylem vessels inside the pith circle, but stele
    (pith parenchyma) cells are present there."""
    pith_r = 0.04
    root = make_star_root(pith_radius=pith_r, radius_valley_side=0.05)

    pith_circle = Point(0.0, 0.0).buffer(pith_r)

    xylem_in_pith = [
        c for c in root.all_cells.cells
        if c.type == "xylem" and pith_circle.contains(Point(c.x, c.y))
    ]
    assert len(xylem_in_pith) == 0, (
        f"Found {len(xylem_in_pith)} xylem vessels inside the pith circle — expected 0"
    )

    stele_in_pith = [
        c for c in root.all_cells.cells
        if c.type == "stele" and pith_circle.contains(Point(c.x, c.y))
    ]
    assert len(stele_in_pith) > 0, "Expected stele (pith) cells inside the pith circle"


def test_star_phloem_sits_between_arms():
    """Phloem strands fall in the valleys (between arms), not on the arm axes."""
    n_peaks = 5
    root = make_star_root(n_vascular_peak=n_peaks)
    phloem = [c for c in root.all_cells.cells if c.type == "phloem"]
    assert phloem, "Expected phloem cells"

    # Arms point along 2*pi*k/n; valleys at the half-offset. Each phloem cell
    # should be nearer a valley angle than an arm angle.
    import numpy as np
    for c in phloem:
        theta = np.arctan2(c.y, c.x) % (2 * np.pi)
        # angular distance to the closest arm axis
        arm_gap = min(abs((theta - 2 * np.pi * k / n_peaks + np.pi) % (2 * np.pi) - np.pi)
                      for k in range(n_peaks))
        half = np.pi / n_peaks
        assert arm_gap > half * 0.5, (
            f"Phloem at theta={theta:.2f} too close to an arm axis (gap={arm_gap:.2f})"
        )
