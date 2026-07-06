"""Geometry tests for secondary phloem generation.

Visual gallery lives in ``example/secondary_phloem_gallery.py``.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath('..'))

from shapely.geometry import Point
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def make_root(**phloem_overrides) -> RootAnatomy:
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    data.set_value("stele", "thickness", 1.2)
    for field, value in phloem_overrides.items():
        data.set_value("secondary_phloem", field, value)
    return RootAnatomy(data, seed=SEED)


def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in list(root.all_cells.cells) + list(root.vascular_cells.cells):
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


def _zone_geometry_ok(root: RootAnatomy) -> tuple[bool, bool]:
    """Return (phloem_at_valley, phloem_at_cambium_arm)."""
    phloem_polys = root.vascular_tissue_polygons.get("secondary_phloem", [])
    if not phloem_polys:
        return False, False
    zone = phloem_polys[0]
    sc = root.secondary_cambium_params
    sp = root.secondary_phloem_params
    n_peaks = root.vascular_params["n_vascular_peak"]

    # Band starts at the cambium valley radius (~inner_distance) and extends
    # outward by `height`; sample a point just inside it at the valley angle.
    r_mid         = sc["inner_distance"] + sp["height"] * 0.4
    valley_angle  = 2 * np.pi * 0.5 / n_peaks   # vessel zone centre (phloem arm)
    cam_arm_angle = 0.0                           # cambium arm centre (parenchyma ray)

    in_valley = zone.contains(Point(r_mid * np.cos(valley_angle), r_mid * np.sin(valley_angle)))
    in_arm    = zone.contains(Point(r_mid * np.cos(cam_arm_angle), r_mid * np.sin(cam_arm_angle)))
    return in_valley, in_arm


def test_secondary_phloem_zone_placement():
    root = make_root()
    root.generate_cells()

    phloem_polys = root.vascular_tissue_polygons.get("secondary_phloem", [])
    assert phloem_polys, "No secondary_phloem polygon registered"

    # Zone must be at the vessel-zone angles (behind secondary xylem),
    # not at the cambium-arm angles (parenchyma ray positions).
    in_valley, in_arm = _zone_geometry_ok(root)
    assert in_valley, "Phloem zone not found at vessel-zone (valley) angle"
    assert not in_arm, "Phloem zone found at cambium-arm (parenchyma ray) angle — wrong position"

    # Sieve elements are always present.
    counts = cell_type_counts(root)
    assert counts.get("phloem", 0) > 0, "No phloem (sieve) cells generated"
