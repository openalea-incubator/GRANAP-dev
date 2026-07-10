"""Tests for dicot star-shaped xylem with Apollonian packing.

Visual scenario gallery lives in ``example/root_dicot_features_gallery.py``.
"""

import os
import sys

sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def make_dicot_root(cambium_kwargs: dict = None, **xylem_kwargs) -> RootAnatomy:
    """Construct a dicot RootAnatomy with custom xylem and optional cambium parameters."""
    data = OrganInputData.for_dicot_root()
    for field, value in xylem_kwargs.items():
        data.set_value("xylem", field, value)
    if cambium_kwargs:
        for field, value in cambium_kwargs.items():
            data.set_value("cambium", field, value)
    return RootAnatomy(data, seed=SEED)


BASE_KWARGS = {
    "arc_peak_side":    0.04,
    "arc_valley_side": 0.06,
}

BASE_CAMBIUM = {
    "cell_diameter":    0.006,
    "cell_width":       0.01,
    "radius_valley_side":   0.11,
    "visible_distance": 0.27,
    "arc_peak_side":          0.05,
    "arc_valley_side":       0.07,
}

scenarios = [
    {"label": "Diarch (2 peaks)",  "kwargs": {**BASE_KWARGS, "n_vascular_peak": 2}, "cambium": BASE_CAMBIUM},
    {"label": "Tétrarch (4 peaks)", "kwargs": {**BASE_KWARGS, "n_vascular_peak": 4}, "cambium": BASE_CAMBIUM},
    {"label": "Heptarch (7 peaks)", "kwargs": {**BASE_KWARGS, "n_vascular_peak": 7}, "cambium": BASE_CAMBIUM},
    {"label": "Narrow peaks",
     "kwargs": {**BASE_KWARGS, "arc_valley_side": 0.05, "arc_peak_side": 0.05,
                "vessel_diameter": 0.07, "vessel_diameter_min": 0.04, "radius_valley_side": 0.04},
     "cambium": BASE_CAMBIUM},
    {"label": "Wide star",
     "kwargs": {**BASE_KWARGS, "radius_valley_side": 0.15, "radius_peak_side": 0.20},
     "cambium": {**BASE_CAMBIUM, "radius_valley_side": 0.19, "visible_distance": 0.40}},
    {"label": "Circle",
     "kwargs": {**BASE_KWARGS, "radius_valley_side": 0.15, "radius_peak_side": 0.15},
     "cambium": {**BASE_CAMBIUM, "radius_valley_side": 0.17, "visible_distance": 0.17}},
]


def test_dicot_star_xylem_size_classification():
    """Every scenario yields xylem cells, and no xylem cell is smaller than
    vessel_diameter_min (small ones must be reclassified as 'stele')."""
    for s in scenarios:
        root = make_dicot_root(cambium_kwargs=s.get("cambium"), **s["kwargs"])
        root.generate_cells()

        xylem = [c for c in root.all_cells.cells if c.type == "xylem"]
        assert len(xylem) > 0, f"Expected at least one xylem cell in scenario '{s['label']}'"

        # Recover vessel_diameter_min for this scenario's parameterisation.
        data_defaults = OrganInputData.for_dicot_root()
        for f, v in s["kwargs"].items():
            data_defaults.set_value("xylem", f, v)
        dmin = data_defaults.get("xylem").vessel_diameter_min

        small_xylem = [c for c in xylem if c.diameter < dmin]
        assert len(small_xylem) == 0, (
            f"[{s['label']}] Found {len(small_xylem)} xylem cell(s) with "
            f"diameter < vessel_diameter_min ({dmin})"
        )
