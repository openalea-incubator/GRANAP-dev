"""Minimal coverage for the newer dicot/monocot stem features:

* mixed-kind ``bundle_pattern`` placement (equal spacing and ``grouped`` clusters),
* the bicollateral inner cambium (a cambium band on *both* faces of the xylem),
* the monocot ``spaced`` (best-candidate) bundle placement,
* the ``CellManager.remove_cells`` identity fix (the epidermis-gap bug),
* the unified absolute peak/valley ``star`` contour.

One assertion per behaviour — enough to catch a regression, not exhaustive.
"""

import os
import sys

import numpy as np
from shapely.ops import unary_union

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.cell_manager import CellManager
from openalea.granap.cell_class import Cell
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.input_data import (
    OrganInputData, VascularBundleParams, BundlePatternParams,
)

SEED = 0


def _bundles(stem):
    return stem.vascular_tissue_polygons.get("bundle", [])


def _pattern_stem(spacing, ring, repeats, sequence):
    """A dicot eustele with two bundle kinds arranged by a bundle_pattern."""
    d = OrganInputData.for_dicot_stem()
    d.params = [p for p in d.params if getattr(p, "name", None) != "vascular_bundle"]
    common = {"ring_shape": ring}
    if ring == "star":
        common.update(n_peaks=repeats, radius_peak_side=0.42, radius_valley_side=0.30,
                      arc_peak_side=0.12, arc_valley_side=0.10)
    d.params += [
        VascularBundleParams(kind="big",   width=0.13, height=0.17, **common),
        VascularBundleParams(kind="small", width=0.07, height=0.11, **common),
        BundlePatternParams(sequence=list(sequence), repeats=repeats,
                            spacing=spacing, align_to_arms=True),
    ]
    return d


# -- mixed-kind bundle pattern ----------------------------------------------

def test_bundle_pattern_places_every_bundle():
    """Equal (distance) spacing places the whole tiled sequence: 2 kinds x 3 = 6."""
    stem = StemAnatomy(_pattern_stem("distance", "circle", 3, ["big", "small"]), seed=SEED)
    stem.generate_cells()
    assert len(_bundles(stem)) == 6


def test_bundle_pattern_grouped_forms_clusters():
    """'grouped' spacing on a 3-arm star clusters the sequence into 3 lobes with
    empty valleys between — so the bundle angles show exactly 3 wide gaps."""
    stem = StemAnatomy(_pattern_stem("grouped", "star", 3, ["big", "small", "small"]), seed=SEED)
    stem.generate_cells()
    bundles = _bundles(stem)
    assert len(bundles) == 9
    ang = np.sort([np.arctan2(b.centroid.y, b.centroid.x) for b in bundles])
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    assert int((gaps > gaps.mean() * 2).sum()) == 3, "grouped must leave one valley per lobe"


# -- bicollateral inner + outer cambium -------------------------------------

def test_bicollateral_has_inner_and_outer_cambium():
    """A bicollateral eustele lays a cambium band on both radial sides of the xylem
    (inner-phloem side + outer side), not just the outer contour."""
    d = OrganInputData.for_dicot_stem()
    d.set_values("vascular_bundle", bundle_type="bicollateral", inner_phloem_fraction=0.18,
                 inner_cambium=True, n_bundles=6)
    stem = StemAnatomy(d, seed=SEED)
    stem.generate_cells()
    xr = [np.hypot(c.x, c.y) for c in stem.all_cells.cells if c.type == "xylem"]
    cr = [np.hypot(c.x, c.y) for c in stem.all_cells.cells if c.type == "cambium"]
    xmean = float(np.mean(xr))
    assert any(r < xmean for r in cr) and any(r > xmean for r in cr), \
        "cambium must sit both inner and outer of the xylem"


# -- monocot 'spaced' placement ---------------------------------------------

def test_monocot_spaced_places_non_overlapping_bundles():
    d = OrganInputData.for_monocot_stem()
    d.set_values("vascular_bundle", placement="spaced", n_bundles=6)
    stem = StemAnatomy(d, seed=SEED)
    stem.generate_cells()
    bundles = _bundles(stem)
    assert len(bundles) >= 4, "spaced should place most of the requested bundles"
    # non-overlapping: the union area equals the sum of the parts (within rounding).
    total = sum(b.area for b in bundles)
    assert unary_union(bundles).area > 0.99 * total, "spaced bundles must not overlap"


def test_bundle_band_can_be_placed_past_the_pith():
    """An 'even' band whose ``radius`` exceeds the pith radius carries its bundles out
    of the ground tissue into the cortex/rind (bundles may be placed in any tissue).

    The pith radius is measured from a baseline build (the pith's parenchyma edge);
    a ring placed a clear step beyond it must land every bundle outside the pith."""
    base = StemAnatomy(OrganInputData.for_monocot_stem(), seed=SEED)
    base.generate_cells()
    pith_r = max(np.hypot(c.x, c.y) for c in base.all_cells.cells if c.type == "parenchyma")

    ring = pith_r + 0.12                       # a clear step into the cortex/rind
    d = OrganInputData.for_monocot_stem()
    d.params = [p for p in d.params if getattr(p, "name", None) != "vascular_bundle"]
    d.params.append(VascularBundleParams(
        placement="even", n_bundles=6, radius=ring, width=0.08, height=0.06))
    stem = StemAnatomy(d, seed=SEED)
    stem.generate_cells()
    bundles = _bundles(stem)
    assert bundles, "the peripheral band must place bundles"
    assert all(np.hypot(b.centroid.x, b.centroid.y) > pith_r for b in bundles), \
        "an 'even' ring with radius past the pith must place bundles outside the pith"


# -- CellManager.remove_cells identity (epidermis-gap fix) ------------------

def test_remove_cells_matches_identity_not_id_cell():
    cm = CellManager()
    keep = Cell(0.0, 0.0, 0.01, type="epidermis", id_cell=5, id_group=1)
    drop = Cell(1.0, 1.0, 0.01, type="air space", id_cell=5, id_group=2)  # same id_cell!
    cm.cells = [keep, drop]
    cm.remove_cells([drop])
    assert cm.cells == [keep], "remove_cells must delete by object identity, not shared id_cell"


# -- unified absolute peak/valley star contour ------------------------------

def test_contour_polygon_star_uses_absolute_radii():
    p = GeometryProcessor.contour_polygon(
        "star", n_branches=4, radius_peak_side=0.5, radius_valley_side=0.3,
        arc_peak_side=0.1, arc_valley_side=0.1)
    assert p.is_valid and not p.is_empty
    r = np.hypot(*np.asarray(p.exterior.coords).T)
    assert abs(r.max() - 0.5) < 0.02, "peak radius honoured (absolute mm)"
    assert abs(r.min() - 0.3) < 0.05, "valley radius honoured (absolute mm)"
