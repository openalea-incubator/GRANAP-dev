"""Shape-first tissue vocabulary + recipe tests.

Model under test:
  * A **tissue** is a tagged anatomical *region* (a shapely shape).  It is
    manipulated as pure geometry — rotate / translate / smooth / boolean-combine
    — entirely before any cell exists.
  * **Filling** a region with cells (``fill_*``) is a separate, terminal step.
  * A **cell** is the filled polygon; the only verb on it is ``retag``.
  * A monocot/dicot stele is assembled from an inspectable ``TissueRecipe``.
"""

import os
import sys

import numpy as np
from shapely.geometry import Point, Polygon

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData
from openalea.granap.cell_manager import CellManager
from openalea.granap.cell_class import Cell
from openalea.granap.tissue_builder import Tissue, TissueRecipe, retag_tissue

SEED = 0


def make_star_root(**xylem_overrides) -> RootAnatomy:
    data = OrganInputData.for_root()
    data.set_value("xylem", "xylem_shape", "star")
    for field, value in xylem_overrides.items():
        data.set_value("xylem", field, value)
    return RootAnatomy(data, seed=SEED)


def cell_type_counts(root) -> dict:
    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Tissue == a tagged region, transformed as pure geometry
# ---------------------------------------------------------------------------

def test_tissue_translate_moves_region():
    t = Tissue("phloem", Point(0.0, 0.0).buffer(1.0))
    c0 = t.shape.centroid
    out = t.translate(2.0, -1.0)
    assert out is t                       # chainable
    assert np.isclose(t.shape.centroid.x - c0.x, 2.0)
    assert np.isclose(t.shape.centroid.y - c0.y, -1.0)
    assert t.tag == "phloem"


def test_tissue_rotate_about_origin_preserves_distance():
    t = Tissue("xylem", Point(2.0, 0.0).buffer(0.3))
    r0 = np.hypot(t.shape.centroid.x, t.shape.centroid.y)
    t.rotate(90.0, origin=(0.0, 0.0))
    r1 = np.hypot(t.shape.centroid.x, t.shape.centroid.y)
    assert np.isclose(r0, r1)
    assert np.isclose(t.shape.centroid.x, 0.0, atol=1e-9)
    assert np.isclose(t.shape.centroid.y, 2.0, atol=1e-9)


def test_tissue_smooth_changes_boundary_keeps_centre():
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    t = Tissue("cortex", square)
    cx0, cy0 = t.shape.centroid.x, t.shape.centroid.y
    t.smooth(0.6)
    assert not t.is_empty
    assert not t.shape.equals(square)                 # corners rounded
    assert np.isclose(t.shape.centroid.x, cx0, atol=0.1)
    assert np.isclose(t.shape.centroid.y, cy0, atol=0.1)


def test_tissue_region_algebra():
    # difference -> annulus
    annulus = Tissue("stele", Point(0, 0).buffer(2.0)).difference(Point(0, 0).buffer(1.0))
    assert np.isclose(annulus.area, np.pi * 4 - np.pi * 1, rtol=2e-2)

    # operands may themselves be Tissues
    annulus2 = (
        Tissue("stele", Point(0, 0).buffer(2.0))
        .difference(Tissue("xylem", Point(0, 0).buffer(1.0)))
    )
    assert np.isclose(annulus2.area, annulus.area, rtol=1e-9)

    # intersection clips
    clipped = Tissue("a", Point(0, 0).buffer(2.0)).intersection(Point(1, 0).buffer(2.0))
    assert 0 < clipped.area < np.pi * 4


# ---------------------------------------------------------------------------
# Cells are the terminal product; the only verb on them is retag
# ---------------------------------------------------------------------------

def test_retag_is_terminal_cell_verb():
    cm = CellManager()
    for i in range(3):
        cm.add_cell(Cell(x=float(i), y=0.0, diameter=0.1, type="phloem"))
    cm.add_cell(Cell(x=9.0, y=9.0, diameter=0.1, type="xylem"))

    n = retag_tissue(cm, "phloem", "sieve")
    assert n == 3
    assert len(cm.get_cells_by_type("phloem")) == 0
    assert len(cm.get_cells_by_type("sieve")) == 3
    assert len(cm.get_cells_by_type("xylem")) == 1


# ---------------------------------------------------------------------------
# Recipe ergonomics: a step declares a region + fill strategy
# ---------------------------------------------------------------------------

def test_recipe_fill_packs_a_region():
    cm = CellManager()
    rng = np.random.default_rng(SEED)
    region = Tissue("phloem", Point(0.0, 0.0).buffer(0.3))

    recorded = []
    recipe = (
        TissueRecipe().bind(cm, rng)
        .fill("phloem", region, strategy="packing",
              record=lambda t, res: recorded.append((t.tag, len(res))),
              proportion=1.0, diameter_max=0.1, diameter_min=0.1)
    )
    assert recipe.describe() == [("phloem", ("phloem",))]
    recipe.build()
    assert cm.cells                                  # cells were placed
    assert all(c.type == "phloem" for c in cm.cells)
    assert recorded and recorded[0][0] == "phloem"   # record hook ran


def test_recipe_fill_each_and_lazy_target():
    cm = CellManager()
    rng = np.random.default_rng(SEED)
    regions = [Tissue("xylem", Point(x, 0.0).buffer(0.2)) for x in (-1.0, 1.0)]

    # bind a *callable* so the fill resolves the current manager at build time
    holder = {"cm": cm}
    recipe = (
        TissueRecipe().bind(lambda: holder["cm"], rng)
        .fill_each("xylem pair", regions, strategy="packing",
                   proportion=1.0, diameter_max=0.08, diameter_min=0.08)
    )
    assert recipe.describe() == [("xylem pair", ("xylem",))]
    recipe.build()
    assert holder["cm"].cells
    assert all(c.type == "xylem" for c in holder["cm"].cells)


def test_recipe_empty_region_is_safe():
    cm = CellManager()
    recipe = (
        TissueRecipe().bind(cm, np.random.default_rng(SEED))
        .fill("nothing", Tissue("x", Polygon()), strategy="packing",
              proportion=1.0, diameter_max=0.1)
    )
    recipe.build()                                   # must not raise
    assert cm.cells == []


def test_recipe_special_and_cleanup_order():
    log = []
    recipe = (
        TissueRecipe()
        .special("place", lambda: log.append("place"), produces=("a",))
        .cleanup("clean", lambda: log.append("clean"))
    )
    assert recipe.describe() == [("place", ("a",)), ("clean", ())]
    recipe.build()
    assert log == ["place", "clean"]


# ---------------------------------------------------------------------------
# The phloem regions really are shape-first Tissues, built by region algebra
# ---------------------------------------------------------------------------

def test_phloem_valley_zones_are_tissue_regions():
    root = make_star_root()
    layers = root.generate_layer_polygons()
    stele = next(p["polygon"] for p in layers if p["name"] == "stele")

    root.all_cells = CellManager()
    root.vascular_cells = CellManager()
    root.vascular_polygons = []
    root.vascular_tissue_polygons = {}
    root.fit_star_shapped_xylem(stele)        # sets root.xylem_star

    tissues, adjustment = root._phloem_valley_zones(stele, type="monocot")
    assert tissues
    assert all(isinstance(t, Tissue) for t in tissues)
    assert all(t.tag == "phloem" for t in tissues)
    assert all(not t.is_empty for t in tissues)
    # carved out of the xylem star: no phloem region overlaps it
    for t in tissues:
        assert t.shape.intersection(root.xylem_star).area < 1e-9


# ---------------------------------------------------------------------------
# Refactor must not change the produced anatomy
# ---------------------------------------------------------------------------

def test_recipe_is_inspectable():
    root = make_star_root()
    layers = root.generate_layer_polygons()
    stele = next(p["polygon"] for p in layers if p["name"] == "stele")
    recipe = root._vascular_recipe(stele)
    names = [name for name, _ in recipe.describe()]
    assert names == ["xylem star", "clear stele under xylem", "phloem in valleys"]
    assert dict(recipe.describe())["phloem in valleys"] == ("phloem",)


def test_monocot_baseline_unchanged():
    root = make_star_root()
    root.generate_cells()
    counts = cell_type_counts(root)
    assert counts["xylem"] == 25
    assert counts["stele"] == 217
    assert counts["phloem"] == 5


if __name__ == "__main__":
    test_tissue_translate_moves_region()
    test_tissue_rotate_about_origin_preserves_distance()
    test_tissue_smooth_changes_boundary_keeps_centre()
    test_tissue_region_algebra()
    test_retag_is_terminal_cell_verb()
    test_recipe_fill_packs_a_region()
    test_recipe_fill_each_and_lazy_target()
    test_recipe_empty_region_is_safe()
    test_recipe_special_and_cleanup_order()
    test_phloem_valley_zones_are_tissue_regions()
    test_recipe_is_inspectable()
    test_monocot_baseline_unchanged()
    print("ALL SHAPE-FIRST TISSUE TESTS PASSED")
