"""Tests for the continuous (non-fascicular) dicot stem vascular cylinder.

Unlike the eustele (discrete bundles), the continuous variant lays an
uninterrupted ring of xylem / cambium / phloem.  The asserts pin the two
properties that define it:

* the three tissue *regions* each form a single connected annulus regardless of
  the xylem layout — the ``"files"`` layout only adds thin radial parenchyma
  strips *inside* the xylem, forcing the vessels into radial files without
  breaking the xylem / phloem / cambium annuli;
* the xylem is endarch (small protoxylem on the inner/pith face grading to large
  metaxylem toward the cambium).
"""

import os
import sys

import numpy as np
from shapely.ops import unary_union

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.stem_dicot_continuous_class import ContinuousDicotStemAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


def _stem(xylem_layout="packed", n_xylem_files=3):
    data = OrganInputData.for_dicot_stem_continuous()
    data.set_value("vascular_cylinder", "xylem_layout", xylem_layout)
    data.set_value("vascular_cylinder", "n_xylem_files", n_xylem_files)
    stem = StemAnatomy(data, seed=SEED)
    stem.generate_cells()
    return stem


def _n_components(polys):
    u = unary_union(list(polys))
    if u.is_empty:
        return 0
    return len(u.geoms) if hasattr(u, "geoms") else 1


# -- factory dispatch --------------------------------------------------------

def test_factory_returns_continuous_variant():
    stem = StemAnatomy(OrganInputData.for_dicot_stem_continuous(), seed=SEED)
    assert isinstance(stem, ContinuousDicotStemAnatomy)
    # The plain dicot preset must still be the discrete-bundle eustele.
    plain = StemAnatomy(OrganInputData.for_dicot_stem(), seed=SEED)
    assert not isinstance(plain, ContinuousDicotStemAnatomy)


# -- continuity --------------------------------------------------------------

def test_packed_is_one_continuous_ring_per_tissue():
    tp = _stem(xylem_layout="packed").vascular_tissue_polygons
    for tissue in ("xylem", "cambium", "phloem"):
        assert _n_components(tp.get(tissue, [])) == 1, \
            f"{tissue} is not a single continuous annulus"


def _xylem_band(stem):
    """Inner / outer radius (from the organ centre) of the registered xylem annulus."""
    xy = unary_union(stem.vascular_tissue_polygons.get("xylem", []))
    ext = np.asarray(xy.exterior.coords)
    d = np.hypot(ext[:, 0], ext[:, 1])
    r_in, r_out = float(d.min()), float(d.max())
    for ring in xy.interiors:
        ic = np.asarray(ring.coords)
        r_in = min(r_in, float(np.hypot(ic[:, 0], ic[:, 1]).min()))
    return r_in, r_out


def _xylem_band_census(stem):
    """(#vessels, #parenchyma) cells whose centre falls in the xylem annulus band."""
    r_in, r_out = _xylem_band(stem)
    def _in(c):
        return c.polygon is not None and r_in - 0.005 <= np.hypot(c.x, c.y) <= r_out + 0.005
    ves = sum(1 for c in stem.all_cells.cells if c.type == "xylem" and _in(c))
    par = sum(1 for c in stem.all_cells.cells if c.type == "parenchyma" and _in(c))
    return ves, par


def test_files_line_up_xylem_only():
    """The 'files' layout keeps all three tissue *regions* continuous (the strips
    are parenchyma inside the xylem, not a segmentation of it) and organises the
    xylem into radial vessel files: thin radial parenchyma strips displace vessels,
    so the xylem band carries fewer vessels and more parenchyma than the packed
    layout, while the phloem and cambium are untouched."""
    files = _stem(xylem_layout="files", n_xylem_files=6)
    tp = files.vascular_tissue_polygons
    # Every tissue region stays one connected annulus (files only add parenchyma
    # *inside* the xylem — they never cut the xylem / phloem / cambium annuli).
    for tissue in ("xylem", "cambium", "phloem"):
        assert _n_components(tp.get(tissue, [])) == 1, \
            f"{tissue} region is not a single continuous annulus"

    # The radial parenchyma strips are the "files" signature: relative to the
    # seamless packed cylinder they trade xylem vessels for parenchyma in the band.
    packed = _stem(xylem_layout="packed")
    ves_f, par_f = _xylem_band_census(files)
    ves_p, par_p = _xylem_band_census(packed)
    assert par_f > par_p, "files layout did not add parenchyma strips in the xylem"
    assert ves_f < ves_p, "files layout did not displace any xylem vessels"


# -- endarch xylem -----------------------------------------------------------

def test_xylem_is_endarch():
    stem = _stem(xylem_layout="packed")
    vessels = [c for c in stem.all_cells.cells
               if c.type == "xylem" and c.polygon is not None]
    assert len(vessels) > 10
    radius = np.array([np.hypot(c.x, c.y) for c in vessels])
    area = np.array([c.polygon.area for c in vessels])
    order = np.argsort(radius)
    half = len(order) // 2
    inner_mean = area[order[:half]].mean()
    outer_mean = area[order[half:]].mean()
    # Metaxylem (toward the cambium, larger radius) is bigger than the protoxylem
    # (toward the pith, smaller radius).
    assert outer_mean > inner_mean


# -- whole-organ smoke -------------------------------------------------------

def test_continuous_stem_generates_all_tissues():
    stem = _stem(xylem_layout="packed")
    types = {c.type for c in stem.all_cells.cells}
    for t in ("xylem", "cambium", "sieve element", "companion cell",
              "parenchyma", "cortex", "epidermis"):
        assert t in types, f"continuous stem missing {t}"
