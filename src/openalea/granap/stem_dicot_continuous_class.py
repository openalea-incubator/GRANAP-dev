"""Continuous (non-fascicular) dicot stem anatomy.

Where :class:`~openalea.granap.stem_dicot_class.DicotStemAnatomy` builds the
textbook *eustele* — discrete collateral bundles separated by interfascicular
parenchyma — this variant builds the **non-fascicular** pattern: an uninterrupted
vascular *cylinder* laid down as a ring from the start, as seen in *Linum*,
*Ricinus* and rapidly-woody dicots (and, developmentally, in what any dicot
becomes once its interfascicular cambium closes the ring).

The cylinder is three concentric annuli on the pith/cortex boundary,
pith -> cortex:

* an **endarch xylem annulus** toward the pith — small protoxylem on the inner
  (pith) face grading to large metaxylem toward the cambium;
* a **continuous cambium ring** on the boundary (the same renderer the eustele
  uses for its closed secondary cambium);
* a **phloem annulus** toward the cortex — sieve elements + companion cells with
  parenchyma around them.

The xylem annulus is optionally organised into radial **files** (lines):
``vascular_cylinder.xylem_layout == "files"`` lays thin radial parenchyma strips
*inside the xylem annulus only* (never the cambium or phloem, which stay
continuous), cutting it into ``n_xylem_files`` tangential compartments; the vessel
packer then fills each compartment in a radial row.  This is the same
parenchyma-first texture the dicot vascular bundle's xylem uses
(``xylem_layout="files"`` there).  ``xylem_layout="packed"`` (the default) leaves a
fully continuous, un-lined ring of vessels.

Instantiate via ``StemAnatomy(input_data)`` with ``vascular_bundle.arrangement``
set to ``"continuous"`` (planttype == 2); the factory in
:mod:`openalea.granap.stem_class` dispatches here.  The cylinder geometry and
ground-cell composition come from the ``vascular_cylinder`` param; vessel / sieve
/ cambium *cell* sizes reuse the ``xylem`` / ``phloem`` / ``cambium`` blocks.
"""

import logging
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from openalea.granap.stem_dicot_class import DicotStemAnatomy
from openalea.granap.tissue_class import TissueRecipe
from openalea.granap.vascular_bundle import (
    _pack_place, _fill_parenchyma, _place_phloem_cells,
)

log = logging.getLogger(__name__)


def _radial_range(zone: Polygon, cx: float, cy: float) -> Tuple[float, float]:
    """Inner / outer radius of ``zone`` from ``(cx, cy)``.

    For a ring the inner radius comes from the hole (interior ring); for a simply
    connected arc-compartment the inner arc is part of the exterior, so the
    exterior min/max already bracket it.
    """
    ext = np.asarray(zone.exterior.coords)
    d = np.hypot(ext[:, 0] - cx, ext[:, 1] - cy)
    r_in, r_out = float(d.min()), float(d.max())
    for ring in zone.interiors:
        ic = np.asarray(ring.coords)
        r_in = min(r_in, float(np.hypot(ic[:, 0] - cx, ic[:, 1] - cy).min()))
    return r_in, r_out


class ContinuousDicotStemAnatomy(DicotStemAnatomy):
    """Dicot stem with a continuous, non-fascicular vascular cylinder (an
    uninterrupted xylem / cambium / phloem ring instead of discrete bundles)."""

    # ------------------------------------------------------------------
    # Vascular tissue
    # ------------------------------------------------------------------

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """One ``continuous vascular cylinder`` step; the cambium ring, xylem and
        phloem annuli are all laid by :meth:`_build_cylinder`."""
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        cyl = self._get_param("vascular_cylinder")
        if not cyl or float(cyl.get("xylem_thickness", 0.0)) <= 0.0:
            return recipe
        recipe.special(
            "continuous vascular cylinder",
            lambda: self._build_cylinder(polygon),
            produces=("xylem", "cambium", "phloem", "parenchyma"),
        )
        return recipe

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _cylinder_contour(self, polygon: Polygon) -> Polygon:
        """The cambium-ring contour the cylinder is built on — the pith/cortex
        boundary in the ``vascular_cylinder.ring_shape`` family."""
        cyl = self._get_param("vascular_cylinder")
        cx, cy = polygon.centroid.x, polygon.centroid.y
        return self._ring_contour(cx, cy, self._primary_ring_radius(polygon), cyl)

    def _n_xylem_files(self, cyl: dict, xylem: dict, circumference: float) -> int:
        """Number of radial xylem files (0 unless the layout is ``files``).

        Explicit ``n_xylem_files >= 1`` is used as given; ``0`` (the default) is
        *auto*: as many files as fit one file per (metaxylem vessel + strip) of the
        cambium circumference, so every xylem pole reads as its own radial line."""
        if cyl.get("xylem_layout", "packed") != "files":
            return 0
        n = int(cyl.get("n_xylem_files", 0))
        if n >= 1:
            return n
        strip_w = 1.2 * float(cyl.get("parenchyma_width", 0.008))
        pitch = float(xylem.get("vessel_diameter", 0.045)) + strip_w
        return max(2, int(round(circumference / pitch))) if pitch > 0 else 2

    def _xylem_file_strips(self, cyl: dict, cx: float, cy: float,
                           r_in: float, r_out: float, n: int) -> List[Polygon]:
        """``n`` thin radial parenchyma strips cutting the xylem annulus into files.

        Each strip is a thin (``~0.6 * parenchyma_width``) radial bar from ``r_in``
        to ``r_out`` at an evenly-spaced angle, overshooting the annulus radially so
        it cuts all the way across; ``xylem_file_jitter`` nudges each file's angle so
        they don't read as a rigid grid.  Mirrors the dicot bundle's file strips."""
        if n < 2 or r_in <= 0.0:
            return []
        half_w = 0.6 * float(cyl.get("parenchyma_width", 0.008))
        jitter = float(cyl.get("xylem_file_jitter", 0.3))
        reach_in = max(r_in - 2.0 * half_w, 0.0)
        reach_out = r_out + 2.0 * half_w
        strips = []
        for k in range(n):
            th = 2.0 * np.pi * k / n
            th += self.rng.uniform(-1.0, 1.0) * jitter * (np.pi / n)  # nudge the file
            d = np.array([np.cos(th), np.sin(th)])
            p_in = (cx + d[0] * reach_in, cy + d[1] * reach_in)
            p_out = (cx + d[0] * reach_out, cy + d[1] * reach_out)
            strips.append(LineString([p_in, p_out]).buffer(half_w, cap_style=2))
        return strips

    def _angular_sectors(self, zone: Polygon, cx: float, cy: float, n: int,
                         r_far: float) -> List[Polygon]:
        """Split an annular ``zone`` into ``n`` adjacent pie-slice sectors (no gaps).

        A purely computational split: unioning the sectors gives back ``zone``.  It
        keeps each ``pack_circles`` call to a small sub-region — the packer is
        superlinear in the number of circles per region, so packing ``n`` sectors is
        an order of magnitude faster than one big annulus (with the same cells).  Any
        packer clearance along a seam is filled afterwards by the parenchyma pass."""
        if n <= 1:
            return self._pieces(zone)
        out = []
        for k in range(n):
            a0, a1 = 2.0 * np.pi * k / n, 2.0 * np.pi * (k + 1) / n
            wedge = Polygon([(cx, cy)] + [(cx + r_far * np.cos(a), cy + r_far * np.sin(a))
                                          for a in np.linspace(a0, a1, 6)])
            out.extend(self._pieces(zone.intersection(wedge)))
        return out

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_cylinder(self, polygon: Polygon) -> None:
        """Lay the three concentric annuli (xylem / cambium / phloem).

        Radial layout about the cambium contour (radius ``r0``, half-band
        ``half_cam``): the xylem annulus is ``[r0 - half_cam - xylem_thickness,
        r0 - half_cam]``, the cambium band ``[r0 - half_cam, r0 + half_cam]``, and
        the phloem annulus ``[r0 + half_cam, r0 + half_cam + phloem_thickness]``, so
        the conducting annuli abut the cambium band without overlapping it.  With
        ``xylem_layout == "files"`` thin radial parenchyma strips cut the xylem
        annulus into ``n_xylem_files`` compartments so the vessels pack into radial
        files (lines); the cambium ring and the phloem annulus stay continuous.
        """
        cyl = self._get_param("vascular_cylinder")
        xylem = self._get_param("xylem")
        phloem = self._get_param("phloem")
        cambium = self._get_param("cambium")
        if not cyl:
            return

        cx, cy = polygon.centroid.x, polygon.centroid.y
        contour = self._cylinder_contour(polygon)
        r0 = self._primary_ring_radius(polygon)
        half_cam = self._cambium_band_thickness(cambium) / 2.0
        xt = float(cyl["xylem_thickness"])
        pt = float(cyl["phloem_thickness"])

        # Concentric annuli, carried in the ring_shape by buffering the contour.
        xylem_annulus = contour.buffer(-half_cam).difference(
            contour.buffer(-half_cam - xt))
        phloem_annulus = contour.buffer(half_cam + pt).difference(
            contour.buffer(half_cam))
        if xylem_annulus.is_empty and phloem_annulus.is_empty:
            return

        p_diam = float(cyl.get("parenchyma_diameter", 0.008))
        p_w = float(cyl.get("parenchyma_width", 0.008))
        vgrow = 0.25 * (p_diam + p_w)
        circ = contour.length
        r_far = (r0 + half_cam + max(xt, pt)) * 2.0

        # --- xylem: endarch-graded vessels, optionally packed as radial files -
        # The endarch gradient is measured over the whole xylem annulus (small
        # protoxylem at the pith face -> large metaxylem toward the cambium).  In
        # 'files' layout, thin radial parenchyma strips cut the annulus into
        # ``_n_xylem_files`` compartments (auto: one file per vessel of circumference,
        # so each pole is its own radial line); each compartment is packed in a row.
        # The compartments also keep every pack_circles call to a small sub-region,
        # which is what makes 'files' fast.  The parenchyma then fills the strips +
        # the space around the vessels in one pass over the whole annulus.
        grr = _radial_range(xylem_annulus, cx, cy) if not xylem_annulus.is_empty else None
        n_files = self._n_xylem_files(cyl, xylem, circ)
        strips = (self._xylem_file_strips(cyl, cx, cy, grr[0], grr[1], n_files)
                  if grr is not None and n_files >= 2 else [])
        strip_union = unary_union(strips) if strips else None
        xylem_zone = (xylem_annulus.difference(strip_union)
                      if strip_union is not None else xylem_annulus)
        vessels = []
        for piece in self._pieces(xylem_zone):
            vs, _ = _pack_place(
                self.vascular_cells, self.rng, piece, "xylem", cx, cy,
                voronoi_grow=vgrow, r_floor=p_diam * 0.4, n_border=25,
                proportion=float(cyl.get("prop_vessel", 0.6)),
                direction="edge",                 # endarch: large toward the cambium
                gradient_center=(cx, cy), gradient_radial_range=grr,
                diameter_max=xylem.get("vessel_diameter", 0.045),
                diameter_min=xylem.get("vessel_diameter_min", 0.012),
                diameter_sd=xylem.get("vessel_diameter_sd", 0.004),
                gradient_function=xylem.get("gradient_function", "five_pl"),
                gradient_inflection=xylem.get("gradient_inflection", 0.5),
                gradient_steepness=xylem.get("gradient_steepness", 3.0),
                gradient_asymmetry=xylem.get("gradient_asymmetry", 1.0),
            )
            vessels.extend(vs)
        self.vascular_polygons.extend(vessels)
        # Fill parenchyma over the whole (un-cut) annulus minus the vessels, so the
        # file strips fill with parenchyma too (the annulus is a single polygon, so
        # _fill_parenchyma's _largest keeps it whole instead of one compartment).
        _fill_parenchyma(self.vascular_cells, xylem_annulus,
                         unary_union(vessels) if vessels else None,
                         "parenchyma", cx, cy, p_diam, p_w)

        # --- phloem: sieve + companion cells, parenchyma around them --------
        # Packed in angular sectors: the phloem annulus holds many small sieve cells
        # and pack_circles is superlinear, so one big annulus is ~20x slower than the
        # same cells packed sector-by-sector.  The sectors are invisible — seams are
        # filled by the parenchyma pass and blend into the (>=50%) phloem parenchyma,
        # so the phloem stays one continuous ring.
        if not phloem_annulus.is_empty:
            n_sec = max(6, int(round(circ / (10.0 * float(phloem.get("sieve_diameter", 0.012))))))
            occupied = []
            for sec in self._angular_sectors(phloem_annulus, cx, cy, n_sec, r_far):
                occ = _place_phloem_cells(self.vascular_cells, self.rng, sec,
                                          cx, cy, phloem, cyl)
                if occ is not None and not occ.is_empty:
                    occupied.append(occ)
            _fill_parenchyma(self.vascular_cells, phloem_annulus,
                             unary_union(occupied) if occupied else None,
                             "parenchyma", cx, cy, p_diam, p_w)

        # --- continuous cambium ring (reuses the eustele's closed-ring path) -
        conducting = [g for g in (xylem_annulus, phloem_annulus) if not g.is_empty]
        self._build_cambium(contour, [], conducting, cambium, secondary=True)

        # --- register removal masks / tissue view ---------------------------
        # The xylem files are parenchyma *inside* the xylem annulus (like the
        # bundle), so the xylem stays one connected annulus region — no separate
        # ray tissue is registered.
        if not xylem_annulus.is_empty:
            self.vascular_tissue_polygons.setdefault("xylem", []).append(xylem_annulus)
        if not phloem_annulus.is_empty:
            self.vascular_tissue_polygons.setdefault("phloem", []).append(phloem_annulus)

    @staticmethod
    def _pieces(geom) -> List[Polygon]:
        """Non-empty Polygon pieces of a (possibly Multi)geometry."""
        if geom is None or geom.is_empty:
            return []
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        return [g for g in parts if g.geom_type == "Polygon" and not g.is_empty]
