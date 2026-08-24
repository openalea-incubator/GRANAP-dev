"""Monocot root anatomy.

``MonocotRootAnatomy`` builds a monocot stele: either the 'default' ring of
discrete metaxylem bundles or the 'arch' variant (an evenly-spaced metaxylem
ring with a stele sheath and graded protoxylem poles).  Instantiate via
``RootAnatomy(input_data)`` — the factory in :mod:`openalea.granap.root_class`
dispatches to this class when ``planttype == 1``.
"""

import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

from shapely.geometry import Point, Polygon, LineString, MultiPoint, box
from shapely.ops import unary_union
from shapely.affinity import translate, scale as affine_scale, rotate
from shapely.prepared import prep

from openalea.granap.organ_class import Organ
from openalea.granap.layer_class import Layer, LayerPolygon
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.tissue_class import (
    place_packed_group, fill_by_packing, fill_along, fill_by_rings,
    TissueRecipe, Tissue,
)
from openalea.granap.input_data import OrganInputData
from openalea.granap.math_functions import GRADIENT_FUNCTIONS, rescale
from openalea.granap.root_class import RootAnatomy



# ---------------------------------------------------------------------------
# Monocot subclass
# ---------------------------------------------------------------------------

class MonocotRootAnatomy(RootAnatomy):
    """Monocot root: 'default' ring of metaxylem bundles, or 'arch' (an
    evenly-spaced metaxylem ring with a stele sheath + graded protoxylem poles)."""

    def _parse_vascular_params(self) -> None:
        xylem  = self._get_param("xylem")
        phloem = self._get_param("phloem")

        self.vascular_params.update({
            "xylem_diameter":         float(xylem.get("vessel_diameter",        0.06)),
            "xylem_diameter_sd":      float(xylem.get("vessel_diameter_sd",     0.005)),
            "protoxylem_diameter":    float(xylem.get("protoxylem_diameter",    0.01)),
            "protoxylem_diameter_sd": float(xylem.get("protoxylem_diameter_sd", 0.002)),
            "protoxylem_width":       float(xylem.get("protoxylem_cluster_width",  0.03)),
            "protoxylem_height":      float(xylem.get("protoxylem_cluster_height", 0.05)),
            "n_vascular_bundles":     int(xylem.get("n_vascular_bundles",       5)),
            "ratio_proto_meta":       float(xylem.get("ratio_proto_meta",       2.2)),
            "phloem_diameter":        float(phloem.get("sieve_diameter",        0.005)),
            "phloem_diameter_sd":     float(phloem.get("sieve_diameter_sd",     0.001)),
            "phloem_width":           float(phloem.get("cluster_width",         0.02)),
            "phloem_height":          float(phloem.get("cluster_height",        0.03)),
            "relative_phloem":        float(phloem.get("relative_distance",     0.5)),
            "xylem_shape":            str(xylem.get("xylem_shape", "default")),
        })

        # Primary phloem is built only when its param entry is present (opt-out).
        self.has_primary_phloem = bool(phloem)

        if self.vascular_params["xylem_shape"] == "star":
            # Star (actinostele) xylem: a star-shaped region packed with
            # size-graded vessels (shared base-class machinery), phloem in the
            # valleys.  Field names mirror the dicot xylem star.
            self.vascular_params.update({
                "n_vascular_peak":           int(xylem.get("n_vascular_peak",          5)),
                "inner_radius_xylem":        float(xylem.get("radius_valley_side",     0.05)),
                "outer_radius_xylem":        float(xylem.get("radius_peak_side",       0.22)),
                "arc_top_xylem":             float(xylem.get("arc_peak_side",          0.03)),
                "arc_bottom_xylem":          float(xylem.get("arc_valley_side",        0.03)),
                "pith_radius":               float(xylem.get("pith_radius",            0.0)),
                "xylem_diameter_max":        float(xylem.get("vessel_diameter",        0.06)),
                "xylem_diameter_min":        float(xylem.get("vessel_diameter_min",    0.01)),
                "xylem_diameter_sd":         float(xylem.get("vessel_diameter_sd",     0.005)),
                "xylem_gradient_function":   str(xylem.get("gradient_function",        "five_pl")),
                "xylem_gradient_inflection": float(xylem.get("gradient_inflection",    0.7)),
                "xylem_gradient_steepness":  float(xylem.get("gradient_steepness",     5.0)),
                "xylem_gradient_asymmetry":  float(xylem.get("gradient_asymmetry",     1.0)),
                "xylem_enforce_gradient_min": float(xylem.get("enforce_gradient_min",  0.0)),
                "xylem_allow_ellipse":       bool(xylem.get("allow_ellipse",           True)),
                "xylem_ellipse_max_aspect":  float(xylem.get("ellipse_max_aspect",     2.0)),
                "xylem_packing_strategy":    str(xylem.get("packing_strategy",         "space")),
                "xylem_first_vessel_shift":  float(xylem.get("first_vessel_shift",     0.7)),
                "xylem_direction":           str(xylem.get("direction",                "center")),
            })

        if self.vascular_params["xylem_shape"] == "arch":
            self.vascular_params.update({
                # Metaxylem ring geometry + size.
                "xylem_diameter_max":        float(xylem.get("vessel_diameter",        0.06)),
                "xylem_diameter_min":        float(xylem.get("vessel_diameter_min",    0.01)),
                "xylem_diameter_sd":         float(xylem.get("vessel_diameter_sd",     0.005)),
                "n_vascular_peak":           int(xylem.get("n_vascular_peak",          5)),
                "n_metaxylem":               int(xylem.get("n_metaxylem",             0)),
                "outer_radius_xylem":        float(xylem.get("outer_radius",           0.15)),
                "pith_radius":               float(xylem.get("pith_radius",           0.0)),
                "xylem_allow_ellipse":       bool(xylem.get("allow_ellipse",          True)),
                "xylem_ellipse_max_aspect":  float(xylem.get("ellipse_max_aspect",   2.0)),
                # Protoxylem chain (outer band) + its size gradient.
                "protoxylem_diameter":       float(xylem.get("protoxylem_diameter",     0.01)),
                "protoxylem_diameter_sd":    float(xylem.get("protoxylem_diameter_sd",  0.001)),
                "protoxylem_band_depth":     float(xylem.get("protoxylem_band_depth",   0.0)),
                "protoxylem_diameter_min":   float(xylem.get("protoxylem_diameter_min", 0.0)),
                "protoxylem_pole_width_inner": float(xylem.get("protoxylem_pole_width_inner", 0.0)),
                "protoxylem_pole_width_outer": float(xylem.get("protoxylem_pole_width_outer", 0.0)),
                "xylem_gradient_function":   str(xylem.get("gradient_function",       "five_pl")),
                "xylem_gradient_inflection": float(xylem.get("gradient_inflection",   0.7)),
                "xylem_gradient_steepness":  float(xylem.get("gradient_steepness",    5.0)),
                "xylem_gradient_asymmetry":  float(xylem.get("gradient_asymmetry",    1.0)),
                "relative_phloem":           float(phloem.get("relative_distance",    0.5)),
            })

    # ------------------------------------------------------------------
    # Vascular tissue
    # ------------------------------------------------------------------

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Declarative description of how a monocot stele is assembled.

        Built and run by the shared ``Organ._create_vascular_tissue`` scaffold;
        remove_cells_in_polygon + extend_cells happens later in
        ``Organ.generate_cells``.  Two variants share the same vocabulary; the
        build order is data, inspectable via ``recipe.describe()`` /
        ``recipe.plan()``.
        """
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        # Developmental series: place the prescribed tracked vessels instead of
        # packing (see ROOT_SERIES_PLAN / RootAnatomy.prescribe_vessels).
        if getattr(self, "_prescribed_vessels", None):
            # Developmental series: the metaxylem are the prescribed (tracked) vessels;
            # the protoxylem + phloem are regenerated per section around them (untracked),
            # reusing the same default-mode steps — their count follows n_vascular_bundles,
            # which the series sets to the metaxylem count for scaling.
            recipe.special("prescribed metaxylem",
                           lambda: self._place_prescribed_xylem(polygon),
                           produces=("metaxylem",))
            recipe.special("metaxylem sheath",
                           lambda: self.fit_metaxylem_sheath(polygon),
                           produces=("stele",))
            recipe.special("phloem + protoxylem bundles",
                           lambda: self.fit_phloem_protoxylem_elements(polygon),
                           produces=("phloem", "protoxylem"))
            return recipe
        shape = self.vascular_params.get("xylem_shape", "default")
        if shape != "star" and self.vascular_params.get("n_vascular_bundles", 0) == 0:
            return recipe                       # no vascular bundles -> empty
        if shape == "star":
            # Star (actinostele) mode: a star-shaped xylem region packed with
            # size-graded vessels (shared base-class machinery, same as the dicot
            # primary xylem), then phloem strands in the valleys between arms.
            # No cambium — the phloem is positioned relative to the xylem star's
            # valley radius (see _phloem_valley_zones).
            if self.vascular_params.get("n_vascular_peak", 0) == 0:
                return recipe                   # no xylem arms -> empty
            recipe.fill("xylem star", self._xylem_star_region(polygon),
                        strategy="packing", produces=("xylem", "stele"),
                        record=self._record_xylem_vessels, **self._xylem_pack_kwargs())
            # Drop stele cells that are mostly engulfed by xylem vessels; the
            # point-level mask in generate_cells clips the rest.
            recipe.cleanup("clear stele engulfed by xylem",
                           self._remove_stele_engulfed_by_xylem)
            if self.has_primary_phloem:
                self._add_phloem_step(recipe, polygon)
        elif self.vascular_params.get("xylem_shape", "default") == "arch":
            # Arch mode, built inside-out: (1) evenly-spaced metaxylem ring,
            # (2) a stele-cell sheath around each metaxylem, (3) a graded
            # protoxylem chain per pole directed to its nearest metaxylem.
            recipe.special("arch metaxylem",
                           lambda: self._fit_arch_metaxylem(polygon),
                           produces=("metaxylem",))
            recipe.special("arch protoxylem",
                           lambda: self._fit_arch_protoxylem(),
                           produces=("protoxylem",))
            recipe.special("arch phloem",
                           lambda: self._fit_arch_phloem(polygon),
                           produces=("phloem",))
        else:
            # Bespoke placements (a single-vessel border fill, a cell-relative
            # sheath, and pizza-sliced bundles) — not plain region+fill, so they
            # stay `special` steps rather than declarative `fill`s.
            recipe.special("metaxylem ring",
                           lambda: self.fit_metaxylem_elements(polygon),
                           produces=("metaxylem",))
            recipe.special("metaxylem sheath",
                           lambda: self.fit_metaxylem_sheath(polygon),
                           produces=("stele",))
            recipe.special("phloem + protoxylem bundles",
                           lambda: self.fit_phloem_protoxylem_elements(polygon),
                           produces=("phloem", "protoxylem"))
        return recipe

    # ------------------------------------------------------------------
    # Star-mode phloem (valleys between the xylem star arms)
    # ------------------------------------------------------------------

    def _phloem_valley_zones(self, stele_polygon: Polygon):
        """Phloem regions in the valleys between the xylem star arms.

        Monocot roots have no cambium, so — unlike the dicot — the phloem band
        is positioned relative to the xylem star's *valley radius* (the arm
        bases) rather than a cambium radius: it sits between that valley radius
        and the stele edge, interpolated by ``relative_distance``.  Shape-first:
        each valley starts as an oriented ellipse, is clipped to the stele and
        carved out of the xylem star.  Returns the list of phloem
        :class:`Tissue`s (the recipe fills them by circle-packing).
        """
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y
        n_peaks = p["n_vascular_peak"]

        width     = p["phloem_width"]
        height    = p["phloem_height"]
        cell_diam = p["phloem_diameter"]
        relative_distance = p["relative_phloem"]

        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)
        # Inner bound of the phloem band = the xylem star's valley radius; there
        # is no cambium to offset from (the dicot uses its cambium valley radius).
        valley_r = min(p["inner_radius_xylem"], stele_r)
        r_center = (
            valley_r + (height / 2)
            + (stele_r - height - valley_r) * relative_distance
        )

        xylem_star = getattr(self, "xylem_star", None)
        min_area = np.pi * (cell_diam / 2) ** 2 * (1 - 0.0015)
        tissues = []
        for k in range(n_peaks):
            theta = 2 * np.pi * (k + 0.5) / n_peaks   # valleys sit between arms

            raw = self._oriented_ellipse(
                cx + r_center * np.cos(theta), cy + r_center * np.sin(theta),
                width, height, np.degrees(theta),
            )

            tissue = Tissue("phloem", raw).intersection(stele_polygon)
            if xylem_star is not None and not xylem_star.is_empty:
                tissue.difference(xylem_star)
            if tissue.is_empty or tissue.area < min_area:
                continue
            tissues.append(tissue)

        return tissues

    def _add_phloem_step(self, recipe: TissueRecipe, stele_polygon: Polygon) -> None:
        """Declarative phloem step: valley *regions* filled by circle-packing.

        Mirrors the dicot phloem step but with no cambium clearance (monocot has
        no cambium): the recorded region — used by the unified vascular mask in
        ``generate_cells`` to clear the stele seeds underneath — is the phloem
        footprint itself.  The regions are built lazily at build time because
        they are carved from the xylem star placed by the earlier fill step.
        """
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y
        cell_diam = p["phloem_diameter"]
        cell_sd   = p["phloem_diameter_sd"]

        def record(tissue, _result):
            self.vascular_tissue_polygons.setdefault(tissue.tag, []).append(tissue.shape)

        recipe.fill_each(
            "phloem in valleys",
            lambda: self._phloem_valley_zones(stele_polygon),
            strategy="packing", produces=("phloem",), record=record,
            n_border=25, angle_center=(cx, cy),
            proportion=1.0, direction=None,
            diameter_max=cell_diam, diameter_min=cell_diam,
            diameter_sd=cell_sd, gradient_function="normal",
        )

    # ------------------------------------------------------------------
    # Default ring-bundle methods
    # ------------------------------------------------------------------

    @staticmethod
    def metaxylem_positions(stele_polygon: Polygon, n: int, vessel_diameter: float):
        """The class's canonical metaxylem *centres* for ``n`` vessels — the same
        geometry as :meth:`fit_metaxylem_elements` (pizza-slice the stele into ``n``
        wedges, inscribe one vessel per wedge), minus the rng size jitter and cell
        seeding.  Pure geometry, so a caller (e.g. the developmental series) can ask
        "where would the class put ``n`` metaxylem?" without running the pipeline.
        Returns a list of ``(x, y)`` centres."""
        if n <= 0 or stele_polygon.is_empty:
            return []
        if n == 1:
            slices = [stele_polygon]
        else:
            slices = GeometryProcessor.pizza_slice(
                stele_polygon.buffer(-vessel_diameter / 4.0), n)
        out = []
        for s in slices:
            if s.is_empty:
                continue
            c = GeometryProcessor.fit_inner_ellipse(s, vessel_diameter / 2.0)["polygon"].centroid
            out.append((float(c.x), float(c.y)))
        return out

    def fit_metaxylem_elements(self, polygon):
        n_xylem_cells = self.vascular_params["n_vascular_bundles"]
        if n_xylem_cells == 0:
            return
        elif n_xylem_cells == 1:
            slices = [polygon]
        else:
            slices = GeometryProcessor.pizza_slice(
                polygon.buffer(-self.vascular_params["xylem_diameter"] / 4), n_xylem_cells
            )
        cells_in_slices, list_xylem_polygons = self.vascular_elements_in_slice(slices)
        self.vascular_cells = cells_in_slices
        self.vascular_polygons = list_xylem_polygons

    def vascular_elements_in_slice(self, slices: List[Polygon]):
        list_xylem_polygons = []
        cells_in_slices = CellManager()
        for i_slice, slice in enumerate(slices):
            xylem_diameter = float(np.clip(
                self.rng.normal(
                    self.vascular_params["xylem_diameter"],
                    self.vascular_params["xylem_diameter_sd"],
                ),
                self.vascular_params["xylem_diameter"] * 0.1,
                np.inf,
            ))

            # Region: a single metaxylem vessel inscribed in the pizza slice.
            # (One big vessel per slice — its cells are seeded along the vessel
            # border, a bespoke fill that does not map onto the packing verbs.)
            metaxylem = Tissue(
                "metaxylem",
                GeometryProcessor.fit_inner_ellipse(slice, xylem_diameter / 2)["polygon"],
            )
            xylem_polygon_buff = GeometryProcessor.buffer_polygon(metaxylem.shape, -(xylem_diameter / 2) * 0.15)
            x, y = xylem_polygon_buff.exterior.coords.xy
            center = metaxylem.shape.centroid
            coords = np.column_stack((x, y))
            coords = GeometryProcessor.resample_coords(coords, target_n_points=25)

            for cell_border_pts in coords[1:]:
                cells_in_slices.add_cell(Cell.radial(
                    metaxylem.tag, cell_border_pts[0], cell_border_pts[1], xylem_diameter,
                    i_slice, center, id_layer=i_slice,
                ))
            list_xylem_polygons.append(metaxylem.shape)
        return cells_in_slices, list_xylem_polygons

    def fit_metaxylem_sheath(self, stele_polygon: Polygon):
        """Add a ring of xylem parenchyma cells around each metaxylem vessel."""
        cell_diameter = self.vascular_params["cell_diameter"]
        center = stele_polygon.centroid

        next_id_group = max((c.id_group for c in self.all_cells.cells), default=0) + 1
        xylem_polygons = list(self.vascular_polygons)

        for xylem_polygon in xylem_polygons:
            outer = xylem_polygon.buffer(cell_diameter*0.8).intersection(stele_polygon)
            if outer.is_empty:
                continue
            mid_ring = xylem_polygon.buffer(cell_diameter / 2).intersection(stele_polygon)
            if mid_ring.is_empty or mid_ring.geom_type != "Polygon":
                continue

            seed_coords = CellGenerator.cells_on_layer(mid_ring, cell_diameter)
            for pt in seed_coords[1:]:
                self.vascular_cells.add_cell(Cell.radial(
                    "stele", pt[0], pt[1], cell_diameter, next_id_group, center,
                ))
                next_id_group += 1

            ring_polygon = outer.difference(xylem_polygon)
            if not ring_polygon.is_empty:
                self.vascular_polygons.append(ring_polygon)

    def fit_phloem_protoxylem_elements(self, polygon):
        n_protoxylem = int(np.ceil(
            self.vascular_params["ratio_proto_meta"] * self.vascular_params["n_vascular_bundles"]
        ))
        n_phloem = n_protoxylem - 1
        buffing_dist = max(
            self.vascular_params["protoxylem_diameter"],
            self.vascular_params["phloem_diameter"],
        )

        polygon = polygon.difference(polygon.buffer(-buffing_dist * 1.1))
        polygon = polygon.difference(unary_union(self.vascular_polygons))

        slices = GeometryProcessor.pizza_slice(polygon, n_phloem + n_protoxylem)

        self.protoxylem_polygons = []
        self.phloem_polygons = []

        for i, poly_slice in enumerate(slices[1:]):
            kind = "protoxylem" if i % 2 == 0 else "phloem"
            cells_in_slice, list_polygons = self._bundle_elements_in_slice(poly_slice, kind)
            self.vascular_cells.extend_cells(cells_in_slice.cells)
            self.vascular_polygons.extend(list_polygons)
            if kind == "protoxylem":
                self.protoxylem_polygons.extend(list_polygons)
            else:
                self.phloem_polygons.extend(list_polygons)

    def _bundle_elements_in_slice(self, slice_poly: Polygon, kind: str):
        """Pack one ``kind`` ('protoxylem' or 'phloem') bundle inscribed in a
        pizza slice.  The two differ only in which ``{kind}_width/height/diameter``
        params size the oriented ellipse and its packed cells.

        No local stele removal here: the bundle region is appended to
        ``vascular_polygons`` by the caller and the unified vascular mask in
        ``Organ.generate_cells`` clears every layer seed inside it.
        """
        p = self.vascular_params
        list_polygons = []
        cells_in_slice = CellManager()

        bundle_cx, bundle_cy, available_r = GeometryProcessor.get_inscribed_circle(slice_poly)
        radial_angle_deg = np.degrees(np.arctan2(bundle_cy, bundle_cx))

        parent_r = max(p[f"{kind}_width"], p[f"{kind}_height"]) / 2
        scale    = min(1.0, available_r / parent_r) if parent_r > 0 else 1.0
        width    = p[f"{kind}_width"]    * scale
        height   = p[f"{kind}_height"]   * scale
        diameter = p[f"{kind}_diameter"] * scale

        bundle = Tissue(kind, self._oriented_ellipse(
            bundle_cx, bundle_cy, width, height, radial_angle_deg))
        if bundle.is_empty or bundle.area < np.pi * (diameter / 2) ** 2 * (1 - 0.0015):
            return cells_in_slice, list_polygons

        fill_by_packing(
            cells_in_slice, bundle.shape, bundle.tag, rng=self.rng,
            n_border=24, id_base=self.vascular_cells.next_group_id(), angle_center=None,
            proportion=1.0, direction=None,
            diameter_max=diameter, diameter_sd=p[f"{kind}_diameter_sd"] * scale,
            gradient_function="normal",
        )

        list_polygons.append(bundle.shape)
        return cells_in_slice, list_polygons

    # ------------------------------------------------------------------
    # Arch mode: one metaxylem per arm + graded protoxylem chain per arm
    # ------------------------------------------------------------------

    def _arch_split_radius(self):
        """Radius separating the inner metaxylem zone from the outer protoxylem
        band, plus ``(pith_r, outer_r)``.  The band depth is measured inward from
        the pericycle side (``outer_r``); 0 defaults to 35%% of the radial span."""
        pith_r, outer_r = self._xylem_gradient_radial_range
        depth = self.vascular_params.get("protoxylem_band_depth", 0.0)
        if depth <= 0.0:
            depth = 0.35 * (outer_r - pith_r)
        return max(pith_r, outer_r - depth), pith_r, outer_r

    def _fit_arch_metaxylem(self, polygon: Polygon) -> None:
        """Place ``n_metaxylem`` metaxylem **evenly spaced** in a central ring,
        independent of the protoxylem poles.

        The metaxylem live in a solid annulus ``[pith_r, r_split]`` (clipped to
        the stele), sliced into ``n_metaxylem`` equal sectors.  In each sector a
        vessel of ``vessel_diameter`` (+/- ``vessel_diameter_sd``) is placed as a
        circle when it fits, or — when the sector is too tight for the vessels to
        fit side by side — an aspect-capped **radial ellipse** (falling back to
        the largest inscribed circle).  So the count is exactly ``n_metaxylem``,
        regularly spaced, and never tied to which arm has a protoxylem pole.
        ``n_metaxylem == 0`` defaults to ``n_vascular_peak`` sectors."""
        p = self.vascular_params
        cx, cy = polygon.centroid.x, polygon.centroid.y
        _, _, stele_r = GeometryProcessor._chebyshev_center(polygon)
        outer_r = min(p["outer_radius_xylem"], stele_r)
        pith_r = max(0.0, min(p.get("pith_radius", 0.0), outer_r))
        self._xylem_gradient_center = (cx, cy)
        self._xylem_gradient_radial_range = (pith_r, outer_r)
        r_split, _, _ = self._arch_split_radius()
        # The vascular region is simply the stele clipped to outer_r (no star
        # outline): metaxylem fill the inner annulus [pith_r, r_split], protoxylem
        # and phloem the outer band [r_split, outer_r].
        self._arch_star = polygon.intersection(Point(cx, cy).buffer(outer_r, resolution=64))
        self._arch_split_r = r_split

        n_meta = p.get("n_metaxylem", 0) or p["n_vascular_peak"]

        # Solid metaxylem ring [pith_r, r_split], clipped to the stele.
        ring = (Point(cx, cy).buffer(r_split, resolution=64)
                .difference(Point(cx, cy).buffer(pith_r, resolution=64)))
        meta_region = polygon.intersection(ring)
        if meta_region.is_empty:
            return

        d_meta, d_sd = p["xylem_diameter_max"], p["xylem_diameter_sd"]
        d_floor = p["xylem_diameter_min"]
        allow_ell = p["xylem_allow_ellipse"]
        max_aspect = p["xylem_ellipse_max_aspect"] if allow_ell else 1.0
        cell_d = self.vascular_params.get("cell_diameter", 0.0)

        # Waist of the egg/teardrop, as a fraction of the major length measured
        # from the OUTER tip -> the widest point sits toward the band, tapering
        # inward.  (Area is independent of this split.)
        waist = 0.35

        eggs = []
        for j in range(n_meta):
            theta = 2.0 * np.pi * j / n_meta
            target_r = 0.5 * float(np.clip(self.rng.normal(d_meta, d_sd), d_floor, np.inf))

            # Full circle if it fits with a gap; otherwise elongate radially,
            # KEEPING the area (a*b = target_r**2), just enough to open the gap,
            # but never past ellipse_max_aspect.  If the cap is reached and the
            # vessels still touch, they are placed at the cap (area kept).  Two
            # passes converge (spacing depends on the ring radius).
            a_ax = b_ax = target_r
            for _ in range(2):
                r_ring = max(r_split - a_ax, cell_d) 
                tang_half = max(np.pi * r_ring / n_meta - (cell_d/8), 1e-6)
                if target_r <= tang_half or not allow_ell:
                    a_ax = b_ax = min(target_r, tang_half)
                else:
                    b_ax = tang_half  
                    a_ax = min(target_r * target_r / b_ax, max_aspect * b_ax) 
                    a_ax = max(a_ax, b_ax)  

            # Teardrop: total major = 2*a_ax, waist offset toward the band; outer
            # tip pinned at r_split.  A circle is just the symmetric case.
            major = 2.0 * a_ax
            a_out = max(waist * major, b_ax) if a_ax > b_ax * 1.02 else b_ax
            a_in = major - a_out
            r_widest = r_split - a_out
            wx, wy = cx + r_widest * np.cos(theta), cy + r_widest * np.sin(theta)
            if not polygon.contains(Point(wx, wy)):
                continue
            egg = GeometryProcessor.egg_polygon(wx, wy, a_out, a_in, b_ax, np.degrees(theta))
            egg = self._largest_polygon(egg.intersection(polygon))
            if egg is not None:
                eggs.append(egg)

        # Seed each vessel's border (bespoke shapes -> not place_packed_group).
        next_id = self.vascular_cells.next_group_id()
        self._arch_meta_polys, self._arch_meta_centers = [], []
        for i, egg in enumerate(eggs):
            eff_d = 2.0 * np.sqrt(egg.area / np.pi)
            inner = egg.buffer(-eff_d * 0.12)
            border_poly = inner if (inner.geom_type == "Polygon" and not inner.is_empty) else egg
            bx, by = border_poly.exterior.coords.xy
            coords = GeometryProcessor.resample_coords(np.column_stack((bx, by)), 28)
            gid = next_id + i
            for pt in coords[1:]:
                self.vascular_cells.add_cell(Cell.radial(
                    "metaxylem", pt[0], pt[1], eff_d, gid, (cx, cy)))
            ctr = egg.centroid
            self._arch_meta_polys.append(egg)
            self._arch_meta_centers.append((ctr.x, ctr.y))
            self.vascular_polygons.append(egg)

    def _fit_arch_protoxylem(self) -> None:
        """Pack a graded protoxylem chain at each of the ``n_vascular_peak`` poles,
        each **directed to its nearest metaxylem**.

        Metaxylem and protoxylem are *not* forced onto the same arm.  For every
        pole (a peak tip at the pericycle) the nearest metaxylem centre is found
        and the protoxylem is packed into a straight **corridor** from that
        metaxylem out to the pole tip — so an aligned pole gets a radial file and
        an orphan pole (no metaxylem in its own arm) leans across to connect to
        the closest one.  Within the corridor the size gradient runs over
        ``[r_split, outer_r]`` (largest inner, smallest at the pericycle) and the
        cells are oriented toward that metaxylem."""
        if not hasattr(self, "_arch_star"):
            return
        cx, cy = self._xylem_gradient_center
        r_split = self._arch_split_r
        _, pith_r, outer_r = self._arch_split_radius()

        disc = Point(cx, cy).buffer(r_split, resolution=64)
        proto_region = self._arch_star.difference(disc)
        # Carve out the metaxylem already placed so the protoxylem never overlaps.
        meta_polys = list(self.vascular_polygons)
        if meta_polys:
            proto_region = proto_region.difference(unary_union(meta_polys))
        if proto_region.is_empty:
            return

        p = self.vascular_params
        d_max = p["protoxylem_diameter"]
        d_min = p["protoxylem_diameter_min"] or d_max * 0.4
        cell_d = self.vascular_params.get("cell_diameter", d_min)
        centers = getattr(self, "_arch_meta_centers", []) or [(cx, cy)]
        # Each pole is a tapered trapezoid: ``inner`` wide at the metaxylem end,
        # ``outer`` wide at the pericycle end.  Narrower poles leave more room for
        # the phloem valleys between them.
        w_inner = (p.get("protoxylem_pole_width_inner", 0.0) or d_max * 3.0)
        w_outer = (p.get("protoxylem_pole_width_outer", 0.0) or d_max * 3.0)
        next_id = self.vascular_cells.next_group_id()

        n = p["n_vascular_peak"]
        for k in range(n):
            theta = 2.0 * np.pi * k / n
            tip = np.array([cx + outer_r * np.cos(theta), cy + outer_r * np.sin(theta)])
            # Nearest metaxylem to this pole tip -> aim the corridor at it.
            mx, my = min(centers, key=lambda c: (c[0] - tip[0]) ** 2 + (c[1] - tip[1]) ** 2)
            m = np.array([mx, my])
            axis = tip - m
            L = np.hypot(*axis)
            if L < 1e-9:
                continue
            perp = np.array([-axis[1], axis[0]]) / L      # unit normal to the pole axis
            corridor = Polygon([
                m + perp * w_inner / 2.0, m - perp * w_inner / 2.0,   # inner end (metaxylem)
                tip - perp * w_outer / 2.0, tip + perp * w_outer / 2.0,  # outer end (pericycle)
            ])
            arm = self._largest_polygon(proto_region.intersection(corridor))
            if arm is None:
                continue
            placed = fill_by_packing(
                self.vascular_cells, arm, "protoxylem", rng=self.rng,
                n_border=20, id_base=next_id, angle_center=(mx, my),
                proportion=1.0, direction="center",
                diameter_max=d_max, diameter_min=d_min,
                diameter_sd=p["protoxylem_diameter_sd"],
                gradient_function=p["xylem_gradient_function"],
                gradient_inflection=p["xylem_gradient_inflection"],
                gradient_steepness=p["xylem_gradient_steepness"],
                gradient_asymmetry=p["xylem_gradient_asymmetry"],
                gradient_center=(cx, cy),
                gradient_radial_range=(r_split, outer_r),
            )
            next_id += len(placed)
            new_polys = [poly for poly, _t, _g in placed]
            self.vascular_polygons.extend(new_polys)
            if new_polys:
                # Clear stele only from the small interstitial gaps *between* the
                # protoxylem cells (a morphological close of the vessel union),
                # not the whole corridor -- otherwise the mask empties the pole and
                # the protoxylem Voronoi cells blow up to fill it.  Then carve the
                # region so neighbouring poles don't double-fill the overlap.
                union = unary_union(new_polys)
                closed = union.buffer(cell_d).buffer(-cell_d)     # fill gaps <= ~2*cell_d
                self.vascular_tissue_polygons.setdefault("protoxylem", []).append(closed)
                proto_region = proto_region.difference(union)
                if proto_region.is_empty:
                    break

    def _fit_arch_phloem(self, polygon: Polygon) -> None:
        """Phloem sits in the outer band, in the valleys *between* the poles.

        Same radial band as the protoxylem (``[r_split, outer_r]``) but at the
        mid-pole angles, so xylem poles and phloem alternate around the ring.
        Each valley cluster is carved clear of every already-placed vessel."""
        if not hasattr(self, "_arch_star"):
            return
        p = self.vascular_params
        cx, cy = self._xylem_gradient_center
        r_split = self._arch_split_r
        _, _, outer_r = self._arch_split_radius()
        n = p["n_vascular_peak"]
        sieve_d = p["phloem_diameter"]

        width, height = p["phloem_width"], p["phloem_height"]
        r_center = 0.5 * (r_split + outer_r)             # mid-band

        band = (polygon.intersection(Point(cx, cy).buffer(outer_r, resolution=64))
                .difference(Point(cx, cy).buffer(r_split, resolution=64)))
        if self.vascular_polygons:
            band = band.difference(
                unary_union([g.buffer(sieve_d * 0.5) for g in self.vascular_polygons]))
        if band.is_empty:
            return

        next_id = self.vascular_cells.next_group_id()
        for k in range(n):
            theta = 2.0 * np.pi * (k + 0.5) / n          # valley between poles k, k+1
            # A bounded ellipse cluster at the valley, clipped to the free band.
            raw = self._oriented_ellipse(
                cx + r_center * np.cos(theta), cy + r_center * np.sin(theta),
                width, height, np.degrees(theta), resolution=48,
            )
            cluster = self._largest_polygon(raw.intersection(band))
            if cluster is None:
                continue
            placed = fill_by_packing(
                self.vascular_cells, cluster, "phloem", rng=self.rng,
                n_border=16, id_base=next_id, angle_center=(cx, cy),
                proportion=1.0, direction=None,
                diameter_max=sieve_d, diameter_sd=p["phloem_diameter_sd"],
                gradient_function="normal",
            )
            next_id += len(placed)
            # Record the whole cluster region (not just the sieve circles) so the
            # vascular mask clears stele parenchyma from the gaps *between* the
            # phloem cells, not only from inside them.
            if placed:
                self.vascular_tissue_polygons.setdefault("phloem", []).append(cluster)
