"""
Root anatomy implementation.

`RootAnatomy` acts as a transparent factory: calling ``RootAnatomy(input_data)``
returns either a ``MonocotRootAnatomy`` or a ``DicotRootAnatomy`` instance
depending on the ``planttype`` value in the input.  Both subclasses are
``isinstance(obj, RootAnatomy)`` == True, so all existing code keeps working.
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


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_planttype(input_data) -> int:
    """Extract the planttype integer (1 = monocot, 2 = dicot) from raw input."""
    if isinstance(input_data, OrganInputData):
        params = input_data.to_dict_list()
    elif isinstance(input_data, list):
        params = input_data
    else:
        return 1
    pt = next((p for p in params if p["name"] == "planttype"), {})
    return int(pt.get("value", 1))


# ---------------------------------------------------------------------------
# Base class — shared geometry, acts as factory via __new__
# ---------------------------------------------------------------------------

class RootAnatomy(Organ):
    """
    Root cross-sectional anatomy.

    Calling ``RootAnatomy(input_data)`` transparently returns a
    ``MonocotRootAnatomy`` (planttype=1) or ``DicotRootAnatomy`` (planttype=2)
    instance.  Subclass directly if you always know the type.
    """

    # Roots: no ring smoothing — keep each peeled layer's thickness exact so
    # the innermost region (the stele) is not shrunk below its nominal size.
    LAYER_SMOOTH_FACTOR: float = 0.0

    # ------------------------------------------------------------------
    # Factory dispatch
    # ------------------------------------------------------------------

    def __new__(cls, input_data=None, seed=None):
        if cls is RootAnatomy:
            planttype = _get_planttype(input_data)
            actual_cls = DicotRootAnatomy if planttype == 2 else MonocotRootAnatomy
            return super().__new__(actual_cls)
        return super().__new__(cls)

    # ------------------------------------------------------------------
    # Initialisation — shared across both plant types
    # ------------------------------------------------------------------

    def __init__(self, input_data=None, seed=None):
        super().__init__(seed=seed)
        if isinstance(input_data, OrganInputData):
            self.params = input_data.to_dict_list()
        elif isinstance(input_data, list):
            self.params = input_data
        else:
            self.params = OrganInputData.for_root().to_dict_list()

        # Initialise secondary-growth dicts so they always exist
        self.secondary_xylem_params: dict = {}
        self.secondary_cambium_params: dict = {}
        self.secondary_phloem_params: dict = {}
        self.medullar_rays_params: dict = {}

        # Geometry shared from fit_secondary_xylem -> fit_secondary_phloem.
        # Vessel-zone angles + half-width position each phloem trapeze; medullar
        # rays are the thin walls that subdivide them.
        self._secondary_vessel_thetas: list = []
        self._secondary_vessel_half_width: float = 0.0
        self._secondary_medullar_thetas: list = []
        self._secondary_medullar_base_width: float = 0.0

        # Containers for vascular tissue building (populated by _create_vascular_tissue)
        self.vascular_cells: CellManager = CellManager()
        self.vascular_polygons: list = []

        self._parse_shared_params()
        self._parse_vascular_params()   # overridden per subclass
        self._initialize_default_layers()

    @property
    def planttype(self) -> int:
        """Plant type: 1 = monocot, 2 = dicot."""
        return int(self.global_params.get("value", 1))

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------

    def _parse_shared_params(self) -> None:
        """Parse the parameters that are common to all root types."""
        self.global_params = self._get_param("planttype")

        stele = self._get_param("stele")
        self.vascular_params = {
            "thickness":                stele["thickness"],
            "cell_diameter":            stele["cell_diameter"],
            "cell_diameter_center":     stele.get("cell_diameter_center", stele["cell_diameter"]),
            "size_gradient_function":   stele.get("size_gradient_function",   "five_pl"),
            "size_gradient_inflection": stele.get("size_gradient_inflection", 0.5),
            "size_gradient_steepness":  stele.get("size_gradient_steepness",  3.0),
            "size_gradient_asymmetry":  stele.get("size_gradient_asymmetry",  1.0),
        }

        self.intercellular_spaces_params = [p for p in self.params if p["name"] == "inter_cellular_spaces"]
        self.aerenchyma_params = self._get_param("aerenchyma")

        self.layers = [
            p for p in self.params
            if "order" in p and p["name"] not in ("stele", "xylem", "phloem", "aerenchyma")
        ]
        self.layers = sorted(self.layers, key=lambda x: float(x["order"]))

    def _parse_vascular_params(self) -> None:
        """Parse plant-type-specific vascular parameters. Overridden in subclasses."""
        pass

    def _initialize_default_layers(self) -> None:
        """Initialise root layers from parsed params."""
        for param in self.layers:
            self.layer_manager.add_layer(Layer.from_dict(param))

    # ------------------------------------------------------------------
    # Geometry — shared
    # ------------------------------------------------------------------

    def _create_base_shape(self) -> Polygon:
        radius = self._calculate_root_radius()
        shape_params = self._get_param("base_shape")
        kind = shape_params.get("shape", "circle")

        if kind == "circle":
            return GeometryProcessor.circle_polygon(radius)

        if kind == "star":
            # Star outline uses the same parameters as the xylem star.
            return GeometryProcessor.star_polygon(
                n_branches=int(shape_params.get("n_peaks", 5)),
                r_min=float(shape_params.get("inner_radius", 0.4)),
                r_max=float(shape_params.get("outer_radius", 0.6)),
                arc_base=float(shape_params.get("arc_bottom", 0.10)),
                arc_top=float(shape_params.get("arc_top", 0.05)),
            )

        # width/height define the bounding box; 0 (auto) falls back to the
        # auto-computed diameter so the shape matches the default circle's size.
        width = float(shape_params.get("width", 0.0)) or 2 * radius
        height = float(shape_params.get("height", 0.0)) or 2 * radius

        if kind == "ellipse":
            return GeometryProcessor.ellipse_to_polygon(0.0, 0.0, width / 2, height / 2, 0.0)
        if kind == "focus_ellipse":
            # Preferred: a measured contour ``profile`` (list of (major_pos,
            # minor_width) mm points) best-fitted to a single superellipse — no
            # exponent to hand-tune.  Major axis runs along +y (height).  Falls
            # back to the width/height bounding box (+ optional exponent) when no
            # profile is given.
            profile = shape_params.get("profile")
            if profile:
                semi_major, semi_minor, exponent = GeometryProcessor.fit_focus_ellipse(profile)
                return GeometryProcessor.focus_ellipse_polygon(
                    0.0, 0.0, semi_minor, semi_major, 0.0, exponent=exponent,
                )
            return GeometryProcessor.focus_ellipse_polygon(
                0.0, 0.0, width / 2, height / 2, 0.0,
                exponent=float(shape_params.get("exponent", 4.0)),
            )
        if kind == "square":
            return GeometryProcessor.rectangle_polygon(width, width)
        if kind == "rectangle":
            return GeometryProcessor.rectangle_polygon(width, height)
        if kind == "triangle":
            return GeometryProcessor.triangle_polygon(width, height)
        # Unknown shape — fall back to circle.
        return GeometryProcessor.circle_polygon(radius)

    def _calculate_root_radius(self) -> float:
        radius = self.vascular_params["thickness"] / 2
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, "n_layers"):
                radius += layer.get_total_thickness()
            elif hasattr(layer, "cell_diameter"):
                radius += layer.cell_diameter
        return radius

    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[LayerPolygon]:
        """Create stele parenchyma rings from the stele edge toward the centre."""
        central_layers = []

        diameter_fn = rescale(
            GRADIENT_FUNCTIONS[self.vascular_params["size_gradient_function"]],
            lo=self.vascular_params["cell_diameter"],
            hi=self.vascular_params["cell_diameter_center"],
            c=self.vascular_params["size_gradient_inflection"],
            b=self.vascular_params["size_gradient_steepness"],
            m=self.vascular_params["size_gradient_asymmetry"],
        )

        stele_radius = np.sqrt(current_polygon.area / np.pi)
        min_order = min(l.order for l in self.layer_manager.get_layers() if l.order > 0)
        space_increment = self.layer_manager.get_layer_by_order(min_order).cell_diameter / 2
        i_layer = len(params)

        while not current_polygon.is_empty and current_polygon.area > 0:
            r_norm = np.clip(np.sqrt(current_polygon.area / np.pi) / stele_radius, 0.0, 1.0)
            cell_diameter = diameter_fn(r_norm)

            if current_polygon.area <= (cell_diameter / 2) ** 2 * np.pi:
                break

            current_polygon = GeometryProcessor.buffer_polygon(
                current_polygon, -space_increment - cell_diameter / 2, smooth_factor=0.0,
            )
            space_increment = cell_diameter / 2

            central_layers.append(LayerPolygon(
                name="stele",
                polygon=current_polygon,
                cell_diameter=cell_diameter,
                id_layer=i_layer + 1,
            ))
            i_layer += 1

        return central_layers

    def reshape_layers(self, layers_polygons: List[LayerPolygon]) -> List[LayerPolygon]:
        return layers_polygons

    def set_vascular_params(self, **kwargs) -> None:
        self.vascular_params.update(kwargs)
        self._invalidate_geometry()

    def _which_layer_for_vascular(self, layers_polygons: List[LayerPolygon]):
        layer_for_vascular = [l["name"] for l in layers_polygons].index("stele")
        return layers_polygons[layer_for_vascular]["polygon"]

    def add_lateral_root_primordium(self, angle: float, distance: float) -> None:
        pass

    # ------------------------------------------------------------------
    # Shared star-xylem helpers (used by MonocotRootAnatomy star variant
    # AND DicotRootAnatomy)
    # ------------------------------------------------------------------

    def _xylem_star_region(self, stele_polygon: Polygon) -> Tissue:
        """Build the star-shaped xylem *region* (pure geometry).

        Shape-first: the star is assembled and clipped by region algebra — placed
        at the stele centre, clipped to the stele, with the pith subtracted.  Sets
        ``self.xylem_star`` (the region later steps carve phloem out of) and
        ``self.pith_polygon``.  Returns the xylem :class:`Tissue`; an *empty*
        result (the star did not fit) leaves ``self.xylem_star`` unset, so the
        downstream phloem valleys have nothing to carve against.
        """
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y

        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)
        outer_r = min(p["outer_radius_xylem"], stele_r)
        inner_r = min(p["inner_radius_xylem"], stele_r)
        pith_r  = max(0.0, min(p.get("pith_radius", 0.0), outer_r))

        # The size gradient runs radially from the stele centre, normalized over
        # [pith_r, outer_r] (i.e. outer_radius -> centre when there is no pith,
        # outer_radius -> pith_radius otherwise) so the innermost vessels reach
        # the full target diameter regardless of the pith.
        self._xylem_gradient_center = (cx, cy)
        self._xylem_gradient_radial_range = (pith_r, outer_r)

        raw_star = GeometryProcessor.star_polygon(
            n_branches=p["n_vascular_peak"],
            r_min=inner_r,
            r_max=outer_r,
            arc_base=p["arc_bottom_xylem"],
            arc_top=p["arc_top_xylem"],
        )
        star_coord = GeometryProcessor.smoothing_polygon(
            np.column_stack(raw_star.exterior.xy), smooth_factor=0.1, iterations=3,
        )

        xylem = (
            Tissue("xylem", Polygon(star_coord).buffer(0))
            .translate(cx, cy)
            .intersection(stele_polygon)
        )
        if xylem.is_empty:
            return xylem

        if pith_r > 0.0:
            self.pith_polygon = Point(cx, cy).buffer(pith_r)
            xylem.difference(self.pith_polygon)
        else:
            self.pith_polygon = None

        self.xylem_star = xylem.shape
        return xylem

    def _xylem_pack_kwargs(self) -> dict:
        """Circle-packing parameters for the star xylem fill (one source)."""
        p = self.vascular_params
        return dict(
            n_border=25, id_base=0, angle_center=None,
            min_diameter=p["xylem_diameter_min"], alt_type="stele",
            proportion=1.0, direction=p["xylem_direction"],
            diameter_max=p["xylem_diameter_max"],
            diameter_min=p["xylem_diameter_min"],
            diameter_sd=p["xylem_diameter_sd"],
            gradient_function=p["xylem_gradient_function"],
            gradient_inflection=p["xylem_gradient_inflection"],
            gradient_steepness=p["xylem_gradient_steepness"],
            gradient_asymmetry=p["xylem_gradient_asymmetry"],
            enforce_gradient_min=p["xylem_enforce_gradient_min"],
            allow_ellipse=p["xylem_allow_ellipse"],
            ellipse_max_aspect=p["xylem_ellipse_max_aspect"],
            pack_strategy=p["xylem_packing_strategy"],
            first_circle_shift=p["xylem_first_vessel_shift"],
            # Measure the gradient from the stele centre over [pith_r, outer_r]
            # so the pith is accounted for (set in _xylem_star_region).
            gradient_center=getattr(self, "_xylem_gradient_center", None),
            gradient_radial_range=getattr(self, "_xylem_gradient_radial_range", None),
        )

    def _record_xylem_vessels(self, tissue: Tissue, placed) -> None:
        """Record only the wide ("xylem") vessels in ``vascular_polygons``.

        The packing splits one region into wide vessels ("xylem") and narrow
        interstitial cells ("stele"); only the vessels feed the vascular mask and
        the metaxylem sheath / stele-clearing logic.
        """
        for placed_poly, rtype, _gid in placed:
            if rtype == "xylem":
                self.vascular_polygons.append(placed_poly)

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

    @staticmethod
    def _largest_polygon(geom):
        """Largest Polygon piece of a (possibly Multi)Polygon, or None."""
        if geom.is_empty:
            return None
        pieces = [g for g in (geom.geoms if hasattr(geom, "geoms") else [geom])
                  if g.geom_type == "Polygon" and not g.is_empty]
        return max(pieces, key=lambda g: g.area) if pieces else None

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
        next_id = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0
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
        next_id = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0

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

        next_id = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0
        for k in range(n):
            theta = 2.0 * np.pi * (k + 0.5) / n          # valley between poles k, k+1
            # A bounded ellipse cluster at the valley, clipped to the free band.
            raw = affine_scale(Point(0, 0).buffer(1, resolution=48), width / 2, height / 2)
            raw = rotate(raw, np.degrees(theta) - 90, origin=(0, 0))
            raw = translate(raw, cx + r_center * np.cos(theta), cy + r_center * np.sin(theta))
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

    def _remove_stele_engulfed_by_xylem(self, area_fraction: float = 0.75) -> None:
        """Drop stele cells whose footprint is mostly covered by xylem vessels.

        A *group-level* cleanup run before the unified point-level vascular mask.
        A stele cell is a ring of border seeds sharing an ``id_group`` (pre-Voronoi);
        its footprint is approximated by the convex hull of those seeds.  The whole
        cell is dropped when at least ``area_fraction`` (default 0.75) of that
        footprint lies inside the xylem vessel union.

        Cells below the threshold are kept whole; the point-level mask in
        ``Organ.generate_cells`` then clips their seeds that poke into a vessel, so
        they abut the vessels cleanly.  This is the middle ground between the old
        rule (drop the cell if *any* seed touched a vessel — over-removes) and the
        point-level mask alone (leaves distorted partial cells / slivers).
        """
        if not getattr(self, "xylem_star", None) or not self.vascular_polygons:
            return

        xylem_union     = unary_union(self.vascular_polygons)
        xylem_star_prep = prep(self.xylem_star)

        groups: Dict[Any, list] = defaultdict(list)
        for c in self.all_cells.cells:
            if c.type == "stele":
                groups[c.id_group].append(c)

        to_delete: set = set()
        for gid, cells in groups.items():
            # Cheap prefilter: only cells reaching into the xylem star can qualify.
            if not any(xylem_star_prep.contains(Point(c.x, c.y)) for c in cells):
                continue
            footprint = MultiPoint([(c.x, c.y) for c in cells]).convex_hull
            if footprint.area <= 0.0:
                # Degenerate footprint (a point/line of seeds) -> approximate as a
                # disc of the cell diameter so the fraction is still well defined.
                r = 0.5 * max((c.diameter for c in cells), default=0.0)
                if r <= 0.0:
                    continue
                footprint = footprint.centroid.buffer(r)
            if footprint.intersection(xylem_union).area / footprint.area >= area_fraction:
                to_delete.add(gid)

        if to_delete:
            self.all_cells.cells = [
                c for c in self.all_cells.cells
                if not (c.type == "stele" and c.id_group in to_delete)
            ]

    def _phloem_valley_zones(self, stele_polygon: Polygon, type="monocot"):
        """Build the phloem regions: one :class:`Tissue` per valley between peaks.

        Shape-first — each valley starts as a ``Tissue("phloem", ellipse)`` and is
        shaped purely by region algebra: clipped to the stele and carved out of
        the xylem star.  Returns ``(tissues, adjustment)``; the cells are filled
        in later by :meth:`fit_phloem_elements`.  ``adjustment`` is the clearance
        used when recording the region for the stele-clearing mask.
        """
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y
        n_peaks = p["n_vascular_peak"]

        width     = p["phloem_width"]
        height    = p["phloem_height"]
        cell_diam = p["phloem_diameter"]
        relative_distance = p["relative_phloem"]

        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)
        if type == "monocot":
            minimal_distance = min(p["inner_radius_xylem"], stele_r * 0.95)
        if type == "dicot":
            minimal_distance = min(p["cambium_primary_inner_distance"], stele_r * 0.95)
        adjustment = self._phloem_adjustment(type)

        r_center = (
            minimal_distance + adjustment + (height / 2)
            + (stele_r - adjustment - height - minimal_distance) * relative_distance
        )

        xylem_star = getattr(self, "xylem_star", None)
        min_area = np.pi * (cell_diam / 2) ** 2 * (1 - 0.0015)
        tissues = []
        for k in range(n_peaks):
            theta = 2 * np.pi * (k + 0.5) / n_peaks

            raw = Point(0, 0).buffer(1, resolution=64)
            raw = affine_scale(raw, width / 2, height / 2)
            raw = rotate(raw, np.degrees(theta) - 90, origin=(0, 0))
            raw = translate(raw, cx + r_center * np.cos(theta), cy + r_center * np.sin(theta))

            tissue = Tissue("phloem", raw).intersection(stele_polygon)
            if xylem_star is not None and not xylem_star.is_empty:
                tissue.difference(xylem_star)
            if tissue.is_empty or tissue.area < min_area:
                continue
            tissues.append(tissue)

        return tissues, adjustment

    def _phloem_adjustment(self, type="monocot") -> float:
        """Clearance buffer used when recording phloem regions for the stele mask.

        One source of truth for both the valley geometry (:meth:`_phloem_valley_zones`)
        and the mask recording (:meth:`_add_phloem_step`).
        """
        p = self.vascular_params
        if type == "dicot":
            return p.get("cambium_cell_diameter", 0.0)
        return p.get("cell_diameter", 0.0)

    def _add_phloem_step(self, recipe: TissueRecipe, stele_polygon: Polygon, type="monocot") -> None:
        """Declarative phloem step: valley *regions* filled by circle-packing.

        Shape-first vocabulary — the regions (:meth:`_phloem_valley_zones`, built
        lazily at recipe-build time because they are carved from the xylem star
        placed by an earlier step) are filled by the recipe's ``packing``
        strategy.  Each filled region is recorded (buffered by the phloem
        clearance) so the unified vascular mask in ``generate_cells`` clears the
        stele seeds underneath it.
        """
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y
        cell_diam = p["phloem_diameter"]
        cell_sd   = p["phloem_diameter_sd"]
        adjustment = self._phloem_adjustment(type)

        def record(tissue, _result, adj=adjustment):
            self.vascular_tissue_polygons.setdefault(tissue.tag, []).append(
                GeometryProcessor.buffer_polygon(tissue.shape, adj / 2)
            )

        recipe.fill_each(
            "phloem in valleys",
            lambda: self._phloem_valley_zones(stele_polygon, type)[0],
            strategy="packing", produces=("phloem",), record=record,
            n_border=25, angle_center=(cx, cy),
            proportion=1.0, direction=None,
            diameter_max=cell_diam, diameter_min=cell_diam,
            diameter_sd=cell_sd, gradient_function="normal",
        )

    # ------------------------------------------------------------------
    # Shared low-level rendering helper
    # ------------------------------------------------------------------

    def _render_layer(
        self,
        geometry,
        cell_type: str,
        cell_diam: float,
        cell_width: float,
        cx: float,
        cy: float,
        xylem_union=None,
    ) -> None:
        """Place cell seeds along a geometry (Polygon exterior or LineString/MultiLineString)."""
        fill_along(
            self.vascular_cells, geometry, cell_type, cell_diam, cell_width,
            cx, cy, xylem_union=xylem_union,
        )


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
            "xylem_shape":            str(xylem.get("xylem_shape", "default")),
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
        if self.vascular_params.get("n_vascular_bundles", 0) == 0:
            return recipe                       # no vascular bundles -> empty
        if self.vascular_params.get("xylem_shape", "default") == "arch":
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
    # Default ring-bundle methods
    # ------------------------------------------------------------------

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
            if i % 2 == 0:
                cells_in_slice, list_protoxylem_polygons = self.protoxylem_elements_in_slice(poly_slice)
                self.vascular_cells.extend_cells(cells_in_slice.cells)
                self.vascular_polygons.extend(list_protoxylem_polygons)
                self.protoxylem_polygons.extend(list_protoxylem_polygons)
            else:
                cells_in_slice, list_phloem_polygons = self.phloem_elements_in_slice(poly_slice)
                self.vascular_cells.extend_cells(cells_in_slice.cells)
                self.vascular_polygons.extend(list_phloem_polygons)
                self.phloem_polygons.extend(list_phloem_polygons)

    def protoxylem_elements_in_slice(self, slice_poly: Polygon):
        p = self.vascular_params
        list_polygons = []
        cells_in_slice = CellManager()

        bundle_cx, bundle_cy, available_r = GeometryProcessor.get_inscribed_circle(slice_poly)
        radial_angle_deg = np.degrees(np.arctan2(bundle_cy, bundle_cx))

        parent_r = max(p["protoxylem_width"], p["protoxylem_height"]) / 2
        scale    = min(1.0, available_r / parent_r) if parent_r > 0 else 1.0
        width    = p["protoxylem_width"]    * scale
        height   = p["protoxylem_height"]   * scale
        diameter = p["protoxylem_diameter"] * scale

        raw = Point(0, 0).buffer(1, resolution=64)
        raw = affine_scale(raw, width / 2, height / 2)
        raw = rotate(raw, radial_angle_deg - 90, origin=(0, 0))
        protoxylem = Tissue("protoxylem", translate(raw, bundle_cx, bundle_cy))
        if protoxylem.is_empty or protoxylem.area < np.pi * (diameter / 2) ** 2 * (1 - 0.0015):
            return cells_in_slice, list_polygons

        # No local stele removal here: the protoxylem region is appended to
        # vascular_polygons below and the unified vascular mask in
        # Organ.generate_cells() clears every layer seed inside it.
        next_id_group = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0
        fill_by_packing(
            cells_in_slice, protoxylem.shape, protoxylem.tag, rng=self.rng,
            n_border=24, id_base=next_id_group, angle_center=None,
            proportion=1.0, direction=None,
            diameter_max=diameter, diameter_sd=p["protoxylem_diameter_sd"] * scale,
            gradient_function="normal",
        )

        list_polygons.append(protoxylem.shape)
        return cells_in_slice, list_polygons

    def phloem_elements_in_slice(self, slice_poly: Polygon):
        p = self.vascular_params
        list_polygons = []
        cells_in_slice = CellManager()

        bundle_cx, bundle_cy, available_r = GeometryProcessor.get_inscribed_circle(slice_poly)
        radial_angle_deg = np.degrees(np.arctan2(bundle_cy, bundle_cx))

        parent_r = max(p["phloem_width"], p["phloem_height"]) / 2
        scale    = min(1.0, available_r / parent_r) if parent_r > 0 else 1.0
        width    = p["phloem_width"]    * scale
        height   = p["phloem_height"]   * scale
        diameter = p["phloem_diameter"] * scale

        raw = Point(0, 0).buffer(1, resolution=64)
        raw = affine_scale(raw, width / 2, height / 2)
        raw = rotate(raw, radial_angle_deg - 90, origin=(0, 0))
        phloem = Tissue("phloem", translate(raw, bundle_cx, bundle_cy))
        if phloem.is_empty or phloem.area < np.pi * (diameter / 2) ** 2 * (1 - 0.0015):
            return cells_in_slice, list_polygons

        next_id_group = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0
        fill_by_packing(
            cells_in_slice, phloem.shape, phloem.tag, rng=self.rng,
            n_border=24, id_base=next_id_group, angle_center=None,
            proportion=1.0, direction=None,
            diameter_max=diameter, diameter_sd=p["phloem_diameter_sd"] * scale,
            gradient_function="normal",
        )

        list_polygons.append(phloem.shape)
        return cells_in_slice, list_polygons


# ---------------------------------------------------------------------------
# Dicot subclass
# ---------------------------------------------------------------------------

class DicotRootAnatomy(RootAnatomy):
    """Dicot root: star-shaped xylem with cambium and phloem; optional secondary growth."""

    def _parse_vascular_params(self) -> None:
        xylem   = self._get_param("xylem")
        phloem  = self._get_param("phloem")
        cambium = self._get_param("cambium")

        self.vascular_params.update({
            "xylem_diameter_max":        float(xylem.get("vessel_diameter",      0.09)),
            "xylem_diameter_min":        float(xylem.get("vessel_diameter_min",  0.01)),
            "xylem_diameter_sd":         float(xylem.get("vessel_diameter_sd",   0.002)),
            "n_vascular_peak":           int(xylem.get("n_vascular_peak",        3)),
            "inner_radius_xylem":        float(xylem.get("inner_radius",         0.05)),
            "outer_radius_xylem":        float(xylem.get("outer_radius",         0.22)),
            "arc_top_xylem":             float(xylem.get("arc_top",              0.03)),
            "arc_bottom_xylem":          float(xylem.get("arc_bottom",           0.03)),
            "xylem_gradient_function":   str(xylem.get("gradient_function",      "five_pl")),
            "xylem_gradient_inflection": float(xylem.get("gradient_inflection",  0.7)),
            "xylem_gradient_steepness":  float(xylem.get("gradient_steepness",   5.0)),
            "xylem_gradient_asymmetry":  float(xylem.get("gradient_asymmetry",   1.0)),
            "xylem_enforce_gradient_min": float(xylem.get("enforce_gradient_min", 0.0)),
            "xylem_allow_ellipse":       bool(xylem.get("allow_ellipse",        False)),
            "xylem_ellipse_max_aspect":  float(xylem.get("ellipse_max_aspect",  2.0)),
            "xylem_packing_strategy":    str(xylem.get("packing_strategy",      "space")),
            "xylem_first_vessel_shift":  float(xylem.get("first_vessel_shift",   0.7)),
            "pith_radius":               float(xylem.get("pith_radius",          0.0)),
            "xylem_direction":           str(xylem.get("direction",              "center")),
            "phloem_diameter":           float(phloem.get("sieve_diameter",      0.005)),
            "phloem_diameter_sd":        float(phloem.get("sieve_diameter_sd",   0.001)),
            "phloem_width":              float(phloem.get("cluster_width",       0.15)),
            "phloem_height":             float(phloem.get("cluster_height",      0.2)),
            "relative_phloem":           float(phloem.get("relative_distance",   0.2)),
            "cambium_cell_diameter":     float(cambium.get("cell_diameter",      0.015)),
            "cambium_cell_width":        float(cambium.get("cell_width",         0.03)),
            "cambium_primary_inner_distance":   float(cambium.get("inner_distance",   0.10)),
            "cambium_primary_outer_distance":   float(cambium.get("outer_distance",   0.28)),
            "cambium_primary_visible_distance": float(cambium.get("visible_distance", 0.15)),
            "cambium_primary_arc_top":    float(cambium.get("arc_top",    0.1)),
            "cambium_primary_arc_bottom": float(cambium.get("arc_bottom", 0.07)),
        })

        sec_growth = self._get_param("secondary_growth")
        self.vascular_params["secondary_growth"] = bool(sec_growth.get("value", False))

        if self.vascular_params["secondary_growth"]:
            sec_xylem = self._get_param("secondary_xylem")
            sec_cam   = self._get_param("secondary_cambium")
            self.secondary_xylem_params = {
                "prop_stele":             float(sec_xylem.get("prop_stele",             0.5)),
                "cell_diameter":          float(sec_xylem.get("cell_diameter",          0.01)),
                "cell_width":             float(sec_xylem.get("cell_width",             0.01)),
                "vessel_diameter":        float(sec_xylem.get("vessel_diameter",        0.06)),
                "vessel_diameter_sd":     float(sec_xylem.get("vessel_diameter_sd",     0.005)),
                "vessel_diameter_min":    float(sec_xylem.get("vessel_diameter_min",    0.02)),
                "gradient_function":      str(sec_xylem.get("gradient_function",        "five_pl")),
                "gradient_inflection":    float(sec_xylem.get("gradient_inflection",    0.7)),
                "gradient_steepness":     float(sec_xylem.get("gradient_steepness",     5.0)),
                "gradient_asymmetry":     float(sec_xylem.get("gradient_asymmetry",     1.0)),
                "enforce_gradient_min":   float(sec_xylem.get("enforce_gradient_min",   0.0)),
                "allow_ellipse":          bool(sec_xylem.get("allow_ellipse",           False)),
                "ellipse_max_aspect":     float(sec_xylem.get("ellipse_max_aspect",     2.0)),
                "packing_strategy":       str(sec_xylem.get("packing_strategy",         "space")),
                "prop_vessel_ring":       float(sec_xylem.get("prop_vessel_ring",       0.5)),
                "n_ring":                 max(1, int(sec_xylem.get("n_ring",            1))),
                "must_be_adjacent":       bool(sec_xylem.get("must_be_adjacent",        False)),
                "parenchyma_diameter":    float(sec_xylem.get("parenchyma_diameter",    0.03)),
                "parenchyma_diameter_sd": float(sec_xylem.get("parenchyma_diameter_sd", 0.002)),
                "parenchyma_width":       float(sec_xylem.get("parenchyma_width",       0.01)),
            }
            self.secondary_cambium_params = {
                "cell_diameter":  float(sec_cam.get("cell_diameter",  0.015)),
                "cell_width":     float(sec_cam.get("cell_width",     0.025)),
                "inner_distance": float(sec_cam.get("inner_distance", 0.30)),
                "outer_distance": float(sec_cam.get("outer_distance", 0.45)),
                "arc_top":        float(sec_cam.get("arc_top",        0.05)),
                "arc_bottom":     float(sec_cam.get("arc_bottom",     0.07)),
                "n_layers":       max(1, int(sec_cam.get("n_layers",  1))),
                "shape":          str(sec_cam.get("shape", "star")),
                "profile":        list(sec_cam.get("profile", []) or []),
            }

            sec_phloem = self._get_param("secondary_phloem")
            self.secondary_phloem_params = {
                "height":              float(sec_phloem.get("height",              0.1)),
                "top_width":           float(sec_phloem.get("top_width",           0.3)),
                "alive_distance":      float(sec_phloem.get("alive_distance",      0.05)),
                "sieve_diameter":      float(sec_phloem.get("sieve_diameter",      0.015)),
                "sieve_diameter_sd":   float(sec_phloem.get("sieve_diameter_sd",   0.001)),
                "sieve_diameter_min":  float(sec_phloem.get("sieve_diameter_min",  0.008)),
                "prop_sieve":          float(sec_phloem.get("prop_sieve",          0.35)),
                "companion_diameter":  float(sec_phloem.get("companion_diameter",  0.008)),
                "companion_width":     float(sec_phloem.get("companion_width",     0.008)),
                "parenchyma_diameter": float(sec_phloem.get("parenchyma_diameter", 0.012)),
                "parenchyma_width":    float(sec_phloem.get("parenchyma_width",    0.012)),
            }

            med_rays = self._get_param("medullar_rays")
            self.medullar_rays_params = {
                "n_medullar":         int(med_rays.get("n_medullar",         0)),
                "base_width":         float(med_rays.get("base_width",       0.005)),
                "cell_diameter":      float(med_rays.get("cell_diameter",    0.025)),
                "cell_width":         float(med_rays.get("cell_width",       0.005)),
                "allow_non_vascular": bool(med_rays.get("allow_non_vascular", False)),
            }

    # ------------------------------------------------------------------
    # Vascular tissue
    # ------------------------------------------------------------------

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Declarative description of how a dicot stele is assembled.

        Built and run by the shared ``Organ._create_vascular_tissue`` scaffold.
        A shared prefix (star xylem + clearing the stele it engulfs) is followed
        by either primary tissue (phloem + primary cambium) or, when
        ``secondary_growth`` is on, secondary xylem + secondary phloem.  The
        build order is data, inspectable via ``recipe.describe()`` / ``plan()``.
        """
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        if self.vascular_params.get("n_vascular_peak", 0) == 0:
            return recipe                       # no xylem peaks -> empty
        recipe.fill("xylem star", self._xylem_star_region(polygon),
                    strategy="packing", produces=("xylem", "stele"),
                    record=self._record_xylem_vessels, **self._xylem_pack_kwargs())
        # Drop stele cells that are >=75% engulfed by xylem vessels (area-based);
        # cells below the threshold are kept whole and merely clipped against the
        # vessels by the unified point-level mask in Organ.generate_cells.
        recipe.cleanup("clear stele engulfed by xylem",
                       self._remove_stele_engulfed_by_xylem)
        if self.vascular_params.get("secondary_growth", False):
            # Secondary growth is bespoke (concentric ring fills + medullar rays +
            # companion cells), kept as `special` steps; see _build_*_polygon.
            recipe.special("secondary xylem",
                           lambda: self.fit_secondary_xylem(polygon),
                           produces=("xylem", "stele", "cambium", "medullar_ray"))
            recipe.special("secondary phloem",
                           lambda: self.fit_secondary_phloem(polygon),
                           produces=("phloem", "companion_cell", "stele"))
        else:
            self._add_phloem_step(recipe, polygon, type="dicot")
            # Primary cambium is a line fill along the star's visible arc (bespoke).
            recipe.special("primary cambium",
                           lambda: self.fit_primary_cambium_elements(polygon),
                           produces=("cambium",))
        return recipe

    # ------------------------------------------------------------------
    # Primary cambium
    # ------------------------------------------------------------------

    def fit_primary_cambium_elements(self, stele_polygon: Polygon):
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y

        inner_r = p["cambium_primary_inner_distance"]
        primary_arc_top    = p["cambium_primary_arc_top"]
        primary_arc_bottom = p["cambium_primary_arc_bottom"]
        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)

        outer_r = min(p["cambium_primary_outer_distance"], stele_r)
        inner_r = min(inner_r, outer_r)

        raw_star = GeometryProcessor.star_polygon(
            n_branches=p["n_vascular_peak"],
            r_min=inner_r, r_max=outer_r,
            arc_base=primary_arc_bottom, arc_top=primary_arc_top,
        )
        star_coord = GeometryProcessor.smoothing_polygon(
            np.column_stack(raw_star.exterior.xy), smooth_factor=0.9, iterations=5,
        )
        # The cambium region is the star polygon; its cells are seeded along the
        # visible arc of the star's exterior (a line fill, via _render_layer).
        cambium = Tissue("cambium", Polygon(star_coord).buffer(0)).translate(cx, cy)
        if cambium.is_empty:
            return

        clip_circle = Point(cx, cy).buffer(p["cambium_primary_visible_distance"])
        visible_boundary = cambium.shape.exterior.intersection(clip_circle)
        if visible_boundary.is_empty:
            return

        self.cambium_star = visible_boundary
        self.vascular_tissue_polygons.setdefault(cambium.tag, []).append(
            visible_boundary.buffer(p["cambium_cell_diameter"] / 2)
        )

        cell_diam  = p["cambium_cell_diameter"]
        cell_width = p["cambium_cell_width"]

        thin_ring = visible_boundary.buffer(cell_diam / 2)
        groups_to_delete = {
            c.id_group
            for c in self.all_cells.cells
            if c.type in ("stele", "pericycle") and thin_ring.intersects(Point(c.x, c.y))
        }
        self.all_cells.cells = [
            c for c in self.all_cells.cells
            if c.type not in ("stele", "pericycle") or c.id_group not in groups_to_delete
        ]

        # Remove any phloem cells that encroach on the cambium ring
        groups_to_delete = {
            c.id_group
            for c in self.vascular_cells.cells
            if c.type == "phloem" and thin_ring.intersects(Point(c.x, c.y))
        }
        self.vascular_cells.cells = [
            c for c in self.vascular_cells.cells
            if c.type != "phloem" or c.id_group not in groups_to_delete
        ]

        xylem_union = unary_union(self.vascular_polygons) if self.vascular_polygons else None
        self._render_layer(visible_boundary, cambium.tag, cell_diam, cell_width, cx, cy, xylem_union)

    # ------------------------------------------------------------------
    # Secondary growth helpers
    # ------------------------------------------------------------------

    def _build_primary_cambium_polygon(self, stele_polygon: Polygon, cx: float, cy: float) -> Polygon:
        p = self.vascular_params
        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)
        outer_r = min(p["cambium_primary_outer_distance"], stele_r)
        inner_r = min(p["cambium_primary_inner_distance"], outer_r)

        raw_star = GeometryProcessor.star_polygon(
            n_branches=p["n_vascular_peak"],
            r_min=inner_r, r_max=outer_r,
            arc_base=p["cambium_primary_arc_bottom"], arc_top=p["cambium_primary_arc_top"],
        )
        star_coords = GeometryProcessor.smoothing_polygon(
            np.column_stack(raw_star.exterior.xy), smooth_factor=0.9, iterations=5,
        )
        star = Polygon(star_coords).buffer(0)
        return translate(star, cx, cy).intersection(stele_polygon)

    def _build_secondary_cambium_polygon(self, stele_polygon: Polygon, cx: float, cy: float) -> Polygon:
        sc = self.secondary_cambium_params
        p  = self.vascular_params

        # 'focus_ellipse' contour: one smooth best-fit superellipse for a mature
        # ring-shaped cambium.  Axes come from the measured profile (semi-minor =
        # widest point, semi-major = tip) and a single exponent is least-squares
        # fitted to the interior points.  Sized in absolute mm (major axis along
        # +y) and clipped to the stele shape — no isotropic stele-radius clamp, so
        # it stays elliptical instead of collapsing to the stele's inscribed circle.
        if sc.get("shape", "star") == "focus_ellipse":
            semi_major, semi_minor, exponent = GeometryProcessor.fit_focus_ellipse(sc["profile"])
            contour = GeometryProcessor.focus_ellipse_polygon(
                0.0, 0.0, semi_minor, semi_major, 0.0, exponent=exponent,
            )
            return translate(contour, cx, cy).intersection(stele_polygon)

        outer_r = sc["outer_distance"]
        inner_r = min(sc["inner_distance"], outer_r)

        n_peaks = p["n_vascular_peak"]
        raw_star = GeometryProcessor.star_polygon(
            n_branches=n_peaks,
            r_min=inner_r, r_max=outer_r,
            arc_base=sc["arc_bottom"], arc_top=sc["arc_top"],
        )
        # Offset the secondary cambium star by half a period so its peaks fall in
        # the *valleys* of the primary xylem star (the secondary xylem vessel
        # zones sit at 2*pi*(k+0.5)/n_peaks).  Botanically the cambium in those
        # valleys produces secondary xylem and is pushed outward far more than the
        # cambium at the primary-xylem peaks, so the secondary cambium bulges
        # there.  For a diarch (n_peaks == 2) root this makes the secondary
        # cambium run perpendicular to the primary xylem.
        raw_star = rotate(raw_star, np.pi / n_peaks, origin=(0.0, 0.0), use_radians=True)
        star_coords = GeometryProcessor.smoothing_polygon(
            np.column_stack(raw_star.exterior.xy), smooth_factor=0.9, iterations=5,
        )
        star = Polygon(star_coords).buffer(0)
        return translate(star, cx, cy).intersection(stele_polygon)

    @staticmethod
    def _radial_strip(cx: float, cy: float, theta: float, width: float, r_outer: float) -> Polygon:
        """Constant-tangential-width radial strip centred on ``theta``.

        A rectangle of tangential width ``width`` running from the stele centre
        out to ``r_outer`` along the ``theta`` direction — the phloem-band
        footprint of a medullar ray, whose cells are filled at constant
        tangential ``base_width`` (so a rectangle matches the ray better than an
        angular wedge, which would taper).
        """
        strip = box(0.0, -width / 2.0, r_outer, width / 2.0)
        strip = rotate(strip, theta, origin=(0.0, 0.0), use_radians=True)
        return translate(strip, cx, cy)

    @staticmethod
    def _cambium_local_frame(cam_ext, cx: float, cy: float, theta: float, r_far: float):
        """Anchor point on the cambium at ``theta`` plus local (tangent, normal).

        The stomata principle (see ``CellGenerator.create_stomata``): orientation
        comes from the *local boundary tangent*, not the radial direction from the
        organ centre.  A ray cast from the centre at ``theta`` meets the cambium
        exterior (star-shaped about the centre, so a single hit) at ``P``; the
        tangent is read from the boundary either side of ``P`` and the outward
        normal is its perpendicular, flipped to point away from the centre.

        Returns ``(P, (tx, ty), (nx, ny))`` or ``None`` if the ray misses.
        """
        ray   = LineString([(cx, cy), (cx + r_far * np.cos(theta), cy + r_far * np.sin(theta))])
        inter = ray.intersection(cam_ext)
        if inter.is_empty:
            return None
        pts = [inter] if inter.geom_type == "Point" else [g for g in inter.geoms if g.geom_type == "Point"]
        if not pts:
            return None
        pt = max(pts, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)

        L   = cam_ext.length
        s   = cam_ext.project(pt)
        eps = max(L * 1e-3, 1e-5)
        a   = cam_ext.interpolate((s - eps) % L)
        b   = cam_ext.interpolate((s + eps) % L)
        tx, ty = b.x - a.x, b.y - a.y
        tn     = np.hypot(tx, ty) or 1.0
        tx, ty = tx / tn, ty / tn
        nx, ny = ty, -tx                       # perpendicular to the tangent
        if nx * (pt.x - cx) + ny * (pt.y - cy) < 0:
            nx, ny = -nx, -ny                  # make it point outward
        return (pt.x, pt.y), (tx, ty), (nx, ny)

    @staticmethod
    def _phloem_trapeze_curved(cam_ext, cx: float, cy: float, P, tangent, normal,
                               base_arc_half_width: float, top_width: float,
                               height: float, n: int = 40) -> Polygon:
        """Trapeze standing on the cambium at ``P``, oriented by the local tangent
        frame (the stomata principle) with its base **following the cambium curve**.

        The base is an arc of the cambium exterior spanning ``2*base_arc_half_width``
        of arc length centred on ``P`` (pushed slightly inward so the caller's band
        intersection clips cleanly); the arm then tapers to ``top_width`` at
        ``height`` along the outward local ``normal``.

        Curving the base along the contour — rather than the earlier straight
        chord along the local tangent — is what fixes the inverted taper: on a
        convex cambium a straight base chord bows out to a larger radius, so once
        clipped to the annular band the wide base landed at the *outer* edge
        (pinched at the cambium, wide at the tip).  The arm still leans along the
        local normal, so it is not forced to point at the stele centre.
        """
        (px, py), (tx, ty), (nx, ny) = P, tangent, normal
        L     = cam_ext.length
        s0    = cam_ext.project(Point(px, py))
        inset = height * 0.3

        # Curved base: sample the cambium exterior over ±base_arc_half_width of arc
        # length around P, each point pushed radially inward by ``inset`` so the
        # base sits just inside the band's inner edge.
        base = []
        for si in np.linspace(s0 - base_arc_half_width, s0 + base_arc_half_width, n):
            q = cam_ext.interpolate(si % L)
            dx, dy = q.x - cx, q.y - cy
            d = np.hypot(dx, dy) or 1.0
            base.append((q.x - dx / d * inset, q.y - dy / d * inset))

        hw = top_width / 2.0
        tcx, tcy = px + nx * (height + inset), py + ny * (height + inset)
        top_r = (tcx + tx * hw, tcy + ty * hw)   # +tangent (increasing-arc) side
        top_l = (tcx - tx * hw, tcy - ty * hw)   # -tangent side
        return Polygon(base + [top_r, top_l]).buffer(0)

    def _phloem_compartments(self) -> list:
        """``(center_theta, base_half_width)`` for each phloem compartment.

        A compartment is the angular sector between two consecutive rays inside a
        vessel zone — bounded by the parenchyma rays at the zone edges
        (``theta_v ± half_width``) and by any medullar rays in between.  Each one
        becomes its own tapering trapeze, so the phloem is split medullar↔medullar
        and medullar↔parenchyma rather than spanning the whole zone as one block.
        """
        hw = self._secondary_vessel_half_width
        comps = []
        for theta_v in self._secondary_vessel_thetas:
            # Medullar-ray offsets (signed, wrapped to ±pi) that fall in this zone.
            offsets = []
            for m in self._secondary_medullar_thetas:
                off = (m - theta_v + np.pi) % (2.0 * np.pi) - np.pi
                if abs(off) < hw:
                    offsets.append(off)
            offsets.sort()

            bounds = [-hw, *offsets, hw]
            for a, b in zip(bounds[:-1], bounds[1:]):
                comp_hw = (b - a) / 2.0
                if comp_hw > 1e-6:
                    comps.append((theta_v + (a + b) / 2.0, comp_hw))
        return comps

    def _fill_zone_with_cells(
        self,
        fill_zone,
        cell_diameter: float,
        cell_width: float,
        cell_type: str,
        cx: float,
        cy: float,
        start_id: int,
        erosion_polygon=None,
    ) -> int:
        """Fill a polygon zone with parenchyma seeds on concentric inward rings."""
        return fill_by_rings(
            self.vascular_cells, fill_zone, cell_diameter, cell_width, cell_type,
            cx, cy, start_id, erosion_polygon=erosion_polygon,
        )

    @staticmethod
    def _angular_wedge(cx: float, cy: float, theta_center: float,
                       half_angle: float, r_outer: float, n_arc: int = 50) -> Polygon:
        """Pie-wedge polygon: apex at ``(cx, cy)``, spanning ``theta_center ± half_angle``
        out to radius ``r_outer`` along ``n_arc`` arc points.

        The shared building block for the secondary-xylem vessel slices and the
        medullar-ray corridors (both then intersected with their annular zone).
        """
        arc_angles = np.linspace(theta_center - half_angle, theta_center + half_angle, n_arc)
        return Polygon([(cx, cy)] + [
            (cx + r_outer * np.cos(a), cy + r_outer * np.sin(a)) for a in arc_angles
        ])

    def _build_medullar_ray_polygons(
        self,
        annular_zone,
        vessel_zones: list,
        primary_cambium_polygon,
        cx: float,
        cy: float,
        r_outer_wedge: float,
        n_peaks: int,
        prop_stele: float,
        mr_params: dict,
    ) -> list:
        """Build wedge-shaped polygons for each medullar ray.

        When allow_non_vascular=False, rays are distributed evenly within the
        secondary xylem pizza slices (n_medullar / n_peaks per slice).
        When allow_non_vascular=True, rays are placed uniformly around the
        full circle (2π / n_medullar spacing) and span the full annular zone.

        The inner boundary comes from intersecting with the primary cambium
        polygon; no explicit r_inner is needed.

        Returns a list of (polygon, theta_c) tuples.
        """
        n_medullar = mr_params["n_medullar"]
        if n_medullar <= 0:
            return []

        base_width         = mr_params["base_width"]
        allow_non_vascular = mr_params["allow_non_vascular"]
        cambium_exterior   = primary_cambium_polygon.exterior

        if allow_non_vascular:
            # Uniform angular spacing around the full annular zone
            thetas    = [2.0 * np.pi * k / n_medullar for k in range(n_medullar)]
            clip_zone = annular_zone
        else:
            valid_zones = [z for z in vessel_zones if z is not None and not z.is_empty]
            if not valid_zones:
                return []
            clip_zone = unary_union(valid_zones)

            if prop_stele >= 1.0:
                # Full ring — distribute uniformly (same as allow_non_vascular=True)
                thetas = [2.0 * np.pi * k / n_medullar for k in range(n_medullar)]
            else:
                # Distribute n_medullar rays evenly across the n_peaks pizza slices
                full_angle  = 2.0 * np.pi / n_peaks
                half_width  = full_angle * prop_stele / 2.0
                rays_pp     = n_medullar // n_peaks      # rays per peak
                extra       = n_medullar % n_peaks       # first `extra` peaks get one more
                thetas = []
                for pk in range(n_peaks):
                    theta_zone = 2.0 * np.pi * (pk + 0.5) / n_peaks
                    n_r        = rays_pp + (1 if pk < extra else 0)
                    for j in range(n_r):
                        # Even spacing including the parenchyma rays at the zone
                        # edges: split the zone into n_r + 1 equal intervals and
                        # place a medullar ray at each internal node, so the
                        # medullar↔parenchyma gaps match the medullar↔medullar gap.
                        offset = (2.0 * (j + 1) / (n_r + 1) - 1.0) * half_width
                        thetas.append(theta_zone + offset)

        result = []
        for theta_c in thetas:
            ray_tip  = (cx + r_outer_wedge * np.cos(theta_c),
                        cy + r_outer_wedge * np.sin(theta_c))
            ray_line = LineString([(cx, cy), ray_tip])
            rim      = cambium_exterior.intersection(ray_line)

            if rim.is_empty:
                r_inner = max(np.hypot(x - cx, y - cy) for x, y in cambium_exterior.coords)
            elif rim.geom_type == "Point":
                r_inner = np.hypot(rim.x - cx, rim.y - cy)
            else:
                pts = [pt for pt in rim.geoms if pt.geom_type == "Point"]
                r_inner = (
                    max(np.hypot(pt.x - cx, pt.y - cy) for pt in pts)
                    if pts else
                    max(np.hypot(x - cx, y - cy) for x, y in cambium_exterior.coords)
                )

            half_angle = base_width / (2.0 * max(r_inner, 1e-9))
            raw_wedge  = self._angular_wedge(cx, cy, theta_c, half_angle, r_outer_wedge)
            poly       = raw_wedge.intersection(clip_zone)
            if not poly.is_empty:
                result.append((poly, theta_c))
        return result

    def _fill_medullar_rays(
        self,
        medullar_poly,
        theta_c: float,
        cx: float,
        cy: float,
        mr_params: dict,
        start_id: int,
    ) -> int:
        """Fill a medullar ray polygon with medullar_ray cells.

        The tangential width is held constant at base_width at every radius by
        recomputing the angular half-extent as base_width / (2 * r) at each
        step.  The number of lanes (= ceil(base_width / cell_width)) is fixed,
        so cells neither grow wider nor require lane splitting as radius increases.
        """
        if medullar_poly is None or medullar_poly.is_empty:
            return start_id

        d_cell     = mr_params["cell_diameter"]
        w_cell     = mr_params["cell_width"]
        base_width = mr_params["base_width"]
        n_lanes    = max(1, int(np.ceil(base_width / max(w_cell, 1e-9))))
        lane_width = base_width / n_lanes   # constant tangential width per lane

        geoms = list(medullar_poly.geoms) if hasattr(medullar_poly, "geoms") else [medullar_poly]

        n_border     = 25
        phi          = np.linspace(0.0, 2.0 * np.pi, n_border, endpoint=False)
        border_cos   = np.cos(phi)
        border_sin   = np.sin(phi)
        border_scale = 0.7

        next_id = start_id

        for geom in geoms:
            if geom.is_empty or geom.geom_type != "Polygon":
                continue
            geom_prep = prep(geom)

            radii   = [np.hypot(x - cx, y - cy) for x, y in geom.exterior.coords]
            r_inner = min(radii)
            r_outer = max(radii)

            # Start half a cell before the estimated inner boundary so the
            # first valid seed sits flush against the primary cambium edge
            # rather than leaving a gap that Voronoi inflates.
            r = max(r_inner - d_cell / 2.0, d_cell / 2.0)
            while r <= r_outer:
                # Recompute angular range at each r to keep arc width = base_width
                half_angle_r = base_width / (2.0 * r)
                theta_lo_r   = theta_c - half_angle_r
                theta_hi_r   = theta_c + half_angle_r

                for lane in range(n_lanes):
                    theta_mid = theta_lo_r + (lane + 0.5) * (theta_hi_r - theta_lo_r) / n_lanes
                    px = cx + r * np.cos(theta_mid)
                    py = cy + r * np.sin(theta_mid)

                    if not geom_prep.contains(Point(px, py)):
                        continue

                    a_rad    = d_cell * 0.5 * border_scale
                    b_tan    = lane_width * 0.5 * border_scale
                    cos_t, sin_t = np.cos(theta_mid), np.sin(theta_mid)

                    id_group = next_id
                    next_id += 1

                    for j in range(n_border):
                        er = a_rad * border_cos[j]
                        et = b_tan * border_sin[j]
                        self.vascular_cells.add_cell(Cell(
                            type="medullar_ray",
                            x=px + er * cos_t - et * sin_t,
                            y=py + er * sin_t + et * cos_t,
                            diameter=d_cell,
                            id_cell=id_group, id_group=id_group,
                            angle=theta_mid, radius=r,
                            area=np.pi * a_rad * b_tan,
                        ))

                r += d_cell

        return next_id

    def _fill_ray_parenchyma(
        self,
        vessel_zones: list,
        annular_zone,
        cx: float,
        cy: float,
        sx: dict,
        r_outer: float,
        n_peaks: int,
        start_id: int,
    ) -> int:
        """Fill angular gaps between pizza slices with radially-oriented ray parenchyma."""
        valid_zones = [z for z in vessel_zones if z is not None and not z.is_empty]
        if not valid_zones or annular_zone is None or annular_zone.is_empty:
            return start_id
        if r_outer <= 0.0:
            return start_id

        d_cell          = sx["parenchyma_diameter"]
        w_cell          = sx["parenchyma_width"]
        sd_cell         = sx["parenchyma_diameter_sd"]
        split_threshold = w_cell + 3.0 * sd_cell
        prop_stele      = sx["prop_stele"]

        full_angle = 2.0 * np.pi / n_peaks
        half_slice = full_angle * prop_stele / 2.0
        gap_half   = full_angle / 2.0 - half_slice
        if gap_half <= 0.0:
            return start_id

        zones_union = unary_union(valid_zones)
        ray_zone    = annular_zone.difference(zones_union)
        if ray_zone.is_empty:
            return start_id

        n_border   = 15
        phi        = np.linspace(0.0, 2.0 * np.pi, n_border, endpoint=False)
        border_cos = np.cos(phi)
        border_sin = np.sin(phi)
        border_scale = 0.7

        next_id = start_id

        for k in range(n_peaks):
            theta_c  = 2.0 * np.pi * k / n_peaks
            theta_lo = theta_c - gap_half
            theta_hi = theta_c + gap_half

            r_start = max(self.vascular_params["cambium_primary_inner_distance"], d_cell)
            init_spacing = w_cell / r_start
            n_init = max(1, int(np.ceil((theta_hi - theta_lo) / init_spacing)))
            lines  = list(np.linspace(theta_lo, theta_hi, n_init + 1))
            thresholds = [
                float(np.clip(
                    self.rng.uniform(0.7, 1.3) * split_threshold,
                    0.5 * split_threshold, 1.5 * split_threshold,
                ))
                for _ in range(len(lines) - 1)
            ]

            r = r_start + d_cell / 2.0
            while r <= r_outer:
                new_lines      = [lines[0]]
                new_thresholds = []
                noise_scale    = 0.1 * split_threshold
                for i in range(len(lines) - 1):
                    a1, a2 = lines[i], lines[i + 1]
                    thr = thresholds[i]
                    if (a2 - a1) * r > thr:
                        new_lines.append((a1 + a2) / 2.0)
                        t_left = float(np.clip(
                            thr + self.rng.normal(0, noise_scale),
                            0.5 * split_threshold, 1.5 * split_threshold,
                        ))
                        t_right = float(np.clip(
                            thr + self.rng.normal(0, noise_scale),
                            0.5 * split_threshold, 1.5 * split_threshold,
                        ))
                        new_thresholds.extend([t_left, t_right])
                    else:
                        new_thresholds.append(thr)
                    new_lines.append(a2)
                lines      = sorted(new_lines)
                thresholds = new_thresholds

                for i in range(len(lines) - 1):
                    theta_mid      = (lines[i] + lines[i + 1]) / 2.0
                    lane_arc_width = (lines[i + 1] - lines[i]) * r
                    px = cx + r * np.cos(theta_mid)
                    py = cy + r * np.sin(theta_mid)

                    if not ray_zone.contains(Point(px, py)):
                        continue

                    a_rad = d_cell * 0.5 * border_scale
                    b_tan = lane_arc_width * 0.5 * border_scale
                    cos_t, sin_t = np.cos(theta_mid), np.sin(theta_mid)

                    id_group = next_id
                    next_id += 1

                    for j in range(n_border):
                        er = a_rad * border_cos[j]
                        et = b_tan * border_sin[j]
                        self.vascular_cells.add_cell(Cell(
                            type="stele",
                            x=px + er * cos_t - et * sin_t,
                            y=py + er * sin_t + et * cos_t,
                            diameter=d_cell,
                            id_cell=id_group, id_group=id_group,
                            angle=theta_mid, radius=r,
                            area=np.pi * a_rad * b_tan,
                        ))

                r += d_cell

        return next_id

    def fit_secondary_xylem(self, stele_polygon: Polygon) -> None:
        """Build secondary xylem between the primary and secondary cambium."""
        p   = self.vascular_params
        sx  = self.secondary_xylem_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y
        n_peaks = p["n_vascular_peak"]

        primary_cambium_polygon = self._build_primary_cambium_polygon(stele_polygon, cx, cy)
        if primary_cambium_polygon is None or primary_cambium_polygon.is_empty:
            return

        secondary_cambium_polygon = self._build_secondary_cambium_polygon(stele_polygon, cx, cy)
        if secondary_cambium_polygon is None or secondary_cambium_polygon.is_empty:
            return

        sc = self.secondary_cambium_params
        # Cambial zone: n_layers concentric cell files, each the cambium polygon
        # buffered inward by one cell diameter.  n_layers == 1 -> a single ring.
        for k in range(sc.get("n_layers", 1)):
            ring = (secondary_cambium_polygon if k == 0
                    else secondary_cambium_polygon.buffer(-k * sc["cell_diameter"]))
            if ring.is_empty:
                break
            for g in (ring.geoms if hasattr(ring, "geoms") else [ring]):
                if g.geom_type == "Polygon" and not g.is_empty:
                    self._render_layer(g, "cambium", sc["cell_diameter"], sc["cell_width"], cx, cy)
        self.vascular_tissue_polygons.setdefault("cambium", []).append(secondary_cambium_polygon)

        # Reserve the full cambial band for the cambium files.  The n_layers
        # concentric files occupy the outer ``n_layers * cell_diameter`` shell of
        # the secondary cambium (buffered inward, above).  The secondary xylem
        # (vessels + axial parenchyma) must stop at the innermost file, otherwise
        # xylem parenchyma packs on top of the inner cambium layers.
        cambium_band_depth = sc.get("n_layers", 1) * sc["cell_diameter"]
        xylem_boundary = secondary_cambium_polygon.buffer(-cambium_band_depth)
        if xylem_boundary.is_empty:
            xylem_boundary = secondary_cambium_polygon

        annular_zone = xylem_boundary.difference(primary_cambium_polygon)
        shrinked_sec_cambium_pol = GeometryProcessor.buffer_polygon(
            secondary_cambium_polygon, +sc["cell_diameter"] / 1.5, 0
        )
        buffed_annular_zone = shrinked_sec_cambium_pol.difference(primary_cambium_polygon)

        if buffed_annular_zone.is_empty:
            return

        # Remove stele parenchyma seeds from the annular zone (secondary growth replaces them)
        self.all_cells.remove_cells_in_polygon(buffed_annular_zone)

        full_angle_per_slice = 2.0 * np.pi / n_peaks
        half_width = full_angle_per_slice * sx["prop_stele"] / 2.0

        # Vessel-zone geometry shared with fit_secondary_phloem: each phloem arm is
        # a trapeze behind a vessel zone, centred on the valley angle
        # 2*pi*(k+0.5)/n_peaks with the same angular half-width.  When prop_stele
        # >= 1 the vessel zone fills the whole annulus (no discrete arms).
        self._secondary_vessel_half_width = half_width
        self._secondary_vessel_thetas = (
            [2.0 * np.pi * (k + 0.5) / n_peaks for k in range(n_peaks)]
            if sx["prop_stele"] < 1.0 else []
        )

        # Outer radius used for wedge construction (large enough to always cover the annular zone)
        minx, miny, maxx, maxy = secondary_cambium_polygon.bounds
        r_outer_wedge = max(maxx - cx, cx - minx, maxy - cy, cy - miny) * 1.5

        vessel_zones: List = []
        if sx["prop_stele"] >= 1.0:
            if not annular_zone.is_empty and annular_zone.area >= np.pi * (sx["vessel_diameter_min"] / 2) ** 2:
                vessel_zones.append(annular_zone)
        else:
            for k in range(n_peaks):
                theta = 2.0 * np.pi * (k + 0.5) / n_peaks
                if half_width < 1e-9:
                    vessel_zones.append(None)
                    continue
                raw_wedge = self._angular_wedge(cx, cy, theta, half_width, r_outer_wedge)
                zone = raw_wedge.intersection(annular_zone)
                if zone.is_empty or zone.area < np.pi * (sx["vessel_diameter_min"] / 2) ** 2:
                    vessel_zones.append(None)
                    continue
                if zone.geom_type == "Polygon":
                    zone_coords = GeometryProcessor.smoothing_polygon(
                        np.column_stack(zone.exterior.xy), smooth_factor=0.3, iterations=3,
                    )
                    smoothed = Polygon(zone_coords).buffer(0)
                    if not smoothed.is_empty and smoothed.geom_type == "Polygon":
                        zone = smoothed
                vessel_zones.append(zone)

        # Build medullar ray polygons BEFORE pack_circles so they can cut the vessel zones
        mr_params = self.medullar_rays_params
        medullar_ray_polys = []
        medullar_union = None
        if mr_params.get("n_medullar", 0) > 0:
            medullar_ray_polys = self._build_medullar_ray_polygons(
                annular_zone, vessel_zones, primary_cambium_polygon,
                cx, cy, r_outer_wedge, n_peaks, sx["prop_stele"], mr_params,
            )
            # Share medullar-ray angles + width with fit_secondary_phloem
            # (they are the thin walls between phloem trapezes).
            self._secondary_medullar_thetas = [theta for _, theta in medullar_ray_polys]
            self._secondary_medullar_base_width = mr_params.get("base_width", 0.0)
            if medullar_ray_polys:
                all_mr_geoms = []
                for poly, _ in medullar_ray_polys:
                    if hasattr(poly, "geoms"):
                        all_mr_geoms.extend(g for g in poly.geoms if not g.is_empty)
                    elif not poly.is_empty:
                        all_mr_geoms.append(poly)
                if all_mr_geoms:
                    medullar_union = unary_union(all_mr_geoms)
                    # Remove cambium seeds that fall inside the medullar ray corridors.
                    mr_cambium_zone = prep(medullar_union.buffer(sc["cell_diameter"]))
                    self.vascular_cells.cells = [
                        c for c in self.vascular_cells.cells
                        if not (c.type == "cambium" and mr_cambium_zone.contains(Point(c.x, c.y)))
                    ]

        all_vessel_polys: List[Polygon] = []
        next_id = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0

        # Annual growth rings: divide the secondary-xylem annulus radially into
        # n_ring bands that follow the secondary-cambium contour (buffered inward
        # in equal steps). 
        n_ring = sx.get("n_ring", 1)
        annual_bands = None
        if n_ring > 1:
            _, _, sc_r = GeometryProcessor._chebyshev_center(secondary_cambium_polygon)
            _, _, pc_r = GeometryProcessor._chebyshev_center(primary_cambium_polygon)
            step = max(sc_r - pc_r, 0.0) / n_ring
            if step > 0:
                annual_bands = []
                prev = secondary_cambium_polygon
                for k in range(1, n_ring):
                    inner_contour = secondary_cambium_polygon.buffer(-k * step)
                    annual_bands.append(
                        (prev.difference(inner_contour), sc_r - k * step, sc_r - (k - 1) * step)
                    )
                    prev = inner_contour
                annual_bands.append((prev, pc_r, sc_r - (n_ring - 1) * step))

        for original_zone in vessel_zones:
            if original_zone is None or original_zone.is_empty:
                continue

            # Split the vessel zone by medullar rays; each fragment is filled independently
            if medullar_union is not None and not medullar_union.is_empty:
                remaining = original_zone.difference(medullar_union)
            else:
                remaining = original_zone

            if remaining.is_empty:
                sub_zones = []
            elif hasattr(remaining, "geoms"):
                sub_zones = [g for g in remaining.geoms if g.geom_type == "Polygon" and not g.is_empty]
            else:
                sub_zones = [remaining] if remaining.geom_type == "Polygon" else []

            for zone in sub_zones:
                # Split the fragment into annual-ring bands (or keep it whole when
                # n_ring == 1); each band is packed + filled independently.
                if annual_bands is None:
                    band_pieces = [(zone, None)]
                else:
                    band_pieces = []
                    for band_poly, r_in_b, r_out_b in annual_bands:
                        zb = zone.intersection(band_poly)
                        if zb.is_empty:
                            continue
                        for g in (zb.geoms if hasattr(zb, "geoms") else [zb]):
                            if g.geom_type == "Polygon" and not g.is_empty:
                                band_pieces.append((g, (r_in_b, r_out_b)))

                for piece, grr in band_pieces:
                    packed = GeometryProcessor.pack_circles(
                        piece,
                        proportion=sx["prop_vessel_ring"],
                        direction="center",
                        diameter_max=sx["vessel_diameter"],
                        diameter_min=sx["vessel_diameter_min"],
                        diameter_sd=sx["vessel_diameter_sd"],
                        gradient_function=sx["gradient_function"],
                        gradient_inflection=sx["gradient_inflection"],
                        gradient_steepness=sx["gradient_steepness"],
                        gradient_asymmetry=sx["gradient_asymmetry"],
                        enforce_gradient_min=sx["enforce_gradient_min"],
                        allow_ellipse=sx["allow_ellipse"],
                        ellipse_max_aspect=sx["ellipse_max_aspect"],
                        pack_strategy=sx["packing_strategy"],
                        adjacent=sx["must_be_adjacent"],
                        gradient_center=(cx, cy),
                        gradient_radial_range=grr,
                        rng=self.rng,
                    )
                    # Seed one vessel per packed circle — the shared pack-and-seed verb.
                    placed_out = place_packed_group(
                        self.vascular_cells, packed, "xylem",
                        n_border=25, id_base=next_id, angle_center=None,
                    )
                    next_id += len(packed)
                    zone_vessel_polys: List[Polygon] = [placed for placed, _t, _g in placed_out]
                    all_vessel_polys.extend(zone_vessel_polys)

                    if zone_vessel_polys:
                        vessel_union_in_zone = unary_union(zone_vessel_polys)
                        axial_zone = piece.difference(vessel_union_in_zone)
                    else:
                        axial_zone = piece

                    erosion_poly = secondary_cambium_polygon if sx["prop_stele"] >= 1.0 else piece
                    next_id = self._fill_zone_with_cells(
                        axial_zone, sx["cell_diameter"], sx["cell_width"], "stele",
                        cx, cy, next_id, erosion_polygon=erosion_poly,
                    )

        r_outer = max(
            np.hypot(x - cx, y - cy)
            for x, y in secondary_cambium_polygon.exterior.coords
        ) - cambium_band_depth

        ray_annular_zone = xylem_boundary.difference(primary_cambium_polygon)
        # Only exclude medullar areas from the ray zone when they can extend
        # into the gap region (allow_non_vascular=True).  When False, medullar
        # rays are fully inside the vessel pizza slices
        if (medullar_union is not None and not medullar_union.is_empty
                and mr_params.get("allow_non_vascular", False)):
            # Small outward buffer prevents ray-parenchyma seeds from landing
            # right on the corridor wall
            mr_exclusion = medullar_union.buffer(mr_params.get("cell_diameter", 0.025) / 2.0)
            ray_annular_zone = ray_annular_zone.difference(mr_exclusion)

        if not ray_annular_zone.is_empty:
            next_id = self._fill_ray_parenchyma(
                vessel_zones, ray_annular_zone, cx, cy, sx, r_outer, n_peaks, next_id,
            )

        # Fill medullar ray zones with medullar_ray cells
        for poly, theta_c in medullar_ray_polys:
            next_id = self._fill_medullar_rays(poly, theta_c, cx, cy, mr_params, next_id)

        self.vascular_polygons.extend(all_vessel_polys)

    def fit_secondary_phloem(self, stele_polygon: Polygon) -> None:
        """Build secondary phloem as tapering trapezes outside the cambium.

        The phloem occupies a band that follows the secondary cambium contour
        (the cambium polygon buffered outward by ``height``).  Within the band,
        one tapering trapeze sits in each *compartment* — the angular sector
        between two consecutive rays (medullar↔medullar or medullar↔parenchyma).
        Each trapeze has a wide base at the cambium narrowing to ``top_width`` at
        the band's outer edge; the thin medullar-ray strips then separate the
        compartment bases.  Every resulting arm is split radially into an alive
        sub-zone (sieve tubes + companion cells + parenchyma) near the cambium
        and a dead sub-zone (sieve tubes + parenchyma) beyond ``alive_distance``.
        """
        sp = self.secondary_phloem_params

        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y

        secondary_cambium_polygon = self._build_secondary_cambium_polygon(stele_polygon, cx, cy)
        if secondary_cambium_polygon is None or secondary_cambium_polygon.is_empty:
            return

        # Band following the cambium contour, buffered outward by the phloem height.
        band = secondary_cambium_polygon.buffer(sp["height"]).difference(secondary_cambium_polygon)
        if band.is_empty:
            return

        # Outer radius large enough for the medullar strips to always cross the band.
        minx, miny, maxx, maxy = band.bounds
        r_outer = max(maxx - cx, cx - minx, maxy - cy, cy - miny) * 1.5

        # One tapering trapeze per compartment (between consecutive rays), each
        # standing perpendicular on the cambium surface
        cam_ext = secondary_cambium_polygon.exterior
        if self._secondary_vessel_thetas:
            masks = []
            for center, comp_hw in self._phloem_compartments():
                frame = self._cambium_local_frame(cam_ext, cx, cy, center, r_outer)
                if frame is None:
                    continue
                P, tangent, normal = frame
                r_P     = np.hypot(P[0] - cx, P[1] - cy)
                base_hw = comp_hw * r_P                       # arc-length half-width
                top_w   = min(sp["top_width"], 2.0 * base_hw)
                masks.append(self._phloem_trapeze_curved(
                    cam_ext, cx, cy, P, tangent, normal, base_hw, top_w, sp["height"],
                ))
            arms = band.intersection(unary_union(masks)) if masks else band
        else:
            arms = band

        # Subtract the thin medullar-ray strips that subdivide each trapeze.
        if self._secondary_medullar_thetas and self._secondary_medullar_base_width > 0:
            strips = [
                self._radial_strip(cx, cy, theta, self._secondary_medullar_base_width, r_outer)
                for theta in self._secondary_medullar_thetas
            ]
            arms = arms.difference(unary_union(strips))

        if arms.is_empty:
            return

        self.vascular_tissue_polygons.setdefault("secondary_phloem", []).append(arms)

        # Radially split into alive (near cambium) and dead (outer) sub-zones.
        alive_annulus = secondary_cambium_polygon.buffer(sp["alive_distance"])
        alive_zone    = arms.intersection(alive_annulus)
        dead_zone     = arms.difference(alive_annulus)

        next_id = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0

        next_id = self._fill_phloem_zone(alive_zone, alive=True,  cx=cx, cy=cy, sp=sp, start_id=next_id)
        next_id = self._fill_phloem_zone(dead_zone,  alive=False, cx=cx, cy=cy, sp=sp, start_id=next_id)

    def _fill_phloem_zone(
        self,
        zone,
        alive: bool,
        cx: float,
        cy: float,
        sp: dict,
        start_id: int,
    ) -> int:
        """Pack sieve tubes (+ companion cells when alive=True) then fill parenchyma."""
        if zone is None or zone.is_empty:
            return start_id

        sub_zones = (
            [g for g in zone.geoms if g.geom_type == "Polygon" and not g.is_empty]
            if hasattr(zone, "geoms")
            else ([zone] if zone.geom_type == "Polygon" else [])
        )

        min_area = np.pi * (sp["sieve_diameter_min"] / 2) ** 2
        next_id  = start_id

        for arm_zone in sub_zones:
            if arm_zone.is_empty or arm_zone.area < min_area:
                continue

            packed = GeometryProcessor.pack_circles(
                arm_zone,
                proportion=sp["prop_sieve"],
                direction=None,
                diameter_max=sp["sieve_diameter"],
                diameter_min=sp["sieve_diameter_min"],
                diameter_sd=sp["sieve_diameter_sd"],
                gradient_function="normal",
                rng=self.rng,
            )

            sieve_polys:    list = []
            sieve_centers:  list = []   # (pcx, pcy, r) of each placed sieve
            companion_polys: list = []

            # Pass 1 — seed every sieve element.
            for pcx, pcy, r in packed:
                actual_diam = r * 2
                placed      = Point(pcx, pcy).buffer(r, resolution=32)
                placed_buff = placed.buffer(-r * 0.15)
                if placed_buff.is_empty:
                    continue

                bx, by = placed_buff.exterior.coords.xy
                border_coords = GeometryProcessor.resample_coords(
                    np.column_stack((bx, by)), target_n_points=25
                )
                id_group = next_id
                next_id += 1
                for border_pt in border_coords[1:]:
                    self.vascular_cells.add_cell(Cell.radial(
                        "phloem", border_pt[0], border_pt[1], actual_diam, id_group, (cx, cy),
                    ))
                sieve_polys.append(placed)
                sieve_centers.append((pcx, pcy, r))

            # Pass 2 — one companion cell beside each sieve (alive only).  Built
            # after all sieves so it can be rejected if it overlaps ANY sieve (not
            # just the ones placed so far), which is what produced companion cells
            # nested inside a neighbouring sieve.
            if alive and sieve_polys:
                sieve_union = unary_union(sieve_polys)
                for pcx, pcy, r in sieve_centers:
                    comp_r    = sp["companion_diameter"] / 2
                    theta_rad = np.arctan2(pcy - cy, pcx - cx)
                    for side in (1, -1):
                        ccx = pcx + (r + comp_r * 1.05) * np.cos(theta_rad + side * np.pi / 2)
                        ccy = pcy + (r + comp_r * 1.05) * np.sin(theta_rad + side * np.pi / 2)
                        comp_pt = Point(ccx, ccy)
                        if not arm_zone.contains(comp_pt):
                            continue
                        comp_circle = comp_pt.buffer(comp_r)
                        if comp_circle.intersects(sieve_union):
                            continue   # would overlap a sieve -> would nest inside it
                        if any(comp_circle.intersects(c) for c in companion_polys):
                            continue
                        comp_buff = comp_circle.buffer(-comp_r * 0.15)
                        if comp_buff.is_empty:
                            continue
                        bx, by = comp_buff.exterior.coords.xy
                        border_c = GeometryProcessor.resample_coords(
                            np.column_stack((bx, by)), target_n_points=16
                        )
                        id_group = next_id
                        next_id += 1
                        for border_pt in border_c[1:]:
                            self.vascular_cells.add_cell(Cell.radial(
                                "companion_cell", border_pt[0], border_pt[1],
                                sp["companion_diameter"], id_group, (cx, cy),
                            ))
                        companion_polys.append(comp_circle)
                        break   # one companion per sieve

            placed_union = unary_union(sieve_polys + companion_polys) if (sieve_polys or companion_polys) else Polygon()
            fill_zone    = arm_zone.difference(placed_union)
            if not fill_zone.is_empty:
                next_id = self._fill_zone_with_cells(
                    fill_zone,
                    sp["parenchyma_diameter"], sp["parenchyma_width"],
                    "stele", cx, cy, next_id,
                    erosion_polygon=arm_zone,
                )

        return next_id
