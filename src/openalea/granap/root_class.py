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
            # Lazy imports: the subclasses live in their own modules and import
            # RootAnatomy from here, so importing them at module load time would
            # be circular.  Resolving them at call time avoids that.
            from openalea.granap.root_monocot_class import MonocotRootAnatomy
            from openalea.granap.root_dicot_class import DicotRootAnatomy
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

    @staticmethod
    def _largest_polygon(geom):
        """Largest Polygon piece of a (possibly Multi)Polygon, or None."""
        if geom.is_empty:
            return None
        pieces = [g for g in (geom.geoms if hasattr(geom, "geoms") else [geom])
                  if g.geom_type == "Polygon" and not g.is_empty]
        return max(pieces, key=lambda g: g.area) if pieces else None

    @staticmethod
    def _oriented_ellipse(tx: float, ty: float, width: float, height: float,
                          angle_deg: float, resolution: int = 64) -> Polygon:
        """Axis-aligned unit disc scaled to ``width``x``height``, rotated so its
        major (height) axis points along ``angle_deg`` (minus the 90° that maps the
        +y major axis to the radial direction), then translated to ``(tx, ty)``.

        The one source for every oriented vascular cluster ellipse (phloem valleys,
        arch phloem, proto/phloem bundles).
        """
        raw = Point(0, 0).buffer(1, resolution=resolution)
        raw = affine_scale(raw, width / 2, height / 2)
        raw = rotate(raw, angle_deg - 90, origin=(0, 0))
        return translate(raw, tx, ty)

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
# Backward-compatible re-exports
# ---------------------------------------------------------------------------
# The concrete subclasses moved to their own modules (``root_monocot_class`` /
# ``root_dicot_class``).  Expose them here lazily (PEP 562 module __getattr__)
# so existing ``from openalea.granap.root_class import DicotRootAnatomy`` keeps
# working without a circular import at load time.

def __getattr__(name):
    if name == "MonocotRootAnatomy":
        from openalea.granap.root_monocot_class import MonocotRootAnatomy
        return MonocotRootAnatomy
    if name == "DicotRootAnatomy":
        from openalea.granap.root_dicot_class import DicotRootAnatomy
        return DicotRootAnatomy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
