"""
Root anatomy implementation.

`RootAnatomy` acts as a transparent factory: calling ``RootAnatomy(input_data)``
returns either a ``MonocotRootAnatomy`` or a ``DicotRootAnatomy`` instance
depending on the ``planttype`` value in the input.  Both subclasses are
``isinstance(obj, RootAnatomy)`` == True, so all existing code keeps working.
"""

import numpy as np
from typing import List, Dict, Any

from shapely.geometry import Point, Polygon, LineString, box
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
        downstream stele-clearing cleanup is skipped.
        """
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y

        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)
        outer_r = min(p["outer_radius_xylem"], stele_r)
        inner_r = min(p["inner_radius_xylem"], stele_r)

        raw_star = GeometryProcessor.star_polygon(
            n_branches=p["n_vascular_peak"],
            r_min=inner_r,
            r_max=outer_r,
            arc_base=p["arc_bottom_xylem"],
            arc_top=p["arc_top_xylem"],
        )
        star_coord = GeometryProcessor.smoothing_polygon(
            np.column_stack(raw_star.exterior.xy), smooth_factor=0.6, iterations=3,
        )

        xylem = (
            Tissue("xylem", Polygon(star_coord).buffer(0))
            .translate(cx, cy)
            .intersection(stele_polygon)
        )
        if xylem.is_empty:
            return xylem

        pith_r = p.get("pith_radius", 0.0)
        if pith_r and pith_r > 0.0:
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
            first_circle_shift=p["xylem_first_vessel_shift"],
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

    def fit_star_shapped_xylem(self, stele_polygon: Polygon):
        """Build the star xylem region and pack vessels into it (region + fill).

        Thin wrapper kept for direct/test use; the recipes drive the same region
        builder + fill declaratively (``recipe.fill(..., strategy="packing")``).
        """
        xylem = self._xylem_star_region(stele_polygon)
        if xylem.is_empty:
            return
        placed = fill_by_packing(
            self.vascular_cells, xylem.shape, xylem.tag, rng=self.rng,
            **self._xylem_pack_kwargs(),
        )
        self._record_xylem_vessels(xylem, placed)

    def _remove_stele_seeds_near_xylem(self) -> None:
        """Remove stele parenchyma cells engulfed by xylem vessels.

        This is a *group-level* cleanup, deliberately NOT replaced by the unified
        region mask in ``Organ.generate_cells``: that mask removes individual
        seeds whose point lies inside a vascular region, which would leave the
        rest of an engulfed parenchyma cell's border seeds in place and produce a
        partial, distorted Voronoi cell straddling the vessels.  Here, if any seed
        of an ``id_group`` lands inside a vessel the whole cell is dropped.
        (Removing this step leaves ~90 such partial cells in the default star
        root — stele 217 -> 307.)
        """
        if not hasattr(self, "xylem_star") or self.xylem_star is None:
            return
        if not self.vascular_polygons:
            return

        xylem_union     = unary_union(self.vascular_polygons)
        xylem_star_prep = prep(self.xylem_star)

        groups_to_delete: set = set()
        for c in self.all_cells.cells:
            if c.type != "stele" or c.id_group in groups_to_delete:
                continue
            pt = Point(c.x, c.y)
            if not xylem_star_prep.contains(pt):
                continue
            if xylem_union.contains(pt):
                groups_to_delete.add(c.id_group)

        self.all_cells.cells = [
            c for c in self.all_cells.cells
            if c.id_group not in groups_to_delete
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
    """Monocot root: ring of metaxylem vessels or star-shaped xylem."""

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

        if self.vascular_params["xylem_shape"] == "star":
            self.vascular_params.update({
                "xylem_diameter_max":        float(xylem.get("vessel_diameter",        0.06)),
                "xylem_diameter_min":        float(xylem.get("vessel_diameter_min",    0.01)),
                "xylem_diameter_sd":         float(xylem.get("vessel_diameter_sd",     0.005)),
                "n_vascular_peak":           int(xylem.get("n_vascular_peak",          5)),
                "inner_radius_xylem":        float(xylem.get("inner_radius",           0.05)),
                "outer_radius_xylem":        float(xylem.get("outer_radius",           0.15)),
                "arc_top_xylem":             float(xylem.get("arc_top",               0.02)),
                "arc_bottom_xylem":          float(xylem.get("arc_bottom",            0.04)),
                "xylem_gradient_function":   str(xylem.get("gradient_function",       "five_pl")),
                "xylem_gradient_inflection": float(xylem.get("gradient_inflection",   0.7)),
                "xylem_gradient_steepness":  float(xylem.get("gradient_steepness",    5.0)),
                "xylem_gradient_asymmetry":  float(xylem.get("gradient_asymmetry",    1.0)),
                "xylem_first_vessel_shift":  float(xylem.get("first_vessel_shift",    0.7)),
                "xylem_direction":           str(xylem.get("direction",               "center")),
                "pith_radius":               float(xylem.get("pith_radius",           0.0)),
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
        if self.vascular_params.get("xylem_shape", "default") == "star":
            recipe.fill("xylem star", self._xylem_star_region(polygon),
                        strategy="packing", produces=("xylem", "stele"),
                        record=self._record_xylem_vessels, **self._xylem_pack_kwargs())
            recipe.cleanup("clear stele under xylem",
                           self._remove_stele_seeds_near_xylem)
            self._add_phloem_step(recipe, polygon, type="monocot")
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
            outer = xylem_polygon.buffer(cell_diameter).intersection(stele_polygon)
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
                "prop_vessel_ring":       float(sec_xylem.get("prop_vessel_ring",       0.5)),
                "n_ring":                 max(1, int(sec_xylem.get("n_ring",            1))),
                "must_be_adjacent":       bool(sec_xylem.get("must_be_adjacent",        False)),
                "parenchyma_diameter":    float(sec_xylem.get("parenchyma_diameter",    0.03)),
                "parenchyma_diameter_sd": float(sec_xylem.get("parenchyma_diameter_sd", 0.002)),
                "parenchyma_width":       float(sec_xylem.get("parenchyma_width",       0.01)),
            }
            self.secondary_cambium_params = {
                "cell_diameter":  float(sec_cam.get("cell_diameter",  0.01)),
                "cell_width":     float(sec_cam.get("cell_width",     0.02)),
                "inner_distance": float(sec_cam.get("inner_distance", 0.30)),
                "outer_distance": float(sec_cam.get("outer_distance", 0.45)),
                "arc_top":        float(sec_cam.get("arc_top",        0.05)),
                "arc_bottom":     float(sec_cam.get("arc_bottom",     0.07)),
                "n_layers":       max(1, int(sec_cam.get("n_layers",  1))),
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
                "n_medullar":         int(med_rays.get("n_medullar",         6)),
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
        recipe.cleanup("clear stele under xylem",
                       self._remove_stele_seeds_near_xylem)
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
        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)

        outer_r = min(sc["outer_distance"], stele_r)
        inner_r = min(sc["inner_distance"], outer_r)

        raw_star = GeometryProcessor.star_polygon(
            n_branches=p["n_vascular_peak"],
            r_min=inner_r, r_max=outer_r,
            arc_base=sc["arc_bottom"], arc_top=sc["arc_top"],
        )
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
    def _phloem_trapeze_local(P, tangent, normal, base_half_width: float,
                              top_width: float, height: float) -> Polygon:
        """Trapeze standing perpendicular on the cambium at ``P``.

        Base of width ``2*base_half_width`` along the local ``tangent``, extending
        ``height`` along the outward ``normal`` and narrowing to ``top_width`` —
        the local-frame analogue of a stomata.  The base/tip are pushed slightly
        in/out so the caller's band intersection clips cleanly to the band.
        """
        (px, py), (tx, ty), (nx, ny) = P, tangent, normal
        inset = height * 0.3
        bcx, bcy = px - nx * inset,            py - ny * inset
        tcx, tcy = px + nx * (height + inset), py + ny * (height + inset)
        hw = top_width / 2.0
        p1 = (bcx - tx * base_half_width, bcy - ty * base_half_width)
        p2 = (bcx + tx * base_half_width, bcy + ty * base_half_width)
        p3 = (tcx + tx * hw,              tcy + ty * hw)
        p4 = (tcx - tx * hw,              tcy - ty * hw)
        return Polygon([p1, p2, p3, p4]).buffer(0)

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

        annular_zone = secondary_cambium_polygon.difference(primary_cambium_polygon)
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
        # in equal steps).  Each band is packed independently with the vessel size
        # gradient reset (large at the band's inner edge -> small at its outer
        # edge), so n_ring repetitions read as successive growth rings.  n_ring==1
        # keeps a single band == the whole zone (original behaviour).
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
        ) - sc["cell_diameter"]

        ray_annular_zone = secondary_cambium_polygon.buffer(
            -sc["cell_diameter"]
        ).difference(primary_cambium_polygon)
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
        # standing perpendicular on the cambium surface (stomata principle) and
        # clipped to the band.  top_width sets the taper, clamped so a narrow
        # compartment can't flare outward.  With no discrete vessel zones
        # (prop_stele >= 1) the band is all phloem.
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
                masks.append(self._phloem_trapeze_local(
                    P, tangent, normal, base_hw, top_w, sp["height"]
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
