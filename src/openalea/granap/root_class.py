"""
Root anatomy implementation.

`RootAnatomy` acts as a transparent factory: calling ``RootAnatomy(input_data)``
returns either a ``MonocotRootAnatomy`` or a ``DicotRootAnatomy`` instance
depending on the ``planttype`` value in the input.  Both subclasses are
``isinstance(obj, RootAnatomy)`` == True, so all existing code keeps working.
"""

import numpy as np
from typing import List, Dict, Any

from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.affinity import translate, scale as affine_scale, rotate
from shapely.prepared import prep

from openalea.granap.organ_class import Organ
from openalea.granap.layer_class import Layer, LayerPolygon
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.tissue_builder import (
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
        self.global_params = next((p for p in self.params if p["name"] == "planttype"), {})

        stele = next((p for p in self.params if p["name"] == "stele"), {})
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
        self.aerenchyma_params = next((p for p in self.params if p["name"] == "aerenchyma"), {})

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
                current_polygon, -space_increment - cell_diameter / 2, smooth_factor=0.6,
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

    def _organ_specific_tissues(self):
        pass

    def _create_vascular_tissue(self, polygon_for_vascular: Polygon, debug: bool = False):
        """Implemented by each subclass."""
        pass

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
        outer_r = min(p["outer_radius_xylem"], stele_r * 0.95)
        inner_r = min(p["inner_radius_xylem"], stele_r * 0.90)

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
        xylem  = next((p for p in self.params if p["name"] == "xylem"),  {})
        phloem = next((p for p in self.params if p["name"] == "phloem"), {})

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

    def _create_vascular_tissue(self, polygon_for_vascular: Polygon, debug: bool = False):
        if self.vascular_params.get("n_vascular_bundles", 0) == 0:
            return
        # Build the monocot stele as an explicit, ordered recipe of tissue steps.
        # remove_cells_in_polygon + extend_cells happens later in Organ.generate_cells().
        self._vascular_recipe(polygon_for_vascular).build()

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Declarative description of how a monocot stele is assembled.

        Two variants share the same vocabulary; the build order is data, so the
        sequence can be inspected via ``recipe.describe()`` and extended with
        edit-verb steps (smooth/rotate/...) without touching the generators.
        """
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
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
        i_cell = 0
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
                i_cell += 1
                cells_in_slices.add_cell(Cell(
                    type=metaxylem.tag,
                    x=cell_border_pts[0], y=cell_border_pts[1],
                    diameter=xylem_diameter,
                    id_cell=i_slice, id_layer=i_slice, id_group=i_slice,
                    angle=np.arctan2(cell_border_pts[1] - center.y, cell_border_pts[0] - center.x),
                    radius=np.sqrt((cell_border_pts[0] - center.x) ** 2 + (cell_border_pts[1] - center.y) ** 2),
                    area=np.pi * (xylem_diameter / 2) ** 2,
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
                self.vascular_cells.add_cell(Cell(
                    type="stele",
                    x=pt[0], y=pt[1],
                    diameter=cell_diameter,
                    id_cell=next_id_group, id_group=next_id_group,
                    angle=np.arctan2(pt[1] - center.y, pt[0] - center.x),
                    radius=np.sqrt((pt[0] - center.x) ** 2 + (pt[1] - center.y) ** 2),
                    area=np.pi * (cell_diameter / 2) ** 2,
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
        xylem   = next((p for p in self.params if p["name"] == "xylem"),   {})
        phloem  = next((p for p in self.params if p["name"] == "phloem"),  {})
        cambium = next((p for p in self.params if p["name"] == "cambium"), {})

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

        sec_growth = next((p for p in self.params if p["name"] == "secondary_growth"), {})
        self.vascular_params["secondary_growth"] = bool(sec_growth.get("value", False))

        if self.vascular_params["secondary_growth"]:
            sec_xylem = next((p for p in self.params if p["name"] == "secondary_xylem"), {})
            sec_cam   = next((p for p in self.params if p["name"] == "secondary_cambium"), {})
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
            }

            sec_phloem = next((p for p in self.params if p["name"] == "secondary_phloem"), {})
            self.secondary_phloem_params = {
                "outer_distance":      float(sec_phloem.get("outer_distance",      0.55)),
                "arc_top":             (float(sec_phloem["arc_top"]) if sec_phloem.get("arc_top") is not None else None),
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

            med_rays = next((p for p in self.params if p["name"] == "medullar_rays"), {})
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

    def _create_vascular_tissue(self, polygon_for_vascular: Polygon, debug: bool = False):
        if self.vascular_params.get("n_vascular_peak", 0) == 0:
            return
        # Build the dicot stele as an explicit, ordered recipe of tissue steps.
        # remove_cells_in_polygon + extend_cells happens later in Organ.generate_cells().
        self._vascular_recipe(polygon_for_vascular).build()

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Declarative description of how a dicot stele is assembled.

        A shared prefix (star xylem + clearing the stele it engulfs) is followed
        by either primary tissue (phloem + primary cambium) or, when
        ``secondary_growth`` is on, secondary xylem + secondary phloem.  The
        build order is data, so it can be inspected via ``recipe.describe()``.
        """
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
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

    def _build_secondary_phloem_polygon(
        self,
        stele_polygon: Polygon,
        secondary_cambium_polygon: Polygon,
        cx: float,
        cy: float,
    ) -> Polygon:
        """Build the phloem zone as one polygon per arm.

        Arms sit at 2π(k+0.5)/n_peaks — the cambium valley centres, which are
        the same angular positions as the secondary xylem vessel zones.

        Base (inner boundary):
            Pizza-slice half-angle = prop_stele × π/n_peaks, same as the xylem
            vessel zones.  The secondary cambium polygon is subtracted to get the
            real inner boundary from the cambium outer surface.

        Tip (outer boundary):
            sp["arc_top"] is None  →  full pizza-slice sector at sp["outer_distance"]
            sp["arc_top"] is float →  explicit trapeze: outer arc narrowed to arc_top,
                                       straight sides connecting to the pizza-slice base.
        """
        sp = self.secondary_phloem_params
        sc = self.secondary_cambium_params
        sx = self.secondary_xylem_params
        p  = self.vascular_params
        n_peaks = p["n_vascular_peak"]

        r_inner = sc["inner_distance"]   # cambium valley radius (phloem base)
        r_outer = sp["outer_distance"]
        if r_outer <= r_inner:
            return Polygon()

        # Base half-angle: same pizza-slice width as the xylem vessel zones.
        half_angle  = (np.pi / n_peaks) * sx["prop_stele"]
        r_wedge     = r_outer * 1.5
        arc_top     = sp["arc_top"]      # None or float
        n_arc       = 50

        arm_polys = []
        for k in range(n_peaks):
            theta = 2.0 * np.pi * (k + 0.5) / n_peaks

            if arc_top is None:
                # Full pizza-slice sector clipped to r_outer
                arc_a    = np.linspace(theta - half_angle, theta + half_angle, n_arc)
                wedge_pts = (
                    [(cx, cy)]
                    + [(cx + r_wedge * np.cos(a), cy + r_wedge * np.sin(a)) for a in arc_a]
                )
                arm = Polygon(wedge_pts).intersection(Point(cx, cy).buffer(r_outer))
            else:
                # Explicit trapeze: wide base (pizza-slice half-angle at r_inner),
                # narrow tip (arc_top at r_outer), straight sides.
                w_outer      = arc_top / r_outer
                outer_angles = np.linspace(theta - w_outer, theta + w_outer, n_arc)
                outer_pts    = np.column_stack([
                    cx + r_outer * np.cos(outer_angles),
                    cy + r_outer * np.sin(outer_angles),
                ])
                inner_angles = np.linspace(theta + half_angle, theta - half_angle, n_arc)
                inner_pts    = np.column_stack([
                    cx + r_inner * np.cos(inner_angles),
                    cy + r_inner * np.sin(inner_angles),
                ])
                arm = Polygon(np.vstack([outer_pts, inner_pts])).buffer(0)
            if not arm.is_empty:
                arm_polys.append(arm)

        if not arm_polys:
            return Polygon()

        # No stele_polygon intersection: phloem sits outside the secondary cambium
        # and may extend beyond the stele polygon boundary.
        return unary_union(arm_polys).difference(secondary_cambium_polygon)

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
                        # Evenly spaced within [-half_width, +half_width]
                        offset = (2 * j + 1 - n_r) / max(n_r, 1) * half_width
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
            arc_angles = np.linspace(theta_c - half_angle, theta_c + half_angle, 50)
            wedge_pts  = [(cx, cy)] + [
                (cx + r_outer_wedge * np.cos(a), cy + r_outer_wedge * np.sin(a))
                for a in arc_angles
            ]
            raw_wedge = Polygon(wedge_pts)
            poly      = raw_wedge.intersection(clip_zone)
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
        self._render_layer(secondary_cambium_polygon, "cambium", sc["cell_diameter"], sc["cell_width"], cx, cy)
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
                arc_angles = np.linspace(theta - half_width, theta + half_width, 50)
                wedge_pts  = [(cx, cy)] + [
                    (cx + r_outer_wedge * np.cos(a), cy + r_outer_wedge * np.sin(a)) for a in arc_angles
                ]
                raw_wedge = Polygon(wedge_pts)
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
                packed = GeometryProcessor.pack_circles(
                    zone,
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
                    rng=self.rng,
                )
                zone_vessel_polys: List[Polygon] = []

                for pcx, pcy, r in packed:
                    actual_diam = r * 2
                    placed      = Point(pcx, pcy).buffer(r, resolution=32)
                    placed_buff = placed.buffer(-r * 0.15)
                    if placed_buff.is_empty:
                        continue
                    bx, by = placed_buff.exterior.coords.xy
                    border_coords = GeometryProcessor.resample_coords(np.column_stack((bx, by)), target_n_points=25)
                    center   = placed.centroid
                    id_group = next_id
                    next_id += 1
                    for border_pt in border_coords[1:]:
                        self.vascular_cells.add_cell(Cell(
                            type="xylem",
                            x=border_pt[0], y=border_pt[1],
                            diameter=actual_diam,
                            id_cell=id_group, id_group=id_group,
                            angle=np.arctan2(border_pt[1] - center.y, border_pt[0] - center.x),
                            radius=np.sqrt((border_pt[0] - center.x) ** 2 + (border_pt[1] - center.y) ** 2),
                            area=np.pi * r ** 2,
                        ))
                    zone_vessel_polys.append(placed)
                    all_vessel_polys.append(placed)

                if zone_vessel_polys:
                    vessel_union_in_zone = unary_union(zone_vessel_polys)
                    axial_zone = zone.difference(vessel_union_in_zone)
                else:
                    axial_zone = zone

                erosion_poly = secondary_cambium_polygon if sx["prop_stele"] >= 1.0 else zone
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
        """Build secondary phloem outside the secondary cambium.

        Each arm sits at the cambium valley angles (same as the secondary xylem
        vessel zones) and is divided radially into an alive sub-zone (sieve tubes
        + companion cells + parenchyma) and a dead sub-zone (sieve tubes +
        parenchyma).  No medullar-ray cells are placed inside the phloem arms —
        the arm boundaries are already the medullar-ray / parenchyma-ray walls.
        """
        sp = self.secondary_phloem_params
        sx = self.secondary_xylem_params
        p  = self.vascular_params

        cx, cy  = stele_polygon.centroid.x, stele_polygon.centroid.y
        n_peaks = p["n_vascular_peak"]

        secondary_cambium_polygon = self._build_secondary_cambium_polygon(stele_polygon, cx, cy)
        if secondary_cambium_polygon is None or secondary_cambium_polygon.is_empty:
            return

        phloem_zone = self._build_secondary_phloem_polygon(
            stele_polygon, secondary_cambium_polygon, cx, cy
        )
        if phloem_zone is None or phloem_zone.is_empty:
            return

        self.vascular_tissue_polygons.setdefault("secondary_phloem", []).append(phloem_zone)

        # Radially split into alive (near cambium) and dead (outer) sub-zones.
        alive_annulus = secondary_cambium_polygon.buffer(sp["alive_distance"])
        alive_zone    = phloem_zone.intersection(alive_annulus)
        dead_zone     = phloem_zone.difference(alive_annulus)

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
            companion_polys: list = []

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
                    self.vascular_cells.add_cell(Cell(
                        type="phloem",
                        x=border_pt[0], y=border_pt[1],
                        diameter=actual_diam,
                        id_cell=id_group, id_group=id_group,
                        angle=np.arctan2(border_pt[1] - cy, border_pt[0] - cx),
                        radius=np.sqrt((border_pt[0] - cx) ** 2 + (border_pt[1] - cy) ** 2),
                        area=np.pi * r ** 2,
                    ))
                sieve_polys.append(placed)

                if alive:
                    # One companion cell placed tangentially adjacent to the sieve.
                    comp_r    = sp["companion_diameter"] / 2
                    theta_rad = np.arctan2(pcy - cy, pcx - cx)
                    for side in (1, -1):
                        ccx = pcx + (r + comp_r * 1.05) * np.cos(theta_rad + side * np.pi / 2)
                        ccy = pcy + (r + comp_r * 1.05) * np.sin(theta_rad + side * np.pi / 2)
                        comp_pt = Point(ccx, ccy)
                        if not arm_zone.contains(comp_pt):
                            continue
                        comp_circle = comp_pt.buffer(comp_r)
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
                            self.vascular_cells.add_cell(Cell(
                                type="companion_cell",
                                x=border_pt[0], y=border_pt[1],
                                diameter=sp["companion_diameter"],
                                id_cell=id_group, id_group=id_group,
                                angle=np.arctan2(border_pt[1] - cy, border_pt[0] - cx),
                                radius=np.sqrt((border_pt[0] - cx) ** 2 + (border_pt[1] - cy) ** 2),
                                area=np.pi * (comp_r) ** 2,
                            ))
                        companion_polys.append(comp_circle)
                        break   # one companion per sieve

            placed_union = unary_union(sieve_polys + companion_polys) if (sieve_polys or companion_polys) else Polygon()
            fill_zone    = arm_zone.difference(placed_union)
            if not fill_zone.is_empty:
                next_id = self._fill_zone_with_cells(
                    fill_zone,
                    sp["parenchyma_diameter"], sp["parenchyma_width"],
                    "phloem_parenchyma", cx, cy, next_id,
                    erosion_polygon=arm_zone,
                )

        return next_id
