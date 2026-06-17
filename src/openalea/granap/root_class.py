"""
Root anatomy implementation.
"""

import numpy as np
from typing import List, Dict, Any

from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.affinity import translate, scale as affine_scale, rotate

from openalea.granap.organ_class import Organ
from openalea.granap.layer_class import Layer
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.input_data import OrganInputData
from openalea.granap.math_functions import GRADIENT_FUNCTIONS, rescale

class RootAnatomy(Organ):
    """
    Root cross-sectional anatomy.

    Implements the typical structure of plant roots with
    circular cross-section and vascular cylinder.
    """

    def __init__(self, input_data: Any = None):
        """
        Initialize root anatomy.
        """
        super().__init__()
        if isinstance(input_data, OrganInputData):
            self.params = input_data.to_dict_list()
        elif isinstance(input_data, list):
            self.params = input_data
        else:
            self.params = OrganInputData.for_root().to_dict_list()

        self._initialize_params()
        self._initialize_default_layers()
        
    @property
    def planttype(self) -> int:
        """Plant type: 1 = monocot, 2 = dicot."""
        return int(self.global_params.get("value", 1))

    def _initialize_params(self) -> None:
        """Parse the structured input and set local attributes."""
        # Initialise secondary-growth dicts so they always exist as attributes
        self.secondary_xylem_params: dict = {}
        self.secondary_cambium_params: dict = {}

        # 1. Global params
        self.global_params = next((p for p in self.params if p["name"] == "planttype"), {})

        # 2. Stele parenchyma params (shared by both plant types)
        stele = next((p for p in self.params if p["name"] == "stele"), {})

        self.vascular_params = {
            "thickness":                stele["thickness"],
            "cell_diameter":            stele["cell_diameter"],
            "cell_diameter_center":     stele.get("cell_diameter_center", stele["cell_diameter"]),
            "size_gradient_function":   stele.get("size_gradient_function", "five_pl"),
            "size_gradient_inflection": stele.get("size_gradient_inflection", 0.5),
            "size_gradient_steepness":  stele.get("size_gradient_steepness",  3.0),
            "size_gradient_asymmetry":  stele.get("size_gradient_asymmetry",  1.0),
        }

        # 3. Type-specific vascular params — read from dedicated xylem / phloem / cambium dicts
        if self.planttype == 1:
            xylem  = next((p for p in self.params if p["name"] == "xylem"),  {})
            phloem = next((p for p in self.params if p["name"] == "phloem"), {})
            self.vascular_params.update({
                "xylem_diameter":         float(xylem.get("cell_diameter",          0.06)),
                "xylem_diameter_sd":      float(xylem.get("cell_diameter_sd",       0.005)),
                "protoxylem_diameter":    float(xylem.get("protoxylem_diameter",    0.01)),
                "protoxylem_diameter_sd": float(xylem.get("protoxylem_diameter_sd", 0.002)),
                "protoxylem_width":       float(xylem.get("protoxylem_width",       0.03)),
                "protoxylem_height":      float(xylem.get("protoxylem_height",      0.05)),
                "n_vascular_bundles":     int(xylem.get("n_vascular_bundles",       5)),
                "ratio_proto_meta":       float(xylem.get("ratio_proto_meta",       2.2)),
                "phloem_diameter":        float(phloem.get("cell_diameter",         0.005)),
                "phloem_diameter_sd":     float(phloem.get("cell_diameter_sd",      0.001)),
                "phloem_width":           float(phloem.get("width",                 0.02)),
                "phloem_height":          float(phloem.get("height",                0.03)),
                "xylem_shape":            str(xylem.get("xylem_shape", "default")),
            })

            if self.vascular_params["xylem_shape"] == "star":
                self.vascular_params.update({
                    "xylem_diameter_max":        float(xylem.get("vessel_diameter",      0.06)),
                    "xylem_diameter_min":        float(xylem.get("vessel_diameter_min",  0.01)),
                    "xylem_diameter_sd":         float(xylem.get("vessel_diameter_sd",   0.005)),
                    "n_vascular_peak":           int(xylem.get("n_vascular_peak",        5)),
                    "inner_radius_xylem":        float(xylem.get("inner_radius", 0.05)),
                    "outer_radius_xylem":        float(xylem.get("outer_radius",         0.15)),
                    "arc_top_xylem":             float(xylem.get("arc_top",              0.02)),
                    "arc_bottom_xylem":          float(xylem.get("arc_bottom",           0.04)),
                    "xylem_gradient_function":   str(xylem.get("gradient_function",      "five_pl")),
                    "xylem_gradient_inflection": float(xylem.get("gradient_inflection",  0.7)),
                    "xylem_gradient_steepness":  float(xylem.get("gradient_steepness",   5.0)),
                    "xylem_gradient_asymmetry":  float(xylem.get("gradient_asymmetry",   1.0)),
                    "xylem_first_vessel_shift":  float(xylem.get("first_vessel_shift",   0.7)),
                    "xylem_direction":           str(xylem.get("direction",              "center")),
                    "pith_radius":               float(xylem.get("pith_radius",          0.0)),
                    "relative_phloem":           float(phloem.get("relative_distance",   0.5)),
                })
        elif self.planttype == 2:
            xylem   = next((p for p in self.params if p["name"] == "xylem"),   {})
            phloem  = next((p for p in self.params if p["name"] == "phloem"),  {})
            cambium = next((p for p in self.params if p["name"] == "cambium"), {})
            self.vascular_params.update({
                "xylem_diameter_max":        float(xylem.get("vessel_diameter",       0.09)),
                "xylem_diameter_min":        float(xylem.get("vessel_diameter_min",   0.01)),
                "xylem_diameter_sd":         float(xylem.get("vessel_diameter_sd",    0.002)),
                "n_vascular_peak":           int(xylem.get("n_vascular_peak",       3)),
                "inner_radius_xylem":        float(xylem.get("inner_radius",        0.05)),
                "outer_radius_xylem":        float(xylem.get("outer_radius",        0.22)),
                "arc_top_xylem":             float(xylem.get("arc_top",             0.03)),
                "arc_bottom_xylem":          float(xylem.get("arc_bottom",          0.03)),
                "xylem_gradient_function":   str(xylem.get("gradient_function",   "five_pl")),
                "xylem_gradient_inflection": float(xylem.get("gradient_inflection", 0.7)),
                "xylem_gradient_steepness":  float(xylem.get("gradient_steepness",  5.0)),
                "xylem_gradient_asymmetry":  float(xylem.get("gradient_asymmetry",  1.0)),
                "xylem_first_vessel_shift":  float(xylem.get("first_vessel_shift",  0.7)),
                "pith_radius":               float(xylem.get("pith_radius",          0.0)),
                "xylem_direction":           str(xylem.get("direction", "center")),
                "phloem_diameter":           float(phloem.get("vessel_diameter",      0.005)),
                "phloem_diameter_sd":        float(phloem.get("vessel_diameter_sd",   0.001)),
                "phloem_width":              float(phloem.get("width",              0.15)),
                "phloem_height":             float(phloem.get("height",             0.2)),
                "relative_phloem":          float(phloem.get("relative_distance",   0.2)),
                "cambium_cell_diameter":     float(cambium.get("cell_diameter",     0.015)),
                "cambium_cell_width":        float(cambium.get("cell_width",        0.03)),
                "cambium_primary_inner_distance":   float(cambium.get("inner_distance",   0.10)),
                "cambium_primary_outer_distance":   float(cambium.get("outer_distance",   0.28)),
                "cambium_primary_visible_distance": float(cambium.get("visible_distance", 0.15)),
                "cambium_primary_arc_top":    float(cambium.get("arc_top",    0.1)),
                "cambium_primary_arc_bottom": float(cambium.get("arc_bottom", 0.07)),
            })

            # Secondary growth flag
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

        # 3. Intercellular spaces / aerenchyma — store raw config dicts directly
        self.intercellular_spaces_params = [p for p in self.params if p["name"] == "inter_cellular_spaces"]
        self.aerenchyma_params = next((p for p in self.params if p["name"] == "aerenchyma"), {})

        # 4. Extract layer definitions (any param with 'order' that is not a vascular zone)
        self.layers = [p for p in self.params if "order" in p and p["name"] not in ("stele", "xylem", "phloem", "aerenchyma")]
        self.layers = sorted(self.layers, key=lambda x: float(x["order"]))

    def _initialize_default_layers(self) -> None:
        """Initialize root layers from parsed params."""
        for param in self.layers:
            self.layer_manager.add_layer(Layer.from_dict(param))
    
    def _create_base_shape(self) -> Polygon:
        """
        Create the circular shape of a root cross-section.
        
        Returns:
            Circular polygon
        """
        radius = self._calculate_root_radius()
        return GeometryProcessor.circle_polygon(radius)
    
    def _calculate_root_radius(self) -> float:
        """Calculate total root radius from layers."""
        radius = self.vascular_params["thickness"] / 2
        
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, 'n_layers'):
                radius += layer.get_total_thickness()
            elif hasattr(layer, 'cell_diameter'):
                radius += layer.cell_diameter
        
        return radius
    
    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create stele parenchyma rings from the stele edge toward the centre.

        Cell diameter follows a size gradient: rings near the periphery receive
        ``cell_diameter`` (lower bound) and rings near the centre receive
        ``cell_diameter_center`` (upper bound).  When both values are equal the
        gradient is flat and behaviour is identical to the uniform distribution.

        Args:
            current_polygon: Innermost polygon after all outer layers have been built.
            params: Layer parameter dictionaries (used only to compute *i_layer* offset).

        Returns:
            List of central layer polygon dictionaries.
        """
        central_layers = []

        diameter_fn = rescale(
            GRADIENT_FUNCTIONS[self.vascular_params["size_gradient_function"]],
            lo=self.vascular_params["cell_diameter"],
            hi=self.vascular_params["cell_diameter_center"],
            c=self.vascular_params["size_gradient_inflection"],
            b=self.vascular_params["size_gradient_steepness"],
            m=self.vascular_params["size_gradient_asymmetry"],
        )

        # Stele radius at entry — used to normalise radial position.
        stele_radius = np.sqrt(current_polygon.area / np.pi)

        # First space increment: half the cell diameter of the innermost non-stele layer.
        min_order = min(l.order for l in self.layer_manager.get_layers() if l.order > 0)
        space_increment = self.layer_manager.get_layer_by_order(min_order).cell_diameter / 2

        i_layer = len(params)

        while not current_polygon.is_empty and current_polygon.area > 0:
            # Normalized radius of the current ring (outer edge of the ring to be placed).
            r_norm = np.clip(np.sqrt(current_polygon.area / np.pi) / stele_radius, 0.0, 1.0)
            cell_diameter = diameter_fn(r_norm)

            # Stop when the remaining area is too small to fit even one cell.
            if current_polygon.area <= (cell_diameter / 2) ** 2 * np.pi:
                break

            current_polygon = GeometryProcessor.buffer_polygon(
                current_polygon,
                -space_increment - cell_diameter / 2,
                smooth_factor=0.6,
            )

            space_increment = cell_diameter / 2

            central_layers.append({
                "name": "stele",
                "polygon": current_polygon,
                "cell_diameter": cell_diameter,
                "id_layer": i_layer + 1,
                "cell_width": 0,
            })

            i_layer += 1

        return central_layers

    def reshape_layers(self, layers_polygons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reshape layers to fit the organ shape.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        
        Returns:
            List of reshaped layer polygon dictionaries
        """
        
        return layers_polygons
    
    def set_vascular_params(self, **kwargs) -> None:
        """
        Update vascular cylinder parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        self.vascular_params.update(kwargs)
        self._invalidate_geometry()
    
    def add_lateral_root_primordium(self, angle: float, distance: float) -> None:
        """
        Add a lateral root primordium (developmental structure).
        
        Args:
            angle: Angular position (radians)
            distance: Distance from center
        
        Note:
            This is a placeholder for future implementation
        """
        # This would require more complex geometry manipulation
        # Left as a placeholder for future enhancement
        pass

    def _create_vascular_tissue(self, polygon_for_vascular: Polygon, debug: bool = False):
        """Dispatch vascular tissue construction based on plant type."""
        if self.planttype == 1:
            if self.vascular_params["n_vascular_bundles"] == 0:
                return
            self._create_vascular_tissue_monocot(polygon_for_vascular, debug)
        elif self.planttype == 2:
            if self.vascular_params["n_vascular_peak"] == 0:
                return
            self._create_vascular_tissue_dicot(polygon_for_vascular, debug)
        else:
            raise ValueError(f"Unknown planttype: {self.planttype}")

    def _create_vascular_tissue_monocot(self, polygon_for_vascular: Polygon, debug: bool = False):
        """Monocot stele: ring of metaxylem vessels with alternating phloem/protoxylem,
        or star-shaped xylem when xylem_shape == 'star'."""
        if self.vascular_params.get("xylem_shape", "default") == "star":
            self.vascular_cells = CellManager()
            self.vascular_polygons = []
            self.fit_star_shapped_xylem(polygon_for_vascular)
            self._remove_stele_seeds_near_xylem()
            self.fit_phloem_elements(polygon_for_vascular, type="monocot")

            vascular_polygons = unary_union(self.vascular_polygons)
            self.all_cells.remove_cells_in_polygon(vascular_polygons)
            self.all_cells.extend_cells(self.vascular_cells.cells)
            self.all_cells.recalculate_cell_properties()
            if debug:
                self.all_cells.plot_cells()
            return

        self.fit_metaxylem_elements(polygon_for_vascular)
        self.fit_metaxylem_sheath(polygon_for_vascular)
        self.fit_phloem_protoxylem_elements(polygon_for_vascular)

        vascular_polygons = unary_union(self.vascular_polygons)
        self.all_cells.remove_cells_in_polygon(vascular_polygons)
        self.all_cells.extend_cells(self.vascular_cells.cells)
        self.all_cells.recalculate_cell_properties()
        if debug:
            self.all_cells.plot_cells()

    def _create_vascular_tissue_dicot(self, polygon_for_vascular: Polygon, debug: bool = False):
        """Dicot stele: star-shaped xylem with cambium and phloem; optionally secondary growth."""
        self.fit_star_shapped_xylem(polygon_for_vascular)
        self._remove_stele_seeds_near_xylem()
        if self.vascular_params.get("secondary_growth", False):
            self.fit_secondary_xylem(polygon_for_vascular)
        else:
            self.fit_primary_cambium_elements(polygon_for_vascular)
            self.fit_phloem_elements(polygon_for_vascular, type="dicot")

        vascular_polygons = unary_union(self.vascular_polygons)
        self.all_cells.remove_cells_in_polygon(vascular_polygons)
        self.all_cells.extend_cells(self.vascular_cells.cells)
        self.all_cells.recalculate_cell_properties()
        if debug:
            self.all_cells.plot_cells()

    def fit_star_shapped_xylem(self, stele_polygon: Polygon):
        """Pack metaxylem vessels inside the star-shaped xylem region (dicot).

        Builds the trapezoid star from vascular params, smooths it, clips it to
        the stele, then fills it with an Apollonian packing that grades from
        xylem_diameter_max at the centre to xylem_diameter_min at the boundary.
        Circles whose diameter is below xylem_diameter_min are labelled 'stele'.
        """
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y

        # Clamp radii so the star never exceeds the stele
        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)
        outer_r = min(p["outer_radius_xylem"], stele_r * 0.95)
        inner_r = min(p["inner_radius_xylem"], stele_r * 0.90)

        # Build star at the origin, smooth, translate to stele centre, clip
        raw_star = GeometryProcessor.star_polygon(
            n_branches=p["n_vascular_peak"],
            r_min=inner_r,
            r_max=outer_r,
            arc_base=p["arc_bottom_xylem"],
            arc_top=p["arc_top_xylem"],
        )

        star_coord = GeometryProcessor.smoothing_polygon(
            np.column_stack(raw_star.exterior.xy),
            smooth_factor=0.6,
            iterations=3,
        )
        star = Polygon(star_coord).buffer(0)
        # Translate star to stele centre, clip to stele boundary
        star = translate(star, cx, cy).intersection(stele_polygon)
        if star.is_empty:
            return

        # Pith: subtract a central circle so no vessel is placed inside it.
        # Stele seeds already generated over the full stele remain inside the
        # pith and become pith parenchyma after Voronoi.
        pith_r = p.get("pith_radius", 0.0)
        if pith_r and pith_r > 0.0:
            pith_circle = Point(cx, cy).buffer(pith_r)
            self.pith_polygon = pith_circle
            star = star.difference(pith_circle)
        else:
            self.pith_polygon = None

        self.xylem_star = star

        packed = GeometryProcessor.pack_circles(
            star,
            proportion=1.0,
            direction=p["xylem_direction"],
            diameter_max=p["xylem_diameter_max"],
            diameter_min=p["xylem_diameter_min"],
            diameter_sd=p["xylem_diameter_sd"],
            gradient_function=p["xylem_gradient_function"],
            gradient_inflection=p["xylem_gradient_inflection"],
            gradient_steepness=p["xylem_gradient_steepness"],
            gradient_asymmetry=p["xylem_gradient_asymmetry"],
            first_circle_shift=p["xylem_first_vessel_shift"],
        )

        self.vascular_cells = CellManager()
        self.vascular_polygons = []   # only placed circles clear parenchyma, not the star

        min_diam = p["xylem_diameter_min"]

        for i_cell, (pcx, pcy, r) in enumerate(packed):
            actual_diam = r * 2
            actual_r = r

            placed = Point(pcx, pcy).buffer(actual_r, resolution=32)
            cell_type = "xylem" if actual_diam >= min_diam else "stele"

            placed_buff = placed.buffer(-actual_r * 0.15)
            if placed_buff.is_empty:
                continue

            bx, by = placed_buff.exterior.coords.xy
            border_coords = GeometryProcessor.resample_coords(
                np.column_stack((bx, by)), target_n_points=25
            )
            center = placed.centroid

            for border_pt in border_coords[1:]:
                new_cell = Cell(
                    type=cell_type,
                    x=border_pt[0],
                    y=border_pt[1],
                    diameter=actual_diam,
                    id_cell=i_cell,
                    id_layer=0,
                    id_group=i_cell,
                    angle=np.arctan2(border_pt[1] - center.y, border_pt[0] - center.x),
                    radius=np.sqrt(
                        (border_pt[0] - center.x) ** 2 + (border_pt[1] - center.y) ** 2
                    ),
                    area=np.pi * actual_r ** 2,
                )
                self.vascular_cells.add_cell(new_cell)

            if cell_type == "xylem":
                self.vascular_polygons.append(placed)

    def _remove_stele_seeds_near_xylem(self) -> None:
        """Remove stele parenchyma seeds from all_cells that are inside the xylem
        star and engulfed by xylem vessels.

        The stele seeds generated by _create_central_layers span the whole stele
        polygon, including the xylem-star region.  Those that fall inside the star
        and have more than 50 % of a probe circle (diameter = xylem_diameter_max)
        covered by xylem vessel polygons are removed so they don't generate
        spurious parenchyma cells between vessels.
        """
        if not hasattr(self, 'xylem_star') or self.xylem_star is None:
            return
        if not self.vascular_polygons:
            return

        diam_max = self.vascular_params["xylem_diameter_max"]
        probe_r = diam_max / 2
        probe_area = np.pi * probe_r ** 2

        xylem_union = unary_union(self.vascular_polygons)
        pith_polygon = getattr(self, "pith_polygon", None)

        self.all_cells.cells = [
            c for c in self.all_cells.cells
            if not (
                c.type == "stele"
                and self.xylem_star.contains(Point(c.x, c.y))
                and (pith_polygon is None or not pith_polygon.contains(Point(c.x, c.y)))
                and Point(c.x, c.y).buffer(probe_r).intersection(xylem_union).area / probe_area > 0.6
            )
        ]

    def fit_primary_cambium_elements(self, stele_polygon: Polygon):
        """Dicot cambium placement — to be implemented."""
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y

        inner_radius_cambium = p["cambium_primary_inner_distance"]
        primary_arc_top = p["cambium_primary_arc_top"]
        primary_arc_bottom = p["cambium_primary_arc_bottom"]
        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)

        outer_r = min(p["cambium_primary_outer_distance"], stele_r)
        inner_r = min(inner_radius_cambium, outer_r)

        # Build star at the origin, smooth, translate to stele centre, clip
        raw_star = GeometryProcessor.star_polygon(
            n_branches=p["n_vascular_peak"],
            r_min=inner_r,
            r_max=outer_r,
            arc_base=primary_arc_bottom,
            arc_top=primary_arc_top,
        )

        star_coord = GeometryProcessor.smoothing_polygon(
            np.column_stack(raw_star.exterior.xy),
            smooth_factor=0.9,
            iterations=5,
        )
        star = Polygon(star_coord).buffer(0)
        # Translate star to stele centre
        star = translate(star, cx, cy)
        if star.is_empty:
            return

        # The cambium star is treated as a BOUNDARY LINE.
        # primary_visible_distance is the maximum radius from the stele centre
        # at which cambium is differentiated. 
        clip_circle = Point(cx, cy).buffer(p["cambium_primary_visible_distance"])
        visible_boundary = star.exterior.intersection(clip_circle)
        if visible_boundary.is_empty:
            return

        self.cambium_star = visible_boundary  # MultiLineString, one arch per peak

        cell_diam  = p["cambium_cell_diameter"]
        cell_width = p["cambium_cell_width"]

        # Remove stele parenchyma and pericycle seeds inside the thin cambium band.
        thin_ring = visible_boundary.buffer(cell_diam/2)
        groups_to_delete = {
            c.id_group
            for c in self.all_cells.cells
            if c.type in ("stele", "pericycle") and thin_ring.intersects(Point(c.x, c.y))
        }
        self.all_cells.cells = [
            c for c in self.all_cells.cells
            if c.type not in ("stele", "pericycle") or c.id_group not in groups_to_delete
        ]

        xylem_union = unary_union(self.vascular_polygons) if self.vascular_polygons else None
        self._render_layer(visible_boundary, "cambium", cell_diam, cell_width, cx, cy, xylem_union)

    def fit_phloem_elements(self, stele_polygon: Polygon, type = "monocot"):
        """Place one phloem ellipse per valley between xylem peaks, filled with Apollonian packing."""
        p = self.vascular_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y
        n_peaks = p["n_vascular_peak"]

        width     = p["phloem_width"]
        height    = p["phloem_height"]
        cell_diam = p["phloem_diameter"]
        cell_sd   = p["phloem_diameter_sd"]
        relative_distance = p["relative_phloem"]

        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)
        if type == "monocot":
            minimal_distance = min(p["inner_radius_xylem"], stele_r * 0.95)
            adjustment = p.get("cell_diameter", 0.0)
        if type == "dicot":
            minimal_distance = min(p["cambium_primary_inner_distance"], stele_r * 0.95)
            adjustment = p.get("cambium_cell_diameter", 0.0)
            
        r_center = minimal_distance + adjustment + (height/2) + (stele_r - adjustment - height - minimal_distance) * relative_distance

        xylem_star = getattr(self, "xylem_star", None) # get the star-shaped xylem region if it exists, to avoid placing phloem cells there
        next_id_group = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0 # unique id_group for each phloem region

        for k in range(n_peaks):
            theta = 2 * np.pi * (k + 0.5) / n_peaks # place phloem in the middle of the valley between xylem peaks

            # Build ellipse at origin (height=radial, width=tangential), then orient and place
            raw = Point(0, 0).buffer(1, resolution=64) # unit circle
            raw = affine_scale(raw, width / 2, height / 2) # scale to ellipse dimensions
            raw = rotate(raw, np.degrees(theta) - 90, origin=(0, 0)) # rotate so the height axis points radially
            raw = translate(raw, cx + r_center * np.cos(theta), cy + r_center * np.sin(theta)) # translate to position

            ellipse = raw.intersection(stele_polygon) # clip to stele boundary
            if xylem_star is not None and not xylem_star.is_empty: 
                ellipse = ellipse.difference(xylem_star) # further remove the star-shaped xylem region from the phloem ellipse to avoid placing phloem cells there
            if ellipse.is_empty or ellipse.area < np.pi * (cell_diam / 2) ** 2:
                continue

            # Remove stele seeds inside this phloem region
            self.all_cells.cells = [
                c for c in self.all_cells.cells
                if not (c.type == "stele" and ellipse.contains(Point(c.x, c.y)))
            ]

            packed = GeometryProcessor.pack_circles(
                ellipse,
                proportion=1.0,
                direction=None,
                diameter_max=cell_diam,
                diameter_min=cell_diam,
                diameter_sd=cell_sd,
                gradient_function="normal",
            )

            for pcx, pcy, r in packed:
                actual_diam = r * 2
                actual_r = r

                placed = Point(pcx, pcy).buffer(actual_r, resolution=32)
                placed_buff = placed.buffer(-actual_r * 0.15)
                if placed_buff.is_empty:
                    continue

                bx, by = placed_buff.exterior.coords.xy
                border_coords = GeometryProcessor.resample_coords(np.column_stack((bx, by)), target_n_points=25) # resample to get evenly spaced seed points along the border

                id_group = next_id_group
                next_id_group += 1
                for border_pt in border_coords[1:]:
                    self.vascular_cells.add_cell(Cell(
                        type="phloem",
                        x=border_pt[0],
                        y=border_pt[1],
                        diameter=actual_diam,
                        id_cell=id_group,
                        id_layer=0,
                        id_group=id_group,
                        angle=np.arctan2(border_pt[1] - cy, border_pt[0] - cx),
                        radius=np.sqrt((border_pt[0] - cx) ** 2 + (border_pt[1] - cy) ** 2),
                        area=np.pi * actual_r ** 2,
                    ))
                # self.vascular_polygons.append(placed)  # TODO: adding phloem to vascular_polygons clears a larger area than seeds cover (placed vs placed_buff), creating a dead zone — investigate

    def fit_phloem_protoxylem_elements(self, polygon):

        n_protoxylem = int(np.ceil(self.vascular_params["ratio_proto_meta"]*self.vascular_params["n_vascular_bundles"]))
        n_phloem = n_protoxylem-1
        buffing_dist = max(self.vascular_params["protoxylem_diameter"], self.vascular_params["phloem_diameter"])

        polygon = polygon.difference(polygon.buffer(-buffing_dist*1.1))
        polygon = polygon.difference(unary_union(self.vascular_polygons))

        slices = GeometryProcessor.pizza_slice(polygon, n_phloem+n_protoxylem)

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

        bundle_cx, bundle_cy, _ = GeometryProcessor.get_inscribed_circle(slice_poly)
        radial_angle_deg = np.degrees(np.arctan2(bundle_cy, bundle_cx))

        raw = Point(0, 0).buffer(1, resolution=64)
        raw = affine_scale(raw, p["protoxylem_width"] / 2, p["protoxylem_height"] / 2)
        raw = rotate(raw, radial_angle_deg - 90, origin=(0, 0))
        ellipse = translate(raw, bundle_cx, bundle_cy).intersection(slice_poly)
        if ellipse.is_empty or ellipse.area < np.pi * (p["protoxylem_diameter"] / 2) ** 2:
            return cells_in_slice, list_polygons

        packed = GeometryProcessor.pack_circles(
            ellipse,
            proportion=1.0,
            direction=None,
            diameter_max=p["protoxylem_diameter"],
            diameter_sd=p["protoxylem_diameter_sd"],
            gradient_function="normal",
        )

        next_id_group = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0
        for i_cell, (pcx, pcy, r) in enumerate(packed):
            cell_diam     = r * 2
            placed        = Point(pcx, pcy).buffer(r, resolution=32)
            placed_buff   = placed.buffer(-r * 0.15)
            if placed_buff.is_empty:
                continue
            bx, by = placed_buff.exterior.coords.xy
            border_coords = GeometryProcessor.resample_coords(
                np.column_stack((bx, by)), target_n_points=24
            )
            center        = placed.centroid
            cell_id_group = next_id_group + i_cell
            for border_pt in border_coords[1:]:
                cells_in_slice.add_cell(Cell(
                    type="protoxylem",
                    x=border_pt[0],
                    y=border_pt[1],
                    diameter=cell_diam,
                    id_cell=cell_id_group,
                    id_layer=0,
                    id_group=cell_id_group,
                    angle=np.arctan2(border_pt[1] - center.y, border_pt[0] - center.x),
                    radius=np.sqrt((border_pt[0] - center.x)**2 + (border_pt[1] - center.y)**2),
                    area=np.pi * r ** 2,
                ))

        list_polygons.append(ellipse)
        return cells_in_slice, list_polygons

    def phloem_elements_in_slice(self, slice_poly: Polygon):
        p = self.vascular_params
        list_polygons = []
        cells_in_slice = CellManager()

        bundle_cx, bundle_cy, _ = GeometryProcessor.get_inscribed_circle(slice_poly)
        radial_angle_deg = np.degrees(np.arctan2(bundle_cy, bundle_cx))

        raw = Point(0, 0).buffer(1, resolution=64)
        raw = affine_scale(raw, p["phloem_width"] / 2, p["phloem_height"] / 2)
        raw = rotate(raw, radial_angle_deg - 90, origin=(0, 0))
        ellipse = translate(raw, bundle_cx, bundle_cy).intersection(slice_poly)
        if ellipse.is_empty or ellipse.area < np.pi * (p["phloem_diameter"] / 2) ** 2:
            return cells_in_slice, list_polygons

        packed = GeometryProcessor.pack_circles(
            ellipse,
            proportion=1.0,
            direction=None,
            diameter_max=p["phloem_diameter"],
            diameter_sd=p["phloem_diameter_sd"],
            gradient_function="normal",
        )

        next_id_group = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0
        for i_cell, (pcx, pcy, r) in enumerate(packed):
            cell_diam     = r * 2
            placed        = Point(pcx, pcy).buffer(r, resolution=32)
            placed_buff   = placed.buffer(-r * 0.15)
            if placed_buff.is_empty:
                continue
            bx, by = placed_buff.exterior.coords.xy
            border_coords = GeometryProcessor.resample_coords(
                np.column_stack((bx, by)), target_n_points=24
            )
            center        = placed.centroid
            cell_id_group = next_id_group + i_cell
            for border_pt in border_coords[1:]:
                cells_in_slice.add_cell(Cell(
                    type="phloem",
                    x=border_pt[0],
                    y=border_pt[1],
                    diameter=cell_diam,
                    id_cell=cell_id_group,
                    id_layer=0,
                    id_group=cell_id_group,
                    angle=np.arctan2(border_pt[1] - center.y, border_pt[0] - center.x),
                    radius=np.sqrt((border_pt[0] - center.x)**2 + (border_pt[1] - center.y)**2),
                    area=np.pi * r ** 2,
                ))

        list_polygons.append(ellipse)
        return cells_in_slice, list_polygons

    def fit_metaxylem_elements(self, polygon):
        # from polygon, fit two ellipses
        n_xylem_cells = self.vascular_params["n_vascular_bundles"]
        if n_xylem_cells == 0:
            return
        elif n_xylem_cells == 1:
            slices = [polygon]
        else:
            slices = GeometryProcessor.pizza_slice(polygon.buffer(-self.vascular_params["xylem_diameter"]/4), n_xylem_cells)
        cells_in_slices, list_xylem_polygons = self.vascular_elements_in_slice(slices)
        self.vascular_cells = cells_in_slices
        self.vascular_polygons = list_xylem_polygons
    
    def vascular_elements_in_slice(self, slices: List[Polygon]):
        list_xylem_polygons = []
        cells_in_slices = CellManager()
        i_cell = 0
        for i_slice, slice in enumerate(slices):
            # Sample vessel diameter from N(mean, sd); clip to a safe minimum
            xylem_diameter = float(np.clip(
                np.random.normal(self.vascular_params["xylem_diameter"],
                                 self.vascular_params["xylem_diameter_sd"]),
                self.vascular_params["xylem_diameter"] * 0.1,
                np.inf,
            ))

            xylem_polygon = GeometryProcessor.fit_inner_ellipse(slice, xylem_diameter / 2)
            xylem_polygon = xylem_polygon["polygon"]
            xylem_polygon_buff = GeometryProcessor.buffer_polygon(xylem_polygon, -(xylem_diameter / 2) * 0.15)
            x, y = xylem_polygon_buff.exterior.coords.xy
            center = xylem_polygon.centroid
            coords = np.column_stack((x, y))
            coords = GeometryProcessor.resample_coords(coords, target_n_points=25)

            for cell_border_pts in coords[1:]:
                i_cell += 1
                new_cell = Cell(
                        type="metaxylem",
                        x=cell_border_pts[0],
                        y=cell_border_pts[1],
                        diameter=xylem_diameter,
                        id_cell=i_slice,
                        id_layer=i_slice,
                        id_group=i_slice,
                        angle=np.arctan2(cell_border_pts[1] - center.y,
                                          cell_border_pts[0] - center.x),
                        radius=np.sqrt((cell_border_pts[0] - center.x)**2 +
                                        (cell_border_pts[1] - center.y)**2),
                        area=np.pi * (xylem_diameter / 2) ** 2,
                    )
                cells_in_slices.add_cell(new_cell)

            list_xylem_polygons.append(xylem_polygon)
        return cells_in_slices, list_xylem_polygons

    def fit_metaxylem_sheath(self, stele_polygon: Polygon):
        """Add a ring of xylem parenchyma cells around each metaxylem vessel.

        For each metaxylem polygon already stored in self.vascular_polygons,
        seeds are placed along the perimeter of a polygon buffered outward by
        half a stele cell_diameter (the midpoint of the ring).  The full ring
        region (from the metaxylem edge to one cell_diameter outward, clipped
        to the stele) is also appended to self.vascular_polygons so that
        competing stele parenchyma seeds are cleared from that annulus.
        """
        cell_diameter = self.vascular_params["cell_diameter"]
        center = stele_polygon.centroid

        # Start id_group values above all existing layer cell groups
        next_id_group = max((c.id_group for c in self.all_cells.cells), default=0) + 1

        # Snapshot the metaxylem polygons only
        xylem_polygons = list(self.vascular_polygons)

        for xylem_polygon in xylem_polygons:
            # Outer boundary of the sheath ring, clipped to the stele
            outer = xylem_polygon.buffer(cell_diameter).intersection(stele_polygon)
            if outer.is_empty:
                continue

            # Mid-ring polygon used for seed placement
            mid_ring = xylem_polygon.buffer(cell_diameter / 2).intersection(stele_polygon)
            if mid_ring.is_empty or mid_ring.geom_type != "Polygon":
                continue

            seed_coords = CellGenerator.cells_on_layer(mid_ring, cell_diameter)

            for pt in seed_coords[1:]:  # seed_coords[0] duplicates the last point
                new_cell = Cell(
                    type="stele",
                    x=pt[0],
                    y=pt[1],
                    diameter=cell_diameter,
                    id_cell=next_id_group,
                    id_layer=0,
                    id_group=next_id_group,
                    angle=np.arctan2(pt[1] - center.y, pt[0] - center.x),
                    radius=np.sqrt((pt[0] - center.x)**2 + (pt[1] - center.y)**2),
                    area=np.pi * (cell_diameter / 2) ** 2,
                )
                self.vascular_cells.add_cell(new_cell)
                next_id_group += 1

            # Clear stele seeds from the ring so only sheath seeds occupy it
            ring_polygon = outer.difference(xylem_polygon)
            if not ring_polygon.is_empty:
                self.vascular_polygons.append(ring_polygon)

    def _which_layer_for_vascular(self, layers_polygons: List[Dict[str, Any]]):
        """
        Find the layer where vascular tissue will be allocated.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        """
        layer_for_vascular = [l["name"] for l in layers_polygons].index("stele")
        polygon_for_vascular = layers_polygons[layer_for_vascular]["polygon"]
        return polygon_for_vascular

    # ------------------------------------------------------------------
    # Secondary growth — helper methods
    # ------------------------------------------------------------------

    def _build_primary_cambium_polygon(
        self, stele_polygon: Polygon, cx: float, cy: float
    ) -> Polygon:
        """Return the primary cambium star as a filled polygon (inner boundary of annular zone)."""
        p = self.vascular_params
        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)

        outer_r = min(p["cambium_primary_outer_distance"], stele_r)
        inner_r = min(p["cambium_primary_inner_distance"], outer_r)

        raw_star = GeometryProcessor.star_polygon(
            n_branches=p["n_vascular_peak"],
            r_min=inner_r,
            r_max=outer_r,
            arc_base=p["cambium_primary_arc_bottom"],
            arc_top=p["cambium_primary_arc_top"],
        )
        star_coords = GeometryProcessor.smoothing_polygon(
            np.column_stack(raw_star.exterior.xy),
            smooth_factor=0.9,
            iterations=5,
        )
        star = Polygon(star_coords).buffer(0)
        return translate(star, cx, cy).intersection(stele_polygon)

    def _build_secondary_cambium_polygon(
        self, stele_polygon: Polygon, cx: float, cy: float
    ) -> Polygon:
        """Return the secondary cambium star as a filled polygon (outer boundary of annular zone)."""
        sc = self.secondary_cambium_params
        p  = self.vascular_params
        _, _, stele_r = GeometryProcessor._chebyshev_center(stele_polygon)

        outer_r = min(sc["outer_distance"], stele_r)
        inner_r = min(sc["inner_distance"], outer_r)

        raw_star = GeometryProcessor.star_polygon(
            n_branches=p["n_vascular_peak"],
            r_min=inner_r,
            r_max=outer_r,
            arc_base=sc["arc_bottom"],
            arc_top=sc["arc_top"],
        )
        star_coords = GeometryProcessor.smoothing_polygon(
            np.column_stack(raw_star.exterior.xy),
            smooth_factor=0.9,
            iterations=5,
        )
        star = Polygon(star_coords).buffer(0)
        return translate(star, cx, cy).intersection(stele_polygon)

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
        if isinstance(geometry, Polygon):
            line_segs = [geometry.exterior]
        elif hasattr(geometry, "geoms"):
            line_segs = list(geometry.geoms)
        else:
            line_segs = [geometry]

        next_id_group = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0

        for line_seg in line_segs:
            raw_coords = np.array(line_seg.coords)
            seg_length = line_seg.length

            n_cells = max(2, int(np.ceil(seg_length / (cell_width or cell_diam))))
            cells_coords = GeometryProcessor.resample_coords(raw_coords, n_cells)

            if len(cells_coords) < 2:
                continue

            cell_borders = CellGenerator.cell_border(
                cells_coords,
                cell_width * 0.7 if cell_width else cell_diam * 0.7,
                cell_diam  * 0.7 if cell_width else 0,
            )

            for i, _coord in enumerate(cells_coords[1:]):
                if xylem_union and xylem_union.contains(Point(_coord[0], _coord[1])):
                    next_id_group += 1
                    continue
                id_group = next_id_group
                next_id_group += 1
                for border_pt in cell_borders[i][1:]:
                    self.vascular_cells.add_cell(Cell(
                        type=cell_type,
                        x=border_pt[0],
                        y=border_pt[1],
                        diameter=cell_diam,
                        id_cell=id_group,
                        id_layer=0,
                        id_group=id_group,
                        angle=np.arctan2(border_pt[1] - cy, border_pt[0] - cx),
                        radius=np.sqrt((border_pt[0] - cx) ** 2 + (border_pt[1] - cy) ** 2),
                        area=np.pi * (cell_diam / 2) ** 2,
                    ))

    def _fill_zone_with_cells(
        self,
        fill_zone,
        cell_diameter: float,
        cell_width: float,
        cell_type: str,
        cx: float,
        cy: float,
        start_id: int,
    ) -> int:
        """Fill a polygon zone with parenchyma seeds on concentric inward rings.

        Reuses the layer-filling algorithm (cells_on_layer + iterative buffer) used
        elsewhere in the project for ring-based cell placement.

        Args:
            fill_zone:     Polygon (or MultiPolygon) to fill.
            cell_diameter: Target cell diameter.
            cell_width:    Tangential cell width (0 = use diameter).
            cell_type:     Cell type string written to each seed.
            cx, cy:        Stele centre, used to compute angle/radius.
            start_id:      id_group / id_cell for the first new cell.

        Returns:
            Next available id (start_id + number of seeds placed).
        """
        if fill_zone is None or fill_zone.is_empty:
            return start_id
        if fill_zone.area < np.pi * (cell_diameter / 2) ** 2:
            return start_id

        next_id = start_id
        space   = cell_diameter / 2
        current = fill_zone

        while not current.is_empty and current.area > (cell_diameter / 2) ** 2 * np.pi:
            current = current.buffer(-space - cell_diameter / 2, resolution=16)
            if current.is_empty:
                break
            space = cell_diameter / 2

            geoms = list(current.geoms) if hasattr(current, 'geoms') else [current]
            for geom in geoms:
                if geom.is_empty or geom.geom_type != "Polygon":
                    continue
                seed_coords = CellGenerator.cells_on_layer(geom, cell_diameter, cell_width)
                for pt in seed_coords[1:]:
                    self.vascular_cells.add_cell(Cell(
                        type=cell_type,
                        x=pt[0],
                        y=pt[1],
                        diameter=cell_diameter,
                        id_cell=next_id,
                        id_layer=0,
                        id_group=next_id,
                        angle=np.arctan2(pt[1] - cy, pt[0] - cx),
                        radius=np.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2),
                        area=np.pi * (cell_diameter / 2) ** 2,
                    ))
                    next_id += 1

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
        """Fill angular gaps between pizza slices with radially-oriented ray parenchyma.

        Strategy: for each gap (one per xylem peak), place radial dividing lines spaced
        parenchyma_width apart at the inner edge. As the radius increases the arc width
        between two adjacent lines grows; when it exceeds parenchyma_width + 3*sd a new
        line is inserted midway (binary split). Each cell is seeded with an elliptical
        ring of border points (major axis = parenchyma_diameter in the radial direction,
        minor axis = current lane arc width in the tangential direction) so that the
        Voronoi tessellation produces radially elongated cell territories.

        Args:
            vessel_zones: List of smoothed pizza-slice polygons (may contain None).
            annular_zone: Full annular zone between primary and secondary cambium.
            cx, cy:       Stele centre coordinates.
            sx:           Secondary xylem parameter dict.
            r_outer:      Outer radius of the annular zone (secondary cambium side).
            n_peaks:      Number of xylem peaks (= number of angular gaps).
            start_id:     Starting id_group for new cells.

        Returns:
            Next available id.
        """
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

        # Pre-compute border seed angles (8 points on the unit ellipse)
        n_border   = 8
        phi        = np.linspace(0.0, 2.0 * np.pi, n_border, endpoint=False)
        border_cos = np.cos(phi)
        border_sin = np.sin(phi)
        border_scale = 0.7

        next_id = start_id

        for k in range(n_peaks):
            theta_c  = 2.0 * np.pi * k / n_peaks
            theta_lo = theta_c - gap_half
            theta_hi = theta_c + gap_half

            # Start from the primary cambium inner radius (the shallowest part of
            # the star) so the loop covers the full radial depth of the gap.
            # ray_zone.contains() is the sole placement guard: seeds that fall
            # inside the primary cambium polygon are automatically skipped, so no
            # explicit ray-cast is needed to find the inner boundary.
            r_start = max(self.vascular_params["cambium_primary_inner_distance"], d_cell)

            # Initial lanes: arc width ≈ w_cell at the calibration radius
            init_spacing = w_cell / r_start
            n_init = max(1, int(np.ceil((theta_hi - theta_lo) / init_spacing)))
            lines  = list(np.linspace(theta_lo, theta_hi, n_init + 1))
            thresholds = [
                float(np.clip(
                    np.random.uniform(0.7, 1.3) * split_threshold,
                    0.5 * split_threshold,
                    1.5 * split_threshold,
                ))
                for _ in range(len(lines) - 1)
            ]

            r = r_start + d_cell / 2.0
            while r <= r_outer:
                # Binary split: insert a midpoint line whenever a lane's arc width
                # exceeds its per-lane threshold. Child thresholds inherit the
                # parent's value plus independent Gaussian noise so lanes split at
                # staggered radii rather than a uniform grid.
                new_lines      = [lines[0]]
                new_thresholds = []
                noise_scale    = 0.1 * split_threshold
                for i in range(len(lines) - 1):
                    a1, a2 = lines[i], lines[i + 1]
                    thr = thresholds[i]
                    if (a2 - a1) * r > thr:
                        new_lines.append((a1 + a2) / 2.0)
                        t_left = float(np.clip(
                            thr + np.random.normal(0, noise_scale),
                            0.5 * split_threshold, 1.5 * split_threshold,
                        ))
                        t_right = float(np.clip(
                            thr + np.random.normal(0, noise_scale),
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

                    # Elliptical border seeds force the Voronoi to produce a
                    # radially elongated territory: major axis = d_cell (radial),
                    # minor axis = lane_arc_width (tangential).
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
                            id_cell=id_group,
                            id_layer=0,
                            id_group=id_group,
                            angle=theta_mid,
                            radius=r,
                            area=np.pi * a_rad * b_tan,
                        ))

                r += d_cell

        return next_id

    def fit_secondary_xylem(self, stele_polygon: Polygon) -> None:
        """Build secondary xylem between the primary and secondary cambium.

        Steps:
        1. Compute primary cambium polygon (inner boundary, no cells rendered).
        2. Build secondary cambium polygon and render its cells.
        3. Compute annular zone and remove stele seeds from it.
        4. Construct pizza-slice vessel zones (one per xylem peak valley).
        5. Pack secondary xylem vessels inside each zone; fill remaining space
           with axial parenchyma cells.
        6. Fill angular gaps between pizza slices with ray parenchyma cells.
        7. Register all vessel polygons for stele seed clearing.

        Args:
            stele_polygon: The stele boundary polygon.
        """
        p   = self.vascular_params
        sx  = self.secondary_xylem_params
        cx, cy = stele_polygon.centroid.x, stele_polygon.centroid.y
        n_peaks = p["n_vascular_peak"]

        # Step 1: primary cambium polygon — inner boundary of annular zone
        primary_cambium_polygon = self._build_primary_cambium_polygon(stele_polygon, cx, cy)
        if primary_cambium_polygon is None or primary_cambium_polygon.is_empty:
            return

        # Step 2: secondary cambium polygon — outer boundary; render cambium cells
        secondary_cambium_polygon = self._build_secondary_cambium_polygon(stele_polygon, cx, cy)
        if secondary_cambium_polygon is None or secondary_cambium_polygon.is_empty:
            return

        sc = self.secondary_cambium_params
        self._render_layer(secondary_cambium_polygon, "cambium", sc["cell_diameter"], sc["cell_width"], cx, cy)

        # Step 3: annular zone between the two cambium boundaries
        annular_zone = secondary_cambium_polygon.difference(primary_cambium_polygon)
        # buffer by half cell width to avoid overlaps
        shrinked_sec_cambium_pol = GeometryProcessor.buffer_polygon(secondary_cambium_polygon, +sc["cell_diameter"] / 1.5, 0)
        buffed_annular_zone = shrinked_sec_cambium_pol.difference(primary_cambium_polygon)
        
        if buffed_annular_zone.is_empty:
            return
        else: 
            # Remove stele parenchyma seeds from the annular zone (secondary growth replaces them)
            self.all_cells.remove_cells_in_polygon(buffed_annular_zone)

        # Step 4: pizza-slice vessel zones — one per valley between xylem peaks.
        # When prop_stele == 1 the slices would tile the full ring with no gaps, so
        # skip the angular split and treat the entire annular zone as one vessel zone.
        full_angle_per_slice = 2.0 * np.pi / n_peaks
        half_width = full_angle_per_slice * sx["prop_stele"] / 2.0

        vessel_zones: List = []
        if sx["prop_stele"] >= 1.0:
            if not annular_zone.is_empty and annular_zone.area >= np.pi * (sx["vessel_diameter_min"] / 2) ** 2:
                vessel_zones.append(annular_zone)
        else:
            minx, miny, maxx, maxy = secondary_cambium_polygon.bounds
            outer_r = max(maxx - cx, cx - minx, maxy - cy, cy - miny) * 1.5

            for k in range(n_peaks):
                theta = 2.0 * np.pi * (k + 0.5) / n_peaks

                if half_width < 1e-9:
                    vessel_zones.append(None)
                    continue

                arc_angles = np.linspace(theta - half_width, theta + half_width, 50)
                wedge_pts  = [(cx, cy)] + [
                    (cx + outer_r * np.cos(a), cy + outer_r * np.sin(a)) for a in arc_angles
                ]
                raw_wedge = Polygon(wedge_pts)

                zone = raw_wedge.intersection(annular_zone)
                if zone.is_empty or zone.area < np.pi * (sx["vessel_diameter_min"] / 2) ** 2:
                    vessel_zones.append(None)
                    continue

                # Smooth only simple polygons to avoid losing multi-part geometry
                if zone.geom_type == "Polygon":
                    zone_coords = GeometryProcessor.smoothing_polygon(
                        np.column_stack(zone.exterior.xy),
                        smooth_factor=0.3,
                        iterations=3,
                    )
                    smoothed = Polygon(zone_coords).buffer(0)
                    if not smoothed.is_empty and smoothed.geom_type == "Polygon":
                        zone = smoothed

                vessel_zones.append(zone)

        # Step 5: pack vessels and fill axial parenchyma within each zone
        all_vessel_polys: List[Polygon] = []
        next_id = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0

        for zone in vessel_zones:
            if zone is None or zone.is_empty:
                continue

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
            )
            zone_vessel_polys: List[Polygon] = []

            for pcx, pcy, r in packed:
                actual_diam = r * 2
                actual_r    = r

                placed      = Point(pcx, pcy).buffer(actual_r, resolution=32)
                placed_buff = placed.buffer(-actual_r * 0.15)
                if placed_buff.is_empty:
                    continue

                bx, by = placed_buff.exterior.coords.xy
                border_coords = GeometryProcessor.resample_coords(
                    np.column_stack((bx, by)), target_n_points=25
                )
                center   = placed.centroid
                id_group = next_id
                next_id += 1

                for border_pt in border_coords[1:]:
                    self.vascular_cells.add_cell(Cell(
                        type="xylem",
                        x=border_pt[0],
                        y=border_pt[1],
                        diameter=actual_diam,
                        id_cell=id_group,
                        id_layer=0,
                        id_group=id_group,
                        angle=np.arctan2(border_pt[1] - center.y, border_pt[0] - center.x),
                        radius=np.sqrt(
                            (border_pt[0] - center.x) ** 2 + (border_pt[1] - center.y) ** 2
                        ),
                        area=np.pi * actual_r ** 2,
                    ))

                zone_vessel_polys.append(placed)
                all_vessel_polys.append(placed)

            # Axial parenchyma: fills the non-vessel area inside the pizza-slice zone
            if zone_vessel_polys:
                vessel_union_in_zone = unary_union(zone_vessel_polys)
                axial_zone = zone.difference(vessel_union_in_zone)
            else:
                axial_zone = zone

            next_id = self._fill_zone_with_cells(
                axial_zone,
                sx["cell_diameter"],
                sx["cell_width"],
                "stele",
                cx, cy,
                next_id,
            )

        # Step 6: ray parenchyma in angular gaps between pizza slices
        # Use the actual maximum radius of the secondary cambium polygon (not the
        # area-equivalent circle, which undershoots the star-peak radius and causes
        # the radial loop to stop before reaching the outer boundary of ray_zone).
        r_outer = max(
            np.hypot(x - cx, y - cy)
            for x, y in secondary_cambium_polygon.exterior.coords
        )

        next_id = self._fill_ray_parenchyma(
            vessel_zones, annular_zone, cx, cy, sx, r_outer, n_peaks, next_id,
        )

        # Step 7: register vessel circles so stele seeds inside them are cleared
        # remove cells inside vessel circles
        # for vessel_poly in all_vessel_polys:
        #     self.all_cells.remove_cells_by_polygon(vessel_poly)

        self.vascular_polygons.extend(all_vessel_polys)

    def _organ_specific_tissues(self):
        """
        Add organ specific tissues.
        """
        pass


