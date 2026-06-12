"""
Root anatomy implementation.
"""

import numpy as np
from typing import List, Dict, Any

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.affinity import translate, scale as affine_scale, rotate

from openalea.granap.organ_class import Organ
from openalea.granap.layer_class import Layer
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.input_data import OrganInputData


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
        
    def _initialize_params(self) -> None:
        """Parse the structured input and set local attributes."""
        # 1. Global params
        self.global_params = next((p for p in self.params if p["name"] == "planttype"), {})

        # 2. Vascular / Stele params — all vascular info lives in the "stele" dict
        stele = next((p for p in self.params if p["name"] == "stele"), {})

        self.vascular_params = {
            "thickness":           stele["thickness"],
            "cell_diameter":       stele["cell_diameter"],
            # 5PL gradient — fall back to flat (no gradient) when the field is absent (e.g. XML input)
            "cell_diameter_max":        stele.get("cell_diameter_max",        stele["cell_diameter"]),
            "size_gradient_inflection": stele.get("size_gradient_inflection", 0.5),
            "size_gradient_steepness":  stele.get("size_gradient_steepness",  3.0),
            "size_gradient_asymmetry":  stele.get("size_gradient_asymmetry",  1.0),
            "xylem_diameter":          stele["xylem_diameter"],
            "xylem_diameter_sd":       float(stele.get("xylem_diameter_sd", 0.0)),
            "protoxylem_diameter":     stele["protoxylem_diameter"],
            "protoxylem_diameter_sd":  float(stele.get("protoxylem_diameter_sd", 0.0)),
            "phloem_diameter":         stele["phloem_diameter"],
            "phloem_diameter_sd":      float(stele.get("phloem_diameter_sd", 0.0)),
            "n_phloem_per_bundle":      int(stele.get("n_phloem_per_bundle", 1)),
            "n_protoxylem_per_bundle":  int(stele.get("n_protoxylem_per_bundle", 1)),
            "n_vascular_bundles":   int(stele["n_vascular_bundles"]),
            "ratio_proto_meta":    stele["ratio_proto_meta"],
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
    
    def _stele_cell_diameter_5pl(self, r_norm: float) -> float:
        """Return the stele cell diameter at normalized radius *r_norm* ∈ [0, 1].

        Uses the 5-parameter logistic (5PL) model::

            f(r) = d + (a - d) / (1 + (r / c)^b)^m

        Parameters
        ----------
        r_norm : float
            Normalized radial position: 0 = stele centre, 1 = stele edge.

        Returns
        -------
        float
            Cell diameter predicted by the 5PL at position *r_norm*.

        Notes
        -----
        Mapping to vascular_params:

        * ``a`` = ``cell_diameter_max``  — upper asymptote (value at r = 0, centre)
        * ``d`` = ``cell_diameter``      — lower asymptote (value at r → ∞, edge)
        * ``c`` = ``size_gradient_inflection`` — inflection position on [0, 1]
        * ``b`` = ``size_gradient_steepness``  — Hill coefficient (transition sharpness)
        * ``m`` = ``size_gradient_asymmetry``  — skew parameter
        """
        a = self.vascular_params["cell_diameter_max"]
        d = self.vascular_params["cell_diameter"]
        if a == d:
            return d  # gradient disabled — fast path

        c = self.vascular_params["size_gradient_inflection"]
        b = self.vascular_params["size_gradient_steepness"]
        m = self.vascular_params["size_gradient_asymmetry"]

        if r_norm <= 0.0:
            return float(a)
        return float(d + (a - d) / (1.0 + (r_norm / c) ** b) ** m)

    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create stele parenchyma rings from the stele edge toward the centre.

        Cell diameter follows the 5PL gradient defined in :meth:`_stele_cell_diameter_5pl`:
        rings near the periphery receive ``cell_diameter`` (lower asymptote) and rings
        near the centre receive ``cell_diameter_max`` (upper asymptote).  When both
        values are equal the gradient is flat and behaviour is identical to the uniform distribution.

        Args:
            current_polygon: Innermost polygon after all outer layers have been built.
            params: Layer parameter dictionaries (used only to compute *i_layer* offset).

        Returns:
            List of central layer polygon dictionaries.
        """
        central_layers = []

        # Stele radius at entry — used to normalise radial position for the 5PL.
        stele_radius = np.sqrt(current_polygon.area / np.pi)

        # First space increment: half the cell diameter of the innermost non-stele layer.
        min_order = min(l.order for l in self.layer_manager.get_layers() if l.order > 0)
        space_increment = self.layer_manager.get_layer_by_order(min_order).cell_diameter / 2

        i_layer = len(params)

        while not current_polygon.is_empty and current_polygon.area > 0:
            # Normalized radius of the current ring (outer edge of the ring to be placed).
            r_norm = np.clip(np.sqrt(current_polygon.area / np.pi) / stele_radius, 0.0, 1.0)
            cell_diameter = self._stele_cell_diameter_5pl(r_norm)

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

    def _create_vascular_tissue(self, polygon_for_vascular: Polygon, debug = False):
        """
        Create vascular tissue (xylem and phloem).
        """
        if self.vascular_params["n_vascular_bundles"] == 0:
            return
        
        self.fit_metaxylem_elements(polygon_for_vascular)
        self.fit_metaxylem_sheath(polygon_for_vascular)

        self.fit_phloem_protoxylem_elements(polygon_for_vascular)
        # remove the cells in the vascular elements
        vascular_polygons = unary_union(self.vascular_polygons)
        self.all_cells.remove_cells_in_polygon(vascular_polygons)

        # add vascular cells to all_cells
        self.all_cells.extend_cells(self.vascular_cells.cells)
        self.all_cells.recalculate_cell_properties()
        if debug:
            self.all_cells.plot_cells()

        

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
        list_polygons = []
        cells_in_slice = CellManager()

        # Sample a diameter independently for each cell in this bundle
        n_protoxylem_per_bundle = self.vascular_params["n_protoxylem_per_bundle"]
        protoxylem_diameters = [
            float(np.clip(
                np.random.normal(self.vascular_params["protoxylem_diameter"],
                                 self.vascular_params["protoxylem_diameter_sd"]),
                self.vascular_params["protoxylem_diameter"] * 0.1,
                np.inf,
            ))
            for _ in range(n_protoxylem_per_bundle)
        ]

        # Available space in this pizza slice
        bundle_cx, bundle_cy, available_r = GeometryProcessor.get_inscribed_circle(slice_poly)

        # Pack with per-cell sizes; parent_r gives the enclosing radius
        small_circles, parent_circle = GeometryProcessor.pack_circles_variable(protoxylem_diameters)
        parent_r = parent_circle.bounds[2]

        # Scale down uniformly only when the pack does not fit in the available space
        scale = min(1.0, available_r / parent_r) if parent_r > 0 else 1.0
        actual_diameters = [d * scale for d in protoxylem_diameters]
        if scale < 1.0:
            small_circles = [affine_scale(c, scale, scale, origin=(0, 0)) for c in small_circles]
            parent_circle  = affine_scale(parent_circle, scale, scale, origin=(0, 0))

        # Each packed cell needs its own id_group so Voronoi dissolve keeps them separate
        next_id_group = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0

        # Rotate the cluster so its first axis aligns with the radial direction
        radial_angle_deg = np.degrees(np.arctan2(bundle_cy, bundle_cx))

        for i_cell, small_circle in enumerate(small_circles):
            placed = translate(
                rotate(small_circle, radial_angle_deg, origin=(0, 0)),
                bundle_cx, bundle_cy,
            )
            cell_id_group = next_id_group + i_cell
            cell_diam = actual_diameters[i_cell]

            # Border seed points around each placed circle (same pattern as phloem)
            placed_buff = placed.buffer(-(cell_diam / 2) * 0.15)
            bx, by = placed_buff.exterior.coords.xy
            border_coords = GeometryProcessor.resample_coords(
                np.column_stack((bx, by)), target_n_points=24
            )
            center = placed.centroid

            for border_pt in border_coords[1:]:
                new_cell = Cell(
                    type="protoxylem",
                    x=border_pt[0],
                    y=border_pt[1],
                    diameter=cell_diam,
                    id_cell=cell_id_group,
                    id_layer=0,
                    id_group=cell_id_group,
                    angle=np.arctan2(border_pt[1] - center.y, border_pt[0] - center.x),
                    radius=np.sqrt((border_pt[0] - center.x)**2 + (border_pt[1] - center.y)**2),
                    area=np.pi * (cell_diam / 2) ** 2,
                )
                cells_in_slice.add_cell(new_cell)

        list_polygons.append(translate(parent_circle, bundle_cx, bundle_cy))
        return cells_in_slice, list_polygons

    def phloem_elements_in_slice(self, slice_poly: Polygon):
        list_polygons = []
        cells_in_slice = CellManager()

        # Sample a diameter independently for each cell in this bundle
        n_phloem_per_bundle = self.vascular_params["n_phloem_per_bundle"]
        phloem_diameters = [
            float(np.clip(
                np.random.normal(self.vascular_params["phloem_diameter"],
                                 self.vascular_params["phloem_diameter_sd"]),
                self.vascular_params["phloem_diameter"] * 0.1,
                np.inf,
            ))
            for _ in range(n_phloem_per_bundle)
        ]

        # Available space in this pizza slice
        bundle_cx, bundle_cy, available_r = GeometryProcessor.get_inscribed_circle(slice_poly)

        # Pack with per-cell sizes; parent_r gives the enclosing radius
        small_circles, parent_circle = GeometryProcessor.pack_circles_variable(phloem_diameters)
        parent_r = parent_circle.bounds[2]

        # Scale down uniformly only when the pack does not fit in the available space
        scale = min(1.0, available_r / parent_r) if parent_r > 0 else 1.0
        actual_diameters = [d * scale for d in phloem_diameters]
        if scale < 1.0:
            small_circles = [affine_scale(c, scale, scale, origin=(0, 0)) for c in small_circles]
            parent_circle  = affine_scale(parent_circle, scale, scale, origin=(0, 0))

        # Each packed cell needs its own id_group so process_voronoi_groups does not
        # dissolve them into one polygon. Start from the current max in vascular_cells.
        next_id_group = (self.vascular_cells.get_last_id_group() + 1) if self.vascular_cells.cells else 0

        # Rotate the cluster so its first axis aligns with the radial direction
        # pointing from the stele centre to this bundle's position.
        radial_angle_deg = np.degrees(np.arctan2(bundle_cy, bundle_cx))

        for i_cell, small_circle in enumerate(small_circles):
            placed = translate(
                rotate(small_circle, radial_angle_deg, origin=(0, 0)),
                bundle_cx, bundle_cy,
            )
            cell_id_group = next_id_group + i_cell
            cell_diam = actual_diameters[i_cell]

            # Border seed points around each placed circle so each cell's Voronoi
            # territory is properly walled off against neighbouring xylem parenchyma seeds.
            placed_buff = placed.buffer(-(cell_diam / 2) * 0.15)
            bx, by = placed_buff.exterior.coords.xy
            border_coords = GeometryProcessor.resample_coords(
                np.column_stack((bx, by)), target_n_points=24
            )
            center = placed.centroid

            for border_pt in border_coords[1:]:
                new_cell = Cell(
                    type="phloem",
                    x=border_pt[0],
                    y=border_pt[1],
                    diameter=cell_diam,
                    id_cell=cell_id_group,
                    id_layer=0,
                    id_group=cell_id_group,
                    angle=np.arctan2(border_pt[1] - center.y, border_pt[0] - center.x),
                    radius=np.sqrt((border_pt[0] - center.x)**2 + (border_pt[1] - center.y)**2),
                    area=np.pi * (cell_diam / 2) ** 2,
                )
                cells_in_slice.add_cell(new_cell)

        # Claim the bundle area as vascular space so stele parenchyma seeds inside
        # it are cleared, regardless of inter-cell gaps.
        list_polygons.append(translate(parent_circle, bundle_cx, bundle_cy))
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

    def _organ_specific_tissues(self):
        """
        Add organ specific tissues.
        """
        pass

    def add_intercellular_spaces(self):
        """Orchestrate intercellular space and aerenchyma generation."""
        self.add_intercellular()
        self.add_aerenchyma()
        self.merge_intercellular_aerenchyma()

    def add_intercellular(self):
        """Compute air spaces for each inter_cellular_spaces entry.

        Each entry may list one or more tissues. When multiple tissues are given,
        cells from all of them are processed together so that intercellular spaces
        are also generated at the boundary between adjacent tissues.
        Smoothness can be a single float (applied to every tissue) or a list with
        one value per tissue.
        """
        for ics in self.intercellular_spaces_params:
            self._apply_intercellular(ics)

    def _apply_intercellular(self, ics: dict) -> None:
        """Apply one inter_cellular_spaces entry to the relevant tissue cells."""
        tissues = ics.get("tissue", [])
        if isinstance(tissues, str):
            tissues = [tissues]
        if not tissues:
            return

        smoothness = ics.get("smoothness", 0)
        if isinstance(smoothness, (int, float)):
            smoothness_per_tissue = [float(smoothness)] * len(tissues)
        else:
            smoothness_per_tissue = [float(s) for s in smoothness]

        if not any(smoothness_per_tissue):
            return

        # Collect cells from all tissues, tracking the smoothness for each cell
        all_tissue_cells = []
        cell_smoothness: dict = {}
        for tissue_name, s in zip(tissues, smoothness_per_tissue):
            cells = self.all_cells.get_cells_by_type(tissue_name)
            for c in cells:
                cell_smoothness[id(c)] = s
            all_tissue_cells.extend(cells)

        tissue_polys = [c.polygon for c in all_tissue_cells if c.polygon is not None]
        if len(tissue_polys) < 2:
            return

        full_union = GeometryProcessor.union_polygons(tissue_polys)
        min_diameter = min(c.diameter for c in all_tissue_cells)
        full_union_buffed = full_union.buffer(-min_diameter * 0.5)

        smoothed = []
        for cell in all_tissue_cells:
            if cell.polygon is None:
                continue
            s = cell_smoothness[id(cell)]
            shrunk = GeometryProcessor.buffer_polygon(cell.polygon, 0, smooth_factor=s)
            if not shrunk.is_empty:
                smoothed.append(shrunk)

        if not smoothed:
            return

        smoothed_union = GeometryProcessor.union_polygons(smoothed)
        air_region = full_union.difference(smoothed_union)

        if isinstance(air_region, MultiPolygon):
            raw_air_polys = list(air_region.geoms)
        elif air_region.is_empty:
            return
        else:
            raw_air_polys = [air_region]

        r_values = [np.sqrt(p.area / np.pi) for p in tissue_polys]
        tol = float(np.median(r_values)) * 0.05

        air_space_polys = []
        for poly in raw_air_polys:
            if poly.intersects(full_union_buffed):
                simplified = poly.simplify(tol, preserve_topology=True)
                if not simplified.is_empty and simplified.area > 1E-6:
                    air_space_polys.append(simplified)

        if not air_space_polys:
            return

        air_union = GeometryProcessor.union_polygons(air_space_polys)

        for cell in all_tissue_cells:
            if cell.polygon is None:
                continue
            carved = cell.polygon.difference(air_union)
            if not carved.is_empty and carved.area > 1E-6:
                cell.polygon = carved
            else:
                cell.polygon = None

        id_cell = len(self.all_cells.cells)
        for air_space_polygon in air_space_polys:
            id_cell += 1
            self.all_cells.cells.append(Cell(
                x=air_space_polygon.centroid.x,
                y=air_space_polygon.centroid.y,
                diameter=np.sqrt(air_space_polygon.area / np.pi) * 2,
                id_cell=id_cell,
                id_layer=0,
                id_group=id_cell,
                type="air space",
                polygon=air_space_polygon,
            ))

        self.all_cells.cells = CellGenerator.simplify_cells(self.all_cells.cells)

    def add_aerenchyma(self):
        """Generate aerenchyma in the tissue defined in aerenchyma_params."""
        aerenchyma_prop = self.aerenchyma_params.get("aerenchyma_proportion", 0)
        if not aerenchyma_prop:
            return

        tissue = self.aerenchyma_params.get("tissue")
        n_files = int(self.aerenchyma_params.get("n_files", 1))
        aerenchyma_type = int(self.aerenchyma_params.get("aerenchyma_type", 1))

        self._aerenchyma_n_files = n_files
        self._aerenchyma_start_angle = np.random.uniform(0, 2 * np.pi)
        start_angle = self._aerenchyma_start_angle

        def cell_quadrant(cell):
            cell_angle = np.arctan2(cell.y, cell.x) % (2 * np.pi)
            rel = (cell_angle - start_angle) % (2 * np.pi)
            return int(rel / (2 * np.pi / n_files)) % n_files

        if aerenchyma_prop > 1:
            print("Aerenchyma proportion is greater than 1, setting it to 1")
            aerenchyma_prop = 1

        tissue_cells = self.all_cells.get_cells_by_type(tissue)
        if not tissue_cells:
            return

        max_tissue_layer = max(c.id_layer for c in tissue_cells)
        candidates = [c for c in tissue_cells if c.id_layer < max_tissue_layer]
        candidates.extend(self.all_cells.get_cells_by_type("air space"))

        if not candidates:
            return

        total_tissue_area = sum(c.polygon.area for c in tissue_cells if c.polygon is not None)
        total_air_area = sum(c.polygon.area for c in self.all_cells.get_cells_by_type("air space") if c.polygon is not None)
        max_possible_area = sum(c.polygon.area for c in candidates if c.polygon is not None)

        target_aerenchyma_area = (total_tissue_area + total_air_area) * aerenchyma_prop

        if target_aerenchyma_area > max_possible_area:
            print(f"Warning: asked proportion ({aerenchyma_prop:.2f}) requires {target_aerenchyma_area:.2f} area, which is greater than available cells ({max_possible_area:.2f}). Lowering aerenchyma_proportion.")
            aerenchyma_prop = max_possible_area / (total_tissue_area + total_air_area)
            target_aerenchyma_area = max_possible_area

        print(f"Targeted aerenchyma prop: {(target_aerenchyma_area / (total_tissue_area + total_air_area)):.3f}")

        target_per_quadrant = (target_aerenchyma_area - total_air_area) / n_files # ((n_files) ** 1.12 + 1)

        quadrant_buckets = [[] for _ in range(n_files)]
        for c in candidates:
            quadrant_buckets[cell_quadrant(c)].append(c)

        if aerenchyma_type == 1:
            for q, bucket in enumerate(quadrant_buckets):
                central_angle = (start_angle + (q + 0.5) * 2 * np.pi / n_files) % (2 * np.pi)
                def _ang_dist(cell, ca=central_angle):
                    a = np.arctan2(cell.y, cell.x) % (2 * np.pi)
                    d = abs(a - ca)
                    return min(d, 2 * np.pi - d)
                bucket.sort(key=_ang_dist)
        elif aerenchyma_type == 2:
            for q, bucket in enumerate(quadrant_buckets):
                if not bucket:
                    continue
                central_angle = (start_angle + (q + 0.5) * 2 * np.pi / n_files) % (2 * np.pi)
                def _ang_dist_seed(cell, ca=central_angle):
                    a = np.arctan2(cell.y, cell.x) % (2 * np.pi)
                    d = abs(a - ca)
                    return min(d, 2 * np.pi - d)
                seed = min(bucket, key=_ang_dist_seed)
                bucket.sort(key=lambda c, s=seed: np.hypot(c.x - s.x, c.y - s.y))

        quadrant_area = [0.0] * n_files
        quadrant_idx = [0] * n_files

        changed = True
        while changed:
            changed = False
            for q in range(n_files):
                if quadrant_area[q] >= target_per_quadrant:
                    continue
                bucket = quadrant_buckets[q]
                while quadrant_idx[q] < len(bucket):
                    cell = bucket[quadrant_idx[q]]
                    quadrant_idx[q] += 1
                    if cell.type != "air space" and cell.polygon is not None:
                        cell.type = "air space"
                        quadrant_area[q] += cell.polygon.area
                        changed = True
                        break

        tissue = self.aerenchyma_params.get("tissue")
        total_tissue_area = sum(c.polygon.area for c in self.all_cells.get_cells_by_type(tissue) if c.polygon is not None)
        total_air_area = sum(c.polygon.area for c in self.all_cells.get_cells_by_type("air space") if c.polygon is not None)
        print(f"Actual aerenchyma prop: {(total_air_area / (total_tissue_area + total_air_area)):.3f}")

    def merge_intercellular_aerenchyma(self):
        """Fuse touching air-space cells within the same angular sector, then carve tissue cells."""
        from collections import defaultdict

        n_files = getattr(self, '_aerenchyma_n_files', 1)
        start_angle = getattr(self, '_aerenchyma_start_angle', 0.0)

        def cell_quadrant(cell):
            cell_angle = np.arctan2(cell.y, cell.x) % (2 * np.pi)
            rel = (cell_angle - start_angle) % (2 * np.pi)
            return int(rel / (2 * np.pi / n_files)) % n_files

        seen_ids: set = set()
        merge_pool = []
        for c in list(self.all_cells.cells):
            if c.type == "air space" and c.polygon is not None:
                oid = id(c)
                if oid not in seen_ids:
                    seen_ids.add(oid)
                    merge_pool.append(c)

        if merge_pool:
            n_pool = len(merge_pool)
            parent = list(range(n_pool))
            cell_quadrants = [cell_quadrant(c) for c in merge_pool]

            def _find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def _union(i, j):
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

            for i in range(n_pool):
                for j in range(i + 1, n_pool):
                    if cell_quadrants[i] != cell_quadrants[j]:
                        continue
                    if merge_pool[i].polygon.touches(merge_pool[j].polygon) or merge_pool[i].polygon.intersects(merge_pool[j].polygon):
                        _union(i, j)

            groups: dict = defaultdict(list)
            for i, c in enumerate(merge_pool):
                groups[_find(i)].append(c)

            fused_cells = []
            for group in groups.values():
                if len(group) == 1:
                    fused_cells.append(group[0])
                    continue
                fused_polygon = unary_union([c.polygon for c in group])
                fused_cells.append(Cell(
                    x=fused_polygon.centroid.x,
                    y=fused_polygon.centroid.y,
                    diameter=np.sqrt(fused_polygon.area / np.pi) * 2,
                    id_cell=min(c.id_cell for c in group),
                    id_layer=int(np.ceil(np.mean([c.id_layer for c in group]))),
                    id_group=min(c.id_group for c in group),
                    type="air space",
                    polygon=fused_polygon,
                ))

            self.all_cells.remove_cells_by_ids([c.id_cell for c in merge_pool])
            self.all_cells.cells.extend(fused_cells)

        self.all_cells.cells = CellGenerator.simplify_cells(self.all_cells.cells)

        # Carve tissue cells that are trapped inside air spaces
        tissue = self.aerenchyma_params.get("tissue")
        air_spaces = self.all_cells.get_cells_by_type("air space")
        tissue_cells = self.all_cells.get_cells_by_type(tissue)
        tissue_cells.extend(a for a in air_spaces if a.id_layer == 0)

        air_union = unary_union([a.polygon for a in air_spaces if a.polygon is not None and a.id_layer != 0])

        for cell in tissue_cells:
            carved = cell.polygon.difference(air_union)
            if not carved.is_empty and carved.area > 1E-6:
                cell.polygon = carved
            else:
                self.all_cells.remove_cells_by_ids([cell.id_cell])


