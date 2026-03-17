"""
Root anatomy implementation.
"""

import numpy as np
from typing import List, Dict, Any

from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely.affinity import translate

from granap.organ_class import Organ
from granap.layer_class import Layer
from granap.cell_class import Cell
from granap.cell_manager import CellManager
from granap.geometry_collection import GeometryProcessor
from granap.generate_cell import CellGenerator


class RootAnatomy(Organ):
    """
    Root cross-sectional anatomy.
    
    Implements the typical structure of plant roots with
    circular cross-section and vascular cylinder.
    """
    
    from granap.input_data import OrganInputData

    def __init__(self, input_data: Any = None):
        """
        Initialize root anatomy.
        """
        super().__init__()
        # Initialize parameters from input_data or default
        if hasattr(input_data, 'params'):
            self.params = input_data.params
        elif isinstance(input_data, list):
            self.params = input_data
        else:
            self.params = None

        if self.params is None:
            self._initialize_default_layers()

        # Root specific parameters
            self.vascular_params = {
                "name": "stele",
                "thickness": 0.27, # mm
                "cell_diameter": 0.01, # mm
                "xylem_diameter": 0.063, # mm
                "protoxylem_diameter": 0.02, # mm
                "phloem_diameter": 0.012, # mm
                "n_vascular_bundles": 5, # number of metaxylem
                "ratio_proto_meta": 2.2 # ratio of protoxylem to metaxylem
            }
            self.intercellular_spaces_params = {
                "name": "inter_cellular_space",
                "cortex": 0,
                "aerenchyma_proportion": 0.0,
                "aerenchyma_type": 1,
                "n_files": 2,
            }
            self.global_params = {
                "name": "planttype",
                "value": 1, # monocot
                "organ": "root"
            }
            self.params = self.make_params()
        else:
            self._initialize_params()
            self._initialize_default_layers()
        
    def make_params(self):
        """Make a list of parameters from the default input data."""
        params = []
        params.append(self.global_params)
        params.append(self.vascular_params)
        params.append(self.intercellular_spaces_params)
        params.append(self.layer_manager.get_layers_params())
        return params

    def _initialize_params(self) -> None:
        """Parse the structured input and set local attributes."""
        # 1. Global params
        self.global_params = next((p for p in self.params if p["name"] == "planttype"), {})

        # 2. Vascular / Stele params
        stele = next((p for p in self.params if p["name"] == "stele"), {})
        xylem = next((p for p in self.params if p["name"] == "xylem"), {})
        phloem = next((p for p in self.params if p["name"] == "phloem"), {})

        self.vascular_params = {
            "thickness": stele.get("layer_diameter", 0.27),
            "cell_diameter": stele.get("cell_diameter", 0.01),
            "xylem_diameter": xylem.get("max_size", 0.063),
            "protoxylem_diameter": xylem.get("cell_diameter", 0.02),
            "phloem_diameter": phloem.get("cell_diameter", 0.012),
            "n_vascular_bundles": int(xylem.get("n_files", 5) if "n_files" in xylem else 5),
            "ratio_proto_meta": xylem.get("ratio", 2.2)
        }

        # 3. Intercellular spaces / aerenchyma
        inter_cellular_space = next((p for p in self.params if p["name"] == "inter_cellular_space"), {})
        aerenchyma = next((p for p in self.params if p["name"] == "aerenchyma"), {})
        self.intercellular_spaces_params = {
            "cortex": inter_cellular_space.get("size", 0),
            "aerenchyma_proportion": aerenchyma.get("proportion", 0.2),
            "aerenchyma_type": aerenchyma.get("type", 1),
            "n_files": aerenchyma.get("n_files", 2),
        }

        # 4. Extract layer definitions (any param with 'order' that is not a vascular zone)
        self.layers = [p for p in self.params if "order" in p and p["name"] not in ("stele", "xylem", "phloem", "aerenchyma")]
        self.layers = sorted(self.layers, key=lambda x: float(x["order"]))

    def _initialize_default_layers(self) -> None:
        """Initialize default root layers."""
        if hasattr(self, 'layers') and self.layers:
            for param in self.layers:
                self.layer_manager.add_layer(Layer(
                    name=param["name"],
                    cell_diameter=param.get("cell_diameter", 0.01),
                    cell_width=param.get("cell_width", param.get("cell_diameter", 0.01)),
                    shift=param.get("shift", 0.0),
                    n_layers=int(param.get("n_layers", 1)),
                    order=param.get("order", 0)
                ))
        else:
            # Outer to inner (order: higher = outer)
            self.layer_manager.add_layer(Layer(
                name="epidermis",
                cell_diameter=0.015,
                n_layers=1,
                shift=0.5,
                order=6
            ))

            self.layer_manager.add_layer(Layer(
                name="exodermis",
                cell_diameter=0.03,
                n_layers=1,
                order=5
            ))
            
            self.layer_manager.add_layer(Layer(
                name="cortex",
                cell_diameter=0.04,
                n_layers=5,
                order=4
            ))
            
            self.layer_manager.add_layer(Layer(
                name="endodermis",
                cell_diameter=0.02,
                cell_width=0.03,
                n_layers=1,
                order=3
            ))
            
            self.layer_manager.add_layer(Layer(
                name="pericycle",
                cell_diameter=0.01,
                cell_width=0.009,
                n_layers=1,
                order=2
            ))
    
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
        """
        Create vascular cylinder (xylem and phloem).
        
        Args:
            current_polygon: Current inner polygon boundary
            params: Parameter dictionaries
        
        Returns:
            List of central layer polygon dictionaries
        """
        central_layers = []
        cell_diameter = self.vascular_params["cell_diameter"]
        # first space increment is the cell diameter of the layer with the smallest order
        min_order = min([l.order for l in self.layer_manager.get_layers() if l.order > 0])
        space_increment = self.layer_manager.get_layer_by_order(min_order).cell_diameter/2
        i_layer = len(params)
        
        # Create vascular parenchyma layers
        while current_polygon.area > (cell_diameter / 2)**2 * np.pi:
            current_polygon = GeometryProcessor.buffer_polygon(
                current_polygon,
                -space_increment - cell_diameter / 2,
                smooth_factor=0.6
            )
            
            space_increment = cell_diameter / 2
            
            central_layers.append({
                "name": "vascular_parenchyma",
                "polygon": current_polygon,
                "cell_diameter": cell_diameter,
                "id_layer": i_layer + 1,
                "cell_width": 0
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

        for i, poly_slice in enumerate(slices[1:]):
            if i % 2 == 0:
                cells_in_slice, list_protoxylem_polygons = self.protoxylem_elements_in_slice(poly_slice, i)
                self.vascular_cells.extend_cells(cells_in_slice.cells)
                self.vascular_polygons.extend(list_protoxylem_polygons)
            else:
                cells_in_slice, list_phloem_polygons = self.phloem_elements_in_slice(poly_slice, i)
                self.vascular_cells.extend_cells(cells_in_slice.cells)
                self.vascular_polygons.extend(list_phloem_polygons)

    def protoxylem_elements_in_slice(self, slice_poly: Polygon, idx: int = 0):
        list_polygons = []
        cells_in_slice = CellManager()
        i_cell = 0
        
        polygon_res = GeometryProcessor.fit_inner_ellipse(slice_poly, self.vascular_params["protoxylem_diameter"]/2)
        polygon = polygon_res["polygon"]
        polygon_buff = polygon.buffer(-(self.vascular_params["protoxylem_diameter"]/2)*0.15)
        x, y = polygon_buff.exterior.coords.xy
        center = polygon.centroid
        coords = np.column_stack((x, y))
        coords = GeometryProcessor.resample_coords(coords, target_n_points=10)

        for cell_border_pts in coords[1:]:
            i_cell += 1
            new_cell = Cell(
                    type="protoxylem",
                    x=cell_border_pts[0],
                    y=cell_border_pts[1],
                    diameter=self.vascular_params["protoxylem_diameter"],
                    id_cell=i_cell,
                    id_layer=0,
                    id_group=idx,
                    angle=np.arctan2(cell_border_pts[1] - center.y, 
                                      cell_border_pts[0] - center.x),
                    radius=np.sqrt((cell_border_pts[0] - center.x)**2 + 
                                    (cell_border_pts[1] - center.y)**2),
                    area=np.pi * (self.vascular_params["protoxylem_diameter"]/2)**2
                )
            cells_in_slice.add_cell(new_cell)

        list_polygons.append(polygon)
        return cells_in_slice, list_polygons

    def phloem_elements_in_slice(self, slice_poly: Polygon, idx: int = 0):
        list_polygons = []
        cells_in_slice = CellManager()
        i_cell = 0
        
        polygon_res = GeometryProcessor.fit_inner_ellipse(slice_poly, self.vascular_params["phloem_diameter"]/2)
        polygon = polygon_res["polygon"]
        polygon_buff = polygon.buffer(-(self.vascular_params["phloem_diameter"]/2)*0.15)
        
        x, y = polygon_buff.exterior.coords.xy
        center = polygon.centroid
        coords = np.column_stack((x, y))
        coords = GeometryProcessor.resample_coords(coords, target_n_points=10)

        for cell_border_pts in coords[1:]:
            i_cell += 1
            new_cell = Cell(
                    type="phloem",
                    x=cell_border_pts[0],
                    y=cell_border_pts[1],
                    diameter=self.vascular_params["phloem_diameter"],
                    id_cell=i_cell,
                    id_layer=0,
                    id_group=idx,
                    angle=np.arctan2(cell_border_pts[1] - center.y, 
                                      cell_border_pts[0] - center.x),
                    radius=np.sqrt((cell_border_pts[0] - center.x)**2 + 
                                    (cell_border_pts[1] - center.y)**2),
                    area=np.pi * (self.vascular_params["phloem_diameter"]/2)**2
                )
            cells_in_slice.add_cell(new_cell)

        list_polygons.append(polygon)
        return cells_in_slice, list_polygons

    def fit_metaxylem_elements(self, polygon):
        # from polygon, fit two ellipses
        n_xylem_cells = self.vascular_params["n_vascular_bundles"]

        slices = GeometryProcessor.pizza_slice(polygon.buffer(-self.vascular_params["xylem_diameter"]/4), n_xylem_cells)
        cells_in_slices, list_xylem_polygons = self.vascular_elements_in_slice(slices)
        self.vascular_cells = cells_in_slices
        self.vascular_polygons = list_xylem_polygons
    
    def vascular_elements_in_slice(self, slices: List[Polygon]):
        list_xylem_polygons = []
        cells_in_slices = CellManager()
        i_cell = 0
        for i_slice, slice in enumerate(slices):
            
            xylem_polygon = GeometryProcessor.fit_inner_ellipse(slice, self.vascular_params["xylem_diameter"]/2)
            xylem_polygon = xylem_polygon["polygon"]
            xylem_polygon_buff = GeometryProcessor.buffer_polygon(xylem_polygon, -(self.vascular_params["xylem_diameter"]/2)*0.15)
            x, y = xylem_polygon_buff.exterior.coords.xy
            center = xylem_polygon.centroid
            coords = np.column_stack((x, y))
            coords = GeometryProcessor.resample_coords(coords, target_n_points=25)

            # Iterate over centers and borders together
            # coords[1:] slices the centers, cell_borders[1:] slices the corresponding borders
            for cell_border_pts in coords[1:]:
                i_cell += 1
                new_cell = Cell(
                        type="metaxylem",
                        x=cell_border_pts[0],
                        y=cell_border_pts[1],
                        diameter=self.vascular_params["xylem_diameter"],
                        id_cell=i_slice,
                        id_layer=i_slice,
                        id_group=i_slice,
                        angle=np.arctan2(cell_border_pts[1] - center.y, 
                                          cell_border_pts[0] - center.x),
                        radius=np.sqrt((cell_border_pts[0] - center.x)**2 + 
                                        (cell_border_pts[1] - center.y)**2),
                        area=np.pi * (self.vascular_params["xylem_diameter"]/2)**2
                    )
                cells_in_slices.add_cell(new_cell)

            list_xylem_polygons.append(xylem_polygon)
        return cells_in_slices, list_xylem_polygons
        
    def _which_layer_for_vascular(self, layers_polygons: List[Dict[str, Any]]):
        """
        Find the layer where vascular tissue will be allocated.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        """
        layer_for_vascular = [l["name"] for l in layers_polygons].index("vascular_parenchyma")
        polygon_for_vascular = layers_polygons[layer_for_vascular]["polygon"]
        return polygon_for_vascular

    def _organ_specific_tissues(self):
        """
        Add organ specific tissues.
        """
        pass

    def add_intercellular_spaces(self):
        """
        Compute and return intercellular (air space) polygons, and generate aerenchyma.
        """
        air_spaces_cells = CellManager()
        
        # 1. Cortex intercellular spaces (air spaces)
        if self.intercellular_spaces_params["cortex"] > 0:
            cortex_cells = self.all_cells.get_cells_by_type("cortex")
            cortex_polys = [
                c.polygon for c in cortex_cells if c.polygon is not None
            ]
            if len(cortex_polys) >= 2:
                full_union = GeometryProcessor.union_polygons(cortex_polys)
                full_union_buffed = full_union.buffer(-cortex_cells[0].diameter*0.5)

                smoothed = []
                for poly in cortex_polys:
                    shrunk = GeometryProcessor.buffer_polygon(poly, 0, smooth_factor=self.intercellular_spaces_params["cortex"])
                    if not shrunk.is_empty:
                        smoothed.append(shrunk)
        
                if smoothed:
                    smoothed_union = GeometryProcessor.union_polygons(smoothed)
                    air_region = full_union.difference(smoothed_union)
            
                    # Decompose into individual polygons
                    if isinstance(air_region, MultiPolygon):
                        raw_air_polys = list(air_region.geoms)
                    elif air_region.is_empty:
                        raw_air_polys = []
                    else:
                        raw_air_polys = [air_region]
            
                    if raw_air_polys:
                        # Simplify each air space polygon (reduce vertex count)
                        # Tolerance ~ 5 % of the median equivalent radius of mesophyll cells
                        r_values = [np.sqrt(p.area / np.pi) for p in cortex_polys]
                        tol = float(np.median(r_values)) * 0.05
                
                        air_space_polys = []
                        for poly in raw_air_polys:
                            if poly.intersects(full_union_buffed):
                                simplified = poly.simplify(tol, preserve_topology=True)
                                if not simplified.is_empty and simplified.area > 1E-6:
                                    air_space_polys.append(simplified)
                
                        air_union = GeometryProcessor.union_polygons(air_space_polys)
                
                        for cell in cortex_cells:
                            carved = cell.polygon.difference(air_union)
                            if not carved.is_empty:
                                cell.polygon = carved
                
                        # create cells for the air spaces
                        id_cell = len(self.all_cells.cells)
                        for air_space_polygon in air_space_polys:
                            id_cell += 1
                            air_space_cell = Cell(
                                x = air_space_polygon.centroid.x,
                                y = air_space_polygon.centroid.y,
                                diameter = np.sqrt(air_space_polygon.area/np.pi)*2,
                                id_cell=id_cell,
                                id_layer=0,
                                id_group=id_cell,
                                type="air space",
                                polygon=air_space_polygon,
                            )
                            air_spaces_cells.cells.append(air_space_cell)

        # 2. Aerenchyma generation logic
        aerenchyma_prop = self.intercellular_spaces_params.get("aerenchyma_proportion", 0)

        self.all_cells.cells.extend(air_spaces_cells.cells)
        self.all_cells.cells = CellGenerator.simplify_cells(self.all_cells.cells)

        if aerenchyma_prop > 0:
            cortex_cells = self.all_cells.get_cells_by_type("cortex")
            if cortex_cells:
                # Exclude the innermost cortex layer (closest to endodermis)
                max_cortex_layer = max(c.id_layer for c in cortex_cells)
                candidates = [c for c in cortex_cells if c.id_layer < max_cortex_layer]
                candidates.extend(self.all_cells.get_cells_by_type("air space"))

                if candidates:
                    total_cortex_area = sum(c.polygon.area for c in cortex_cells if c.polygon is not None)
                    total_air_area = sum(c.polygon.area for c in self.all_cells.get_cells_by_type("air space") if c.polygon is not None)
                    target_aerenchyma_area = (total_cortex_area + total_air_area) * aerenchyma_prop

                    n_files = int(self.intercellular_spaces_params.get("n_files", 1))
                    aerenchyma_type = int(self.intercellular_spaces_params.get("aerenchyma_type", 1))

                    # Per-quadrant target: each of the n_files sectors contributes equally
                    target_per_quadrant = target_aerenchyma_area / n_files

                    # Define n_files angular sectors uniformly spaced from a random start
                    start_angle = np.random.uniform(0, 2 * np.pi)

                    def cell_quadrant(cell):
                        """Return the sector index [0, n_files) for a cell."""
                        cell_angle = np.arctan2(cell.y, cell.x) % (2 * np.pi)
                        rel = (cell_angle - start_angle) % (2 * np.pi)
                        return int(rel / (2 * np.pi / n_files)) % n_files

                    # Partition candidates into per-quadrant buckets
                    quadrant_buckets = [[] for _ in range(n_files)]
                    for c in candidates:
                        q = cell_quadrant(c)
                        quadrant_buckets[q].append(c)

                    if aerenchyma_type == 1:
                        # Type 1: Radial files — sort each bucket by angular closeness
                        # to the sector's central angle
                        for q, bucket in enumerate(quadrant_buckets):
                            central_angle = (start_angle + (q + 0.5) * 2 * np.pi / n_files) % (2 * np.pi)
                            def _ang_dist(cell, ca=central_angle):
                                a = np.arctan2(cell.y, cell.x) % (2 * np.pi)
                                d = abs(a - ca)
                                return min(d, 2 * np.pi - d)
                            bucket.sort(key=_ang_dist)

                    elif aerenchyma_type == 2:
                        # Type 2: Patch — one seed per quadrant (closest to central angle),
                        # then grow by Euclidean distance from that seed
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

                    # Round-robin across quadrants: pick one cell per quadrant per round
                    # until each quadrant reaches its per-quadrant target area
                    quadrant_area = [0.0] * n_files
                    quadrant_idx = [0] * n_files  # pointer into each bucket

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

        # 3. Fuse touching aerenchyma / air-space cells using union-find
        from collections import defaultdict

        # Collect all aerenchyma/air-space cells (deduplicated by object identity)
        seen_ids: set = set()
        merge_pool = []
        for c in list(self.all_cells.cells):
            if c.type in ("air space") and c.polygon is not None:
                oid = id(c)
                if oid not in seen_ids:
                    seen_ids.add(oid)
                    merge_pool.append(c)

        if merge_pool:
            n_pool = len(merge_pool)
            parent = list(range(n_pool))

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
                    pi = merge_pool[i].polygon
                    pj = merge_pool[j].polygon
                    if pi.touches(pj) or pi.intersects(pj):
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
                fused_type = "air space"
                fused_cell = Cell(
                    x=fused_polygon.centroid.x,
                    y=fused_polygon.centroid.y,
                    diameter=np.sqrt(fused_polygon.area / np.pi) * 2,
                    id_cell=min(c.id_cell for c in group),
                    id_layer=int(round(np.mean([c.id_layer for c in group]))),
                    id_group=min(c.id_group for c in group),
                    type=fused_type,
                    polygon=fused_polygon,
                )
                fused_cells.append(fused_cell)

            # Update self.all_cells: replace merged members with fused cells
            self.all_cells.remove_cells_by_ids([c.id_cell for c in merge_pool])
            self.all_cells.cells.extend(fused_cells)

            self.all_cells.cells = CellGenerator.simplify_cells(self.all_cells.cells)

        
