"""
Plant anatomy base module providing abstract interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import geopandas as gpd
import logging
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from collections import defaultdict
from scipy.sparse import lil_matrix
import time
from openalea.granap.layer_class import Layer, LayerPolygon
from openalea.granap.layer_manager import LayerManager

log = logging.getLogger(__name__)
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.network_base import AbstractNetwork
from openalea.granap.input_data import OrganInputData
from openalea.granap.tissue_class import TissueRecipe, retag_tissue

class Organ(AbstractNetwork, ABC):
    """
    Abstract base class for plant anatomical structures.
    
    Defines the interface and common functionality for generating
    cross-sectional anatomy of different plant types.
    Inherits from AbstractNetwork for hydraulic network construction.
    """

    #: Smoothing applied when peeling each concentric layer ring in
    #: :meth:`_build_layer_polygons`. Subclasses override per organ shape
    #: (e.g. roots use 0.0 to keep ring thickness exact; needles round corners).
    LAYER_SMOOTH_FACTOR: float = 0.5

    def __init__(self, randomness: float = 1.0, seed: Optional[int] = None):
        """
        Initialize the anatomy structure.

        Args:
            randomness: Degree of randomness in cell placement (0-3)
            seed:       Optional integer seed for reproducible cell placement.
        """
        AbstractNetwork.__init__(self)
        self.rng = np.random.default_rng(seed)
        self.layer_manager = LayerManager()
        self.randomness = randomness
        self.params: List[Dict[str, Any]] = []
        self._base_polygon: Optional[Polygon] = None
        self._layers_polygons: List[LayerPolygon] = []
        self._cells_gdf: Optional[gpd.GeoDataFrame] = None
        self.all_cells = CellManager()

    @classmethod
    def create_from_input(cls, input_data: OrganInputData) -> "Organ":
        """
        Factory method to initialize the appropriate Organ subclass 
        (RootAnatomy or NeedleAnatomy) from an OrganInputData instance.
        """
        # Normalize params to plain dicts for uniform access
        params = input_data.to_dict_list() if isinstance(input_data, OrganInputData) else input_data.params

        # Determine the organ type from the parameters
        ptype_param = next((p for p in params if p["name"] == "planttype"), None)
        organ_type = None

        if ptype_param:
            if ptype_param.get("organ") == "needle" or ptype_param.get("value") == 3:
                organ_type = "needle"
            elif ptype_param.get("organ") == "root" or ptype_param.get("value") in [1, 2, 1.0, 2.0]:
                organ_type = "root"

        # Fallback to duck-typing the input parameters if 'organ' isn't explicitly defined
        if not organ_type:
            names = {p["name"] for p in params}
            if "stele" in names or "cortex" in names:
                organ_type = "root"
            else:
                organ_type = "needle"

        if organ_type == "needle":
            from openalea.granap.needle_class import NeedleAnatomy
            return NeedleAnatomy(input_data)
        else:
            from openalea.granap.root_class import RootAnatomy
            return RootAnatomy(input_data)
    
    def add_layer(self, layer: Layer, position: Optional[int] = None) -> None:
        """
        Add a tissue layer to the anatomy.
        
        Args:
            layer: Layer object to add
            position: Optional position index (None = append)
        """
        self.layer_manager.add_layer(layer, position)
        self._invalidate_geometry()
    
    def update_params(self, param_name: str, attribute: str, value: Any) -> None:
        """
        Update a parameter of the organ.
    
        self.params = [{"name": "param_name_1", "attribute_1": 0.0, "attribute_2": 0.0, ...},
                       {"name": "param_name_2", "attribute_1": 0.0, "attribute_2": 0.0, ...},
                       ...]
    
        Args:
            param_name: Name of the parameter to update
            attribute: Name of the attribute to update
            value: New value of the parameter
        """
        for p in self.params:
            if p["name"] == param_name:
                p[attribute] = value
                # Sync the corresponding Layer object in layer_manager if one exists
                layer = self.layer_manager.get_layer(param_name)
                if layer is not None:
                    if hasattr(layer, attribute):
                        setattr(layer, attribute, value)
                    else:
                        layer.additional_params[attribute] = value
                self._invalidate_geometry()
                return
        raise ValueError(f"Parameter '{param_name}' not found in params.")

    
    def remove_layer(self, name: str) -> Layer:
        """
        Remove a tissue layer by name.
        
        Args:
            name: Name identifier of the layer
        
        Returns:
            The removed Layer object    
        """
        removed = self.layer_manager.remove_layer(name)
        self._invalidate_geometry()
        return removed
    
    def get_layer(self, name: str) -> Optional[Layer]:
        """Get a layer by name."""
        return self.layer_manager.get_layer(name)
    
    def list_layers(self) -> List[str]:
        """List all layer names."""
        return [layer.name for layer in self.layer_manager.get_layers()]
    
    def _invalidate_geometry(self) -> None:
        """Invalidate cached geometry after layer changes."""
        self._base_polygon = None
        self._layers_polygons = []
        self._cells_gdf = None
    
    def generate_base_shape(self) -> Polygon:
        """
        Generate or retrieve the base shape.
        
        Returns:
            Base polygon
        """
        if self._base_polygon is None:
            self._base_polygon = self._create_base_shape()
        return self._base_polygon
    
    def generate_layer_polygons(self) -> List[LayerPolygon]:
        """
        Generate polygons for all layers.
        
        Returns:
            List of layer polygon dictionaries
        """
        if not self._layers_polygons:
            self._layers_polygons = self._build_layer_polygons()
        return self._layers_polygons
    
    def _build_layer_polygons(self) -> List[LayerPolygon]:
        """Build layer polygons from current layer configuration."""
        layers_polygons = []
        layer_array = self.layer_manager.expand_layers()
        
        polygon = self.generate_base_shape()
        
        for i_layer, layer in enumerate(layer_array):
            if i_layer == 0:
                # Add outside layer
                space_increment = layer["cell_diameter"] / 2
                polygon = GeometryProcessor.buffer_polygon(
                    polygon, space_increment, smooth_factor=0.01
                )
                layers_polygons.append(LayerPolygon(
                    name="outside",
                    polygon=polygon,
                    cell_diameter=layer["cell_diameter"] / 3,
                    id_layer=i_layer,
                ))

            # Add the layer polygon.  Keep smoothing light: smoothing_polygon
            # corner-cuts slightly *past* the requested buffer distance, so a high
            # factor applied once per peeled ring accumulates and shrinks the
            # innermost region (the stele) well below its nominal thickness.
            polygon = GeometryProcessor.buffer_polygon(
                polygon,
                -space_increment - layer["cell_diameter"] / 2,
                smooth_factor=self.LAYER_SMOOTH_FACTOR,
            )

            space_increment = layer["cell_diameter"] / 2

            layers_polygons.append(LayerPolygon(
                name=layer["name"],
                polygon=polygon,
                cell_diameter=layer["cell_diameter"],
                id_layer=i_layer + 1,
                cell_width=layer["cell_width"],
                shift=layer["shift"],
            ))
        
        # Add central layers (vascular, parenchyma, etc.)
        params = [l.to_dict() for l in self.layer_manager.get_layers()]
        central_layers = self._create_central_layers(polygon, params)
        layers_polygons.extend(central_layers)

        # Optional reshape: let subclasses morph layer polygons
        layers_polygons = self.reshape_layers(layers_polygons)
        
        return layers_polygons
    
    def generate_cells(self) -> gpd.GeoDataFrame:
        """
        Generate cell geometries using Voronoi tessellation.
        
        Returns:
            GeoDataFrame with cell geometries
        """
        if self._cells_gdf is None:
            t_start = time.time()
            layers_polygons = self.generate_layer_polygons()
            log.info("Layer polygons:          %.3fs", time.time() - t_start)
            center = layers_polygons[0]["polygon"].centroid

            t_start = time.time()
            for layer in self.layer_manager.get_layers():
                layer.cells = []
            self.all_cells = CellGenerator.generate_cells_info(layers_polygons, center, rng=self.rng)
            log.info("Cell seeds:              %.3fs", time.time() - t_start)

            t_start = time.time()
            self.vascular_cells = CellManager()
            self.vascular_polygons = []
            self.vascular_tissue_polygons: Dict[str, list] = {}
            self.allocate_vascular_tissue(layers_polygons)

            # Vascular elements take priority: remove every layer seed that falls
            # inside any vascular zone (xylem vessels + phloem + any other named
            # tissue tracked in vascular_tissue_polygons).
            all_vascular_polys = list(self.vascular_polygons)
            for poly_list in self.vascular_tissue_polygons.values():
                all_vascular_polys.extend(poly_list)
            if all_vascular_polys:
                vascular_mask = unary_union(all_vascular_polys)
                self.all_cells.remove_cells_in_polygon(vascular_mask)

            if self.vascular_cells.cells:
                self.all_cells.extend_cells(self.vascular_cells.cells)
            self._organ_specific_tissues()
            log.info("Vascular + organ tissues: %.3fs", time.time() - t_start)

            t_start = time.time()
            vor = CellGenerator.voronoi_diagram(self.all_cells, rng=self.rng)
            log.info("Voronoi diagram:         %.3fs", time.time() - t_start)

            t_start = time.time()
            grouped_cells = CellGenerator.process_voronoi_groups(self.all_cells, vor).cells
            grouped_cells = CellGenerator.simplify_cells(grouped_cells)
            self.all_cells = CellManager()
            self.all_cells.cells = grouped_cells
            log.info("Voronoi grouping:        %.3fs", time.time() - t_start)

            t_start = time.time()
            self.add_intercellular_spaces()
            log.info("Intercellular spaces:    %.3fs", time.time() - t_start)

            t_start = time.time()
            for cell in self.all_cells.cells:
                if 0 <= cell.id_layer < len(layers_polygons):
                    layer_name = layers_polygons[cell.id_layer]["name"]
                    if layer_name != "outside":
                        layer = self.get_layer(layer_name)
                        if layer:
                            layer.cells.append(cell)
            log.info("Layer population:        %.3fs", time.time() - t_start)

            t_start = time.time()
            self.all_cells.recalculate_cell_properties()
            log.info("Cell properties:         %.3fs", time.time() - t_start)

            t_start = time.time()
            cell_dicts = [c.cell_to_dict() for c in self.all_cells.cells]
            for i, c in enumerate(self.all_cells.cells):
                cell_dicts[i]['geometry'] = c.polygon
            self._cells_gdf = gpd.GeoDataFrame(cell_dicts)
            log.info("GeoDataFrame export:     %.3fs", time.time() - t_start)
        
        return self._cells_gdf

    def retag_cells(self, old_tag: str, new_tag: str) -> int:
        """Rename every cell tagged ``old_tag`` to ``new_tag``.

        Retags the live cells in ``all_cells`` and, when the cells have
        already been materialised, the cached ``_cells_gdf`` too — so plots
        and exports reflect the new tag without a full regeneration.
        Returns the number of cells retagged.
        """
        n = retag_tissue(self.all_cells, old_tag, new_tag)
        if self._cells_gdf is not None and "type" in self._cells_gdf.columns:
            self._cells_gdf.loc[self._cells_gdf["type"] == old_tag, "type"] = new_tag
        return n

    @abstractmethod
    def reshape_layers(self, layers_polygons: List[LayerPolygon]) -> List[LayerPolygon]:
        """
        Optionally reshape layer polygons after they have been built.

        The default implementation is a no-op (returns the list unchanged).
        Subclasses can override this to morph each layer's polygon — for
        example, interpolating between the outer organ shape and an inner
        ellipse so that the central cylinder has a different cross-section.

        Args:
            layers_polygons: Layer polygons as produced by ``_build_layer_polygons``.

        Returns:
            The (potentially modified) list of LayerPolygon objects.
        """
        return layers_polygons

    def allocate_vascular_tissue(self, layers_polygons: List[LayerPolygon]):
        """
        Allocate vascular tissue.
        Define the region where vascular tissue will be allocated.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        """
        # Find the layer where vascular tissue will be allocated
        polygon_for_vascular = self._which_layer_for_vascular(layers_polygons)
        # Create vascular tissue
        self._create_vascular_tissue(polygon_for_vascular)

    @abstractmethod
    def _which_layer_for_vascular(self, layers_polygons: List[LayerPolygon]):
        """
        Find the layer where vascular tissue will be allocated.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        """
        pass

    def _create_vascular_tissue(self, polygon: Polygon):
        """Create vascular tissue by building this organ's vascular recipe.

        Shared scaffold for every organ: each subclass supplies only
        :meth:`_vascular_recipe`; the remove-mask + extend step runs once in
        :meth:`generate_cells` after this returns.
        """
        self._vascular_recipe(polygon).build()

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Return the ordered recipe that builds this organ's vascular tissue.

        Default: an empty recipe (no vascular tissue).  Monocot / dicot / needle
        override this; the build order is data, inspectable via
        ``recipe.describe()`` / ``recipe.plan()``.
        """
        return TissueRecipe()

    def _organ_specific_tissues(self):
        """Add organ-specific tissues by building this organ's organ recipe."""
        self._organ_recipe().build()

    def _organ_recipe(self) -> TissueRecipe:
        """Return the recipe of organ-specific (post-fill) tissues.

        Default: empty (root organs have none).  Needle overrides it with resin
        ducts + stomata.
        """
        return TissueRecipe()

    def _extra_tissue_polygons(self, layers_polygons: List[LayerPolygon]) -> Dict[str, list]:
        """Return extra tissue polygons for visualization without placing cells.
        Subclasses override to expose organ-specific structures (e.g. stomata, resin ducts)."""
        return {}

    def _get_param(self, name: str) -> dict:
        """Return the params dict whose 'name' key matches, or an empty dict."""
        return next((p for p in self.params if p["name"] == name), {})

    def add_intercellular_spaces(self):
        """Orchestrate intercellular space and aerenchyma generation."""
        self.add_intercellular()
        self.add_aerenchyma()
        self.merge_intercellular_aerenchyma()

    def add_intercellular(self):
        """Compute air spaces for each inter_cellular_spaces entry."""
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

    def _aerenchyma_target_denominator(self, n_files: int) -> float:
        """Denominator for the per-quadrant aerenchyma target area. Override in subclasses."""
        return float(n_files)

    def add_aerenchyma(self):
        """Generate aerenchyma in the tissue defined in aerenchyma_params."""
        aerenchyma_prop = self.aerenchyma_params.get("aerenchyma_proportion", 0)
        if not aerenchyma_prop:
            return

        tissue = self.aerenchyma_params.get("tissue")
        n_files = int(self.aerenchyma_params.get("n_files", 1))
        aerenchyma_type = int(self.aerenchyma_params.get("aerenchyma_type", 1))

        self._aerenchyma_n_files = n_files
        self._aerenchyma_start_angle = self.rng.uniform(0, 2 * np.pi)
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
            log.warning(
                "Requested aerenchyma proportion %.2f requires %.4f area but only %.4f is available; clamping.",
                aerenchyma_prop, target_aerenchyma_area, max_possible_area,
            )
            aerenchyma_prop = max_possible_area / (total_tissue_area + total_air_area)
            target_aerenchyma_area = max_possible_area

        log.debug("Targeted aerenchyma proportion: %.3f", target_aerenchyma_area / (total_tissue_area + total_air_area))

        target_per_quadrant = (target_aerenchyma_area - total_air_area) / self._aerenchyma_target_denominator(n_files)

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
        log.debug("Actual aerenchyma proportion: %.3f", total_air_area / (total_tissue_area + total_air_area))

    def merge_intercellular_aerenchyma(self):
        """Fuse touching air-space cells within the same angular sector, then carve tissue cells."""

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

            # Only test pairs whose bounding boxes overlap (an STRtree query);
            # touching/intersecting polygons always have overlapping bboxes, so
            # the resulting union-find partition is identical to the old O(n²)
            # all-pairs scan.  ``touches or intersects`` collapses to just
            # ``intersects`` (touches ⟹ intersects).
            pool_polys = [c.polygon for c in merge_pool]
            tree = STRtree(pool_polys)
            for i in range(n_pool):
                poly_i = pool_polys[i]
                quad_i = cell_quadrants[i]
                for j in tree.query(poly_i):
                    if j <= i or cell_quadrants[j] != quad_i:
                        continue
                    if poly_i.intersects(pool_polys[j]):
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


    def plot_layers(self, show: bool = True, **kwargs) -> Optional[plt.Figure]:
        """
        Plot layer boundaries.
        
        Args:
            show: Whether to display the plot
        
        Returns:
            Matplotlib figure
        """
        
        layers_polygons = self.generate_layer_polygons()
        
        ax = kwargs.get('ax')
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        colors = plt.cm.viridis(np.linspace(0, 1, len(layers_polygons)))
        
        for polygon_data, color in zip(layers_polygons, colors):
            ax.plot(*polygon_data["polygon"].exterior.xy, 
                   color=color, label=polygon_data["name"])
        
        ax.set_aspect('equal')
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(kwargs.get('title', f"{self.__class__.__name__} - Layer Boundaries"))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if fig is not None:
            plt.tight_layout()
            if show:
                plt.show()
            return fig
        return None

    
    def plot_cells(self, show: bool = True, **kwargs) -> Optional[plt.Figure]:
        """
        Plot cell geometries.
        
        Args:
            show: Whether to display the plot
        
        Returns:
            Matplotlib figure
        """
        cells_gdf = self.generate_cells()
        
        ax = kwargs.get('ax')
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        cells_gdf.plot(
            ax=ax,
            column='type',
            categorical=True,
            cmap='tab20',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.5,
            legend=True,
            legend_kwds={'title': 'Cell Type', 'loc': 'best'}
        )
        
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(kwargs.get('title', f"{self.__class__.__name__} - Cross Section"))
        
        if fig is not None:
            plt.tight_layout()
            if show:
                plt.show()
            return fig
        return None
    
    def export_to_geopandas(self) -> gpd.GeoDataFrame:
        """
        Export cell geometries as GeoDataFrame.
        
        Returns:
            GeoDataFrame with cell data
        """
        return self.generate_cells()
    
    def export_to_csv(self, filepath: str) -> None:
        """
        Export cell data to CSV file.
        
        Args:
            filepath: Output file path
        """
        cells_gdf = self.generate_cells()
        # Drop geometry column for CSV export
        cells_df = cells_gdf.drop(columns=['geometry'])
        cells_df.to_csv(filepath, index=False)

    def write_to_xml(self, path: str, **kwargs):
        """Write anatomy cross section as .xml file."""
        from openalea.granap.anatomy_writer import AnatomyWriter
        AnatomyWriter(self).write_to_xml(path, **kwargs)

    def write_xml_geometry(self, path: str, **kwargs):
        """Write anatomy cross section as .xml file for MECHA."""
        from openalea.granap.anatomy_writer import AnatomyWriter
        AnatomyWriter(self).write_xml_geometry(path, **kwargs)
        
    def write_to_obj(self, path: str, **kwargs):
        """Write anatomy cross section as .obj file."""
        from openalea.granap.anatomy_writer import AnatomyWriter
        AnatomyWriter(self).write_to_obj(path, **kwargs)

    def write_to_svg(self, path: str, **kwargs):
        """Write anatomy cross section as .svg file."""
        from openalea.granap.anatomy_writer import AnatomyWriter
        AnatomyWriter(self).write_to_svg(path, **kwargs)
        
    def write_to_geo(self, path: str, **kwargs):
        """Write anatomy cross section as .geo file for GMSH."""
        from openalea.granap.anatomy_writer import AnatomyWriter
        AnatomyWriter(self).write_to_geo(path, **kwargs)


    def build_anatomy_tissues(self) -> List[Dict[str, Any]]:
        """Return tissue zone descriptors (layer rings + vascular polygons).
        See :func:`openalea.granap.visualization.build_anatomy_tissues`."""
        from openalea.granap.visualization import build_anatomy_tissues
        return build_anatomy_tissues(self)

    def plot_tissues(self, ax=None, show: bool = True, labels: bool = True,
                     show_effective: bool = False,
                     fuse: bool = False) -> Optional[plt.Figure]:
        """Plot every tissue zone before placing any cell.
        See :func:`openalea.granap.visualization.plot_tissues`."""
        from openalea.granap.visualization import plot_tissues
        return plot_tissues(self, ax=ax, show=show, labels=labels,
                            show_effective=show_effective, fuse=fuse)

    def export_to_adjencymatrix(self) -> lil_matrix:
        """
        Build the hydraulic network from cell geometry and return
        the sparse adjacency matrix.

        Returns
        -------
        lil_matrix
            Sparse adjacency matrix (n_total x n_total).
        """
        # Ensure cells are generated before building the network
        self.generate_cells()
        return super().export_to_adjencymatrix()

    # ------------------------------------------------------------------
    # Network construction from Voronoi cell geometry
    # ------------------------------------------------------------------
    def _build_anatnetwork(self) -> None:
        """
        Populate ``self.graph`` from the cell GeoDataFrame.
        Delegated to AnatomyWriter's NetworkExporter.
        """
        from openalea.granap.anatomy_writer import NetworkExporter
        NetworkExporter(self).export(self)

    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate anatomical statistics.
        
        Returns:
            Dictionary with statistics
        """
        cells_gdf = self.generate_cells()
        
        stats = {
            "total_cells": len(cells_gdf),
            "cell_types": cells_gdf['type'].unique().tolist(),
            "cells_per_type": cells_gdf['type'].value_counts().to_dict(),
            "total_area": cells_gdf.geometry.area.sum(),
            "mean_cell_area": cells_gdf['area'].mean(),
            "n_layers": len(self.layer_manager)
        }
        
        return stats
    
    @abstractmethod
    def _create_base_shape(self) -> Polygon:
        """
        Create the base shape for the organ.
        
        This method must be implemented by subclasses to define
        the characteristic shape of each organ type.
        
        Returns:
            Base polygon shape
        """
        pass
    
    @abstractmethod
    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[LayerPolygon]:
        """
        Create central tissue layers (vascular, parenchyma, etc.).

        Args:
            current_polygon: Current inner polygon boundary
            params: Parameter dictionaries

        Returns:
            List of LayerPolygon objects for the central zone.
        """
        pass
