"""
Plant anatomy base module providing abstract interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import geopandas as gpd
import logging
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from collections import defaultdict
from scipy.sparse import coo_matrix, lil_matrix
import hashlib
import networkx as nx
import shapely
import copy
import heapq
import time
from openalea.granap.layer_class import Layer, LayerPolygon
from openalea.granap.layer_manager import LayerManager

log = logging.getLogger(__name__)
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager, set_polygons
from openalea.granap.metafspm.build import single_plant_mpg, write_cells
from openalea.granap.metafspm.cells import CellStore, MPGCells
from openalea.granap.metafspm.topology import store_topology, topology_from_mpg
from openalea.granap.metafspm.views import cells_from_mpg, gdf_from_cells
from openalea.granap.network_base import AbstractNetwork
from openalea.granap.input_data import OrganInputData
from openalea.granap.tissue_class import TissueRecipe, retag_tissue

_ITERATIONS = 10                 # GeometryProcessor.smoothing_polygon's default passes
_POINTS = 200                    # its resampling


def _topology_key(polys, cell_ids, protect_ids) -> tuple:
    """The cache key of a topology request: the polygons (their WKB), the ids, the protected ids."""
    digest = hashlib.blake2b(digest_size=16)
    for wkb in shapely.to_wkb(list(polys), hex=False):      # None for missing polygons
        digest.update(wkb if wkb is not None else b"-")
    return tuple(cell_ids), frozenset(protect_ids or ()), digest.digest()


def _copies(cells):
    """Cells copied one level deep: new Cell objects sharing their (immutable) attribute values."""
    return [copy.copy(c) for c in cells]


def _smoothed_polygons(polygons, smooth_factors) -> list:
    """[GeometryProcessor.buffer_polygon(p, 0, smooth_factor=s)] for each polygon and factor (all > 0), in bulk."""
    buffered = shapely.buffer(np.asarray(polygons, dtype=object), 0., quad_segs=16)
    out = [None] * len(polygons)
    batch, rows, factors = [], [], []
    for i, (polygon, b, s) in enumerate(zip(polygons, buffered, smooth_factors)):
        if b.geom_type != "Polygon" or b.is_empty:
            out[i] = GeometryProcessor.buffer_polygon(polygon, 0, smooth_factor=s)   # GRANAP's own path
            continue
        coords = shapely.get_coordinates(b.exterior)
        resampled = GeometryProcessor.resample_coords(coords, target_n_points=_POINTS)
        if not np.allclose(resampled[0], resampled[-1]):
            out[i] = GeometryProcessor.buffer_polygon(polygon, 0, smooth_factor=s)   # an open ring: GRANAP's
            continue
        batch.append(resampled)
        rows.append(i)
        factors.append(s)
    if batch:
        coords = np.stack(batch).astype(float)                   # (cells, points, 2), closed rings
        s = np.asarray(factors, dtype=float)[:, None, None]
        for _ in range(_ITERATIONS):
            pts = coords[:, :-1]
            prev_pts = np.roll(pts, 1, axis=1)
            next_pts = np.roll(pts, -1, axis=1)
            smoothed_pts = (1 - s) * pts + s * (prev_pts + next_pts) / 2.0
            coords = np.concatenate([smoothed_pts, smoothed_pts[:, :1]], axis=1)
        polygons_out = shapely.polygons(coords)
        for i, p in zip(rows, polygons_out):
            out[i] = p
    return out


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

    #: Whether :meth:`generate_cells` absorbs empty gaps into their neighbouring
    #: cell of the same tissue (:meth:`fuse_gaps`) as a final step, so every cross
    #: section is gap-free by default.  Set False (per subclass or instance) to keep
    #: the raw tessellation with its small uncovered slivers.
    AUTO_FUSE_GAPS: bool = True

    #: Whether :meth:`fuse_gaps` only absorbs gaps *enclosed by cells* and leaves the
    #: uncovered ribbon along the organ surface alone.  That ribbon — between the
    #: outermost Voronoi edges and the outline — runs unbroken past many cells and
    #: survives the sliver opening wherever the surface curves sharply (leaf tips,
    #: blunt margins), so fusing it stretches one surface cell right across its
    #: neighbours.  Set False to fuse every gap, surface ones included.
    FUSE_ENCLOSED_GAPS_ONLY: bool = True

    #: Whether ordinary intercellular air-space cells (see :meth:`_apply_intercellular`)
    #: are flagged ``protect_topology`` so NetworkExporter wires them as dedicated
    #: wall_air network nodes.  False for root/stem; leaves override this True.
    PROTECT_AIR_TOPOLOGY: bool = False

    #: Cell types that must NEVER carry a symplastic (plasmodesmata) edge,
    #: even when they legitimately flank a real two-cell wall -- apoplastic-only
    #: pathways (e.g. a dead conduit with no living cytoplasm to connect
    #: through). Enforced generically in ``NetworkExporter.export``'s Phase-6
    #: symplastic block; membrane/wall edges are always left untouched. Empty
    #: for every organ by default; ``NeedleAnatomy`` sets
    #: ``("transfusion tracheid",)``.
    APOPLASTIC_ONLY_TYPES: tuple = ()

    # The section in its MPG (openalea.metafspm): built by generate_cells, stage by stage
    mpg = None                       # the MPG and the section's SubOrgan vertex
    segment = None
    _mpg_written = None              # the record of what the stages wrote (vertices by kind, uids, the topology)
    mpg_stage_hooks = ()             # callables (organ, stage) after each stage write (tests, diagnostics)
    stores_network = True            # the network stored at generation (False: the cells only, e.g. a donor section)

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
    def create_from_input(cls, input_data: OrganInputData, seed=None) -> "Organ":
        """
        Factory method to initialize the appropriate Organ subclass 
        (RootAnatomy or NeedleAnatomy) from an OrganInputData instance, with *seed* (None: not reproducible).
        """
        # Normalize params to plain dicts for uniform access
        params = input_data.to_dict_list() if isinstance(input_data, OrganInputData) else input_data.params

        # Determine the organ type from the parameters
        ptype_param = next((p for p in params if p["name"] == "planttype"), None)
        organ_type = None

        if ptype_param:
            # Check 'stem'/'leaf' before the value-based root test: they share
            # planttype values 1/2 with roots, so they are distinguished only by
            # the explicit organ tag.
            if ptype_param.get("organ") == "stem":
                organ_type = "stem"
            elif ptype_param.get("organ") == "leaf":
                organ_type = "leaf"
            elif ptype_param.get("organ") == "needle" or ptype_param.get("value") == 3:
                organ_type = "needle"
            elif ptype_param.get("organ") == "root" or ptype_param.get("value") in [1, 2, 1.0, 2.0]:
                organ_type = "root"

        # Fallback to duck-typing the input parameters if 'organ' isn't explicitly defined
        if not organ_type:
            names = {p["name"] for p in params}
            if "pith" in names:
                organ_type = "stem"
            elif "palisade" in names or "spongy" in names:
                organ_type = "leaf"
            elif "stele" in names or "cortex" in names:
                organ_type = "root"
            else:
                organ_type = "needle"

        if organ_type == "needle":
            from openalea.granap.needle_class import NeedleAnatomy
            return NeedleAnatomy(input_data, seed=seed)
        elif organ_type == "stem":
            from openalea.granap.stem_class import StemAnatomy
            return StemAnatomy(input_data, seed=seed)
        elif organ_type == "leaf":
            from openalea.granap.leaf_class import LeafAnatomy
            return LeafAnatomy(input_data, seed=seed)
        else:
            from openalea.granap.root_class import RootAnatomy
            return RootAnatomy(input_data, seed=seed)
    
    def add_layer(self, layer: Layer, position: Optional[int] = None) -> None:
        """
        Add a tissue layer to the anatomy.
        
        Args:
            layer: Layer object to add
            position: Optional position index (None = append)
        """
        self.layer_manager.add_layer(layer, position)
        self._invalidate_geometry()

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
        base_center = polygon.centroid

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
            polygon = self._offset_layer_polygon(
                polygon,
                -space_increment - layer["cell_diameter"] / 2,
                layer,
                self.LAYER_SMOOTH_FACTOR,
                center=base_center,
            )

            space_increment = layer["cell_diameter"] / 2

            layers_polygons.append(LayerPolygon(
                name=layer["name"],
                polygon=polygon,
                cell_diameter=layer["cell_diameter"],
                id_layer=i_layer + 1,
                cell_width=layer["cell_width"],
                shift=layer["shift"],
                n_points=layer.get("n_points"),
                protect_shape=layer.get("protect_shape", False),
            ))
        
        # Add central layers (vascular, parenchyma, etc.)
        params = [l.to_dict() for l in self.layer_manager.get_layers()]
        central_layers = self._create_central_layers(polygon, params)
        layers_polygons.extend(central_layers)

        # Optional reshape: let subclasses morph layer polygons
        layers_polygons = self.reshape_layers(layers_polygons)

        return layers_polygons

    def _offset_layer_polygon(self, polygon: Polygon, distance: float, layer: dict,
                              smooth_factor: float, center: Optional[Point] = None) -> Polygon:
        """
        Shrink/expand one ring layer's polygon by ``distance``.

        Default implementation: a plain uniform buffer, identical to the
        previous inline call — every organ keeps today's behavior unchanged.
        Subclasses (e.g. ``NeedleAnatomy``) may override to offset by a
        per-angle distance instead (see ``GeometryProcessor.variable_buffer_polygon``),
        driven by an optional ``"thickness_profile"`` entry in ``layer``.
        """
        return GeometryProcessor.buffer_polygon(polygon, distance, smooth_factor=smooth_factor)

    def generate_cells(self) -> gpd.GeoDataFrame:
        """
        Generate the cells (once), into the section's MPG; returns their GeoDataFrame, a view built from the cells when
        first read.
        """
        self.generate_section()
        return self._cells_gdf

    def generate_section(self) -> None:
        """
        The section generated once into its MPG (self.mpg, self.segment): the pipeline stage by stage (each stage's
        cells written as Layers, Cells and protoplasts), the final cells, and the stored network (walls, junctions,
        Connections) unless stores_network is False.
        """
        if self.__dict__.get("_mpg_generated"):
            return
        try:
            self._generate_pipeline()
        finally:
            self.__dict__.pop("_topology_cache", None)
        self._mpg_stage("final")
        self._read_cells_back()
        self.__dict__["_mpg_generated"] = True                  # the cells are final; the network follows them
        if self.stores_network:
            self._store_network()

    def _generate_pipeline(self) -> None:
        """The cells generated by Voronoi tessellation of the layers, stage by stage into the section's MPG."""
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
        self.vascular_tissue_polygons = {}
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
        grouped_cells = CellGenerator.simplify_cells(grouped_cells, snap=self.section_topology)
        # simplify_cells rebuilds each polygon independently, which can distort
        # small cells (few vertices) into a neighbour and leave them rendered
        # one-inside-another; drop those, keeping the larger cell.
        grouped_cells = CellGenerator.remove_nested_cells(grouped_cells)
        self.all_cells = CellManager()
        self.all_cells.cells = grouped_cells
        log.info("Voronoi grouping:        %.3fs", time.time() - t_start)
        self._grouped()                                        # stage "grouping"; from here on, views of the MPG

        t_start = time.time()
        self.add_intercellular_spaces()
        log.info("Intercellular spaces:    %.3fs", time.time() - t_start)

        if self.AUTO_FUSE_GAPS:
            t_start = time.time()
            n_fused = self.fuse_gaps()
            log.info("Fuse gaps:               %.3fs (%d fused)",
                     time.time() - t_start, n_fused)
            self._mpg_stage("fused_gaps")

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
    @property
    def _cells_gdf(self):
        """The cell GeoDataFrame: once generated, built from the cells at its first read (then cached)."""
        gdf = self.__dict__.get("_gdf")
        if gdf is None and self.__dict__.get("_mpg_generated"):
            gdf = self.__dict__["_gdf"] = gdf_from_cells(self.all_cells.cells, self.__dict__.get("_gdf_n_real"))
        return gdf

    @_cells_gdf.setter
    def _cells_gdf(self, value):
        self.__dict__["_gdf"] = value                          # direct writes (bridge rows, retagging)
        if value is None:                                      # _invalidate_geometry: to be generated again
            self.__dict__["_mpg_generated"] = False

    # ── The stages: the pipeline's cells written into the MPG at each boundary ──

    def _mpg_stage(self, stage: str) -> None:
        """Write the current cells into the section's MPG (created at the first stage) as the record of *stage*."""
        if self.mpg is None:
            self.mpg, self.segment = single_plant_mpg()
        self._mpg_written = write_cells(self.mpg, self.segment, self, self._mpg_written, stage)
        store = getattr(self, "_cell_store", None)
        if store is not None:                                    # views of removed cells are forgotten
            alive = self._mpg_written["cell_of_uid"]
            for uid in [u for u in store.views if u not in alive]:
                del store.views[uid]
        for hook in getattr(self, "mpg_stage_hooks", ()):
            hook(self, stage)

    def _grouped(self) -> None:
        """The Voronoi groups, simplified, nested cells removed: stage "grouping"; from here on the cells are views of
        the MPG (their attributes read from and written to their Cell vertices)."""
        self._mpg_stage("grouping")
        self._cell_store = CellStore(self.mpg, self.segment, self._mpg_written)
        self.all_cells = MPGCells(self._cell_store, self.all_cells.cells)

    def _read_cells_back(self, n_real=None):
        """all_cells and the layers' cell lists from the MPG (the same cells, the same order), as plain Cells; the
        GeoDataFrame follows them (with *n_real*, the cells after it are the exporter's bridge cells)."""
        old = list(self.all_cells.cells)
        cells = cells_from_mpg(self.mpg, self._mpg_written)
        replaced = {id(a): b for a, b in zip(old, cells)}
        if isinstance(self.all_cells, MPGCells):               # the pipeline is over: plain Cells
            self.all_cells = CellManager()
            self._cell_store = None
        self.all_cells.cells[:] = cells
        for layer in getattr(self.layer_manager, "layers", []):
            layer.cells[:] = [replaced.get(id(c), c) for c in layer.cells]
        self.__dict__["_gdf"], self.__dict__["_gdf_n_real"] = None, n_real      # the view, rebuilt when read

    # ── The section's topology: stored in the MPG, asked of the organ ──

    def section_topology(self, polys, cell_ids, protect_ids=None):
        """
        CellGenerator._build_topology for this section: the rings stored in the MPG when asked for the stored network's
        cells (the exporter, the XML writer); otherwise computed, through the organ's cache (the same input asked again,
        e.g. by the calibration's attempts, gets the same result without a new snap).
        """
        written = self._mpg_written or {}
        ids = written.get("topology", {}).get("ids")
        if ids is not None and [int(i) for i in cell_ids] == ids:
            self.mpg_topology_reads = getattr(self, "mpg_topology_reads", 0) + 1
            return topology_from_mpg(self.mpg, self.segment, written)
        cache = self.__dict__.setdefault("_topology_cache", {})
        key = _topology_key(polys, cell_ids, protect_ids)
        if key not in cache:
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            cache[key] = CellGenerator._build_topology(polys, cell_ids, protect_ids=protect_ids)
        return cache[key]

    def _store_network(self) -> None:
        """
        The hydraulic network of the final cells, stored in the MPG when the cells are generated, without bridges
        (which would add cells): the section's rings snapped once, then its walls, junctions and edges built in the MPG
        from them (Compartments and Connections, their quantities as numeric columns).
        """
        from openalea.granap.metafspm.network import build_network
        store_topology(self.mpg, self.segment, self, self._mpg_written)          # the one snapping of the section
        build_network(self)

    def _stored_network_matches(self, air_link_radius, exporter_kwargs) -> bool:
        """Whether the export asked for is the stored one: default exporter settings, and no bridge to add."""
        if air_link_radius is not None or set(exporter_kwargs) - {"bridges", "bridge_radius"}:
            return False
        wants_bridges = exporter_kwargs.get("bridges", True)
        return not (wants_bridges and self.network_bridge_specs())

    @property
    def graph(self):
        """The exporter's networkx graph, built from the MPG at its first read after the network is (re)built."""
        graph = self.__dict__.get("_graph")
        if graph is None:
            from openalea.granap.metafspm.views import network_from_mpg
            graph = self.__dict__["_graph"] = network_from_mpg(self.mpg, self._mpg_written)
        return graph

    @graph.setter
    def graph(self, value):
        self.__dict__["_graph"] = value

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

    def _empty_void_polygons(self) -> List[Polygon]:
        """Regions that are *meant* to hold no cell (so they are not gaps).

        The base organ has none; the stem's medullary cavity is the usual one
        (registered both as ``pith_cavity_polygon`` and under the
        ``medullary cavity`` key of ``vascular_tissue_polygons``).  Protoxylem
        lacunae and intercellular / aerenchyma spaces are seeded as real ``air
        space`` cells, so they are covered and never reported as gaps.
        """
        voids: List[Polygon] = []
        cav = getattr(self, "pith_cavity_polygon", None)
        if cav is not None and not cav.is_empty:
            voids.append(cav)
        vtp = getattr(self, "vascular_tissue_polygons", None) or {}
        voids.extend(v for v in vtp.get("medullary cavity", []) if v is not None and not v.is_empty)
        return voids

    @staticmethod
    def _filled_cell_mass(polys: List[Polygon]):
        """The union of ``polys`` with its internal pockets filled in.

        Anything inside this outline but not covered by a cell is a hole the tissue
        encloses; anything outside it is the ribbon along the organ surface.
        """
        covered = unary_union(polys)
        parts = covered.geoms if hasattr(covered, "geoms") else [covered]
        return unary_union([Polygon(p.exterior) for p in parts
                            if p.geom_type == "Polygon"])

    @staticmethod
    def _filled_mass(covered):
        """The union of the cells, *covered*, with its internal pockets filled in (_filled_cell_mass from the union)."""
        parts = covered.geoms if hasattr(covered, "geoms") else [covered]
        return unary_union([Polygon(p.exterior) for p in parts if p.geom_type == "Polygon"])

    def find_gaps(self, sliver_width: float = None, min_area: float = None,
                  exclude_voids: bool = True, enclosed_only: bool = False, unions: dict = None) -> List[Polygon]:
        """Detect empty space inside the organ that no cell was assigned to.

        A gap is the organ outline minus the union of every cell polygon (minus the
        declared voids — the medullary cavity — when ``exclude_voids``).  These arise
        where the pipeline drops or clips cells: ``simplify_cells`` /
        ``remove_nested_cells`` deleting a small cell, Voronoi clipping at the
        boundary, or a bundle removal mask leaving a sliver between the bundle sheath
        and the surrounding tissue.

        Hairline slivers along tissue borders (sub-cell width, a meshing artefact
        rather than a real hole) are removed by a morphological *opening*: erode then
        dilate by ``sliver_width / 2``, so anything thinner than ``sliver_width``
        disappears while cell-sized gaps survive.  ``sliver_width`` defaults to 15 %
        of the median cell diameter; ``min_area`` (optional) drops pieces below an
        area.  ``enclosed_only`` keeps just the holes the tissue closes around,
        dropping the ribbon that runs along the organ surface (see
        :attr:`FUSE_ENCLOSED_GAPS_ONLY`); the default reports every gap, so this stays
        a full diagnostic.  Returns the gap polygons, largest first — feed them to a
        plot or read their ``.area`` / ``.centroid`` to locate each hole.

        Call after :meth:`generate_cells` (it reads the materialised cells).

            *unions* ({"covered", "mass"}) keeps the cells' union and filled mass computed here, for the caller.
            """
        polys = [c.polygon for c in self.all_cells.get_cells()
                 if c.polygon is not None and not c.polygon.is_empty]
        if not polys:
            return []
        unions = {} if unions is None else unions
        outline = self.generate_base_shape()
        if "covered" not in unions:
            unions["covered"] = unary_union(polys)
        covered = unions["covered"]
        gaps = outline.difference(covered)
        if exclude_voids:
            for v in self._empty_void_polygons():
                gaps = gaps.difference(v)
        if gaps.is_empty:
            return []

        if sliver_width is None:
            med_area = float(np.median([p.area for p in polys]))
            med_diam = 2.0 * np.sqrt(med_area / np.pi) if med_area > 0 else 0.0
            sliver_width = 0.15 * med_diam
        if sliver_width > 0:
            gaps = gaps.buffer(-0.5 * sliver_width).buffer(0.5 * sliver_width)

        pieces = [g for g in (gaps.geoms if hasattr(gaps, "geoms") else [gaps])
                  if g.geom_type == "Polygon" and not g.is_empty]
        if enclosed_only:
            # representative_point() is guaranteed inside the piece, unlike a centroid
            # on a crescent, so a surface gap curving around a tip is not mistaken for
            # an enclosed one.
            if "mass" not in unions:
                unions["mass"] = self._filled_mass(covered)
            mass = unions["mass"]
            pieces = [g for g in pieces if g.representative_point().within(mass)]
        if min_area is not None:
            pieces = [g for g in pieces if g.area >= min_area]
        pieces.sort(key=lambda g: g.area, reverse=True)
        return pieces

    def fuse_gaps(self, sliver_width: float = None, min_area: float = None) -> int:
        """Absorb each detected gap into the nearest cell of its own tissue zone.

        For every gap from :meth:`find_gaps`, the surrounding **tissue zone** is the
        cell type sharing the most boundary with the hole (so a pith gap goes to the
        pith, a sheath gap to the sheath, never into a vessel it merely touches).  The
        hole is then unioned into the neighbouring cell of that zone with the longest
        shared border — the nearest cell of the same tissue — so no empty space is
        left and the cell simply grows to fill it.

        Only gaps the tissue closes around are absorbed, unless
        :attr:`FUSE_ENCLOSED_GAPS_ONLY` is cleared; the ribbon along the organ surface
        is left empty rather than smeared into one surface cell.

        The full (un-opened) uncovered area around each gap is recovered first, so the
        fusion reaches the real hole edge rather than the eroded outline
        :meth:`find_gaps` reports.  Returns the number of gaps fused and refreshes the
        materialised ``_cells_gdf`` in place (no full regeneration).

            The cells' union and filled mass are computed once (find_gaps keeps them for this method).
            """
        cells = [c for c in self.all_cells.get_cells()
                 if c.polygon is not None and not c.polygon.is_empty]
        if not cells:
            return 0
        enclosed_only = self.FUSE_ENCLOSED_GAPS_ONLY
        unions: Dict[str, Any] = {}
        gaps = self.find_gaps(sliver_width=sliver_width, min_area=min_area, enclosed_only=enclosed_only,
                         unions=unions)
        if not gaps:
            return 0

        med_area = float(np.median([c.polygon.area for c in cells]))
        med_d = 2.0 * np.sqrt(med_area / np.pi) if med_area > 0 else 0.0
        eps = max(med_d * 1e-3, 1e-9)
        # How far to dilate each reported gap to recover the true (un-opened) hole.
        recover = 1.5 * sliver_width if sliver_width else 0.25 * med_d

        covered = unions["covered"]                     # the union find_gaps made of the same cells' polygons
        raw = self.generate_base_shape().difference(covered)
        for v in self._empty_void_polygons():
            raw = raw.difference(v)
        if enclosed_only:
            # Clip the recoverable area too. Without this the ``recover`` dilation
            # below reconnects an enclosed hole near the surface to the ribbon, and
            # the cell fused with it smears along the ribbon anyway — defeating the
            # filter find_gaps just applied.
            if "mass" not in unions:
                unions["mass"] = self._filled_mass(covered)
            raw = raw.intersection(unions["mass"])

        tree = STRtree([c.polygon for c in cells])
        fused = 0
        for g in gaps:
            hole = raw.intersection(g.buffer(recover)) if recover > 0 else g
            if hole.is_empty:
                hole = g
            probe = hole.buffer(eps)
            cand = [cells[i] for i in tree.query(probe) if cells[i].polygon.intersects(probe)]
            if not cand:
                continue

            # Dominant bordering tissue = the zone the gap belongs to; then the cell
            # of that zone with the longest shared border (its nearest cell).
            by_type: Dict[str, float] = defaultdict(float)
            best: Dict[str, Any] = {}
            for c in cand:
                # .boundary (not .exterior) so a MultiPolygon cell doesn't crash.
                shared = c.polygon.boundary.intersection(probe).length
                by_type[c.type] += shared
                if c.type not in best or shared > best[c.type][0]:
                    best[c.type] = (shared, c)
            if by_type and max(by_type.values()) > 0:
                zone = max(by_type, key=by_type.get)
                target = best[zone][1]
            else:                                   # no shared border -> nearest overall
                target = min(cand, key=lambda c: c.polygon.distance(hole))

            merged = unary_union([target.polygon, hole])
            if merged.geom_type != "Polygon":
                parts = [p for p in merged.geoms if p.geom_type == "Polygon"]
                merged = max(parts, key=lambda p: p.area) if parts else target.polygon
            target.polygon = merged
            fused += 1

        self.all_cells.recalculate_cell_properties()
        if self._cells_gdf is not None:
            cell_dicts = [c.cell_to_dict() for c in self.all_cells.cells]
            for i, c in enumerate(self.all_cells.cells):
                cell_dicts[i]["geometry"] = c.polygon
            self._cells_gdf = gpd.GeoDataFrame(cell_dicts)
        return fused

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
        ducts, transfusion tissue, the corner-parenchyma retag, stomata and
        layer-count zoning -- in that order, which is load-bearing (see
        ``NeedleAnatomy._organ_recipe``).
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
        """Orchestrate intercellular space and aerenchyma generation (stages "intercellular" and "aerenchyma")."""
        self.add_intercellular()
        self._mpg_stage("intercellular")
        self._add_aerenchyma_to_proportion()
        self._mpg_stage("aerenchyma")

    def _measure_air_proportion(self, tissues: List[str]) -> float:
        """Air-space fraction of the tissue band, ``air / (air + tissue)``,
        measured from the current cell polygons. This is the quantity the
        aerenchyma request refers to."""
        tissue_area = sum(
            c.polygon.area
            for t in tissues
            for c in self.all_cells.get_cells_by_type(t)
            if c.polygon is not None
        )
        air_area = sum(
            c.polygon.area
            for c in self.all_cells.get_cells_by_type("air space")
            if c.polygon is not None
        )
        denom = tissue_area + air_area
        return air_area / denom if denom > 0 else 0.0

    def _add_aerenchyma_to_proportion(self, tol: float = 5e-3, max_iter: int = 20):
        """Place aerenchyma so the realized air proportion matches the request.

        The fuse-and-smooth step in :meth:`merge_intercellular_aerenchyma` grows
        the air lacunae, so simply filling to the requested area overshoots
        the final proportion. Because that overshoot depends on the emergent
        geometry it cannot be predicted cleanly, so we calibrate: fill + fuse +
        smooth for a trial target, measure the *realized* proportion, and bisect
        the internal target until it lands on the requested value. Each attempt
        restores the clean pre-aerenchyma tessellation, and the sector angle is
        fixed once, so the attempts differ only in how much air is placed and the
        realized proportion is monotone in the internal target.

            Each attempt starts from shallow copies of the cells (a Cell's attributes are immutable values), the fill
            order is planned at the first attempt and reused (_fill_plan), and the best attempt's cells are kept
            instead of running that attempt again.
            """
        requested = self.aerenchyma_params.get("aerenchyma_proportion", 0)
        if not requested:
            self.merge_intercellular_aerenchyma()
            return

        tissue = self.aerenchyma_params.get("tissue")
        tissues = list(tissue) if isinstance(tissue, (list, tuple)) else [tissue]

        snapshot = _copies(self.all_cells.cells)
        self._aerenchyma_start_angle = float(self.rng.uniform(0, 2 * np.pi))

        plan = [None]                    # B2b: the fill order, computed at the first attempt, the same for all

        def attempt(internal_target: float) -> float:
            self.all_cells.cells = _copies(snapshot)
            plan[0] = self.add_aerenchyma(prop_override=internal_target, plan=plan[0])
            self.merge_intercellular_aerenchyma()
            return self._measure_air_proportion(tissues)

        best_target, best_realized = requested, attempt(requested)
        best_err = abs(best_realized - requested)
        best_cells = _copies(self.all_cells.cells)          # the best attempt's cells, restored at the end
        lo, hi = 0.0, requested
        if best_realized < requested:
            lo, hi = requested, 1.0

        for _ in range(max_iter):
            if best_err <= tol or (hi - lo) < 1e-4:
                break
            mid = 0.5 * (lo + hi)
            realized = attempt(mid)
            err = abs(realized - requested)
            if err < best_err:
                best_target, best_realized, best_err = mid, realized, err
                best_cells = _copies(self.all_cells.cells)
            if realized > requested:
                hi = mid
            else:
                lo = mid

        self.all_cells.cells = best_cells           # was attempt(best_target): the same cells, computed again
        log.debug("Calibrated aerenchyma: requested %.3f -> realized %.3f (internal target %.3f)",
                  requested, best_realized, best_target)

    def add_intercellular(self):
        """Compute air spaces for each inter_cellular_spaces entry."""
        for ics in self.intercellular_spaces_params:
            self._apply_intercellular(ics)

    _REFLEX_CORNER_DEG: float = 185.0

    def _center_air_pocket(self, pocket, nb_tree, nb_polys):
        """Rebuild a pocket at a *reflex* junction as a 4-point quad that hugs the
        reflex cell instead of a thin, mis-oriented triangle.
        
        Pockets with only convex neighbours (the common case) are returned unchanged.
        """
        ctr = np.array([pocket.centroid.x, pocket.centroid.y])
        pverts = np.asarray(pocket.exterior.coords)[:-1]
        # A reflex neighbour can *wrap around* the pocket with no area overlap, and its
        # bbox need not overlap the tiny pocket — so probe a buffered pocket and select
        # by distance, and keep the most-reflex neighbour as A.
        adj_tol = max(1e-6, 0.5 * np.sqrt(pocket.area / np.pi))
        probe = pocket.buffer(adj_tol)
        best = None                                    # (interior, V, u1, u2, A_poly)
        for i in nb_tree.query(probe):
            poly = nb_polys[i]
            if poly is None or poly.is_empty or poly.geom_type != "Polygon" \
                    or poly.distance(pocket) > adj_tol:
                continue
            P = np.asarray(poly.exterior.coords)[:-1]
            if len(P) < 3:
                continue
            k = int(np.argmin(np.hypot(P[:, 0] - ctr[0], P[:, 1] - ctr[1])))
            V, prev, nxt = P[k], P[(k - 1) % len(P)], P[(k + 1) % len(P)]
            a = np.arctan2(V[1] - prev[1], V[0] - prev[0])
            b = np.arctan2(nxt[1] - V[1], nxt[0] - V[0])
            turn = (np.degrees(b - a) + 180.0) % 360.0 - 180.0
            if not poly.exterior.is_ccw:               # normalise to CCW so interior is 180-turn
                turn = -turn
            interior = 180.0 - turn
            if interior < self._REFLEX_CORNER_DEG:     # convex enough: not the reflex wall
                continue
            if best is None or interior > best[0]:
                n1, n2 = np.hypot(*(prev - V)), np.hypot(*(nxt - V))
                if n1 <= 0 or n2 <= 0:
                    continue
                best = (interior, V, (prev - V) / n1, (nxt - V) / n2, poly)

        if best is None:                               # no reflex neighbour: keep the pocket
            return pocket

        _, V, u1, u2, A = best
        r = float(np.max(np.hypot(pverts[:, 0] - V[0], pverts[:, 1] - V[1])))
        if r <= 0:
            return pocket
        AB = V + u1 * r
        AC = V + u2 * r
        BC = AB + AC - V                               # opposite corner, between B and C
        quad = Polygon([tuple(V), tuple(AB), tuple(BC), tuple(AC)])
        if not quad.is_valid:
            quad = quad.buffer(0)
        if quad.is_empty or quad.area <= 0:
            return pocket
        # Never take space from A (its walls only bound the quad); keep the original
        # pocket's coverage by unioning it in, then drop any A overlap.
        quad = unary_union([quad, pocket]).difference(A)
        if quad.geom_type != "Polygon":
            parts = [g for g in quad.geoms if g.geom_type == "Polygon" and not g.is_empty]
            quad = max(parts, key=lambda g: g.area) if parts else pocket
        return quad if not quad.is_empty else pocket

    def _apply_intercellular(self, ics: dict) -> None:
        """Apply one inter_cellular_spaces entry to the relevant tissue cells.

            The cells are smoothed in bulk (_smoothed_polygons), and the lacunae seated with the organ's topology.
            """
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

        # The cells' smoothing in bulk: buffer_polygon(polygon, 0, smooth_factor=s) for each (s > 0), the buffer alone
        # for s == 0, as buffer_polygon returns it
        with_polygon = [cell for cell in all_tissue_cells if cell.polygon is not None]
        factors = [cell_smoothness[id(cell)] for cell in with_polygon]
        smooth = [i for i, s in enumerate(factors) if s > 0]
        shrunk_all = [None] * len(with_polygon)
        for i, p in zip(smooth, _smoothed_polygons([with_polygon[i].polygon for i in smooth], [factors[i] for i in smooth])):
            shrunk_all[i] = p
        for i, s in enumerate(factors):
            if not s > 0:
                shrunk_all[i] = with_polygon[i].polygon.buffer(0, resolution=16)
        smoothed = [shrunk for shrunk in shrunk_all if not shrunk.is_empty]

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

        # Centre any lopsided pocket: where a neighbour meets it with a near-straight
        # or reflex corner (which doesn't retreat under the corner-rounding, leaving a
        # flat wall), clip that corner into the pocket so every side retreats and the
        # pocket sits centered on the junction node.
        nb_polys = [c.polygon for c in all_tissue_cells if c.polygon is not None]
        if nb_polys:
            nb_tree = STRtree(nb_polys)
            air_space_polys = [self._center_air_pocket(p, nb_tree, nb_polys)
                               for p in air_space_polys]

        air_union = GeometryProcessor.union_polygons(air_space_polys)
        from openalea.granap.special_tissues import seat_air_spaces
        seat_air_spaces(self.all_cells, all_tissue_cells, air_union, air_space_polys,
                        protect_topology=self.PROTECT_AIR_TOPOLOGY, snap=self.section_topology)

    def _aerenchyma_target_denominator(self, n_files: int) -> float:
        """Denominator for the per-quadrant aerenchyma target area. Override in subclasses."""
        return float(n_files)

    @staticmethod
    def _connected_radial_order(cells: list, central_angle: float) -> list:
        """Order ``cells`` so that every prefix is a connected region, growing
        outward from the cell nearest ``central_angle``.

        Used to fill an aerenchyma file as one connected radial channel. Cells
        are treated as adjacent when their polygons touch (across tissue types),
        so growth bridges tissue boundaries; expansion is prioritised by angular
        distance to ``central_angle`` to keep the channel centred. When a
        connected component is exhausted before all cells are placed, growth
        restarts from the next-nearest unused cell (a fresh component).
        """
        n = len(cells)
        if n <= 1:
            return list(cells)

        def ang_dist(cell):
            a = np.arctan2(cell.y, cell.x) % (2 * np.pi)
            d = abs(a - central_angle)
            return min(d, 2 * np.pi - d)

        polys = [c.polygon for c in cells]
        tree = STRtree(polys)
        neighbors: list = [[] for _ in range(n)]
        for i, poly_i in enumerate(polys):
            if poly_i is None:
                continue
            for j in tree.query(poly_i):
                if j != i and polys[j] is not None and poly_i.intersects(polys[j]):
                    neighbors[i].append(j)

        priorities = [ang_dist(c) for c in cells]
        seeds = sorted(range(n), key=lambda i: priorities[i])
        visited = [False] * n
        heap: list = []
        counter = 0
        order: list = []
        seed_ptr = 0

        def _push(i):
            nonlocal counter
            heapq.heappush(heap, (priorities[i], counter, i))
            counter += 1

        while len(order) < n:
            if not heap:
                while seed_ptr < n and visited[seeds[seed_ptr]]:
                    seed_ptr += 1
                if seed_ptr >= n:
                    break
                _push(seeds[seed_ptr])
            _, _, i = heapq.heappop(heap)
            if visited[i]:
                continue
            visited[i] = True
            order.append(cells[i])
            for j in neighbors[i]:
                if not visited[j]:
                    _push(j)

        return order

    def add_aerenchyma(self, prop_override: Optional[float] = None, plan: dict = None):
        """Generate aerenchyma in the tissue defined in aerenchyma_params.

        ``prop_override`` replaces the requested ``aerenchyma_proportion`` with an
        internal working target; :meth:`_add_aerenchyma_to_proportion` uses it to
        calibrate the fill so the *realized* proportion (measured after the fuse
        and smoothing) matches what the user asked for. The sector start angle is
        drawn once and then reused, so repeated calibration attempts share the
        same geometry and only differ in how much air is placed.

            *plan*: what the fill depends on besides the target (_fill_plan), from an earlier call on the same cells
            (the calibration's attempts); computed when None. Returns the plan.
            """
        aerenchyma_prop = (
            prop_override if prop_override is not None
            else self.aerenchyma_params.get("aerenchyma_proportion", 0)
        )
        if not aerenchyma_prop:
            return plan

        n_files = int(self.aerenchyma_params.get("n_files", 1))
        self._aerenchyma_n_files = n_files
        if getattr(self, "_aerenchyma_start_angle", None) is None:
            self._aerenchyma_start_angle = float(self.rng.uniform(0, 2 * np.pi))

        if aerenchyma_prop > 1:
            print("Aerenchyma proportion is greater than 1, setting it to 1")
            aerenchyma_prop = 1

        if plan is None:
            plan = self._fill_plan()
        if plan is None:
            return plan
        total_tissue_area, total_air_area = plan["total_tissue_area"], plan["total_air_area"]
        max_possible_area = plan["max_possible_area"]

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

        cells = self.all_cells.cells
        quadrant_buckets = [[cells[k] for k in bucket] for bucket in plan["buckets"]]
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

        if log.isEnabledFor(logging.DEBUG):
            tissues = plan["tissues"]
            total_tissue_area = sum(
                c.polygon.area
                for t in tissues
                for c in self.all_cells.get_cells_by_type(t)
                if c.polygon is not None
            )
            total_air_area = sum(c.polygon.area for c in self.all_cells.get_cells_by_type("air space") if c.polygon is not None)
            log.debug("Actual aerenchyma proportion: %.3f", total_air_area / (total_tissue_area + total_air_area))
        return plan

    def _fill_plan(self):
        """
        What GRANAP's add_aerenchyma computes before filling, which depends only on the cells it starts from: the
        candidates and the areas, and each file's fill order (quadrant buckets, ordered as a connected radial channel or by
        distance), as positions in organ.all_cells.cells. None where add_aerenchyma would return without filling.
        """
        tissue = self.aerenchyma_params.get("tissue")
        tissues = list(tissue) if isinstance(tissue, (list, tuple)) else [tissue]
        n_files = int(self.aerenchyma_params.get("n_files", 1))
        aerenchyma_type = int(self.aerenchyma_params.get("aerenchyma_type", 1))
        start_angle = self._aerenchyma_start_angle

        def cell_quadrant(cell):
            cell_angle = np.arctan2(cell.y, cell.x) % (2 * np.pi)
            rel = (cell_angle - start_angle) % (2 * np.pi)
            return int(rel / (2 * np.pi / n_files)) % n_files

        tissue_cells = [c for t in tissues for c in self.all_cells.get_cells_by_type(t)]
        if not tissue_cells:
            return None

        # id_layer grows radially inward, so the single global max is the
        # innermost ring of the combined region: listed tissues act as one
        # contiguous band and only that inner boundary ring is preserved.
        max_tissue_layer = max(c.id_layer for c in tissue_cells)
        candidates = [c for c in tissue_cells if c.id_layer < max_tissue_layer]
        candidates.extend(self.all_cells.get_cells_by_type("air space"))

        if not candidates:
            return None

        total_tissue_area = sum(c.polygon.area for c in tissue_cells if c.polygon is not None)
        total_air_area = sum(c.polygon.area for c in self.all_cells.get_cells_by_type("air space") if c.polygon is not None)
        max_possible_area = sum(c.polygon.area for c in candidates if c.polygon is not None)

        quadrant_buckets = [[] for _ in range(n_files)]
        for c in candidates:
            quadrant_buckets[cell_quadrant(c)].append(c)

        if aerenchyma_type == 1:
            # Order each file as a *connected* radial channel rather than by raw angular distance.
            for q, bucket in enumerate(quadrant_buckets):
                central_angle = (start_angle + (q + 0.5) * 2 * np.pi / n_files) % (2 * np.pi)
                quadrant_buckets[q] = self._connected_radial_order(bucket, central_angle)
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

        position = {id(c): k for k, c in enumerate(self.all_cells.cells)}
        return dict(tissues=tissues, n_files=n_files, total_tissue_area=total_tissue_area,
                    total_air_area=total_air_area, max_possible_area=max_possible_area,
                    buckets=[[position[id(c)] for c in bucket] for bucket in quadrant_buckets])

    def merge_intercellular_aerenchyma(self):
        """Fuse touching air-space cells within the same angular sector, then carve tissue cells.

            The tissue is carved in bulk (the same GEOS difference on each cell the air reaches).
            """

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
                    protect_topology=any(getattr(c, "protect_topology", False) for c in group),
                    protect_shape=any(getattr(c, "protect_shape", False) for c in group),
                ))

            self.all_cells.remove_cells(merge_pool)
            self.all_cells.cells.extend(fused_cells)

        self.all_cells.cells = CellGenerator.simplify_cells(self.all_cells.cells, snap=self.section_topology)

        tissue = self.aerenchyma_params.get("tissue")
        air_spaces = self.all_cells.get_cells_by_type("air space")
        tissue_cells = self.all_cells.get_cells_by_type(tissue)
        tissue_cells.extend(a for a in air_spaces if a.id_layer == 0)

        air_union = unary_union([a.polygon for a in air_spaces if a.polygon is not None and a.id_layer != 0])

        # The carving, in bulk: a cell without polygon is dropped; a cell the air reaches is carved, and dropped if
        # nothing (or under 1e-6) is left; a cell the air never touches is left intact.
        polygons = np.array([cell.polygon for cell in tissue_cells], dtype=object)
        present = np.array([p is not None for p in polygons], dtype=bool)
        reached = np.zeros(len(tissue_cells), dtype=bool)
        if not air_union.is_empty and present.any():
            reached[present] = shapely.intersects(polygons[present], air_union)
        carved = shapely.difference(polygons[reached], air_union) if reached.any() else np.empty(0, dtype=object)
        kept = ~shapely.is_empty(carved) & (shapely.area(carved) > 1E-6) if carved.size else np.zeros(0, dtype=bool)
        reached_rows = np.flatnonzero(reached)
        set_polygons([tissue_cells[i] for i in reached_rows[kept]], carved[kept])
        dropped = [tissue_cells[i] for i in np.flatnonzero(~present)] + [tissue_cells[i] for i in reached_rows[~kept]]
        if dropped:
            self.all_cells.remove_cells(dropped)


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
        """Write anatomy cross section as .xml file (its walls numbered from the topology stored in the MPG)."""
        from openalea.granap.anatomy_writer import AnatomyWriter
        self.generate_section()
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

    def network_bridge_specs(self) -> List[Dict[str, Any]]:
        """Extra plasmodesmata/membrane network bridges beyond real shared walls.

        Each spec describes a NODE bridge (see ``NetworkExporter.export``'s
        Phase 9, and the reference diagram
        ``example/needle/Transfusion_tissue_network.png``): for a
        ``source_types`` cell A close to a ``target_types`` cell B with only
        one or more ``blocker_types`` cells geometrically between them (A-B
        straight line covered by A U B U blockers), a virtual NODE V_T is
        created per distinct blocker T actually crossed -- standing in for a
        real, out-of-plane third cell -- and chained
        ``A <-> V_T1 <-> ... <-> V_Tn <-> B`` via ``plasmodesmata`` edges,
        with ``membrane`` edges from the two end nodes to the EXISTING wall
        each already shares with its neighbouring blocker
        (``wall(A, T1)``/``wall(Tn, B)``). No new wall or junction is ever
        created, and the virtual node carries no polygon -- just a position
        (the blocker's own centroid) and a nominal area. One virtual node is
        shared per blocker across every path that crosses it. A spec dict
        has the keys ``name``, ``source_types``, ``target_types``,
        ``blocker_types``, ``bridge_type``, ``radius`` and ``max_links``
        (a fan-out cap: at most this many accepted bridge paths per source
        AND per target cell, shortest gaps preferred -- independent of node
        sharing); see ``NeedleAnatomy.network_bridge_specs`` for a concrete
        example. Default: no bridges for any organ.
        """
        return []

    def export_to_adjencymatrix(self, air_link_radius: Optional[float] = None, **exporter_kwargs) -> lil_matrix:
        """
        Build the hydraulic network from cell geometry and return
        the sparse adjacency matrix.

        ``air_link_radius`` bridges protected air-space cells that lie within
        that distance of each other but aren't directly adjacent (see
        ``NetworkExporter.export``); ``None`` uses its own default.

        ``**exporter_kwargs`` (e.g. ``bridges: bool``, ``bridge_radius:
        float``) are forwarded to ``NetworkExporter.export`` — see that
        method's docstring for the full set.

        Returns
        -------
        lil_matrix
            Sparse adjacency matrix (n_total x n_total).
        """
        self.generate_section()
        if self._matrix is not None:
            return self._matrix
        if not self.__dict__.get("_network_built"):
            self._build_anatnetwork(air_link_radius=air_link_radius, **exporter_kwargs)
        g, written = self.mpg, self._mpg_written
        u_of, v_of = g.property("edge_u"), g.property("edge_v")
        u = np.array([u_of[e] for e in written["connections"]], dtype=np.int64)
        v = np.array([v_of[e] for e in written["connections"]], dtype=np.int64)
        n = self.n_total
        self._matrix = coo_matrix((np.ones(2 * u.size), (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tolil()
        return self._matrix

    # ------------------------------------------------------------------
    # Network construction from Voronoi cell geometry
    # ------------------------------------------------------------------
    def _build_anatnetwork(self, air_link_radius: Optional[float] = None, **exporter_kwargs) -> None:
        """
        The network: the one stored in the MPG at generation, or, for other exporter settings (bridges, air link radius,
        wall thicknesses), the exporter's, written into the MPG in its place. The graph is a view of the MPG.
        """
        from openalea.granap.anatomy_writer import NetworkExporter
        from openalea.granap.metafspm.build import write_network
        self.generate_section()
        written = self._mpg_written
        if not self._stored_network_matches(air_link_radius, exporter_kwargs):
            self.graph = nx.Graph()                                               # the exporter's own target
            NetworkExporter(self).export(self, air_link_radius=air_link_radius, **exporter_kwargs)
            written = write_network(self.mpg, self.segment, self, written)        # replaces the stored one
            n_real = len(written["cells"]) - len(written.get("bridge_cells", []))
            if len(written["cells"]) > n_real or "bridge" in self._cells_gdf.columns:
                self._read_cells_back(n_real)
        self.graph = None                                                         # a view of the MPG, when read
        self.n_walls, self.n_junctions, self.n_cells = written["n_walls"], written["n_junctions"], written["n_cells"]
        self._wall_to_cells = dict(written["wall_to_cells"])
        self._network_built = True

    
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
