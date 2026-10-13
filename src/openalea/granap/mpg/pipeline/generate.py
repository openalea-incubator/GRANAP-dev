"""
The cell generation, converted (devplan_metafspm.md, §28, N2): GRANAP's Organ.generate_cells (organ_class.py),
copied, the stages it goes through written into the MPG where they end instead of through method overrides, and no
GeoDataFrame built at the end (the organ's _cells_gdf is a view of the MPG, built when read). Every other line is
GRANAP's.

    the layers' polygons, the seeds, the vascular and organ-specific tissues     (no cells yet: arrays and polygons)
    the Voronoi diagram and its grouping into cells        -> stage "grouping"; the cells become views of the MPG
    the intercellular spaces (and in them, the aerenchyma) -> stages "intercellular", "aerenchyma" (written by the
                                                              calibration, which the organs' own methods call)
    the gaps fused                                         -> stage "fused_gaps"
    the layers populated, the cells' properties            (the stage "final" follows, in MPGBacked.generate_cells)
"""
import time

from shapely.ops import unary_union

from openalea.granap.cell_manager import CellManager
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.organ_class import log
from openalea.granap.mpg.pipeline.gaps import recalculate_cell_properties
from openalea.granap.mpg.pipeline.merge import simplify_cells


def generate_cells(organ) -> None:
    """Generate the organ's cells (Voronoi tessellation of its layers), stage by stage into its MPG."""
    t_start = time.time()
    layers_polygons = organ.generate_layer_polygons()
    log.info("Layer polygons:          %.3fs", time.time() - t_start)
    center = layers_polygons[0]["polygon"].centroid

    t_start = time.time()
    for layer in organ.layer_manager.get_layers():
        layer.cells = []
    organ.all_cells = CellGenerator.generate_cells_info(layers_polygons, center, rng=organ.rng)
    log.info("Cell seeds:              %.3fs", time.time() - t_start)

    t_start = time.time()
    organ.vascular_cells = CellManager()
    organ.vascular_polygons = []
    organ.vascular_tissue_polygons = {}
    organ.allocate_vascular_tissue(layers_polygons)

    # Vascular elements take priority: remove every layer seed that falls
    # inside any vascular zone (xylem vessels + phloem + any other named
    # tissue tracked in vascular_tissue_polygons).
    all_vascular_polys = list(organ.vascular_polygons)
    for poly_list in organ.vascular_tissue_polygons.values():
        all_vascular_polys.extend(poly_list)
    if all_vascular_polys:
        vascular_mask = unary_union(all_vascular_polys)
        organ.all_cells.remove_cells_in_polygon(vascular_mask)

    if organ.vascular_cells.cells:
        organ.all_cells.extend_cells(organ.vascular_cells.cells)
    organ._organ_specific_tissues()
    log.info("Vascular + organ tissues: %.3fs", time.time() - t_start)

    t_start = time.time()
    vor = CellGenerator.voronoi_diagram(organ.all_cells, rng=organ.rng)
    log.info("Voronoi diagram:         %.3fs", time.time() - t_start)

    t_start = time.time()
    grouped_cells = CellGenerator.process_voronoi_groups(organ.all_cells, vor).cells
    grouped_cells = simplify_cells(grouped_cells)                   # B2a: the polygons rebuilt in bulk
    # simplify_cells rebuilds each polygon independently, which can distort
    # small cells (few vertices) into a neighbour and leave them rendered
    # one-inside-another; drop those, keeping the larger cell.
    grouped_cells = CellGenerator.remove_nested_cells(grouped_cells)
    organ.all_cells = CellManager()
    organ.all_cells.cells = grouped_cells
    log.info("Voronoi grouping:        %.3fs", time.time() - t_start)
    organ._grouped()                                       # stage "grouping"; from here on, views of the MPG

    t_start = time.time()
    organ.add_intercellular_spaces()
    log.info("Intercellular spaces:    %.3fs", time.time() - t_start)

    if organ.AUTO_FUSE_GAPS:
        t_start = time.time()
        n_fused = organ.fuse_gaps()
        log.info("Fuse gaps:               %.3fs (%d fused)",
                 time.time() - t_start, n_fused)
        organ._mpg_stage("fused_gaps")

    t_start = time.time()
    for cell in organ.all_cells.cells:
        if 0 <= cell.id_layer < len(layers_polygons):
            layer_name = layers_polygons[cell.id_layer]["name"]
            if layer_name != "outside":
                layer = organ.get_layer(layer_name)
                if layer:
                    layer.cells.append(cell)
    log.info("Layer population:        %.3fs", time.time() - t_start)

    t_start = time.time()
    recalculate_cell_properties(organ.all_cells.cells)         # B2d: the geometry in bulk
    log.info("Cell properties:         %.3fs", time.time() - t_start)
