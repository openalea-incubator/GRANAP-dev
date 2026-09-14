import os
import io
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely as sp
from typing import Dict, Any, Union, List, Optional
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.affinity import scale
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from openalea.granap.organ_class import Organ
from openalea.granap.network_base import AbstractNetwork
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.cell_class import Cell


DEFAULT_CELL_WALL_THICKNESS: Dict[str, float] = {
    "epidermis": 2,
    "exodermis": 2,
    "hypodermis": 2,
    "endodermis": 1.5,
    "cortex": 1,
    "mesophyll": 1,
    "parenchyma": 1,
    "stele": 1,
    "pericycle": 1,
    "phloem": 1,
    "xylem": 1.5,
    "protoxylem": 1.5,
    "metaxylem": 2,
    "cambium": 1,
    "duct": 5,
    "resin duct epithelium": 1,   # thin-walled, secretory (Ep)
    "resin duct sheath": 2,        # thicker-walled (Sh)
    "guard cell": 2,
    "Strasburger cell": 1,
    "Str. Interstitial cell": 1,
    "outerwall": 2,
    "air space": 0.001,
    "pore": 0.001,
    "aerenchyma": 0.001,
}

# ---------------------------------------------------------------------------
# GRANAP cell ``type`` string -> MECHA ``cgroup`` integer.
#
# Mirrors MECHA's own authoritative maps (read-only, for reference):
#   - ``hydraulic_cell.py``'s ``CGROUP_TO_TYPE`` (cgroup -> name, used to build
#     ``HydraulicCell`` objects -- ``hydraulic_cell.py:925`` does a bare
#     ``CGROUP_TO_TYPE[cgroup]`` and raises ``KeyError`` on an unknown cgroup).
#   - ``network_builder.py``'s ``populate_from_network`` fallback ``type_mapper``
#     (name -> cgroup, only consulted when GRANAP supplies no valid int cgroup).
#
# GRANAP previously never populated ``cgroup`` at all, so every cell fell back
# to ``type_mapper``'s default of 4 (cortex) -- including "transfusion
# parenchyma"/"transfusion tracheid", neither of which are in that fallback
# dict, so both silently became living, plasmodesmata-connected cortex cells.
# This map is consulted directly by ``NetworkExporter.export`` (bypassing the
# MECHA-side fallback entirely) whenever a cell's own ``Cell.cgroup`` is unset
# (0 / falsy).
# ---------------------------------------------------------------------------
CGROUP_MAP: Dict[str, int] = {
    "exodermis": 1,
    "hypodermis": 1,
    "hypodermis_corner": 1,  # same tissue as "hypodermis", just the corner-thickened rows
    "epidermis": 2,
    "endodermis": 3,
    "passage_cell": 3,
    "cortex": 4,
    "mesophyll": 4,
    "mesophyll_abaxial": 4,  # same tissue as "mesophyll", the abaxial-only extra ring
    "palisade": 4,           # needle palisade mesophyll -- same rank as the rest of the mesophyll
    "air space": 4,
    "pore": 4,
    "duct": 4,
    "aerenchyma": 4,
    "resin duct epithelium": 4,
    "resin duct sheath": 4,
    "stele": 5,
    "pith": 5,
    "parenchyma": 5,
    "phloem": 11,
    "sieve": 11,
    "companion_cell": 12,
    "cambium": 12,
    "guard cell": 12,
    "Strasburger cell": 12,
    "Str. Interstitial cell": 12,
    "xylem": 13,
    "protoxylem": 13,
    "metaxylem": 13,
    "pericycle": 16,
    "transfusion parenchyma": 17,
    "transfusion tracheid": 18,
    "protosieve": 23,
    "protophloem": 23,
}

class AnatomyWriter:
    """
    Class to export Organ anatomy to various formats (XML, OBJ, GEO).
    """

    def __init__(self, organ: Organ):
        self.organ = organ
        # Ensure cells are generated
        self.organ.generate_cells()
        self.cells = self.organ.all_cells.cells

    def write_to_xml(self, path: str):
        """
        Write the root anatomy as an XML file matching GRANAR format.

        Cell ordering and wall IDs are derived from ``self.organ.generate_cells()``
        using the same GeoDataFrame index as ``NetworkExporter.export``, so that
        XML cell id attributes map 1-to-1 to the GRANAP graph cell node ids.
        """

        # One mapping for both exporters: CGROUP_MAP is what NetworkExporter.export
        # resolves cell nodes with, so the XML and the graph agree by construction.
        # (This used to be a separate local dict that drifted from it -- it mapped
        # "cambium" to 11 instead of 12, listed "stele" twice, and sent an unknown
        # tag to group 0, which MECHA reads as no tissue at all.)
        cellgroups = CGROUP_MAP

        # Use the same GeoDataFrame (and same index) as NetworkExporter.export().
        # IMPORTANT: apply the IDENTICAL valid-geometry filter so that both paths
        # call _build_topology with the exact same polygon set -> same KD-tree
        # snap_tol, same clusters, same canonical junction coordinates.
        cells_gdf = self.organ.generate_cells()
        valid_mask = cells_gdf["geometry"].notna() & cells_gdf["geometry"].apply(
            lambda g: g is not None and not g.is_empty
        )
        valid_gdf = cells_gdf[valid_mask]

        polys    = list(valid_gdf["geometry"])
        cell_ids = list(valid_gdf.index)

        # Cells flagged ``protect_topology`` (needle mesophyll air-space rhombi)
        # keep their walls from collapsing into a single node.
        # (see CellGenerator._build_topology's ``protect_ids``).
        protect_ids = (
            set(valid_gdf.index[valid_gdf["protect_topology"]])
            if "protect_topology" in valid_gdf.columns else None
        )

        cell_vkeys, _, _, junction_set, protected_shape_set = CellGenerator._build_topology(
            polys, cell_ids, protect_ids=protect_ids
        )

        wall_registry = {}
        next_wall_id  = 0
        cell_walls    = {idx: [] for idx in cell_ids}

        for row_idx, vkeys in cell_vkeys.items():
            n = len(vkeys)
            junc_positions = [i for i in range(n) if vkeys[i] in junction_set]

            if len(junc_positions) < 2:
                # no junctions → single wall loop
                shape_signature = tuple(vk for vk in vkeys if vk in protected_shape_set)
                wall_key = (tuple(sorted(vkeys)), shape_signature,)
                if wall_key not in wall_registry:
                    wall_registry[wall_key] = {"id": next_wall_id, "points": list(vkeys) + [vkeys[0]]}
                    next_wall_id += 1
                cell_walls[row_idx].append(wall_registry[wall_key]["id"])
                continue

            for jp in range(len(junc_positions)):
                start_idx = junc_positions[jp]
                end_idx   = junc_positions[(jp + 1) % len(junc_positions)]

                segment = []
                i = start_idx
                while True:
                    segment.append(vkeys[i])
                    if i == end_idx:
                        break
                    i = (i + 1) % n

                if len(segment) < 2:
                    continue

                junc_start = segment[0]
                junc_end   = segment[-1]
                wall_key   = tuple(sorted((junc_start, junc_end)))

                if wall_key not in wall_registry:
                    wall_registry[wall_key] = {"id": next_wall_id, "points": segment}
                    next_wall_id += 1

                cell_walls[row_idx].append(wall_registry[wall_key]["id"])

        xml_lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<granardata>',
            '\t<metadata>',
            '\t\t<parameters>',
            '\t\t\t<parameter io="0" name="python_export" type="default" value="1"/>',
            '\t\t</parameters>',
            '\t</metadata>',
            f'\t<cells count="{len(valid_gdf)}">'
        ]

        # Write cells using GDF index as id — matches graph node numbering
        for row_idx, row in valid_gdf.iterrows():
            # Same precedence NetworkExporter.export uses for cell nodes: an
            # explicit Cell.cgroup wins (that is how imported real data keeps its
            # own group integers), else the tag is looked up. The old default of
            # 0 was not a soft failure -- MECHA does a bare CGROUP_TO_TYPE[cgroup]
            # lookup and 0 is not a key.
            group_id = int(row.get("cgroup") or cellgroups.get(row.get("type", ""), 4))
            xml_lines.append(f'\t\t<cell id="{row_idx}" group="{group_id}" truncated="false" >')
            xml_lines.append(f'\t\t\t<walls>')
            for wid in cell_walls[row_idx]:
                xml_lines.append(f'\t\t\t\t<wall id="{wid}"/>')
            xml_lines.append(f'\t\t\t</walls>')
            xml_lines.append(f'\t\t</cell>')

        xml_lines.append('\t</cells>')

        xml_lines.append(f'\t<walls count="{len(wall_registry)}">')
        for wdict in wall_registry.values():
            wid = wdict["id"]
            xml_lines.append(f'\t\t<wall id="{wid}" group="0" edgewall="false" >')
            xml_lines.append(f'\t\t\t<points>')
            for pt in wdict["points"]:
                xml_lines.append(f'\t\t\t\t<point x="{pt[0]}" y="{pt[1]}"/>')
            xml_lines.append(f'\t\t\t</points>')
            xml_lines.append(f'\t\t</wall>')
        xml_lines.append('\t</walls>')

        xml_lines.append('\t<groups>')
        xml_lines.append('\t\t<cellgroups>')
        for cname, cid in cellgroups.items():
            xml_lines.append(f'\t\t\t<group id="{cid}" name="{cname}" />')
        xml_lines.append('\t\t</cellgroups>')
        xml_lines.append('\t\t<wallgroups>')
        xml_lines.append('\t\t\t<group id="0" name="unassigned" />')
        xml_lines.append('\t\t</wallgroups>')
        xml_lines.append('\t</groups>')

        xml_lines.append('</granardata>\n')

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(xml_lines))
        print(f"XML saved to {path}")

    def write_xml_geometry(self, path: str, barrier: List= [1,3]):
        """
        write the MECHA geometry files (retro-compatible)

        """

        cells_gdf = self.organ.generate_cells()
        valid_mask = cells_gdf["geometry"].notna() & cells_gdf["geometry"].apply(
            lambda g: g is not None and not g.is_empty
        )
        valid_gdf = cells_gdf[valid_mask]

        passage_cells = valid_gdf[valid_gdf["type"] == "passage_cell"].index.tolist()
        aerenchyma_cells = valid_gdf[valid_gdf["type"].isin(["aerenchyma", "air space", "pore"])].index.tolist()

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<param>',
            '\t<!-- Plant type -->',
            '\t<Plant value="Root"/>',
            '\t<!-- Image path and properties -->',
            '\t<path value=""/>',
            '\t<im_scale value="1000"/>',
            '\t<Maturityrange>'
        ]
        
        for b in barrier:
            xml_lines.append(f'\t\t<Maturity Barrier="{b}" height="200" Nlayers="1"/>')
            
        xml_lines.extend([
            '\t</Maturityrange>',
            '\t<Printrange>',
            '\t\t<Print_layer value="0"/>',
            '\t</Printrange>',
            '\t<Xwalls value="1"/>',
            '\t<PileUp value="0"/>',
            '\t<passage_cell_range>'
        ])

        if not passage_cells:
            xml_lines.append('\t\t<passage_cell id="-1"/>')
        else:
            for pid in passage_cells:
                xml_lines.append(f'\t\t<passage_cell id="{pid}"/>')

        xml_lines.append('\t</passage_cell_range>')
        xml_lines.append('\t<aerenchyma_range>')

        for aid in aerenchyma_cells:
            xml_lines.append(f'\t\t<aerenchyma id="{aid}"/>')
        
        xml_lines.extend([
            '\t</aerenchyma_range>',
            '\t<InterC_perim_search value="0"/>',
            '\t<InterC_perim1 value="0"/>',
            '\t<InterC_perim2 value="0"/>',
            '\t<InterC_perim3 value="0"/>',
            '\t<InterC_perim4 value="0"/>',
            '\t<InterC_perim5 value="0"/>',
            '\t<kInterC value="0.0"/>',
            '\t<cell_per_layer cortex="nan" stele="nan"/>',
            '\t<diffusion_length cortex="nan" stele="nan"/>',
            '\t<thickness value="1.5"/>',
            '\t<PD_section value="7.47E-5"/>',
            '\t<Xylem_pieces flag="0"/>',
            '</param>'
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(xml_lines) + "\n")
        
        print(f"Geometry XML saved to {path}")

    def write_to_obj(self, path: str, membrane: bool = True, wall: bool = True, shrink_factor: float = 0.001):
        """
        Write a .obj from the generated cross section geometry.
        If membrane is True, write shrank cell polygons as faces.
        If False, write the cell borders as lines.
        """
        obj_lines = ['# Wavefront OBJ file']

        vertex_registry = {}
        v_idx = 1
    
        def get_v_idx(x, y):
            nonlocal v_idx
            key = (round(x, 6), round(y, 6))
            if key not in vertex_registry:
                vertex_registry[key] = v_idx
                obj_lines.append(f"v {key[0]} {key[1]} 0.0")
                v_idx += 1
            return vertex_registry[key]
    
        def process_polygon(poly):
            faces_lines = []
    
            # --- shrink polygon ---
            buffed_poly = poly.buffer(-shrink_factor)
    
            # ----------------------
            # MEMBRANE (filled face)
            # ----------------------
            if membrane and not buffed_poly.is_empty:
                if isinstance(buffed_poly, MultiPolygon):
                    polys = buffed_poly.geoms
                else:
                    polys = [buffed_poly]
    
                for p in polys:
                    coords = list(p.exterior.coords[:-1])
                    v_indices = [str(get_v_idx(x, y)) for x, y in coords]
                    if len(v_indices) >= 3:
                        faces_lines.append("f " + " ".join(v_indices))
            if wall:
                wall_poly = poly.difference(buffed_poly)
    
                if wall_poly.is_empty:
                    return faces_lines
    
                if isinstance(wall_poly, MultiPolygon):
                    polys = wall_poly.geoms
                else:
                    polys = [wall_poly]
    
                for p in polys:
                    # outer ring
                    outer = list(p.exterior.coords)
                    for i in range(len(outer) - 1):
                        v1 = get_v_idx(*outer[i])
                        v2 = get_v_idx(*outer[i + 1])
                        faces_lines.append(f"l {v1} {v2}")
    
                    # inner rings (holes)
                    for interior in p.interiors:
                        inner = list(interior.coords)
                        for i in range(len(inner) - 1):
                            v1 = get_v_idx(*inner[i])
                            v2 = get_v_idx(*inner[i + 1])
                            faces_lines.append(f"l {v1} {v2}")
            else:
                coords = list(poly.exterior.coords)
                for i in range(len(coords) - 1):
                    v1 = get_v_idx(*coords[i])
                    v2 = get_v_idx(*coords[i + 1])
                    faces_lines.append(f"l {v1} {v2}")
    
            return faces_lines
    
        all_faces = []
    
        for cell in self.cells:
            if cell.polygon is None:
                continue

            if cell.type in ["air space", "pore", "xylem"]:
                continue

    
            poly = cell.polygon
    
            if isinstance(poly, MultiPolygon):
                for p in poly.geoms:
                    all_faces.extend(process_polygon(p))
            else:
                all_faces.extend(process_polygon(poly))
    
        obj_lines.extend(all_faces)
        obj_lines.append("")
    
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(obj_lines))
    
        print(f"OBJ saved to {path}")

    def write_to_svg(self, path: str, shrink_factor: Union[float, Dict[str, float]] = DEFAULT_CELL_WALL_THICKNESS, 
                        corner_smoothing: Union[float, Dict[str, float]] = 0.5):
        """
        Write a .svg from the generated cross section geometry.
        Uses prep_geo logic for cell rendering.
        """
        inner_polygons, final_polygon = self.prep_geo(self.cells, cell_wall_thickness=shrink_factor, corner_smoothing=corner_smoothing)
        
        svg_lines = []
        
        valid_cells = [c for c in self.cells if c.polygon is not None]
        if not valid_cells:
            print("No valid cells to export.")
            return

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        # Colors for cell groups
        viridis = plt.get_cmap("viridis")

        cell_types_list = []
        for cell in valid_cells:
            if cell.type not in cell_types_list:
                cell_types_list.append(cell.type)
        
        # shuffle cell types list (1, last, 2, last-1, ...)
        shuffled_cell_types_list = [""]*len(cell_types_list)
        for i in range(len(cell_types_list)//2 + 1):
            if not cell_types_list:
                break
            shuffled_cell_types_list[i*2] = cell_types_list[0]
            cell_types_list.remove(cell_types_list[0])
            if len(cell_types_list) > 0:
                shuffled_cell_types_list[i*2+1] = cell_types_list[-1]
                cell_types_list.remove(cell_types_list[-1])

        
        cell_colors = {"default": "#440154"}
        for i, cell_type in enumerate(shuffled_cell_types_list):
            cell_colors[cell_type] = to_hex(viridis(i / max(1, len(shuffled_cell_types_list) - 1)))

        # Calculate bounding box
        bounds = final_polygon.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
            
        width = (max_x - min_x)
        height = (max_y - min_y)
        
        pad_x, pad_y = width * 0.05, height * 0.05
        min_x, min_y = min_x - pad_x, min_y - pad_y
        width, height = width + 2*pad_x, height + 2*pad_y
        
        svg_lines.append(f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
        svg_lines.append(f'<svg viewBox="{min_x} {min_y} {width} {height}" xmlns="http://www.w3.org/2000/svg">')
        svg_lines.append(f'\t<rect x="{min_x}" y="{min_y}" width="{width}" height="{height}" fill="white" />')

        def get_svg_points(poly):
            return " ".join([f"{x},{y}" for x, y in poly.exterior.coords])
            
        def get_svg_path(poly):
            d = f"M {poly.exterior.coords[0][0]} {poly.exterior.coords[0][1]} "
            for x, y in list(poly.exterior.coords)[1:]:
                d += f"L {x} {y} "
            for interior in poly.interiors:
                d += f"M {interior.coords[0][0]} {interior.coords[0][1]} "
                for x, y in list(interior.coords)[1:]:
                    d += f"L {x} {y} "
            return d

        svg_lines.append(f'\t<polygon points="{get_svg_points(final_polygon)}" fill="black" stroke="none" />')

        for cell in inner_polygons:
            color = cell_colors.get(cell["type"], cell_colors["default"])
            poly = cell["polygon"]
            svg_lines.append(f'\t<polygon points="{get_svg_points(poly)}" fill="{color}" stroke="none" />')
                
        svg_lines.append("</svg>")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_lines))
        print(f"SVG saved to {path}")

    @staticmethod
    def prep_geo(cells: List, cell_wall_thickness: Union[float, Dict[str, float]], 
                 corner_smoothing: Union[float, Dict[str, float]]):
        """
        Pre-proc for .geo file generation.
        Returns list of shrunken inner luminal polygons and one full tissue outer boundary.
        Uses GeometryProcessor for buffering and smoothing.
        Geometry is scaled by 1000 (microns).
        """
        
        inner_polygons = []
        outer_tissue_polygons = []

        def get_thickness(c_type):
            if isinstance(cell_wall_thickness, dict):
                val = cell_wall_thickness.get(c_type, cell_wall_thickness.get("default", 1))
            else:
                val = cell_wall_thickness
            # No conversion, assumed scaling to microns
            return val
            
        def get_smoothing(c_type):
            if isinstance(corner_smoothing, dict):
                return corner_smoothing.get(c_type, corner_smoothing.get("default", 5))
            return corner_smoothing

        for cell in cells:
            if cell.polygon is None:
                continue
            
            poly = cell.polygon
            if not poly.is_valid:
                poly = poly.buffer(0)
                
            # Scale coordinates by 1000 to match GMSH expected micron scale
            r_poly = scale(poly, xfact=1000, yfact=1000, origin=(0, 0))
            # improve the resolution of the polygon
            coords = GeometryProcessor.resample_coords(r_poly.exterior.coords, int(len(r_poly.exterior.coords)*5))
            r_poly = Polygon(coords)
            
            thickness = get_thickness(cell.type)
            smoothing = get_smoothing(cell.type)

            r_poly_smooth = GeometryProcessor.buffer_polygon(r_poly, -thickness, smooth_factor=smoothing)
            
            if not r_poly_smooth.is_empty:
                inner_polygons.append({
                    "id_cell": cell.id_cell,
                    "type": cell.type,
                    "polygon": r_poly_smooth
                })

            # For outer tissue boundary, slightly swell the polygon and add to list for union
            outer_thickness = get_thickness("outerwall")*1.1
            swollen_polygon = GeometryProcessor.buffer_polygon(r_poly, outer_thickness, 0.001)
            outer_tissue_polygons.append(swollen_polygon)

        # Union to get the final tissue polygon
        final_polygon = sp.ops.unary_union(outer_tissue_polygons)

        return inner_polygons, final_polygon

    def write_to_geo(self, path: str, dim: int = 2, celldomain: bool = False,
                     cell_wall_thickness: Union[float, Dict[str, float]] = DEFAULT_CELL_WALL_THICKNESS, 
                     corner_smoothing: Union[float, Dict[str, float]] = 0.5):
        """
        Write .geo file for GMSH.
        Calls prep_geo to compute cell lumina and the outer boundary.
        """
        inner_polygons, final_polygon = self.prep_geo(self.cells, cell_wall_thickness, corner_smoothing)
        
        geo_lines = [
            '// Gmsh project',
            'SetFactory("OpenCASCADE");',
            '//+'
        ]

        vertex_registry = {}
        v_idx = 1
        l_idx = 1
        s_idx = 1
        c_loop = 1

        def register_polygon_edges(poly, tol=0.2):
            nonlocal v_idx, l_idx
            
            # Use shapely's Douglas-Peucker simplification to reduce points drastically
            poly_simplified = poly.simplify(tol, preserve_topology=True)
            if poly_simplified.geom_type != 'Polygon' or poly_simplified.is_empty:
                poly_simplified = poly

            coords = list(poly_simplified.exterior.coords)[:-1]

            v_start_idx = v_idx
            for c in coords:
                geo_lines.append(f"Point({v_idx}) = {{{round(c[0], 2)}, {round(c[1], 2)}, 0, 1.0}};")
                v_idx += 1
                
            line_ids = []
            n_pts = len(coords)
            for i in range(n_pts):
                curr = v_start_idx + i
                nxt = v_start_idx + ((i + 1) % n_pts)
                geo_lines.append(f"Line({l_idx}) = {{{curr}, {nxt}}};")
                geo_lines.append("//+")
                line_ids.append(l_idx)
                l_idx += 1
                
            return line_ids

        # which cell is at centroid closer to (0,0) of the cross-section
        center_cell = self.cells[0]
        for cell in self.cells:
            if cell.polygon.centroid.distance(Point(0,0)) < center_cell.polygon.centroid.distance(Point(0,0)):
                center_cell = cell

        # Write each inner cell
        cell_curves = []
        air_space_curves = []
        for item in inner_polygons:
            poly = item["polygon"]
            id_cell = item["id_cell"]
            id_type = item["type"]
            if poly.geom_type == 'MultiPolygon':
                geoms = list(poly.geoms)
            else:
                geoms = [poly]
                
            for geom in geoms:
                line_ids = register_polygon_edges(geom)
                
                cl_idx = c_loop
                geo_lines.append(f"Curve Loop({cl_idx}) = {{{', '.join(map(str, line_ids))}}};")
                if id_cell == center_cell.id_cell:
                    center_curve = [cl_idx]
                elif id_type in ["air space", "pore"]:
                    air_space_curves.append(cl_idx)
                else:
                    cell_curves.append(cl_idx)
                geo_lines.append("//+")
                geo_lines.append(f"Surface({s_idx}) = {{{cl_idx}}};")
                geo_lines.append("//+")
                
                if celldomain:
                    geo_lines.append(f"Physical Surface({s_idx}) = {{{s_idx}}};")
                else:
                    geo_lines.append(f"//Physical Surface({s_idx}) = {{{s_idx}}};")
                    
                s_idx += 1
                c_loop += 2

        # Write final outer domain
        if final_polygon.geom_type == 'MultiPolygon':
            p_geoms = list(final_polygon.geoms)
        else:
            p_geoms = [final_polygon]
            
        for geom in p_geoms:
            line_ids = register_polygon_edges(geom)
            
            cl_idx = c_loop
            geo_lines.append(f"Curve Loop({cl_idx}) = {{{', '.join(map(str, line_ids))}}};")
            geo_lines.append("//+")
            
            # Plane Surface mapping to inner holes + boundary
            plane_surfaces = list(range(1, cl_idx+1, 2))
            plane_surfaces.sort(reverse=True)
            
            # Replicate Plane Surface format from R code
            geo_lines.append(f"Plane Surface({s_idx}) = {{{', '.join(map(str, plane_surfaces))}}};")
            geo_lines.append("//+")
            geo_lines.append(f"Physical Surface(0) = {{{s_idx}}};")
            geo_lines.append(f'Physical Curve("cells", 1) = {{{", ".join(map(str, cell_curves))}}};')
            geo_lines.append(f'Physical Curve("air space", 2) = {{{", ".join(map(str, air_space_curves))}}};')
            geo_lines.append(f'Physical Curve("center", 3) = {{{", ".join(map(str, center_curve))}}};')
            
            s_idx += 1
            c_loop += 2

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(geo_lines))
        print(f"GEO saved to {path}")

class NetworkExporter:
    """
    Class to export Organ anatomy to an AbstractNetwork topological graph.
    """

    def __init__(self, organ: Organ):
        self.organ = organ

    def export(self, network: AbstractNetwork, cell_wall_thickness: Union[float, Dict[str, float]] = DEFAULT_CELL_WALL_THICKNESS, air_link_radius: Optional[float] = None,
               bridges: bool = True, bridge_radius: Optional[float] = None) -> None:
        """
        Populate the provided network graph from the cell GeoDataFrame.

        Algorithm
        ---------
        1. Delegate vertex snapping, vertex/edge maps, and junction
           detection to :meth:`CellGenerator._build_topology`.
        2. Walk each cell boundary between consecutive junctions to
           define **walls** (one wall per cell-pair interface).
        3. Register any organ-declared network bridges (see
           :meth:`Organ.network_bridge_specs`) as extra virtual CELL rows
           into ``cells_gdf`` -- no polygon, no new wall, no new junction --
           *before* node indices are assigned, so the MECHA
           ``[walls][junctions][cells]`` contract stays contiguous with no
           renumbering. Each bridge is a virtual node standing in for an
           out-of-plane cell, chained via ``plasmodesmata`` to its real
           source/target cells and wired via ``membrane`` to the EXISTING
           wall it already shares with its neighbouring blocker cell.
        4. Assign MECHA-compatible node indices and build the graph
           (including the bridge ``membrane``/``plasmodesmata`` edges).

        ``bridges``/``bridge_radius`` control step 3 — set ``bridges=False``
        to reproduce the pre-bridge graph exactly, or override every spec's
        own radius with ``bridge_radius``.
        """
        cells_gdf = self.organ.generate_cells()

        # Filter to cells with valid (non-null, non-empty) geometry — must match
        # the same filter used in write_to_xml so that both paths call
        # _build_topology with exactly the same polygon set, guaranteeing
        # identical KD-tree snapping results and canonical junction coordinates.
        valid_mask = cells_gdf["geometry"].notna() & cells_gdf["geometry"].apply(
            lambda g: g is not None and not g.is_empty
        )

        # Exclude any *leftover* bridge rows from a previous export() call on
        if "bridge" in cells_gdf.columns:
            valid_mask = valid_mask & ~cells_gdf["bridge"].astype(bool)

        cells_gdf = cells_gdf[valid_mask].copy()

        # Mesophyll air-space nodes requiring special rewiring
        protected_air_cells = set(
            cells_gdf.index[ (cells_gdf.get("protect_topology", False))
                           & (cells_gdf["type"] == "air space")
                           ]
        )

        # Cells whose type must NEVER carry a symplastic (plasmodesmata) edge
        apoplastic_cells = set(
            cells_gdf.index[
                cells_gdf["type"].isin(getattr(self.organ, "APOPLASTIC_ONLY_TYPES", ()))
            ]
        )


        # Phases 0–2 — snapping, topology maps, junction detection
        polys    = list(cells_gdf["geometry"])
        cell_ids = list(cells_gdf.index)

        # Cells flagged ``protect_topology`` (needle mesophyll air-space rhombi)
        # keep their walls from collapsing into a single node.
        # (see CellGenerator._build_topology's ``protect_ids``).
        protect_ids = (
            set(cells_gdf.index[cells_gdf["protect_topology"]])
            if "protect_topology" in cells_gdf.columns else None
        )

        def get_thickness(c_type: str) -> float:
            if isinstance(cell_wall_thickness, dict):
                return float(cell_wall_thickness.get(c_type, cell_wall_thickness.get("default", 1)))
            return float(cell_wall_thickness)

        cell_vkeys, _, edge_to_cells, junction_set, protected_shape_set = (
            CellGenerator._build_topology(polys, cell_ids, protect_ids=protect_ids)
        )

        if not cell_vkeys:
            return

        # Phase 3 — walk cell boundaries to define walls
        # A "wall" = the polyline segment between two consecutive
        # junction vertices along one cell boundary.  Two cells that
        # share the same (juncA, juncB) segment share a wall.
        wall_registry: Dict[tuple, dict] = {}  # wall_key -> wall info
        next_wall_id = 0

        for row_idx, vkeys in cell_vkeys.items():
            n = len(vkeys)
            junc_positions = [i for i in range(n) if vkeys[i] in junction_set]

            if len(junc_positions) < 2:
                # Fewer than 2 junctions -> treat entire boundary as one wall
                wall_key = tuple(sorted(vkeys))
                if wall_key not in wall_registry:
                    length = sum(
                        np.hypot(vkeys[(k+1) % n][0] - vkeys[k][0],
                                 vkeys[(k+1) % n][1] - vkeys[k][1])
                        for k in range(n)
                    )
                    mid_x = np.mean([v[0] for v in vkeys])
                    mid_y = np.mean([v[1] for v in vkeys])
                    wall_registry[wall_key] = {
                        "id": next_wall_id,
                        "junc_start": vkeys[0],
                        "junc_end": vkeys[0],
                        "midpoint": (mid_x, mid_y),
                        "length": length,
                        "wall_thickness": 0.0,
                        "cells": [],
                    }
                    next_wall_id += 1
                if row_idx not in wall_registry[wall_key]["cells"]:
                    wall_registry[wall_key]["cells"].append(row_idx)
                continue

            for jp in range(len(junc_positions)):
                start_idx = junc_positions[jp]
                end_idx = junc_positions[(jp + 1) % len(junc_positions)]

                # Collect vertices along the segment
                segment: List[tuple] = []
                i = start_idx
                while True:
                    segment.append(vkeys[i])
                    if i == end_idx:
                        break
                    i = (i + 1) % n

                if len(segment) < 2:
                    continue

                junc_start = segment[0]
                junc_end = segment[-1]

                shape_signature = tuple(vk for vk in segment[1:-1] if vk in protected_shape_set)

                wall_key = (tuple(sorted((junc_start, junc_end))), shape_signature,)

                if wall_key not in wall_registry:
                    length = sum(
                        np.hypot(segment[k+1][0] - segment[k][0],
                                 segment[k+1][1] - segment[k][1])
                        for k in range(len(segment) - 1)
                    )
                    mid_x = np.mean([v[0] for v in segment])
                    mid_y = np.mean([v[1] for v in segment])
                    wall_registry[wall_key] = {
                        "id": next_wall_id,
                        "junc_start": junc_start,
                        "junc_end": junc_end,
                        "midpoint": (mid_x, mid_y),
                        "length": length,
                        "wall_thickness": 0.0,
                        "cells": [],
                        "shape_signature": shape_signature,
                    }
                    next_wall_id += 1

                if row_idx not in wall_registry[wall_key]["cells"]:
                    wall_registry[wall_key]["cells"].append(row_idx)

        # ------------------------------------------------------------------
        # organ-declared network bridges: detection
        #
        # For a path A -> B crossing tracheids T1..Tn (in that order along
        # the straight A-B line):
        #   plasmodesmata  A <-> V_T1 <-> V_T2 <-> ... <-> V_Tn <-> B
        #   membrane       V_T1 <-> wall(A, T1)   and   V_Tn <-> wall(Tn, B)
        # after cell boundaries have been walked into
        # wall_registry (so wall(A, T)/wall(T, B) lookups and "already
        # adjacent" pairs can be resolved) but *before* Phase 4 assigns node
        # indices off ``len(wall_registry)`` / ``cells_gdf.index``. It also
        # runs after ``CellGenerator._build_topology`` (already called
        # above), so nothing here can perturb that call's snap-tolerance or
        # junction clustering -- virtual cells never carry a polygon, so
        # there is no geometry left that could do so anyway.
        # ------------------------------------------------------------------

        bridge_rows: List[dict] = []
        # frozenset({node_a, node_b}) -> deduped, in cells_gdf-index space
        # (real row ids and/or virtual v_idx). Realised into graph edges in
        # Phase 9b, after Phase 5 has created every node.
        bridge_plasmodesmata_pairs: set = set()
        # (v_idx, wall_id) -> deduped. wall_id is an *existing* wall's own
        # id (from wall_registry), never a new one.
        bridge_membrane_pairs: set = set()
        # Diagnostics for the report: how many virtual nodes this would be
        # under each sharing strategy -- shared-per-tracheid (what is
        # actually built) vs. one-per-path (what it would be without the
        # sharing).
        n_bridge_paths = 0
        # Sum, over every accepted path, of the number of distinct
        # tracheids it traverses -- the virtual-node count under a
        # one-per-path (no sharing) strategy, for comparison against the
        # shared-per-tracheid count actually built.
        n_bridge_traversals = 0

        if bridges:
            bridge_specs = self.organ.network_bridge_specs() if hasattr(self.organ, "network_bridge_specs") else []

            if bridge_specs:
                # Pre-fill the "bridge" column for every *real* row before any
                # virtual row is appended.
                if "bridge" not in cells_gdf.columns:
                    cells_gdf["bridge"] = False

                # Pairs that already share a real wall get a plasmodesmata
                # edge for free once Phase 6 runs -- no bridge needed.
                adjacent_pairs = {
                    frozenset(wd["cells"]) for wd in wall_registry.values() if len(wd["cells"]) == 2
                }

                # Lookup: real cell pair (2-cell wall only) -> that wall's
                # own id. This is the ONLY thing a bridge is allowed to
                # attach a membrane edge to -- no new wall is ever created,
                # so ``next_wall_id`` never advances for a bridge.
                existing_wall_lookup: Dict[frozenset, int] = dict(
                    (frozenset(wd["cells"]), wd["id"])
                    for wd in wall_registry.values() if len(wd["cells"]) == 2
                )

                # New virtual-cell row indices must never collide with any
                # real cell's DataFrame index
                unfiltered_gdf = self.organ.generate_cells()
                if "bridge" in unfiltered_gdf.columns:
                    unfiltered_gdf = unfiltered_gdf[~unfiltered_gdf["bridge"].astype(bool)]
                next_bridge_id = int(unfiltered_gdf.index.max()) + 1 if len(unfiltered_gdf) else 0

                # Shared across every spec
                tracheid_to_vidx: Dict[Any, Any] = {}
                # v_idx -> the real cells it is *directly* plasmodesmata-
                # connected to (the A/B endpoint(s) of whichever chain(s) it
                # terminates) -- used only for the nominal area estimate.
                vidx_real_neighbours: Dict[Any, set] = {}

                for spec in bridge_specs:
                    radius = spec.get("radius") or 0.0
                    if radius <= 0:
                        continue
                    max_links = int(spec.get("max_links", 4))
                    bridge_type = spec.get("bridge_type", spec.get("name", "bridge"))
                    bridge_cgroup = CGROUP_MAP.get(bridge_type, 4)

                    # A virtual bridge cell (``bridge == True``) must never
                    # itself become a source/target/blocker for *another*
                    # bridge -- it has no real anatomical identity of its
                    # own beyond standing in for the out-of-plane cell that
                    # created it.
                    not_bridge = ~cells_gdf["bridge"].astype(bool)
                    source_rows = list(cells_gdf.index[cells_gdf["type"].isin(spec.get("source_types", ())) & not_bridge])
                    target_rows = list(cells_gdf.index[cells_gdf["type"].isin(spec.get("target_types", ())) & not_bridge])
                    blocker_rows = list(cells_gdf.index[cells_gdf["type"].isin(spec.get("blocker_types", ())) & not_bridge])

                    if not source_rows or not target_rows or not blocker_rows:
                        continue

                    target_geoms = [cells_gdf.loc[r, "geometry"] for r in target_rows]
                    target_tree = STRtree(target_geoms)

                    blocker_geoms = [cells_gdf.loc[r, "geometry"] for r in blocker_rows]
                    blocker_tree = STRtree(blocker_geoms)

                    # (gap, row_a, row_b, ordered_tracheid_rows) -- the
                    # candidate-pair detection and the
                    # covered_by(A U B U blockers) "separated by a tracheid"
                    # test are unchanged from before.
                    candidates = []
                    seen_pairs = set()

                    for row_a in source_rows:
                        geom_a = cells_gdf.loc[row_a, "geometry"]
                        if geom_a is None or geom_a.is_empty:
                            continue
                        centroid_a = geom_a.centroid
                        search_area = geom_a.buffer(radius)

                        for pos in target_tree.query(search_area):
                            row_b = target_rows[pos]
                            if row_b == row_a:
                                continue
                            pair = frozenset((row_a, row_b))
                            if pair in adjacent_pairs or pair in seen_pairs:
                                continue
                            geom_b = cells_gdf.loc[row_b, "geometry"]
                            if geom_b is None or geom_b.is_empty:
                                continue
                            gap = geom_a.distance(geom_b)
                            if gap <= 1e-9 or gap > radius:
                                continue

                            centroid_b = geom_b.centroid
                            segment_ab = LineString([centroid_a.coords[0], centroid_b.coords[0]])

                            # Every blocker (tracheid) whose footprint the
                            # A-B segment could plausibly cross.
                            crossing = [
                                (blocker_rows[bpos], blocker_geoms[bpos])
                                for bpos in blocker_tree.query(segment_ab)
                                if blocker_geoms[bpos].intersects(segment_ab)
                            ]
                            if not crossing:
                                continue
                            blockers_union = sp.ops.unary_union([g for _, g in crossing])

                            # "Separated by a blocker", not merely close: the
                            # straight centroid-to-centroid line must be
                            # entirely covered by A, B and the blocker(s).
                            if not segment_ab.covered_by(
                                sp.ops.unary_union([geom_a, geom_b, blockers_union])
                            ):
                                continue

                            # Order the distinct tracheids crossed along the
                            # segment, A-ward to B-ward, so a path that
                            # crosses more than one tracheid chains its
                            # virtual nodes in the right order.
                            def _along(item):
                                _, geom = item
                                inter = segment_ab.intersection(geom)
                                pt = inter.centroid if not inter.is_empty else geom.centroid
                                return segment_ab.project(pt)

                            ordered = sorted(crossing, key=_along)
                            ordered_rows = []
                            for r, _ in ordered:
                                if r not in ordered_rows:
                                    ordered_rows.append(r)

                            seen_pairs.add(pair)
                            candidates.append((gap, row_a, row_b, ordered_rows))

                    # accept shortest gaps first, at most
                    # ``max_links`` accepted paths per source AND per target
                    # cell -- keeps a parenchyma in a dense tracheid matrix
                    # from bridging to every neighbour within radius. This
                    # is independent of node *sharing*: several accepted
                    # paths that happen to cross the same tracheid still
                    # reuse that tracheid's one virtual node (see below).
                    candidates.sort(key=lambda c: c[0])
                    degree: Dict[Any, int] = {}

                    for gap, row_a, row_b, ordered_rows in candidates:
                        if degree.get(row_a, 0) >= max_links or degree.get(row_b, 0) >= max_links:
                            continue
                        degree[row_a] = degree.get(row_a, 0) + 1
                        degree[row_b] = degree.get(row_b, 0) + 1
                        n_bridge_paths += 1
                        n_bridge_traversals += len(ordered_rows)

                        v_chain = []
                        for t_row in ordered_rows:
                            v_idx = tracheid_to_vidx.get(t_row)
                            if v_idx is None:
                                v_idx = next_bridge_id
                                next_bridge_id += 1
                                tracheid_to_vidx[t_row] = v_idx
                                t_centroid = cells_gdf.loc[t_row, "geometry"].centroid
                                cells_gdf.loc[v_idx, "type"] = bridge_type
                                cells_gdf.loc[v_idx, "x"] = t_centroid.x
                                cells_gdf.loc[v_idx, "y"] = t_centroid.y
                                cells_gdf.loc[v_idx, "cgroup"] = bridge_cgroup
                                cells_gdf.loc[v_idx, "geometry"] = None
                                cells_gdf.loc[v_idx, "protect_topology"] = False
                                cells_gdf.loc[v_idx, "bridge"] = True
                                vidx_real_neighbours[v_idx] = set()
                            v_chain.append(v_idx)

                        # plasmodesmata chain: A - V_T1 - V_T2 - ... - V_Tn - B
                        full_chain = [row_a] + v_chain + [row_b]
                        for i in range(len(full_chain) - 1):
                            bridge_plasmodesmata_pairs.add(frozenset((full_chain[i], full_chain[i + 1])))

                        # membrane: only the two chain ENDS reuse a real,
                        # already-registered wall -- V_T1<->wall(A,T1) and
                        # V_Tn<->wall(Tn,B). No membrane edge is added for
                        # an interior tracheid-tracheid boundary.
                        first_v, first_t = v_chain[0], ordered_rows[0]
                        wall_id = existing_wall_lookup.get(frozenset((row_a, first_t)))
                        if wall_id is not None:
                            bridge_membrane_pairs.add((first_v, wall_id))
                        vidx_real_neighbours[first_v].add(row_a)

                        last_v, last_t = v_chain[-1], ordered_rows[-1]
                        wall_id2 = existing_wall_lookup.get(frozenset((last_t, row_b)))
                        if wall_id2 is not None:
                            bridge_membrane_pairs.add((last_v, wall_id2))
                        vidx_real_neighbours[last_v].add(row_b)

                # Nominal area for every virtual node (MECHA's capacitance/
                # volume terms need a number even without a polygon): the
                # mean of the equivalent radii of the real cells it is
                # directly plasmodesmata-connected to (falls back to the
                # tracheid's own real area for the rare interior-of-a-chain
                # node with no direct real neighbour).
                for t_row, v_idx in tracheid_to_vidx.items():
                    neigh = vidx_real_neighbours.get(v_idx, set())
                    radii = [
                        math.sqrt(cells_gdf.loc[r, "geometry"].area / math.pi)
                        for r in neigh
                        if cells_gdf.loc[r, "geometry"] is not None and cells_gdf.loc[r, "geometry"].area > 0
                    ]
                    if radii:
                        r_equiv = sum(radii) / len(radii)
                        area = math.pi * r_equiv ** 2
                    else:
                        area = float(cells_gdf.loc[t_row, "geometry"].area)
                    cells_gdf.loc[v_idx, "area"] = area

                    bridge_rows.append({
                        "id": v_idx, "type": cells_gdf.loc[v_idx, "type"],
                        "x": cells_gdf.loc[v_idx, "x"], "y": cells_gdf.loc[v_idx, "y"],
                        "cgroup": cells_gdf.loc[v_idx, "cgroup"], "geometry": None,
                        "area": area,
                        "diameter": 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0,
                        "tracheid": t_row,
                    })

                self.organ._bridge_report = {
                    "n_virtual_shared": len(tracheid_to_vidx),
                    "n_virtual_per_path": n_bridge_traversals,
                    "n_paths": n_bridge_paths,
                }

        # Compute true wall_thickness based on adjacent cells
        for wd in wall_registry.values():
            w_thick = 0.0
            for r in wd["cells"]:
                row = cells_gdf.loc[r]
                c_type = row.get("type", "")
                w_thick += get_thickness(c_type)
            
            if len(wd["cells"]) == 1:
                w_thick += get_thickness("outerwall")
                
            wd["wall_thickness"] = w_thick

        # Phase 4 — assign MECHA-compatible node indices
        network.n_walls = len(wall_registry)

        # Only keep junctions actually referenced by walls
        used_junctions: set = set()
        for wd in wall_registry.values():
            used_junctions.add(wd["junc_start"])
            used_junctions.add(wd["junc_end"])
        junction_list = sorted(used_junctions)
        junction_vk_to_id = {vk: i for i, vk in enumerate(junction_list)}

        network.n_junctions = len(junction_list)
        network.n_cells = len(cells_gdf)

        cell_row_to_node = {
            row_idx: network.n_walls + network.n_junctions + i
            for i, row_idx in enumerate(cells_gdf.index)
        }

        # Phase 5 — add nodes to graph
        # Wall nodes
        for wd in wall_registry.values():
            network.graph.add_node(
                wd["id"],
                indice=wd["id"],
                type="apo",
                position=wd["midpoint"],
                length=wd["length"],
                wall_thickness=wd["wall_thickness"],
            )

        # Junction nodes
        for vk in junction_list:
            node_id = network.n_walls + junction_vk_to_id[vk]
            network.graph.add_node(
                node_id,
                indice=node_id,
                type="apo",
                position=vk,
                length=0,
            )

        # Cell nodes
        for row_idx, row in cells_gdf.iterrows():
            node_id = cell_row_to_node[row_idx]
            centroid = row["geometry"].centroid if row["geometry"] is not None else None
            # A virtual (bridge) cell carries no polygon at all 
            # -- fall back to its own precomputed ``area`` column value
            # (the mean-of-real-neighbours estimate) instead of losing it to
            # None, which would otherwise zero out MECHA's capacitance/
            # volume terms for every bridge node.
            area = row["geometry"].area if row["geometry"] is not None else row.get("area")
            cx = centroid.x if centroid else row["x"]
            cy = centroid.y if centroid else row["y"]
            network.graph.add_node(
                node_id,
                indice=node_id,
                type="cell",
                cgroup=row.get("cgroup") or CGROUP_MAP.get(row.get("type", ""), 4),
                cell_type=row.get("type", ""),
                protect_topology=bool(row.get("protect_topology", False)),
                bridge=bool(row.get("bridge", False)),
                position=(cx, cy),
                area=area,
            )

        air_nodes = { cell_row_to_node[row_idx] for row_idx in protected_air_cells}   

        # Phase 6 — add edges
        network._wall_to_cells = {
            wd["id"]: [cell_row_to_node[r] for r in wd["cells"]]
            for wd in wall_registry.values()
        }

        for wd in wall_registry.values():
            wall_id = wd["id"]
            cell_nodes = network._wall_to_cells[wall_id]
            wall_length = wd["length"]
            wall_thickness = wd["wall_thickness"]
            pos_wall = wd["midpoint"]

            # Transmembrane: cell <-> wall
            #
            # NOTE: this is not a value-preserving bugfix. For every
            # 2-cell (interior) wall, of every organ, ``lateral_distance``/
            # ``distnode_wall_cell`` now differ numerically from the old
            # last-cell-wins behaviour (the mean of both flanking cells'
            # distances instead of whichever cell happened to be visited
            # second) -- and those values feed MECHA's apoplastic wall
            # conductance. Sanctioned by the plan and the mean is the
            # defensible representative value, but flagging it here so it
            # is never mistaken for a pure no-op bugfix later.
            dist_wall_cell_values = []
            for cn in cell_nodes:
                pos_cell = network.graph.nodes[cn]["position"]

                dist_wall_cell = np.hypot(
                    pos_wall[0] - pos_cell[0],
                    pos_wall[1] - pos_cell[1],
                )
                dist_wall_cell_values.append(dist_wall_cell)
                d_vec = np.array([
                    pos_wall[0] - pos_cell[0],
                    pos_wall[1] - pos_cell[1],
                ])

                # Identify edges connecting to protected air spaces.

                if (
                    cn in air_nodes
                    and bool(wd.get("shape_signature"))
                ):

                    network.graph.add_edge(
                        cn,
                        wall_id,
                        path="wall_air",
                        length=wall_length,
                        dist=dist_wall_cell,
                        d_vec=d_vec,
                        wall_thickness=wall_thickness,
                    )
                else:
                    network.graph.add_edge(
                        cn,
                        wall_id,
                        path="membrane",
                        length=wall_length,
                        dist=dist_wall_cell,
                        d_vec=d_vec,
                        wall_thickness=wall_thickness,
                    )

            mean_dist_wall_cell = (
                float(np.mean(dist_wall_cell_values)) if dist_wall_cell_values else 0.0
            )

            # each junction connected to the wall node
            for junc in ["junc_start", "junc_end"]:
                junc_id = network.n_walls + junction_vk_to_id[wd[junc]]
                pos_junc = network.graph.nodes[junc_id]["position"]
                dist_junc_wall_node = np.hypot(pos_junc[0] - pos_wall[0], pos_junc[1] - pos_wall[1])
                lateral_distance = mean_dist_wall_cell + dist_junc_wall_node
                d_vec = np.array([pos_junc[0] - pos_wall[0], pos_junc[1] - pos_wall[1]])

                # Apoplastic: wall <-> junction
                network.graph.add_edge(
                        junc_id,
                        wall_id,
                        path = 'wall',
                        length = wall_length / 2.0,
                        lateral_distance = lateral_distance,
                        d_vec = d_vec,
                        distnode_wall_cell = mean_dist_wall_cell,
                        wall_thickness=wall_thickness,
                )

            # Symplastic: cell <-> cell
            if len(cell_nodes) == 2:

                # only connect cells symplastically if they are not special air spaces
                # with protect_topology (mesophyll rhombic air spaces)

                cid_a, cid_b = wd["cells"]

                a_special = cid_a in protected_air_cells
                b_special = cid_b in protected_air_cells

                if a_special != b_special:
                    continue

                # Apoplastic-only cells (e.g. needle transfusion tracheids)
                # never get a symplastic edge, regardless of what flanks
                # them -- membrane/wall edges above are unaffected.
                if cid_a in apoplastic_cells or cid_b in apoplastic_cells:
                    continue

                pos_a = network.graph.nodes[cell_nodes[0]]["position"]
                pos_b = network.graph.nodes[cell_nodes[1]]["position"]
                dist = np.hypot(
                    pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
                )
                d_vec = np.array([pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]])
                network.graph.add_edge(
                    cell_nodes[0], cell_nodes[1],
                    path="plasmodesmata",
                    length=wall_length,
                    dist=dist,
                    d_vec=d_vec,
                )

        # ------------------------------------------------------------------
        # organ-declared network bridges: edges
        #
        # Realises the pairs 
        # ------------------------------------------------------------------

        for pair in bridge_plasmodesmata_pairs:
            row_x, row_y = tuple(pair)
            node_x = cell_row_to_node[row_x]
            node_y = cell_row_to_node[row_y]
            pos_x = network.graph.nodes[node_x]["position"]
            pos_y = network.graph.nodes[node_y]["position"]
            dist = np.hypot(pos_y[0] - pos_x[0], pos_y[1] - pos_x[1])
            d_vec = np.array([pos_y[0] - pos_x[0], pos_y[1] - pos_x[1]])
            network.graph.add_edge(
                node_x, node_y,
                path="plasmodesmata",
                length=dist,
                dist=dist,
                d_vec=d_vec,
            )

        for v_row, wall_id in bridge_membrane_pairs:
            v_node = cell_row_to_node[v_row]
            pos_v = network.graph.nodes[v_node]["position"]
            wall_node = network.graph.nodes[wall_id]
            pos_wall = wall_node["position"]
            dist = np.hypot(pos_wall[0] - pos_v[0], pos_wall[1] - pos_v[1])
            d_vec = np.array([pos_wall[0] - pos_v[0], pos_wall[1] - pos_v[1]])
            network.graph.add_edge(
                v_node, wall_id,
                path="membrane",
                length=wall_node["length"],
                dist=dist,
                d_vec=d_vec,
                wall_thickness=wall_node["wall_thickness"],
            )

        # ------------------------------------------------------------------
        # Phase 8 — construct direct air-space connectivity ("air_link")
        #
        # Motif:
        # air space -> new junction -> old junction -> new junction -> air space
        #
        # The "new junctions" are induced by protected topology handling
        # (mesophyll rhombic air spaces). These are identified through
        # wall segments carrying a non-empty shape_signature derived from
        # protected_shape_set.
        #
        # Existing topology generation remains untouched.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Step 1 — classify junctions
        # ------------------------------------------------------------------

        new_junctions = set()

        for wd in wall_registry.values():

            # shape_signature comes from protected topology handling
            if wd.get("shape_signature"):

                new_junctions.add(wd["junc_start"])
                new_junctions.add(wd["junc_end"])

        old_junctions = set(junction_list) - new_junctions

        # Convert junction vertex-keys -> graph node ids
        new_junction_nodes = {
            network.n_walls + junction_vk_to_id[vk]
            for vk in new_junctions
        }

        old_junction_nodes = {
            network.n_walls + junction_vk_to_id[vk]
            for vk in old_junctions
        }

        # ------------------------------------------------------------------
        # Step 2 — map each air space to its adjacent NEW junctions
        # ------------------------------------------------------------------

        air_to_new_junctions = {}

        for air_node in air_nodes:

            attached_new_junctions = set()

            # cell -> wall edges are membrane edges [1]
            for wall_node in network.graph.neighbors(air_node):

                if network.graph.nodes[wall_node].get("type") != "apo":
                    continue

                # wall -> junction edges are apoplastic wall edges [1]
                for junc_node in network.graph.neighbors(wall_node):

                    if junc_node in new_junction_nodes:
                        attached_new_junctions.add(junc_node)

            air_to_new_junctions[air_node] = attached_new_junctions

        # ------------------------------------------------------------------
        # Step 3 — construct air_link edges
        # ------------------------------------------------------------------

        for air_a in air_nodes:

            for air_b in air_nodes:

                if air_a >= air_b:
                    continue

                # avoid duplicate creation
                if network.graph.has_edge(air_a, air_b):
                    continue

                # NEW junctions attached to each air space
                juncs_a = air_to_new_junctions.get(air_a, set())
                juncs_b = air_to_new_junctions.get(air_b, set())

                if not juncs_a or not juncs_b:
                    continue

                # Search motif:
                # airA -> wall -> new junction J1 -> wall -> old junction Jmid -> wall -> new junction J2 -> wall -> airB

                valid_connection = False

                for j1 in juncs_a:

                    # NEW junction -> wall
                    for wall_1 in network.graph.neighbors(j1):

                        if network.graph.nodes[wall_1].get("type") != "apo":
                            continue

                        # wall -> OLD junction
                        for old_j in network.graph.neighbors(wall_1):

                            if old_j not in old_junction_nodes:
                                continue

                            # OLD junction -> wall
                            for wall_2 in network.graph.neighbors(old_j):

                                if wall_2 == wall_1:
                                    continue

                                if network.graph.nodes[wall_2].get("type") != "apo":
                                    continue

                                # wall -> NEW junction
                                for j2 in network.graph.neighbors(wall_2):

                                    if j2 == j1:
                                        continue

                                    if j2 not in juncs_b:
                                        continue

                                    valid_connection = True

                                    pos_a = network.graph.nodes[air_a]["position"]
                                    pos_b = network.graph.nodes[air_b]["position"]

                                    dist = np.hypot(
                                        pos_b[0] - pos_a[0],
                                        pos_b[1] - pos_a[1]
                                    )

                                    d_vec = np.array([
                                        pos_b[0] - pos_a[0],
                                        pos_b[1] - pos_a[1]
                                    ])

                                    network.graph.add_edge(
                                        air_a,
                                        air_b,
                                        path="air_link",
                                        length=dist,
                                        dist=dist,
                                        d_vec=d_vec,
                                        via_new_junction_a=j1,
                                        via_old_junction=old_j,
                                        via_new_junction_b=j2,
                                    )

                                    break

                                if valid_connection:
                                    break

                            if valid_connection:
                                break

                        if valid_connection:
                            break

                    if valid_connection:
                        break

        # ------------------------------------------------------------------
        # Step 4 — bridge air_link within a given radius
        #
        # Two protected air spaces (e.g. the sub-stomatal chamber and a
        # mesophyll rhombus) are not always adjacent and so never match the
        # junction motif above. Link every pair of protected air spaces whose
        # polygons lie within ``air_link_radius`` of each other (true
        # boundary-to-boundary gap, not centroid distance — elongated rhombi
        # can have a centroid well outside the radius while their nearest tip
        # is right next to the other air space). Defaults to 3x the median
        # wall length so the radius scales with the organ's own cell size.
        # ------------------------------------------------------------------

        radius = air_link_radius
        if radius is None:
            wall_lengths = [wd["length"] for wd in wall_registry.values() if wd["length"] > 0]
            radius = float(np.median(wall_lengths)) if wall_lengths else 0.0

        protected_air_rows = list(protected_air_cells)
        for i in range(len(protected_air_rows)):
            row_a = protected_air_rows[i]
            air_a = cell_row_to_node[row_a]
            geom_a = cells_gdf.loc[row_a, "geometry"]
            for j in range(i + 1, len(protected_air_rows)):
                row_b = protected_air_rows[j]
                air_b = cell_row_to_node[row_b]
                if network.graph.has_edge(air_a, air_b):
                    continue
                geom_b = cells_gdf.loc[row_b, "geometry"]
                if geom_a.distance(geom_b) > radius:
                    continue

                pos_a = network.graph.nodes[air_a]["position"]
                pos_b = network.graph.nodes[air_b]["position"]
                dist = np.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1])
                d_vec = np.array([pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]])
                network.graph.add_edge(
                    air_a, air_b,
                    path="air_link",
                    length=dist,
                    dist=dist,
                    d_vec=d_vec,
                )

        # ------------------------------------------------------------------
        # Keep ``organ.all_cells`` / ``organ._cells_gdf`` aligned with the
        # graph for MECHA (network_builder.py's ``populate_from_network``
        # walks ``src.all_cells.cells`` *positionally* against the cell-node
        # order established above -- a bridge cell that exists as a graph
        # node but not in ``all_cells`` would read back as an empty type and
        # zero area). Idempotent per organ instance (``_bridge_cells_added``)
        # so a second ``export()`` call -- e.g. a cached-graph
        # ``export_to_adjencymatrix()`` re-entry, or a fresh ``NetworkExporter``
        # on the same organ -- never appends the same cells twice.
        #
        # Deliberately NOT using ``CellManager.extend_cells``: that helper
        # renumbers ``id_cell``/``id_group`` to merge an independently-seeded
        # batch, which would break the identity we rely on here (bridge row
        # index == graph cell-node position == ``Cell.id_cell``).
        #
        # This only ever runs from inside ``export()`` -- a
        # ``generate_cells()``-only run (e.g. the golden regression tests)
        # never reaches this code and stays completely unaffected. Every
        # bridge row/cell is flagged ``bridge=True`` so a later census,
        # plot, or geometry export can filter them back out.
        # ------------------------------------------------------------------
        if bridge_rows and not getattr(self.organ, "_bridge_cells_added", False):
            new_cells = []
            for br in bridge_rows:
                cell = Cell(
                    x=br["x"], y=br["y"], diameter=br["diameter"],
                    type=br["type"], id_cell=br["id"], id_layer=-1, id_group=br["id"],
                    area=br["area"], polygon=br["geometry"], cgroup=br["cgroup"],
                )
                new_cells.append(cell)

            for cell in new_cells:
                self.organ.all_cells.add_cell(cell)
            self.organ._bridge_cells_added = True

            if self.organ._cells_gdf is not None:
                if "bridge" not in self.organ._cells_gdf.columns:
                    self.organ._cells_gdf["bridge"] = False
                extra_dicts = []
                for cell in new_cells:
                    d = cell.cell_to_dict()
                    d["geometry"] = cell.polygon
                    d["bridge"] = True
                    extra_dicts.append(d)
                extra_gdf = gpd.GeoDataFrame(extra_dicts, index=[c.id_cell for c in new_cells])
                self.organ._cells_gdf = pd.concat(
                    [self.organ._cells_gdf, extra_gdf], ignore_index=False
                )

