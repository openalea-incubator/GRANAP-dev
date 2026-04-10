import os
import io
import math
import numpy as np
import shapely as sp
from typing import Dict, Any, Union, List
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import scale
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from granap.organ_class import Organ
from granap.network_base import AbstractNetwork
from granap.geometry_collection import GeometryProcessor


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
    "guard cell": 2,
    "Strasburger cell": 1,
    "outerwall": 2,
    "air space": 0.001,
    "pore": 0.001,
    "aerenchyma": 0.001,
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
        """
        from granap.generate_cell import CellGenerator
        
        cellgroups = {
            "exodermis": 1, "epidermis": 2, "endodermis": 3, "passage_cell": 3, "cortex": 4,
            "stele": 5, "xylem": 13, "pericycle": 16, "companion_cell": 12, "phloem": 11,
            "inter_cellular_space": 4, "aerenchyma": 4, "cambium": 11, "metaxylem": 13,
            "protoxylem": 13, "air space": 4, "stele": 5
        }

        valid_cells = [c for c in self.cells if c.polygon is not None]
        polys = [c.polygon for c in valid_cells]
        cell_ids = list(range(len(valid_cells)))
        
        cell_vkeys, _, _, junction_set = CellGenerator._build_topology(polys, cell_ids)

        wall_registry = {}
        next_wall_id = 0
        cell_walls = {i: [] for i in cell_ids}

        for row_idx, vkeys in cell_vkeys.items():
            n = len(vkeys)
            junc_positions = [i for i in range(n) if vkeys[i] in junction_set]

            if len(junc_positions) < 2:
                # no junctions -> single wall loop
                wall_key = tuple(sorted(vkeys))
                if wall_key not in wall_registry:
                    wall_registry[wall_key] = {"id": next_wall_id, "points": list(vkeys) + [vkeys[0]]}
                    next_wall_id += 1
                cell_walls[row_idx].append(wall_registry[wall_key]["id"])
                continue

            for jp in range(len(junc_positions)):
                start_idx = junc_positions[jp]
                end_idx = junc_positions[(jp + 1) % len(junc_positions)]

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
                junc_end = segment[-1]
                wall_key = tuple(sorted((junc_start, junc_end)))

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
            f'\t<cells count="{len(valid_cells)}">'
        ]

        for i, cell in enumerate(valid_cells):
            group_id = cellgroups.get(cell.type, 0)
            xml_lines.append(f'\t\t<cell id="{i}" group="{group_id}" truncated="false" >')
            xml_lines.append(f'\t\t\t<walls>')
            
            for wid in cell_walls[i]:
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

    def export(self, network: AbstractNetwork) -> None:
        """
        Populate the provided network graph from the cell GeoDataFrame.

        Algorithm
        ---------
        1. Delegate vertex snapping, vertex/edge maps, and junction
           detection to :meth:`CellGenerator._build_topology`.
        2. Walk each cell boundary between consecutive junctions to
           define **walls** (one wall per cell-pair interface).
        3. Assign MECHA-compatible node indices and build the graph.
        """
        from granap.generate_cell import CellGenerator
        cells_gdf = self.organ.generate_cells()

        # Phases 0–2 — snapping, topology maps, junction detection
        polys    = list(cells_gdf["geometry"])
        cell_ids = list(cells_gdf.index)

        cell_vkeys, _, edge_to_cells, junction_set = (
            CellGenerator._build_topology(polys, cell_ids)
        )

        if not cell_vkeys:
            return

        # Phase 3 — walk cell boundaries to define walls
        # A "wall" = the polyline segment between two consecutive
        # junction vertices along one cell boundary.  Two cells that
        # share the same (juncA, juncB) segment share a wall.
        wall_registry: Dict[tuple, dict] = {}  # wall_key → wall info
        next_wall_id = 0

        for row_idx, vkeys in cell_vkeys.items():
            n = len(vkeys)
            junc_positions = [i for i in range(n) if vkeys[i] in junction_set]

            if len(junc_positions) < 2:
                # Fewer than 2 junctions → treat entire boundary as one wall
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
                wall_key = tuple(sorted((junc_start, junc_end)))

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
                        "cells": [],
                    }
                    next_wall_id += 1

                if row_idx not in wall_registry[wall_key]["cells"]:
                    wall_registry[wall_key]["cells"].append(row_idx)

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
            area = row["geometry"].area if row["geometry"] is not None else None
            cx = centroid.x if centroid else row["x"]
            cy = centroid.y if centroid else row["y"]
            network.graph.add_node(
                node_id,
                indice=node_id,
                type="cell",
                cgroup=row.get("cgroup", ""),
                cell_type=row.get("type", ""),
                position=(cx, cy),
                area=area,
            )

        # Phase 6 — add edges
        network._wall_to_cells = {
            wd["id"]: [cell_row_to_node[r] for r in wd["cells"]]
            for wd in wall_registry.values()
        }

        for wd in wall_registry.values():
            wall_id = wd["id"]
            cell_nodes = network._wall_to_cells[wall_id]
            wall_length = wd["length"]

            # Transmembrane: cell ↔ wall
            for cn in cell_nodes:
                pos_cell = network.graph.nodes[cn]["position"]
                pos_wall = wd["midpoint"]
                dist_wall_cell = np.hypot(
                    pos_wall[0] - pos_cell[0],
                    pos_wall[1] - pos_cell[1],
                )
                d_vec = np.array([pos_wall[0] - pos_cell[0], pos_wall[1] - pos_cell[1]])
                network.graph.add_edge(
                    cn, wall_id,
                    path="membrane",
                    length=wall_length,
                    dist=dist_wall_cell,
                    d_vec=d_vec,
                )
            
            # each junction connected to the wall node
            for junc in ["junc_start", "junc_end"]:
                junc_id = network.n_walls + junction_vk_to_id[wd[junc]]
                pos_junc = network.graph.nodes[junc_id]["position"]
                dist_junc_wall_node = np.hypot(pos_junc[0] - pos_wall[0], pos_junc[1] - pos_wall[1])
                lateral_distance = dist_wall_cell + dist_junc_wall_node
                d_vec = np.array([pos_junc[0] - pos_wall[0], pos_junc[1] - pos_wall[1]])
                
                # Apoplastic: wall ↔ junction
                network.graph.add_edge(
                        junc_id,
                        wall_id,
                        path = 'wall',
                        length = wall_length / 2.0,
                        lateral_distance = lateral_distance,
                        d_vec = d_vec,
                        distnode_wall_cell = dist_wall_cell,
                )
            
            # Symplastic: cell ↔ cell
            if len(cell_nodes) == 2:
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

