import os
import io
import math
import numpy as np
import shapely as sp
from typing import Dict, Any, Union, List
from shapely.geometry import Polygon, MultiPolygon, Point

from granap.organ_class import Organ
from granap.geometry_collection import GeometryProcessor


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
            "protoxylem": 13, "air space": 4, "vascular_parenchyma": 5
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

    def prep_geo(self, cell_wall_thickness: Union[float, Dict[str, float]] = 0.5, 
                 corner_smoothing: Union[float, Dict[str, float]] = 5):
        """
        Pre-proc for .geo file generation.
        Returns list of shrunken inner luminal polygons and one full tissue outer boundary.
        Uses GeometryProcessor for buffering and smoothing.
        Geometry is scaled by 1000 (microns).
        """
        import shapely.affinity
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

        for cell in self.cells:
            if cell.polygon is None:
                continue
            
            poly = cell.polygon
            if not poly.is_valid:
                poly = poly.buffer(0)
                
            # Scale coordinates by 1000 to match GMSH expected micron scale
            r_poly = shapely.affinity.scale(poly, xfact=1000, yfact=1000, origin=(0, 0))
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
                     cell_wall_thickness: Union[float, Dict[str, float]] = 1, 
                     corner_smoothing: Union[float, Dict[str, float]] = 0.5):
        """
        Write .geo file for GMSH.
        Calls prep_geo to compute cell lumina and the outer boundary.
        """
        inner_polygons, final_polygon = self.prep_geo(cell_wall_thickness, corner_smoothing)
        
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
