import os
import re
import glob
from typing import List, Dict, Any, Optional
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import roifile

from granap.organ_class import Organ
from granap.cell_class import Cell
from granap.geometry_collection import GeometryProcessor

class RoiOrgan(Organ):
    """
    An Organ class generated from a folder of ImageJ ROI files.
    Bypasses procedural generation to directly populate the Cell network.
    """
    def __init__(self, folder_path: str, mm_per_pixel: float = 0.001, **kwargs):
        """
        Initialize the RoiOrgan anatomy structure.
        
        Args:
            folder_path: Path to the directory containing .roi files.
            mm_per_pixel: Scaling factor to convert ROI pixel coordinates to mm.
        """
        super().__init__(**kwargs)
        self.folder_path = folder_path
        self.mm_per_pixel = mm_per_pixel
        # Immediately parse the folder to set up cells
        self._load_rois()

    def _load_rois(self):
        """Parse all .roi files in the target folder and create Cell objects."""
        roi_files = glob.glob(os.path.join(self.folder_path, '*.roi'))
        
        if not roi_files:
            print(f"Warning: No .roi files found in {self.folder_path}")
            return

        new_cells = []
        cell_id_counter = 0

        for file_path in roi_files:
            basename = os.path.basename(file_path)
            stem = os.path.splitext(basename)[0]
            
            # Infer type: split by "_" and take first part. If it doesn't exist/empty, default to parenchyma
            parts = stem.split('_')
            cell_type = parts[0].strip()
            
            # Remove digits from cell_type in case they just named them "epidermis1.roi"
            cell_type_alpha = re.sub(r'[^a-zA-Z]', '', cell_type)
            if not cell_type_alpha:
                cell_type = "parenchyma"
            else:
                cell_type = cell_type_alpha

            # Load ROI coordinates
            roi = roifile.ImagejRoi.fromfile(file_path)
            coords = roi.coordinates()
            
            if coords is None or len(coords) < 3:
                continue

            # Scale coordinates and close the polygon loop
            scaled_coords = [(pt[0] * self.mm_per_pixel, pt[1] * self.mm_per_pixel) for pt in coords]
            scaled_coords.append(scaled_coords[0])  # Close the loop
            smooth_coords = GeometryProcessor.smoothing_polygon(scaled_coords, 0.5)

            poly = Polygon(smooth_coords)
            
            # Geometric properties for the Cell
            area = poly.area
            if area < 1e-5:
                continue
            diameter = 2 * np.sqrt(area / np.pi)
            centroid = poly.centroid
            x, y = centroid.x, centroid.y
            
            # Initialize the Cell
            cell = Cell(
                x=x,
                y=y,
                diameter=diameter,
                type=cell_type,
                id_cell=cell_id_counter,
                polygon=poly
            )
            new_cells.append(cell)
            cell_id_counter += 1

        self.all_cells.cells = new_cells

    def generate_cells(self) -> gpd.GeoDataFrame:
        """
        Override the default Voronoi/geometric generation of Granap.
        Returns a GeoDataFrame mapping the loaded ROI cells.
        """
        if self._cells_gdf is None:
            if not self.all_cells.cells:
                self._load_rois()
                
            cell_dicts = [c.cell_to_dict() for c in self.all_cells.cells]
            for i, c in enumerate(self.all_cells.cells):
                cell_dicts[i]['geometry'] = c.polygon
                
            self._cells_gdf = gpd.GeoDataFrame(cell_dicts) if cell_dicts else gpd.GeoDataFrame()
        
        return self._cells_gdf
        
    def _create_base_shape(self) -> Polygon:
        """Define the basic shape of the root/needle from loaded polygons."""
        # Provide the convex hull of all internal cells as the outer shape 
        # (useful for plotting or downstream references expecting a main boundary)
        if not self.all_cells.cells:
            return Polygon()
        
        polys = [c.polygon for c in self.all_cells.cells if c.polygon is not None]
        if not polys:
            return Polygon()
            
        union_geom = gpd.GeoSeries(polys).union_all()
        return union_geom.convex_hull

    # ---------- Stubs for purely abstract methods in Organ ----------
    
    def _which_layer_for_vascular(self, layers_polygons: List[Dict[str, Any]]):
        pass

    def _create_vascular_tissue(self, polygon: Polygon):
        pass

    def _organ_specific_tissues(self):
        pass

    def add_intercellular_spaces(self):
        pass

    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []

    def reshape_layers(self, layers_polygons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return layers_polygons
