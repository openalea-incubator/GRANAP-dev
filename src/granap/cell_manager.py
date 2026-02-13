
from typing import List, Optional
from granap.cell_class import Cell
from shapely.geometry import Polygon, Point
import numpy as np

class CellManager:
    def __init__(self):
        self.cells: List[Cell] = []

    def add_cell(self, cell: Cell):
        self.cells.append(cell)

    def get_cells(self):
        return self.cells

    def get_cell_by_id(self, id_cell: int):
        for cell in self.cells:
            if cell.id_cell == id_cell:
                return cell
        return None

    def extend_cells(self, cells: List[Cell]):

        # add vascular cells
        for cell in cells:
            max_id_layer = max([c.id_layer for c in self.cells])
            cell.id_layer = max_id_layer + cell.id_layer
            cell.id_cell = len(self.cells) + cell.id_cell
            cell.id_group = len(self.cells) + cell.id_cell
            self.cells.append(cell)

    def get_cells_by_type(self, type: str):
        return [cell for cell in self.cells if cell.type == type]

    def get_cells_by_layer(self, id_layer: int):
        return [cell for cell in self.cells if cell.id_layer == id_layer]

    def get_cells_by_group(self, id_group: int):
        return [cell for cell in self.cells if cell.id_group == id_group]

    def get_cells_by_polygon(self, polygon: Polygon):
        # Check if cell has polygon attribute and it is not None
        return [cell for cell in self.cells if cell.polygon is not None and cell.polygon.intersects(polygon)]
    
    def remove_cells_by_polygon(self, polygon: Polygon):
        if not self.cells:
            return

        # Check the first cell to decide strategy, assuming homogeneity
        # Or better, handle both cases robustly
        
        cells_to_keep = []
        for cell in self.cells:
            if cell.polygon is not None:
                if not cell.polygon.intersects(polygon):
                    cells_to_keep.append(cell)
            else:
                point = Point(cell.x, cell.y)
                if not point.intersects(polygon):
                    cells_to_keep.append(cell)
        
        self.cells = cells_to_keep
    
    def recalculate_cell_properties(self):
        """Recalculate the properties of all cells in the list."""
        for i, cell in enumerate(self.cells):
            cell.angle = np.arctan2(cell.y, cell.x)
            if cell.polygon is not None:
                cell.radius = cell.polygon.centroid.distance(Point(0, 0))
                cell.area = cell.polygon.area
            else:
                cell.radius = np.sqrt(cell.x**2 + cell.y**2)
                cell.area = cell.diameter**2 * np.pi / 4
            cell.id_cell = i

    def remove_cells_in_polygon(self, polygon: Polygon):
        # Filter cells that do not intersect the polygon
        # This creates a new list, avoiding modification during iteration
        self.cells = [cell for cell in self.cells if not cell.point.intersects(polygon)]
