
from typing import List, Optional
from granap.cell_class import Cell
from shapely.geometry import Polygon, Point, MultiPoint
from scipy.spatial import Delaunay
from shapely.affinity import translate
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

    def get_cells_by_ids(self, ids: List[int]):
        return [cell for cell in self.cells if cell.id_cell in ids]

    def extend_cells(self, cells: List[Cell]):

        max_id_layer = max([c.id_layer for c in self.cells])
        max_id_cell = max([c.id_cell for c in self.cells])
        max_id_group = max([c.id_group for c in self.cells])

        # add a list of cells to the current list
        for cell in cells:
            cell.id_layer = max_id_layer + cell.id_layer +1
            cell.id_cell = max_id_cell + cell.id_cell+1 
            cell.id_group = max_id_group + cell.id_group+1
            self.cells.append(cell)

    def get_cells_by_type(self, type: str):
        return [cell for cell in self.cells if cell.type == type]
    
    def get_all_types(self):
        # return a list of unique cell type for all cells
        return list(set([cell.type for cell in self.cells]))

    def get_cells_by_layer(self, id_layer: int):
        return [cell for cell in self.cells if cell.id_layer == id_layer]

    def get_cells_by_group(self, id_group: int):
        return [cell for cell in self.cells if cell.id_group == id_group]

    def get_cells_by_groups(self, id_groups: List[int]):
        return [cell for cell in self.cells if cell.id_group in id_groups]

    def get_cells_by_polygon(self, polygon: Polygon):
        # Check if cell has polygon attribute and it is not None
        return [cell for cell in self.cells if cell.polygon is not None and cell.polygon.intersects(polygon)]

    def get_centroid_of_group(self, id_group: int):
        group_cells = self.get_cells_by_group(id_group)
        cx = np.mean([cell.x for cell in group_cells])
        cy = np.mean([cell.y for cell in group_cells])
        return cx, cy

    def get_polygons(self):
        return [cell.polygon for cell in self.cells if cell.polygon is not None]
    
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

    def remove_cells_by_ids(self, ids: []):
        # filter cells
        self.cells = [cell for cell in self.cells if not cell.id_cell in ids]
    
    def get_last_id_group(self):
        return max([cell.id_group for cell in self.cells])

    def recenter_cells(self):
        # re position cells to the center of the global cell population
        x_center = np.mean([c.x for c in self.cells])
        y_center = np.mean([c.y for c in self.cells])
        for cell in self.cells:
            cell.x = cell.x - x_center
            cell.y = cell.y - y_center
            cell.angle = np.arctan2(cell.y, cell.x)
            if cell.polygon is not None:
                cell.polygon = translate(cell.polygon, xoff = -x_center, yoff = -y_center)

    def plot_cells(self, ax = None):
        # plot cells coordinates
        # color code by type
        # legend
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if ax is None:
            fig, ax = plt.subplots()
        
        types = self.get_all_types()
        # Create a color map
        colors = cm.viridis(np.linspace(0, 1, len(types)))
        type_to_color = dict(zip(types, colors))

        # Plot cells
        for cell in self.cells:
            if cell.type in type_to_color:
                c = type_to_color[cell.type]
                ax.plot(cell.x, cell.y, marker='o', linestyle='', c=c, label=cell.type)
        
        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())
        
        ax.set_aspect('equal', adjustable='box')
        plt.show()

