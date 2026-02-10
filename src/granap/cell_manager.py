class CellManager:
    def __init__(self):
        self.cells = []

    def add_cell(self, cell: Cell):
        self.cells.append(cell)

    def get_cells(self):
        return self.cells

    def get_cell_by_id(self, id_cell: int):
        for cell in self.cells:
            if cell.id_cell == id_cell:
                return cell
        return None

    def get_cells_by_type(self, type: str):
        return [cell for cell in self.cells if cell.type == type]

    def get_cells_by_layer(self, id_layer: int):
        return [cell for cell in self.cells if cell.id_layer == id_layer]

    def get_cells_by_group(self, id_group: int):
        return [cell for cell in self.cells if cell.id_group == id_group]

    def get_cells_by_polygon(self, polygon: Polygon):
        return [cell for cell in self.cells if cell.polygon.intersects(polygon)]
    
    def remove_cells_by_polygon(self, polygon: Polygon):
        # if self.cells have polygon
        if self.cells[0].polygon is not None:
            self.cells = [cell for cell in self.cells if not cell.polygon.intersects(polygon)]
        else:
            for cell in self.cells:
                point = Point(cell.x, cell.y)
                if point.intersects(polygon):
                    self.cells.remove(cell)
    
    def recalculate_cell_properties(self):
        """Recalculate the properties of all cells in the list."""
        for i, cell in enumerate(self.cells):
            cell.angle = np.arctan2(cell.y, cell.x)
            cell.radius = cell.polygon.centroid.distance(Point(0, 0))
            cell.area = cell.polygon.area
            cell.id_cell = i
