import copy
from typing import List, Optional
from openalea.granap.cell_class import Cell
from shapely.geometry import Polygon, Point, MultiPoint
from scipy.spatial import Delaunay
from shapely.affinity import translate
import numpy as np
import shapely

class CellManager:
    def __init__(self):
        self._cells: List[Cell] = []
        # Cache for get_last_id_group: the running max id_group and the list length
        # it was computed at.  next_group_id() is called once per seeded Voronoi
        # group all through vascular building, so scanning every cell each time was
        # O(n^2).  The cache is O(1) on the append-only build path and self-heals
        # (recomputes) whenever the list is mutated behind our back (a direct
        # ``cells = [...]`` / ``cells.append`` changes the length).
        self._max_gid: int = 0
        self._gid_len: int = 0

    @property
    def cells(self) -> List[Cell]:
        return self._cells

    @cells.setter
    def cells(self, value: List[Cell]):
        self._cells = value
        self._max_gid = max((c.id_group for c in value), default=0)
        self._gid_len = len(value)

    def add_cell(self, cell: Cell):
        self._cells.append(cell)
        if cell.id_group > self._max_gid:
            self._max_gid = cell.id_group
        self._gid_len = len(self._cells)

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
        if not self.cells:
            self.cells.extend(copy.copy(c) for c in cells)
            return

        max_id_cell  = max(c.id_cell  for c in self.cells)
        max_id_group = max(c.id_group for c in self.cells)

        for cell in cells:
            new_cell = copy.copy(cell)
            # id_layer is an absolute layer index — preserve it as-is
            new_cell.id_cell  = max_id_cell  + cell.id_cell  + 1
            new_cell.id_group = max_id_group + cell.id_group + 1
            self.cells.append(new_cell)

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
        if not group_cells:
            raise KeyError(f"No cells found for id_group={id_group}")
        return np.mean([c.x for c in group_cells]), np.mean([c.y for c in group_cells])

    def get_polygons(self):
        return [cell.polygon for cell in self.cells if cell.polygon is not None]
    
    def remove_cells_by_polygon(self, polygon: Polygon):
        """Drop every cell that intersects ``polygon``.

        A cell's footprint is its Voronoi polygon if it has one, else its seed
        point.  Both predicates are evaluated in bulk (vectorised shapely) rather
        than one shapely call per cell — same result, far fewer Python-level
        shapely dispatches.
        """
        if not self.cells:
            return

        remove = np.zeros(len(self.cells), dtype=bool)

        poly_idx  = [i for i, c in enumerate(self.cells) if c.polygon is not None]
        point_idx = [i for i, c in enumerate(self.cells) if c.polygon is None]

        if poly_idx:
            polys = np.array([self.cells[i].polygon for i in poly_idx], dtype=object)
            remove[poly_idx] = shapely.intersects(polys, polygon)
        if point_idx:
            xs = np.array([self.cells[i].x for i in point_idx], dtype=float)
            ys = np.array([self.cells[i].y for i in point_idx], dtype=float)
            remove[point_idx] = shapely.intersects_xy(polygon, xs, ys)

        self.cells = [c for c, drop in zip(self.cells, remove) if not drop]
    
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
        """Drop every cell whose seed point intersects ``polygon`` (bulk predicate)."""
        if not self.cells:
            return
        xs = np.fromiter((c.x for c in self.cells), dtype=float, count=len(self.cells))
        ys = np.fromiter((c.y for c in self.cells), dtype=float, count=len(self.cells))
        remove = shapely.intersects_xy(polygon, xs, ys)
        self.cells = [c for c, drop in zip(self.cells, remove) if not drop]

    def remove_cells_by_ids(self, ids: []):
        # filter cells
        self.cells = [cell for cell in self.cells if not cell.id_cell in ids]

    def remove_cells(self, cells):
        """Remove exactly these cell objects, matched by identity.

        Unlike :meth:`remove_cells_by_ids`, this is safe when ``id_cell`` is not
        unique across the list (e.g. air-space cells seeded with ``id_cell =
        len(cells) + k`` can collide with a contiguous block of earlier layer-seed
        ids): removing by value would then delete unrelated cells that happen to
        share an id.  Matching on ``id(cell)`` deletes only the intended objects.
        """
        drop = {id(c) for c in cells}
        self.cells = [cell for cell in self.cells if id(cell) not in drop]

    def get_last_id_group(self):
        # O(1) on the append-only build path; recompute only if the list was
        # mutated outside add_cell / the cells setter (detected by a length change).
        if len(self._cells) != self._gid_len:
            self._max_gid = max((c.id_group for c in self._cells), default=0)
            self._gid_len = len(self._cells)
        return self._max_gid

    def next_group_id(self):
        """Next free ``id_group`` for appending a new cell group.

        ``get_last_id_group() + 1`` when there are cells, else 0 — the idiom
        used everywhere vascular tissue is seeded group by group.
        """
        return self.get_last_id_group() + 1 if self.cells else 0

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
