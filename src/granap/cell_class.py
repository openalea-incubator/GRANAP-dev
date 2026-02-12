import numpy as np
from shapely.geometry import Polygon

class Cell:
    def __init__(self, x: float, y: float, diameter: float, width: float=0, height: float=0, 
                type: str="", id_cell: int=-1, id_layer: int=-1, id_group: int=-1,
                angle: float=None, radius: float=None, area: float=None, polygon: Polygon=None):
        self.x = x # cell center x-coordinate
        self.y = y # cell center y-coordinate
        self.diameter = diameter # cell diameter
        self.width = width if width != 0 else diameter # cell width
        self.height = height if height != 0 else diameter # cell height
        self.type = type # cell type
        self.id_cell = id_cell # cell id
        self.id_layer = id_layer # layer id
        self.id_group = id_group # group id
        self.angle = angle if angle != None else np.arctan2(y, x) # angle of the cell center from the center of the organ
        self.radius = radius if radius != None else np.sqrt(x**2 + y**2) # distance of the cell center from the center of the organ
        self.area = area if area != None else np.pi * (diameter/2)**2 # approximate area of the cell
        self.polygon = polygon if polygon != None else None # polygon of the cell

    def jitter(self, shift: float = 0.001):
        """Jitter the cell position."""
        self.x += np.random.uniform(-shift, shift)
        self.y += np.random.uniform(-shift, shift)
        self.angle = np.arctan2(self.y, self.x)
        self.radius = np.sqrt(self.x**2 + self.y**2)

    def cell_to_dict(self):
        return {"type": self.type, "x": self.x, "y": self.y, 
                "cell_diameter": self.diameter,
                "cell_width": self.width,
                "cell_height": self.height,
                "id_cell": self.id_cell,
                "id_layer": self.id_layer,
                "id_group": self.id_group,
                "angle": self.angle,
                "radius": self.radius,
                "area": self.area,
                }
