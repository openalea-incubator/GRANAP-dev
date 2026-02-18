import numpy as np
from shapely.geometry import Polygon, Point

class Cell:

    def __init__(self, x: float, y: float, diameter: float, width: float=0, height: float=0, 
                type: str="", id_cell: int=-1, id_layer: int=-1, id_group: int=-1,
                angle: float=None, radius: float=None, area: float=None, polygon: Polygon=None):

        self.x = x
        self.y = y
        self.point = Point(x, y)
        self.diameter = diameter
        self.width = width if width != 0 else diameter
        self.height = height if height != 0 else diameter
        self.type = type
        self.id_cell = id_cell
        self.id_layer = id_layer
        self.id_group = id_group
        self.angle = angle if angle != None else np.arctan2(y, x)
        self.radius = radius if radius != None else np.sqrt(x**2 + y**2)
        self.area = area if area != None else np.pi * (diameter/2)**2
        self.polygon = polygon if polygon != None else None
        

    def jitter(self, shift: float = 0.0001):
        """Jitter the cell position."""
        if shift != 0:
            self.x += np.random.uniform(-shift, shift)*self.diameter
            self.y += np.random.uniform(-shift, shift)*self.diameter
            self.angle = np.arctan2(self.y, self.x)
            self.radius = np.sqrt(self.x**2 + self.y**2)
            self.point = Point(self.x, self.y)

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
