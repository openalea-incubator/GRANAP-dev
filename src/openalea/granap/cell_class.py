import numpy as np
from shapely.geometry import Polygon, Point
from openalea.granap.geometry_collection import GeometryProcessor

class Cell:

    def __init__(self, x: float, y: float, diameter: float, width: float=0, height: float=0,
                type: str="", id_cell: int=-1, id_layer: int=-1, id_group: int=-1,
                angle: float=None, radius: float=None, area: float=None, polygon: Polygon=None, axis: float=None,
                protect_topology: bool = False, protect_shape: bool = False, track_id=None,
                z: float = 0.0, axial_height: float = None):

        self.x = x
        self.y = y
        self.diameter = diameter
        self.width = width if width != 0 else diameter
        self.height = height if height != 0 else diameter
        # Out-of-plane (longitudinal/axial, Z) extent — see ROOT_3D_PLAN. Distinct
        # from `height` above, which is an in-plane axis kept for 2D back-compat.
        # None means "no 3D axial subdivision configured" (2D behaviour).
        self.z = z
        self.axial_height = axial_height
        self.type = type
        self.id_cell = id_cell
        self.id_layer = id_layer
        self.id_group = id_group
        # Persistent developmental-series id (see ROOT_SERIES_PLAN): a tracked xylem
        # vessel keeps the same track_id across the apex->collet series; None for
        # everything untracked.  Must survive Voronoi grouping and reach the gdf.
        self.track_id = track_id
        self.angle = angle if angle != None else np.arctan2(y, x)
        self.radius = radius if radius != None else np.sqrt(x**2 + y**2)
        self.area = area if area != None else np.pi * (diameter/2)**2
        self.polygon = polygon if polygon != None else None
        self.axis = axis if axis != None else None
        self.protect_topology = protect_topology
        self.protect_shape = protect_shape

    @classmethod
    def radial(cls, type: str, x: float, y: float, diameter: float,
               id_group: int, center, *, id_cell: int = None, id_layer: int = -1,
               track_id=None, z: float = 0.0, axial_height: float = None) -> "Cell":
        """Build a seed cell whose ``angle``/``radius`` are measured from ``center``.

        This is the ubiquitous seeding idiom — a cell placed at ``(x, y)`` with its
        polar attributes relative to a tissue centre ``center`` (a shapely
        ``Point`` or an ``(cx, cy)`` tuple).  ``area`` defaults to a disc of
        ``diameter`` (as everywhere it is used); ``id_cell`` defaults to
        ``id_group`` (one seed per Voronoi group) unless an explicit running id is
        given.  ``track_id`` tags a persistent tracked vessel (default None).
        ``z``/``axial_height`` are the out-of-plane counterparts, unused until a
        3D pipeline places seeds along the organ's longitudinal axis.
        """
        cx, cy = (center.x, center.y) if hasattr(center, "x") else center
        return cls(
            type=type, x=x, y=y, diameter=diameter,
            id_cell=id_group if id_cell is None else id_cell,
            id_layer=id_layer, id_group=id_group, track_id=track_id,
            angle=np.arctan2(y - cy, x - cx),
            radius=np.sqrt((x - cx) ** 2 + (y - cy) ** 2),
            z=z, axial_height=axial_height,
        )

    @property
    def point(self):
        return Point(self.x, self.y)

    def jitter(self, shift: float = 0.0001, rng=None):
        """Jitter the cell position.

        Uses ``rng`` (the organ's seeded generator) when supplied so the
        perturbation is reproducible; falls back to the global ``np.random``.
        """
        if shift != 0:
            _rng = rng if rng is not None else np.random
            self.x += _rng.uniform(-shift, shift)*self.diameter
            self.y += _rng.uniform(-shift, shift)*self.diameter
            self.angle = np.arctan2(self.y, self.x)
            self.radius = np.sqrt(self.x**2 + self.y**2)

    def cell_to_dict(self):
        return {"type": self.type, "x": self.x, "y": self.y, "z": self.z,
                "cell_diameter": self.diameter,
                "cell_width": self.width,
                "cell_height": self.height,
                "axial_height": self.axial_height,
                "id_cell": self.id_cell,
                "id_layer": self.id_layer,
                "id_group": self.id_group,
                "track_id": self.track_id,
                "angle": self.angle,
                "radius": self.radius,
                "area": self.area,
                "protect_topology": self.protect_topology,
                "protect_shape": self.protect_shape,
                }
    
    def smooth(self, smooth_factor: float = 0.01):
        """Smooth the cell polygon."""
        self.polygon = GeometryProcessor.buffer_polygon(self.polygon, 0, smooth_factor=smooth_factor)

