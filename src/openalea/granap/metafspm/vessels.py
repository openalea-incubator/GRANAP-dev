"""
Vessel and pole ids of a generated section: GRANAP keeps no grouping on its cells,
so the ids are recovered from what the generation leaves on the organ.

    vessel_id   every conducting xylem and phloem cell (CONDUCTING types), numbered in cell order; -1 otherwise
    pole_id     its bundle or pole: the bundle envelope containing it (stems, leaves), the phloem-pole polygon
                containing it (dicot root phloem), else the angular sector of the n_vascular_bundles poles (root
                xylem, monocot phloem); -1 for non-conducting cells
    family      0 xylem, 1 phloem, -1 neither: vessels are only matched within a family between sections
"""
import numpy as np
from shapely import STRtree, points

XYLEM_TYPES = ("metaxylem", "protoxylem", "xylem")
PHLOEM_TYPES = ("phloem", "sieve element")
CONDUCTING = XYLEM_TYPES + PHLOEM_TYPES


def family_of(cell_type) -> int:
    return 0 if cell_type in XYLEM_TYPES else 1 if cell_type in PHLOEM_TYPES else -1


def section_centre(cells) -> np.ndarray:
    """The area-weighted mean of the cell centroids (the organ's axis)."""
    real = [c for c in cells if c.polygon is not None and not c.polygon.is_empty]
    areas = np.array([c.polygon.area for c in real])
    centres = np.array([[c.polygon.centroid.x, c.polygon.centroid.y] for c in real])
    return (centres * areas[:, None]).sum(axis=0) / areas.sum()


def _containing(polygons, xy) -> np.ndarray:
    """Index of the polygon containing each point (the nearest one when none does)."""
    tree = STRtree(polygons)
    pts = points(xy)
    found = np.full(len(xy), -1, dtype=np.int64)
    inside, which = tree.query(pts, predicate="within")
    found[inside] = which
    missing = np.flatnonzero(found < 0)
    if missing.size:
        found[missing] = tree.nearest(pts[missing])
    return found


def _n_poles(organ) -> int:
    xylem = organ._get_param("xylem") if hasattr(organ, "_get_param") else {}
    return int(xylem.get("n_vascular_bundles") or xylem.get("n_vascular_peak") or 1)


def vessel_ids(organ) -> dict:
    """{"vessel_id", "pole_id", "family", "angle", "relative_radius"}: arrays over organ.all_cells, in cell order."""
    cells = organ.all_cells.cells
    n = len(cells)
    family = np.array([family_of(c.type) for c in cells], dtype=np.int64)
    conducting = family >= 0
    vessel = np.full(n, -1, dtype=np.int64)
    vessel[conducting] = np.arange(int(conducting.sum()))
    centre = section_centre(cells)
    xy = np.array([[c.polygon.centroid.x, c.polygon.centroid.y] if c.polygon is not None else [c.x, c.y]
                   for c in cells]) - centre
    angle = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2 * np.pi)
    radius = np.hypot(xy[:, 0], xy[:, 1])
    section_radius = radius.max() if n else 1.
    pole = np.full(n, -1, dtype=np.int64)
    zones = getattr(organ, "vascular_tissue_polygons", {}) or {}
    rows = np.flatnonzero(conducting)
    if rows.size:
        if zones.get("bundle"):
            pole[rows] = _containing(zones["bundle"], xy[rows] + centre)
        else:
            sectors = _n_poles(organ)
            pole[rows] = np.floor(angle[rows] / (2 * np.pi / sectors)).astype(np.int64) % sectors
            phloem_zones = zones.get("phloem")
            phloem_rows = rows[family[rows] == 1]
            if phloem_zones and phloem_rows.size:
                pole[phloem_rows] = _containing(phloem_zones, xy[phloem_rows] + centre)
    return {"vessel_id": vessel, "pole_id": pole, "family": family, "angle": angle,
            "relative_radius": radius / section_radius}
