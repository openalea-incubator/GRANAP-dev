"""
3D cell generation via literal-copy 2D extrusion.

Supersedes an earlier 3D-Voronoi approach (seed an ellipsoidal point cloud per
cell, tessellate all of them together in 3D, merge via ridge-facet classification)
that proved too fragile and expensive for what it bought: near-tangent
vessel/seed Voronoi degeneracies, visibly faceted walls needing a smoothing
pass, gap-welding across independently-built per-cell vertex arrays, 10+
minute builds, large files. See the project's root-3D-extension notes for the
full history.

This approach instead reuses the mature, fast 2D pipeline (``Organ.generate_cells``)
as the sole source of cell geometry: generate ONE 2D cross-section, then build
3D purely by stacking literal copies of each cell's real 2D polygon along Z at
its own tissue-type height. No 3D Voronoi, no border-point clouds, no
smoothing needed — extruded polygon prisms are watertight and flat-walled by
construction. Generic across organs (root/stem/leaf/needle): everything used
here (``rng``, ``generate_layer_polygons()``, ``all_cells``,
``intercellular_spaces_params``/``aerenchyma_params``) lives on the shared
``Organ`` base class.

Every cell type (including vessels) goes through the same extrusion loop —
the only thing that differs per type is its axial "row height":
  - Ordinary tissue (epidermis, cortex, endodermis, pericycle, stele, ...):
    its own axial_height (explicit, or DEFAULT_AXIAL_HEIGHT_RATIO * cell_diameter),
    repeated along Z — each repeat is a distinct output cell, with its OWN
    random Z-phase so neighbouring cells' row boundaries don't all land on the
    same plane (the "height shift inside the same tissue" that avoids a
    barcode-striped look without needing a new 2D tessellation per row).
  - Vessels (xylem/metaxylem/protoxylem/phloem/sieve elements): row height =
    the whole segment span, so each vessel gets exactly one extrusion covering
    the full height — its REAL 2D shape (not a synthetic circle), since the
    2D vascular recipe already merges each vessel's border points into one
    true polygon in Organ.all_cells.

"No tissue above/behind a vessel" falls out for free: every extruded row
reuses the SAME 2D cross-section, which already excludes vascular footprints
from surrounding tissue (Organ.generate_cells' existing vascular-mask step) —
no separate 3D masking code needed at all.

Ordinary intercellular ("air space") cells are excluded from the 2D
generation itself, not merely filtered afterward — see generate_cells_3d,
which clears intercellular_spaces_params before calling organ.generate_cells().
3D intercellular space is deliberately out of scope here; a later, separate
addition once solid-cell extrusion is validated.

Aerenchyma is deliberately left ENABLED and extruded like any other tissue.
Organ.add_aerenchyma() doesn't create new cells: it retypes existing real
tissue cells (their true 2D polygon, diameter, id_layer all untouched) to
type "air space" (organ_class.py's add_aerenchyma, ``cell.type = "air
space"``) to represent the schizogenous lacunae real aerenchyma actually is.
With ordinary intercellular generation disabled, "air space" can therefore
only mean an aerenchyma-converted cell here — there is nothing left to
distinguish it from, so it goes through the same extrusion path as every
other cell, at its own (original tissue's) axial_height.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from shapely.geometry import Polygon

from openalea.granap.mesh_utils_3d import write_obj, Mesh

# Vessel/sieve-element type tags across the recipes (root/stem/leaf, monocot +
# dicot): one extrusion spanning the whole segment, not repeated rows.
VESSEL_TYPES = {
    "xylem", "metaxylem", "protoxylem", "phloem", "sieve element", "sieve tube",
    "companion cell",
}

# Most non-vascular tissue is axially elongated in real anatomy, not
# isotropic — when a tissue's own axial_height isn't explicitly configured,
# default it to this multiple of that tissue's own cell_diameter.
DEFAULT_AXIAL_HEIGHT_RATIO = 5.0


@dataclass
class Cells3DResult:
    cells: List[dict]   # [{"type", "vertices", "faces"}]
    z_min: float
    z_max: float

    def export_obj(self, path: str) -> None:
        by_type: Dict[str, List[Mesh]] = {}
        for cell in self.cells:
            by_type.setdefault(cell["type"], []).append((cell["vertices"], cell["faces"]))
        write_obj(path, by_type)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cell in self.cells:
            counts[cell["type"]] = counts.get(cell["type"], 0) + 1
        return counts


def extrude_polygon(polygon: Polygon, z0: float, z1: float) -> Mesh:
    """A literal-copy prism: the polygon's exterior ring repeated at z0 and
    z1, connected by side quads, capped top and bottom by the polygon itself.

    Holes (interior rings) are not handled — out of scope for ordinary tissue
    cells at this stage. Face winding: bottom cap reversed / side quads / top
    cap in the ring's original order, so normals point outward (matches the
    convention the retired mesh_utils_3d.cylinder_mesh used to use).
    """
    coords = np.array(polygon.exterior.coords)
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]  # drop the closing duplicate vertex
    n = len(coords)
    bottom = np.column_stack((coords, np.full(n, z0)))
    top = np.column_stack((coords, np.full(n, z1)))
    vertices = np.vstack((bottom, top))

    faces: List[List[int]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])       # side quad
    faces.append(list(range(n - 1, -1, -1)))      # bottom cap, reversed (outward = -Z)
    faces.append(list(range(n, 2 * n)))           # top cap, original order (outward = +Z)

    return vertices, faces


def _resolve_axial_height(cell_diameter: float, layer_axial_height: Optional[float],
                          default_axial_height: Optional[float]) -> float:
    return (layer_axial_height or default_axial_height
            or cell_diameter * DEFAULT_AXIAL_HEIGHT_RATIO)


def generate_cells_3d(organ, n_axial_repeats: float = 8.0,
                      default_axial_height: Optional[float] = None,
                      seed: Optional[int] = None) -> Cells3DResult:
    """Build a 3D segment of ``organ`` by stacking literal copies of each 2D
    cell's polygon along Z at its own tissue-type height.

    Generates the 2D cross-section itself — with *ordinary* intercellular
    space disabled, aerenchyma left enabled (see module docstring) —
    discarding any 2D generation already cached on ``organ``
    (``organ._invalidate_geometry()``). Called via
    ``Organ.generate_cells_3d(...)``; see that method for the public API.
    """
    rng = np.random.default_rng(seed) if seed is not None else organ.rng

    # Ordinary intercellular space disabled at the source -- not generated
    # then filtered, never generated. Aerenchyma stays whatever organ was
    # already configured with (untouched here) -- it's real tissue cells
    # retyped to "air space" by Organ.add_aerenchyma, not a separate feature,
    # so there is nothing to disable/skip for it; it extrudes like any other
    # cell below.
    organ.intercellular_spaces_params = []
    organ._invalidate_geometry()
    organ.generate_cells()

    layers_polygons = organ.generate_layer_polygons()
    layer_axial_height = {i: lp.get("axial_height") for i, lp in enumerate(layers_polygons)}

    # Segment height: sized off the tallest ordinary tissue's own
    # axial_height (vessels don't count -- their "height" is the whole span
    # by definition, so including them would be circular).
    heights = [
        _resolve_axial_height(c.diameter, layer_axial_height.get(c.id_layer), default_axial_height)
        for c in organ.all_cells.cells
        if c.type not in VESSEL_TYPES
    ]
    base_height = max(heights) if heights else 0.02
    z_span = n_axial_repeats * base_height
    z_min, z_max = -z_span / 2, z_span / 2

    cells: List[dict] = []
    for cell in organ.all_cells.cells:
        if cell.polygon is None or cell.polygon.is_empty:
            continue
        polygon = cell.polygon
        if polygon.geom_type != "Polygon":
            polygon = max(polygon.geoms, key=lambda g: g.area)  # MultiPolygon -> largest piece

        is_vessel = cell.type in VESSEL_TYPES
        if is_vessel:
            # One extrusion covering the whole segment -- no phase, no repeats.
            vertices, faces = extrude_polygon(polygon, z_min, z_max)
            cells.append({"type": cell.type, "vertices": vertices, "faces": faces})
            continue

        height = _resolve_axial_height(cell.diameter, layer_axial_height.get(cell.id_layer), default_axial_height)
        phase = rng.uniform(0, height)  # per-cell Z-shift -- the "height shift inside the same tissue"
        z = z_min + phase - height
        while z < z_max:
            row_z0, row_z1 = max(z, z_min), min(z + height, z_max)
            if row_z1 > row_z0:
                vertices, faces = extrude_polygon(polygon, row_z0, row_z1)
                cells.append({"type": cell.type, "vertices": vertices, "faces": faces})
            z += height

    return Cells3DResult(cells=cells, z_min=z_min, z_max=z_max)
