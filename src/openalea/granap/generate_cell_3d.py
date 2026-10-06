"""
3D cell generation via literal-copy 2D extrusion.

Supersedes an earlier 3D-Voronoi approach (seed an ellipsoidal point cloud per
cell, tessellate all of them together in 3D, merge via ridge-facet classification)
that proved too fragile and expensive for what it bought: near-tangent
vessel/seed Voronoi degeneracies, visibly faceted walls needing a smoothing
pass, gap-welding across independently-built per-cell vertex arrays, 10+
minute builds, large files.

This approach instead reuses the mature, fast 2D pipeline (``Organ.generate_cells``)
as the sole source of cell geometry: generate ONE 2D cross-section, then build
3D purely by stacking literal copies of each cell's real 2D polygon along Z at
its own tissue-type height. No 3D Voronoi, no border-point clouds, no
smoothing needed — extruded polygon prisms are watertight and flat-walled by
construction. Generic across organs (root/stem/leaf/needle): everything used
here (``rng``, ``generate_layer_polygons()``, ``all_cells``,
``intercellular_spaces_params``, ``aerenchyma_params``) lives on the shared
``Organ`` base class.

Axial row height
----------------
Every cell type goes through the same extrusion loop — the only thing that
differs per type is its axial "row height":

  - Ordinary tissue (epidermis, cortex, endodermis, pericycle, stele, ...):
    its own axial_height (explicit, or DEFAULT_AXIAL_HEIGHT_RATIO * cell_diameter),
    repeated along Z — each repeat is a distinct output cell, with its OWN
    random Z-phase so neighbouring cells' row boundaries don't all land on the
    same plane (the "height shift inside the same tissue" that avoids a
    barcode-striped look without needing a new 2D tessellation per row).
  - Axially continuous structures (vessels, sieve elements, and — in
    ``aerenchyma_mode="before"`` — aerenchyma lacunae): row height = the whole
    segment span, so each gets exactly one extrusion covering the full height,
    using its REAL 2D shape (not a synthetic circle), since the 2D vascular
    recipe already merges each vessel's border points into one true polygon in
    ``Organ.all_cells``.

"No tissue above/behind a vessel" falls out for free: every extruded row
reuses the SAME 2D cross-section, which already excludes vascular footprints
from surrounding tissue (``Organ.generate_cells``' existing vascular-mask
step) — no separate 3D masking code needed at all.

Air space: ordinary intercellular vs aerenchyma
-----------------------------------------------
These are two different things and are handled differently.

*Ordinary* intercellular space (``Organ.add_intercellular``, driven by
``intercellular_spaces_params``) is always disabled for the 3D build — at the
source, not generated then filtered. Small interstitial 2D wedges do not
extrude into anything meaningful, and true 3D intercellular space is a
separate, later piece of work.

*Aerenchyma* is not a separate feature but a retyping of real tissue cells:
``Organ.add_aerenchyma`` sets ``cell.type = "air space"`` on existing cortical
(or other) cells, leaving their polygon, diameter and id_layer untouched, to
represent the schizogenous lacunae real aerenchyma is. Because the lacuna's
3D form differs by organ, ``aerenchyma_mode`` selects when it is applied:

  - ``"before"`` (default) — aerenchyma is built in 2D, then its cells are
    extruded as ONE full-span prism each, giving continuous axial lacunae.
    This is the rice-root case: cortical aerenchyma really is a set of long
    uninterrupted channels running the length of the root, so the lacuna
    should be one big file rather than a stack of independent rows.
  - ``"after"`` — aerenchyma is suppressed during the 2D pass and recreated on
    the extruded stack instead, by retyping individual prisms to "air space".
    This is the leaf case: mesophyll air space is a distributed, lobed void
    network, not a set of axial channels, so converting per-prism after
    extrusion gives a much better representation than extruding a 2D lacuna.
  - ``"none"`` — no aerenchyma at all.

Note that ``"air space"`` in the 2D output does not *only* mean aerenchyma:
leaf and needle also produce air space from substomatal chambers and the
stomatal recipe (``produces=("guard cell", "air space", "pore")``), by paths
independent of ``intercellular_spaces_params``. In ``"before"`` mode those
chambers are extruded full-span along with the aerenchyma; use ``"after"``
(or ``"none"``) on leaf/needle if that is not what you want.
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

# The tag Organ.add_aerenchyma / the stomatal recipes give to air.
AIR_SPACE_TYPE = "air space"

# When aerenchyma is applied relative to the extrusion. See the module
# docstring for why the choice is organ-dependent.
AERENCHYMA_MODES = ("before", "after", "none")

# Most non-vascular tissue is axially elongated in real anatomy, not
# isotropic — when a tissue's own axial_height isn't explicitly configured,
# default it to this multiple of that tissue's own cell_diameter.
DEFAULT_AXIAL_HEIGHT_RATIO = 5.0


@dataclass
class Cells3DResult:
    cells: List[dict]   # [{"type", "vertices", "faces", "volume"}]
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
    cap in the ring's original order, so normals point outward.
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
    """First configured height wins: the layer's own, then the caller's
    default, then a multiple of this cell's diameter. Explicit ``is not None``
    checks — a configured height of 0 must not silently fall through."""
    if layer_axial_height is not None:
        return layer_axial_height
    if default_axial_height is not None:
        return default_axial_height
    return cell_diameter * DEFAULT_AXIAL_HEIGHT_RATIO


def _aerenchyma_tissues(organ) -> List[str]:
    """The tissue name(s) ``aerenchyma_params`` targets, always as a list."""
    tissue = (organ.aerenchyma_params or {}).get("tissue")
    if tissue is None:
        return []
    return list(tissue) if isinstance(tissue, (list, tuple)) else [tissue]


def _apply_aerenchyma_after(cells: List[dict], organ, requested: float, rng) -> None:
    """Recreate aerenchyma on the extruded stack, in place.

    Retypes whole prisms of the target tissue to "air space" — chosen at
    random, by volume — until the requested air fraction of that tissue band
    is reached. Mirrors ``Organ._measure_air_proportion``'s definition of the
    quantity (air / (air + tissue)), but in 3D volume rather than 2D area.

    Deliberately per-prism and independent: unlike the ``"before"`` path this
    does NOT produce continuous axial channels, which is the point — leaf
    mesophyll air space is a distributed void network, not a set of files.
    The voids are not connected to each other here; a lacuna-network model is
    a separate piece of work.
    """
    if requested <= 0:
        return

    tissues = set(_aerenchyma_tissues(organ))
    if not tissues:
        return

    candidates = [c for c in cells if c["type"] in tissues]
    if not candidates:
        return

    band_volume = sum(c["volume"] for c in candidates)
    # Air already present in the band (e.g. substomatal chambers) counts
    # toward the request, exactly as the 2D measurement does.
    air_volume = sum(c["volume"] for c in cells if c["type"] == AIR_SPACE_TYPE)
    total = band_volume + air_volume
    if total <= 0:
        return

    target_air = requested * total
    order = rng.permutation(len(candidates))
    for idx in order:
        if air_volume >= target_air:
            break
        cell = candidates[idx]
        cell["type"] = AIR_SPACE_TYPE
        air_volume += cell["volume"]


def generate_cells_3d(organ, n_axial_repeats: float = 8.0,
                      default_axial_height: Optional[float] = None,
                      aerenchyma_mode: str = "before",
                      seed: Optional[int] = None) -> Cells3DResult:
    """Build a 3D segment of ``organ`` by stacking literal copies of each 2D
    cell's polygon along Z at its own tissue-type height.

    Generates the 2D cross-section itself, with ordinary intercellular space
    disabled and aerenchyma handled per ``aerenchyma_mode`` (see the module
    docstring). Called via ``Organ.generate_cells_3d(...)``; see that method
    for the public API.

    ``organ`` is left as it was found: the parameters this function overrides
    for the 2D pass are restored, and the cached geometry is invalidated, so a
    later ``organ.generate_cells()`` returns the organ's own 2D section rather
    than the 3D-flavoured one built here.
    """
    if aerenchyma_mode not in AERENCHYMA_MODES:
        raise ValueError(
            f"aerenchyma_mode must be one of {AERENCHYMA_MODES}, got {aerenchyma_mode!r}"
        )

    rng = np.random.default_rng(seed) if seed is not None else organ.rng

    aerenchyma_params = organ.aerenchyma_params or {}
    saved_ics = organ.intercellular_spaces_params
    saved_proportion = aerenchyma_params.get("aerenchyma_proportion")
    # The 2D pass below draws from organ.rng. Restoring its state keeps this
    # function free of side effects: the organ's own later generate_cells()
    # produces exactly the section it would have without the 3D build, and
    # repeated 3D builds of one organ are reproducible.
    saved_rng_state = organ.rng.bit_generator.state

    try:
        # Ordinary intercellular space disabled at the source -- not generated
        # then filtered, never generated.
        organ.intercellular_spaces_params = []
        # "after"/"none" suppress the 2D aerenchyma pass; "before" leaves the
        # organ's own configuration alone so the lacunae are built in 2D.
        if aerenchyma_mode in ("after", "none") and "aerenchyma_proportion" in aerenchyma_params:
            aerenchyma_params["aerenchyma_proportion"] = 0.0

        organ._invalidate_geometry()
        organ.generate_cells()

        layers_polygons = organ.generate_layer_polygons()
        layer_axial_height = {i: lp.get("axial_height") for i, lp in enumerate(layers_polygons)}

        # Types that get one full-span extrusion instead of repeated rows.
        # In "before" mode the aerenchyma lacunae join the vessels: a rice
        # cortical lacuna is one long channel, not a stack of rows.
        full_span_types = set(VESSEL_TYPES)
        if aerenchyma_mode == "before":
            full_span_types.add(AIR_SPACE_TYPE)

        # Segment height: sized off the tallest ordinary tissue's own
        # axial_height (full-span types don't count -- their "height" is the
        # whole span by definition, so including them would be circular).
        heights = [
            _resolve_axial_height(c.diameter, layer_axial_height.get(c.id_layer), default_axial_height)
            for c in organ.all_cells.cells
            if c.type not in full_span_types
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

            if cell.type in full_span_types:
                # One extrusion covering the whole segment -- no phase, no repeats.
                vertices, faces = extrude_polygon(polygon, z_min, z_max)
                cells.append({"type": cell.type, "vertices": vertices, "faces": faces,
                              "volume": polygon.area * (z_max - z_min)})
                continue

            height = _resolve_axial_height(cell.diameter, layer_axial_height.get(cell.id_layer), default_axial_height)
            phase = rng.uniform(0, height)  # per-cell Z-shift -- the "height shift inside the same tissue"
            z = z_min + phase - height
            while z < z_max:
                row_z0, row_z1 = max(z, z_min), min(z + height, z_max)
                if row_z1 > row_z0:
                    vertices, faces = extrude_polygon(polygon, row_z0, row_z1)
                    cells.append({"type": cell.type, "vertices": vertices, "faces": faces,
                                  "volume": polygon.area * (row_z1 - row_z0)})
                z += height

        if aerenchyma_mode == "after":
            # The requested proportion as configured, not the 0.0 written above
            # to suppress the 2D pass.
            _apply_aerenchyma_after(cells, organ, float(saved_proportion or 0.0), rng)

        return Cells3DResult(cells=cells, z_min=z_min, z_max=z_max)

    finally:
        organ.intercellular_spaces_params = saved_ics
        if saved_proportion is not None:
            aerenchyma_params["aerenchyma_proportion"] = saved_proportion
        organ.rng.bit_generator.state = saved_rng_state
        # The cached 2D section built above is 3D-specific (no intercellular
        # space, possibly no aerenchyma); drop it so the organ regenerates its
        # own section on next use.
        organ._invalidate_geometry()
