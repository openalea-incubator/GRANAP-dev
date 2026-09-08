"""CellSet XML ingestion — parse a digitized cross-section into polygons + tags.

CellSet (``<cellsetdata>``) stores a section as a *shared-wall* mesh, not as a set
of independent outlines:

    <cells count="40">
      <cell id="1" group="2" truncated="false">
        <walls><wall id="0"/><wall id="6"/>...</walls>
      </cell>
      ...
    </cells>
    <walls count="115">
      <wall id="0" group="0" edgewall="true">
        <points><point x="126.7055" y="93.3115"/>...</points>
      </wall>
      ...
    </walls>
    <groups>
      <cellgroups><group id="2" name="epidermis"/>...</cellgroups>
      <wallgroups><group id="0" name="unassigned"/></wallgroups>
    </groups>

Each cell lists the walls that bound it; each wall owns the polyline, and an
interior wall is referenced by exactly two cells.  That is why this reader must
keep the wall points **verbatim** — no smoothing, no resampling.  Two adjacent
cells then still share their wall's vertices exactly, which is what lets
``CellGenerator._build_topology`` pair their walls and give MECHA real
``membrane``/``plasmodesmata`` connections.  (Contrast ``roi_organ.RoiOrgan``,
which pushes every outline through ``GeometryProcessor.smoothing_polygon`` and
therefore exports a network with one private wall per cell and no cell-to-cell
edges at all.)

A cell whose walls assemble into more than one closed ring is **annular**: the
extra rings are holes punched by cells drawn inside it.  MECHA cannot handle such
a cell, which is the whole motivation for ``cellset_organ.CellSetOrgan`` — see
``Andrea/Mixing_CellSet_GRANAP.md``.

CellSet's ``group`` integers are the same integers MECHA calls ``cgroup``
(``anatomy_writer.CGROUP_MAP`` / MECHA's ``hydraulic_cell.CGROUP_TO_TYPE``), so
they are carried straight through onto ``Cell.cgroup`` rather than being
re-derived from the tag string.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon

Point2 = Tuple[float, float]
Ring = List[Point2]

# Fallback tag for a CellSet cell group whose <cellgroups> entry is missing or
# unusable.  Mirrors anatomy_writer.CGROUP_MAP read backwards; only the ids a
# real section is likely to carry are listed.
CELLSET_TAG_BY_GROUP: Dict[int, str] = {
    1: "exodermis",
    2: "epidermis",
    3: "endodermis",
    4: "cortex",
    5: "stele",
    11: "phloem",
    12: "companion_cell",
    13: "xylem",
    16: "pericycle",
    17: "transfusion parenchyma",
    18: "transfusion tracheid",
    19: "xylem",
    20: "xylem",
    21: "pericycle",
    23: "protophloem",
    26: "companion_cell",
}

# CellSet group names that GRANAP spells differently.
_TAG_ALIASES: Dict[str, str] = {
    "companion": "companion_cell",
    "companion cell": "companion_cell",
    "sieve tube": "sieve",
    "passage cell": "passage_cell",
    "intercellular space": "air space",
    "unassigned": "parenchyma",
}

# Rounding used when matching polyline endpoints during ring assembly.  CellSet
# writes 4 decimals in micrometres; 9 decimals of the scaled value is well below
# that yet still tolerant of float round-off from the scaling multiply.
_MATCH_DECIMALS: int = 9


def _key(pt: Point2) -> Point2:
    return (round(pt[0], _MATCH_DECIMALS), round(pt[1], _MATCH_DECIMALS))


def _ring_area(ring: Ring) -> float:
    """Absolute shoelace area of a closed ring."""
    area = 0.0
    for i in range(len(ring) - 1):
        area += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return abs(area) / 2.0


def assemble_rings(polylines: Sequence[Sequence[Point2]]) -> List[Ring]:
    """Assemble unordered wall polylines into closed rings, largest ring first.

    A CellSet cell is given as a bag of wall polylines in arbitrary order and
    arbitrary direction.  This walks them end-to-end, reversing a polyline when
    that is what makes it join, and starts a new ring whenever the current one
    closes.  A cell that yields more than one ring is annular: ring 0 is the
    outer boundary and the rest are holes.

    Polylines that cannot be joined to anything are returned as their own
    (unclosed) ring so the caller can notice rather than silently lose them.
    """
    remaining: List[List[Point2]] = [list(p) for p in polylines if len(p) >= 2]
    rings: List[Ring] = []

    while remaining:
        current = remaining.pop(0)
        joined = True
        while joined:
            joined = False
            if _key(current[0]) == _key(current[-1]):
                break  # already closed -- do not keep growing this ring
            for i, seg in enumerate(remaining):
                for candidate in (seg, seg[::-1]):
                    if _key(current[-1]) == _key(candidate[0]):
                        current.extend(candidate[1:])
                    elif _key(current[0]) == _key(candidate[-1]):
                        current[:0] = candidate[:-1]
                    else:
                        continue
                    remaining.pop(i)
                    joined = True
                    break
                if joined:
                    break
        rings.append(current)

    rings.sort(key=_ring_area, reverse=True)
    return rings


def _polygon_from_rings(rings: Sequence[Ring]) -> Optional[Polygon]:
    """Build a Polygon from assembled rings (ring 0 outer, the rest holes)."""
    usable = [r for r in rings if len(r) >= 4 and _key(r[0]) == _key(r[-1])]
    if not usable:
        return None
    shell, holes = usable[0], usable[1:]
    poly = Polygon(shell, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            return None
    return poly


@dataclass
class CellSetCell:
    """One digitized cell: its CellSet identity plus the geometry it assembles to."""

    id: int
    group: int
    tag: str
    wall_ids: List[int]
    rings: List[Ring]
    polygon: Optional[Polygon]

    @property
    def is_annular(self) -> bool:
        """True when the cell encloses holes -- the case MECHA cannot handle."""
        return len(self.rings) > 1

    @property
    def outer_ring_polygon(self) -> Optional[Polygon]:
        """The outer boundary alone, holes filled in."""
        if not self.rings:
            return None
        return _polygon_from_rings(self.rings[:1])


@dataclass
class CellSetSection:
    """A parsed CellSet file, in GRANAP's units (mm by default)."""

    cells: List[CellSetCell]
    walls: Dict[int, Ring] = field(default_factory=dict)
    group_names: Dict[int, str] = field(default_factory=dict)
    scale: float = 1.0
    path: Optional[str] = None

    @property
    def annular_cells(self) -> List[CellSetCell]:
        return [c for c in self.cells if c.is_annular]

    def census(self) -> Dict[str, Tuple[int, float]]:
        """``{tag: (n_cells, total_polygon_area)}`` -- judge tissues by area."""
        out: Dict[str, Tuple[int, float]] = {}
        for cell in self.cells:
            n, area = out.get(cell.tag, (0, 0.0))
            out[cell.tag] = (n + 1, area + (cell.polygon.area if cell.polygon else 0.0))
        return out


def normalize_tag(name: Optional[str], group: int) -> str:
    """CellSet group name -> GRANAP cell tag.

    Falls back to :data:`CELLSET_TAG_BY_GROUP` when the file gives no usable
    name, and finally to ``"parenchyma"`` (cgroup 5, stele) so an unmapped tissue
    never silently lands on cortex the way ``CGROUP_MAP``'s default would.
    """
    if name:
        cleaned = name.strip()
        lowered = cleaned.lower()
        if lowered in _TAG_ALIASES:
            return _TAG_ALIASES[lowered]
        if lowered not in ("", "unassigned", "none"):
            return cleaned
    return CELLSET_TAG_BY_GROUP.get(group, "parenchyma")


def read_cellset(path: str, *, scale: float = 0.001) -> CellSetSection:
    """Parse a CellSet ``<cellsetdata>`` file into polygons, tags and cgroups.

    Args:
        path:  Path to the CellSet XML file.  The root element name is not
               checked -- GRANAP's own ``AnatomyWriter.write_to_xml`` emits the
               same schema under ``<granardata>``, and MECHA's ``parse_cellset``
               likewise only uses relative xpaths, so both are readable here.
        scale: Multiplier applied to every coordinate.  The default 0.001
               converts CellSet's micrometres to GRANAP's millimetres, which is
               also what MECHA's ``GeometryData.im_scale`` default of 1000
               expects to read back.

    Returns:
        A :class:`CellSetSection`.  Coordinates are exact wall points times
        ``scale`` -- deliberately unsmoothed, see the module docstring.
    """
    root = ET.parse(path).getroot()

    walls: Dict[int, Ring] = {}
    walls_node = root.find("walls")
    if walls_node is not None:
        for wall in walls_node.findall("wall"):
            points_node = wall.find("points")
            if points_node is None:
                continue
            pts = [
                (float(p.get("x")) * scale, float(p.get("y")) * scale)
                for p in points_node.findall("point")
            ]
            walls[int(wall.get("id"))] = pts

    group_names: Dict[int, str] = {}
    groups_node = root.find("groups")
    if groups_node is not None:
        cellgroups = groups_node.find("cellgroups")
        if cellgroups is not None:
            for group in cellgroups.findall("group"):
                group_names[int(group.get("id"))] = group.get("name") or ""

    cells: List[CellSetCell] = []
    cells_node = root.find("cells")
    if cells_node is not None:
        for cell in cells_node.findall("cell"):
            wall_ids = [
                int(w.get("id"))
                for w in cell.findall("walls/wall")
                if w.get("id") is not None
            ]
            polylines = [walls[wid] for wid in wall_ids if wid in walls]
            rings = assemble_rings(polylines)
            group = int(cell.get("group", 0))
            cells.append(
                CellSetCell(
                    id=int(cell.get("id")),
                    group=group,
                    tag=normalize_tag(group_names.get(group), group),
                    wall_ids=wall_ids,
                    rings=rings,
                    polygon=_polygon_from_rings(rings),
                )
            )

    return CellSetSection(
        cells=cells, walls=walls, group_names=group_names, scale=scale, path=path
    )
