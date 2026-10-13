"""
The needle's intercellular spaces, converted (devplan_metafspm.md, §47, P5.6 a): GRANAP's NeedleAnatomy.
_apply_intercellular, _apply_mesophyll_wall_rhombi and _apply_palisade_wall_slits (needle_class.py), copied, with
three edits: the remaining entries go to the converted intercellular spaces (Organ's, intercellular.py), and the
lacunae are seated by the converted seat_air_spaces with the organ's topology (no class method called). Every other
line is GRANAP's; *self* is the organ.
"""
from openalea.granap.needle_class import GeometryProcessor, List, MultiPolygon, Polygon, STRtree, np
from openalea.granap.mpg.pipeline.intercellular import apply_intercellular, seat_air_spaces


def apply_needle_intercellular(self, ics: dict) -> None:
    """Needle override of the intercellular-space geometry.

    In the needle mesophyll the intercellular air spaces are small rhombic
    (diamond-shaped) lacunae seated *on the walls* between adjacent mesophyll
    cells. This override intercepts the ``mesophyll`` tissue and builds those
    wall-centred rhombi (see :meth:`_apply_mesophyll_wall_rhombi`).

    It also intercepts ``palisade``: instead of rhombi, thin full-height
    air *slits* are carved across every ``slit_every``-th wall between
    angularly-adjacent palisade cells (see
    :meth:`_apply_palisade_wall_slits`), read from the optional
    ``slit_width``/``slit_every`` per-tissue lists.

    Any other tissue is delegated to the shared base implementation unchanged.
    """
    tissues = ics.get("tissue", [])
    if isinstance(tissues, str):
        tissues = [tissues]
    if "mesophyll" not in tissues and "palisade" not in tissues:
        apply_intercellular(self, ics)
        return

    smoothness = ics.get("smoothness", 0)
    if isinstance(smoothness, (int, float)):
        smoothness_per_tissue = [float(smoothness)] * len(tissues)
    else:
        smoothness_per_tissue = [float(s) for s in smoothness]
    smoothness_by_tissue = dict(zip(tissues, smoothness_per_tissue))

    # slit_width / slit_every: optional parallel per-tissue lists (only
    # consumed for "palisade"). Default to 0 (off) so tissue entries that
    # don't set them -- and every pre-existing config, which never had
    # these keys at all -- see no behaviour change.
    slit_width = ics.get("slit_width", 0.0)
    if isinstance(slit_width, (int, float)):
        slit_width_per_tissue = [float(slit_width)] * len(tissues)
    else:
        slit_width_per_tissue = [float(w) for w in slit_width]
    slit_width_by_tissue = dict(zip(tissues, slit_width_per_tissue))

    slit_every = ics.get("slit_every", 0)
    if isinstance(slit_every, (int, float)):
        slit_every_per_tissue = [int(slit_every)] * len(tissues)
    else:
        slit_every_per_tissue = [int(e) for e in slit_every]
    slit_every_by_tissue = dict(zip(tissues, slit_every_per_tissue))

    # Build the wall-centred rhombic lacunae for the mesophyll only.
    if "mesophyll" in tissues:
        apply_mesophyll_wall_rhombi(self, smoothness_by_tissue.get("mesophyll", 0.0))

    # Carve the thin wall slits for the palisade only.
    if "palisade" in tissues:
        apply_palisade_wall_slits(self, 
            slit_width_by_tissue.get("palisade", 0.0),
            slit_every_by_tissue.get("palisade", 0),
        )

    # Delegate the remaining tissues (if any) to the generic implementation.
    other_tissues = [t for t in tissues if t not in ("mesophyll", "palisade")]
    if other_tissues:
        remaining = dict(ics)
        remaining["tissue"] = other_tissues
        remaining["smoothness"] = [smoothness_by_tissue[t] for t in other_tissues]
        remaining.pop("slit_width", None)
        remaining.pop("slit_every", None)
        apply_intercellular(self, remaining)


def apply_mesophyll_wall_rhombi(self, smoothness: float) -> None:
    """Insert small rhombic air spaces on the walls between mesophyll cells.

    For every wall shared by two adjacent mesophyll cells a rhombus (a
    four-vertex diamond) is placed, centred on the wall midpoint and aligned
    with the wall: its principal diagonal is parallel to the wall and spans
    about ``MAJOR_WALL_FRACTION`` of the wall length, and its
    principal-to-secondary diagonal ratio is ``AXIS_RATIO``.

    The air-space cells use the same labelling conventions as the base
    intercellular routine (``id_layer=0``, ``id_group=id_cell``).
    """
    MAJOR_WALL_FRACTION = 1.0 / 3.0   # principal diagonal ≈ 1/3 of the wall length
    AXIS_RATIO = 2.0                  # principal : secondary diagonal = 2 : 1

    # Any mesophyll-family type (plain "mesophyll" plus variants like the
    # adaxial-only extra ring) gets the same wall-rhombi treatment -- they
    # are all spongy mesophyll tissue.
    mesophyll_cells = [
        c for c in self.all_cells.cells
        if c.polygon is not None and (c.type == "mesophyll" or c.type.startswith("mesophyll_"))
    ]
    if len(mesophyll_cells) < 2:
        return

    polys = [c.polygon for c in mesophyll_cells]
    tree = STRtree(polys)

    rhombus_polys: List[Polygon] = []
    seen_pairs: set = set()
    for i, poly_i in enumerate(polys):
        for j in tree.query(poly_i):
            if j <= i:
                continue
            pair = (i, j)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            poly_j = polys[j]
            if not poly_i.intersects(poly_j):
                continue
            shared = poly_i.intersection(poly_j)
            for wall in self._iter_wall_segments(shared):
                rhombus = self._wall_rhombus(wall, MAJOR_WALL_FRACTION, AXIS_RATIO)
                if rhombus is not None and not rhombus.is_empty and rhombus.area > 1e-6:
                    rhombus_polys.append(rhombus)

    if not rhombus_polys:
        return

    # Union all lacunae once. Carving the cells and building the air-space
    # cells from this *same* geometry keeps their shared boundaries
    # vertex-for-vertex identical, so the network sees real walls and
    # junctions between each lacuna and its mesophyll hosts.
    air_union = GeometryProcessor.union_polygons(rhombus_polys)

    # Split the air union into its connected lacunae; each becomes one cell.
    if isinstance(air_union, MultiPolygon):
        air_faces = [g for g in air_union.geoms if not g.is_empty and g.area > 1e-6]
    elif not air_union.is_empty and air_union.area > 1e-6:
        air_faces = [air_union]
    else:
        return

    # Carve the lacunae out of the mesophyll cells and insert them as
    # air-space cells (shared post-fill placement). ``protect_topology`` keeps
    # each rhombus's straight sides as distinct walls, so the neighbouring mesophyll cell keeps the matching
    # notch instead of being straightened across it (see ``CellGenerator._build_topology``).

    seat_air_spaces(
        self.all_cells, mesophyll_cells, air_union, air_faces,
        protect_topology=True, snap=self.section_topology,
    )


def apply_palisade_wall_slits(self, width: float, every: int) -> None:
    """Carve thin, full-height air slits across every ``every``-th wall
    between angularly-adjacent palisade cells.

    The palisade is a single ring, so its cells form one closed angular
    sequence around the needle. Cells are ordered by the polar angle
    (``atan2``) of their own polygon centroid about the organ centre --
    by the time intercellular spaces are computed the whole cell
    population has already been recentred on the origin (``add_stomata``
    calls ``CellManager.recenter_cells`` earlier in ``_organ_recipe``,
    well before Voronoi grouping / ``add_intercellular_spaces`` run), so
    this angle is a genuine position around the ring, not an arbitrary
    seeding index.

    ``id_group`` is deliberately NOT used to pick "every Nth cell":
    stomata carving and ``_restrict_zoned_layers`` both remove palisade
    cells after they are first seeded, opening gaps in the ``id_group``
    sequence. Taking parity on ``id_group`` would drift across those gaps
    and skip walls unevenly instead of a true "every Nth cell" pattern.
    The angular ordering has no such gaps -- it is recomputed from the
    cells that actually survived to this point.

    Only every ``every``-th wall of the ring's ``n`` walls is slit (wall
    ``k`` sits between the ``k``-th and ``(k+1)``-th cell in angular
    order); with ``every=2`` this pairs up angularly-adjacent cells
    (0-1, 2-3, 4-5, ...), giving one slit per two cells as the bug
    report asks for, and leaves every other wall solid so no palisade
    cell loses more than one of its walls to a slit.
    """
    if width <= 0 or every < 2:
        return

    palisade_cells = [
        c for c in self.all_cells.cells
        if c.polygon is not None and c.type == "palisade"
    ]
    if len(palisade_cells) < every:
        return

    # Order the ring by angular position about the (recentred) organ centre.
    palisade_cells.sort(key=lambda c: np.arctan2(c.polygon.centroid.y, c.polygon.centroid.x))
    polys = [c.polygon for c in palisade_cells]
    n = len(polys)

    slit_polys: List[Polygon] = []
    for k in range(0, n, every):
        j = (k + 1) % n
        poly_i, poly_j = polys[k], polys[j]
        if not poly_i.intersects(poly_j):
            continue
        shared = poly_i.intersection(poly_j)
        for wall in self._iter_wall_segments(shared):
            # Slightly short of the full wall length (not exactly 1.0):
            # a slit reaching all the way to the wall's own endpoints
            # lands its corners exactly on existing cell-junction
            # vertices, which can turn the carve difference into a
            # degenerate GeometryCollection instead of a clean polygon.
            slit = self._wall_slit(wall, 0.9, width)
            if slit is not None and not slit.is_empty and slit.area > 1e-9:
                slit_polys.append(slit)

    if not slit_polys:
        return

    # Union all slits once, same reasoning as the mesophyll rhombi: carving
    # the cells and building the air-space cells from the same geometry
    # keeps the shared boundaries vertex-for-vertex identical.
    slit_union = GeometryProcessor.union_polygons(slit_polys)

    if isinstance(slit_union, MultiPolygon):
        slit_faces = [g for g in slit_union.geoms if not g.is_empty and g.area > 1e-9]
    elif not slit_union.is_empty and slit_union.area > 1e-9:
        slit_faces = [slit_union]
    else:
        return

    # protect_topology keeps each slit's straight sides as distinct walls
    # so the neighbouring palisade cell keeps the matching notch instead
    # of being straightened across it.
    seat_air_spaces(
        self.all_cells, palisade_cells, slit_union, slit_faces,
        protect_topology=True, snap=self.section_topology,
    )
