"""
OrganDomain — bvpy-compatible Gmsh domain built from a GRANAP Organ.

Usage
-----
    from granap.root_class import RootAnatomy
    from granap.organ_domain import OrganDomain

    root = RootAnatomy()
    domain = OrganDomain(root, cell_wall_thickness=1, cell_size=50, dim=2, clear=True)
    domain.discretize()          # triggers Gmsh meshing -> FEniCS mesh
    domain.info()                # prints mesh stats
    print(domain.sub_domain_names)   # {tag: cell_type_name}

Or via the convenience helper on Organ:
    domain = root.to_domain(cell_wall_thickness=1, cell_size=50, dim=2, clear=True)
"""

from __future__ import annotations
from typing import Union, Dict

from bvpy.domains.abstract import OccModel, AbstractDomain
from granap.anatomy_writer import AnatomyWriter
from granap.organ_class import Organ
from shapely.geometry import Point as _Pt


# Cell-type → integer tag used for Physical Groups (Mecha convention)
_CELL_TYPE_TAGS: Dict[str, int] = {
    'exodermis': 1,
    'epidermis': 2,
    'hypodermis': 1,
    'endodermis': 3,
    'passage': 3,
    'cortex': 4,
    'mesophyll':4,
    'stele': 5,
    'pith': 5,
    'parenchyma': 5,
    'vascular_parenchyma': 5,
    'phloem': 11,
    'companion': 12,               
    'cambium':12,
    'guard cell': 12,
    'Strasburger cell': 12,
    'xylem': 13,
    'protoxylem': 13,
    'metaxylem': 13,
    'pericycle': 16,
    'air space': 9,
    'pore': 9,
    'inter_cellular_space': 9,
    'aerenchyma': 9,
}

_OUTER_DOMAIN_TAG = 0
_CELLS_BOUNDARY_TAG = 1
_AIR_BOUNDARY_TAG = 2
_CENTER_BOUNDARY_TAG = 3
_XYLEM_BOUNDARY_TAG = 4


class OrganDomain(AbstractDomain, OccModel):
    """bvpy-compatible 2-D meshed domain for a GRANAP cross-section.

    Converts a GRANAP :class:`Organ` (root or needle) into a live Gmsh OCC
    model that can be meshed by :meth:`discretize` and used directly in FEniCS /
    bvpy simulations.

    The geometry pipeline re-uses :meth:`AnatomyWriter.prep_geo` for polygon
    buffering, smoothing and coordinate scaling (×1000, mm → μm), then builds
    the Gmsh entities programmatically via :pyobj:`self.factory` (``gmsh.model.occ``).

    Parameters
    ----------
    organ : Organ
        A generated GRANAP organ (RootAnatomy or NeedleAnatomy).
    cell_wall_thickness : float or dict, optional
        Wall thickness passed to :meth:`~AnatomyWriter.prep_geo` (μm after scaling).
        A dict ``{cell_type: thickness}`` overrides per type. Default ``1``.
    corner_smoothing : float or dict, optional
        Smoothing factor passed to :meth:`~AnatomyWriter.prep_geo`.
        A dict ``{cell_type: factor}`` overrides per type.  Default ``0.5``.
    celldomain : bool, optional
        * ``True``  – each individual cell surface gets its own Physical Group
          (fine-grained; useful when you need per-cell BCs).
        * ``False`` (default) – cells of the same type share a Physical Group tag.
    simplify_tol : float, optional
        Douglas–Peucker tolerance (in μm) applied to polygon rings before
        sending coordinates to Gmsh.  Reduces point count and speeds up meshing.
        Default ``1.0``.
    **kwargs
        Forwarded to :class:`~bvpy.domains.abstract.AbstractDomain` (e.g.
        ``cell_size``, ``cell_type``, ``dim``, ``clear``, ``algorithm``).

    Attributes
    ----------
    sub_domain_names : dict
        ``{int_tag: human_readable_name}`` populated after :meth:`discretize`.
    surfaces : dict
        Gmsh surface tags grouped by cell type (set during :meth:`geometry`).

    Examples
    --------
    >>> from granap.needle_class import NeedleAnatomy
    >>> needle = NeedleAnatomy()
    >>> domain = needle.to_domain(cell_wall_thickness=1, cell_size=50, dim=2, clear=True)
    >>> domain.discretize()
    >>> print(domain.sub_domain_names)
    """

    def __init__(
        self,
        organ: Organ,
        cell_wall_thickness: Union[float, Dict[str, float]] = 1.0,
        corner_smoothing: Union[float, Dict[str, float]] = 0.5,
        celldomain: bool = False,
        simplify_tol: float = 0.2,
        symplast: bool = True,
        **kwargs,
    ):
        # Default to 2-D triangular mesh if caller did not specify
        kwargs.setdefault("dim", 2)
        kwargs.setdefault("cell_type", "triangle")

        super().__init__(**kwargs)
        self.geometry(organ, cell_wall_thickness, corner_smoothing, celldomain, simplify_tol, symplast)

    # ------------------------------------------------------------------
    # bvpy interface
    # ------------------------------------------------------------------

    def geometry(
        self,
        organ: Organ,
        cell_wall_thickness: Union[float, Dict[str, float]] = 1.0,
        corner_smoothing: Union[float, Dict[str, float]] = 3,
        celldomain: bool = False,
        simplify_tol: float = 0.05,
        symplast: bool = True,
    ) -> None:
        """Build the Gmsh OCC model from a GRANAP Organ.

        Called automatically from :meth:`__init__`.  You can call it again to
        rebuild the geometry with different parameters without creating a new
        instance (after ``gmsh.clear()``).

        Parameters
        ----------
        symplast : bool
            * ``True`` (default) – inner cell surfaces are meshed and labelled
              with Physical Groups.  The full cross-section (apoplast + symplast)
              is simulated.
            * ``False`` – inner cell surfaces are used only as *holes* in the
              apoplast surface and are **not** meshed or labelled.  Only the
              apoplast (cell-wall) layer is simulated.  This avoids the
              partial-labelling Warning raised by bvpy when some dim-2 surfaces
              are untagged.
        """
        writer = AnatomyWriter(organ)
        inner_polygons, final_polygon = writer.prep_geo(cell_wall_thickness, corner_smoothing)

        # Gmsh counter state
        v_idx = 1   # Point tag counter
        l_idx = 1   # Line tag counter
        cl_idx = 1  # CurveLoop tag counter
        s_idx = 1   # Surface tag counter

        # ---------- helpers ----------

        def _add_polygon_edges(poly):
            """Register a Shapely polygon as Gmsh Points + Lines.

            Returns the list of Line tags for the exterior ring.
            """
            nonlocal v_idx, l_idx

            simplified = poly.simplify(simplify_tol, preserve_topology=True)
            if simplified.is_empty or simplified.geom_type != "Polygon":
                simplified = poly

            coords = list(simplified.exterior.coords)[:-1]  # drop closing duplicate
            if len(coords) < 3:
                return []

            first_v = v_idx
            for x, y in coords:
                self.factory.addPoint(round(x, 4), round(y, 4), 0.0, tag=v_idx)
                v_idx += 1

            line_tags = []
            n = len(coords)
            for i in range(n):
                curr = first_v + i
                nxt = first_v + (i + 1) % n
                self.factory.addLine(curr, nxt, tag=l_idx)
                line_tags.append(l_idx)
                l_idx += 1

            return line_tags

        # ---------- inner cell surfaces ----------
        # Track: cell-type → list of surface tags
        type_to_surfaces: Dict[str, list] = {}
        # Track: all inner curve-loop tags (for the outer Plane Surface hole list)
        inner_cl_tags = []
        # Track boundary curve tags by purpose
        cell_curves = []
        air_curves = []
        xylem_curves = []

        center_curve = None

        # Find the cell whose polygon centroid is closest to (0, 0) — the "center" cell

        origin = _Pt(0, 0)
        center_id = min(
            [ip for ip in inner_polygons if ip["polygon"] is not None],
            key=lambda ip: ip["polygon"].centroid.distance(origin),
            default=None,
        )
        center_id = center_id["id_cell"] if center_id else None

        for item in inner_polygons:
            poly = item["polygon"]
            id_cell = item["id_cell"]
            cell_type = item["type"]

            geoms = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]

            for geom in geoms:
                line_tags = _add_polygon_edges(geom)
                if not line_tags:
                    continue

                # Curve loop — always needed as a hole reference for the apoplast
                self.factory.addCurveLoop(line_tags, tag=cl_idx)
                inner_cl_tags.append(cl_idx)

                if symplast:
                    # Plane surface for each inner cell (meshed)
                    self.factory.addPlaneSurface([cl_idx], tag=s_idx)

                    # Track surface by cell type
                    type_to_surfaces.setdefault(cell_type, []).append(cl_idx)
                    s_idx += 1

                # Categorise curves for boundary labels (always, for ds markers)
                if id_cell == center_id:
                    center_curve = line_tags[:]
                elif cell_type in ("air space", "aerenchyma", "pore"):
                    air_curves.extend(line_tags)
                elif cell_type in ("xylem", "protoxylem", "metaxylem"):
                    xylem_curves.extend(line_tags)
                else:
                    cell_curves.extend(line_tags)

                cl_idx += 1

        # ---------- outer tissue boundary ----------
        outer_geoms = (
            list(final_polygon.geoms)
            if final_polygon.geom_type == "MultiPolygon"
            else [final_polygon]
        )

        outer_surface_tags = []
        for geom in outer_geoms:
            line_tags = _add_polygon_edges(geom)
            if not line_tags:
                continue

            # Outer curve loop
            self.factory.addCurveLoop(line_tags, tag=cl_idx)
            outer_cl_idx = cl_idx
            cl_idx += 1

            # Plane Surface with ALL inner curve loops as holes
            hole_refs = [outer_cl_idx] + inner_cl_tags
            self.factory.addPlaneSurface(hole_refs, tag=s_idx)
            outer_surface_tags.append(s_idx)
            s_idx += 1

        # ---------- synchronise ----------
        self.factory.synchronize()

        # ---------- Physical Groups (surfaces = dim 2) ----------
        all_typed_surface_tags = []

        if symplast:
            if celldomain:
                # One Physical Group per cell surface
                for ctype, stags in type_to_surfaces.items():
                    base_tag = _CELL_TYPE_TAGS.get(ctype, 99)

                    if base_tag in all_typed_surface_tags:  # avoid collision
                        base_tag += 21
                    while base_tag in all_typed_surface_tags:
                        base_tag += 1

                    self.model.addPhysicalGroup(2, stags, tag=base_tag, name=f"{ctype}")
                    all_typed_surface_tags.append(base_tag)
            else:
                # One Physical Group per cell type (merge same-type cells)
                tag_to_surfaces: Dict[int, list] = {}
                tag_to_name: Dict[int, str] = {}
                next_unknown_tag = max(list(_CELL_TYPE_TAGS.values()) + [99]) + 1

                for ctype, stags in type_to_surfaces.items():
                    pg_tag = _CELL_TYPE_TAGS.get(ctype)
                    if pg_tag is None:
                        pg_tag = next_unknown_tag
                        next_unknown_tag += 1

                    if pg_tag not in tag_to_surfaces:
                        tag_to_surfaces[pg_tag] = []
                        tag_to_name[pg_tag] = ctype
                    else:
                        tag_to_name[pg_tag] += f"_{ctype}"

                    tag_to_surfaces[pg_tag].extend(stags)

                for pg_tag, stags in tag_to_surfaces.items():
                    self.model.addPhysicalGroup(2, stags, tag=pg_tag, name=f"{tag_to_name[pg_tag]}")
                    all_typed_surface_tags.extend(stags)

        # Outer (wall/apoplast) domain — always tagged
        if outer_surface_tags:
            self.model.addPhysicalGroup(2, outer_surface_tags, tag=_OUTER_DOMAIN_TAG, name="apoplast")
            all_typed_surface_tags.extend(outer_surface_tags)

        # ---------- Physical Groups (curves = dim 1) ----------
        if cell_curves:
            self.model.addPhysicalGroup(1, cell_curves, tag=_CELLS_BOUNDARY_TAG, name="cells")
        if air_curves:
            self.model.addPhysicalGroup(1, air_curves, tag=_AIR_BOUNDARY_TAG, name = "air_space")
        if center_curve:
            self.model.addPhysicalGroup(1, center_curve, tag=_CENTER_BOUNDARY_TAG, name = "center")
        if xylem_curves:
            self.model.addPhysicalGroup(1, xylem_curves, tag=_XYLEM_BOUNDARY_TAG, name = "xylem")

        # Store surface tags for inspection / CSG operations
        self.surfaces = {
            ctype: stags for ctype, stags in type_to_surfaces.items()
        }
        self.surfaces["apoplast"] = outer_surface_tags
        
        # Expose boundary tags for easy BC setup
        self.symplast_boundary_tag = _CELLS_BOUNDARY_TAG
        self.air_boundary_tag = _AIR_BOUNDARY_TAG
        self.center_boundary_tag = _CENTER_BOUNDARY_TAG
        self.xylem_boundary_tag = _XYLEM_BOUNDARY_TAG