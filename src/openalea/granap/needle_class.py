"""
Needle anatomy implementation.
"""

import dataclasses
import numpy as np
from typing import List, Dict, Any, Optional
from shapely.geometry import Polygon, Point, LineString, MultiLineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.strtree import STRtree

from openalea.granap.organ_class import Organ
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.layer_class import Layer, LayerPolygon
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.shapes import PolygonInterpolator
from openalea.granap.input_data import OrganInputData
from openalea.granap.special_tissues import place_resin_duct, place_stomata, seat_air_spaces
from openalea.granap.tissue_class import TissueRecipe, fill_by_packing
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Module-level constants — geometry tuning parameters
# ---------------------------------------------------------------------------
# Number of epidermis border-point cells to skip at the start of the boundary
_STOMATA_SKIP_BORDER_PTS: int = 300

# The mesophyll ring used for duct placement is the outer annulus whose inner
# edge is 1.2x duct diameters from the mesophyll boundary.
_DUCT_RING_BUFFER_FACTOR: float = 1.2

# The parenchyma-ring polygon is obtained by shrinking the fitted ellipse
# inward by this fraction of cell_diameter (prevents ring cells from sitting
# on the exact ellipse edge).
_DUCT_RING_INNER_SHRINK: float = 0.15

# Fixed placement order for resin ducts within the 7 mesophyll slices.
# Positions 3 and 6 are the edge positions (placed first, as in real anatomy);
# the rest fill in around the ring in an evenly distributed pattern.
_DUCT_PLACEMENT_ORDER: list = [3, 6, 0, 2, 4, 1, 5]

class NeedleAnatomy(Organ):
    """
    Needle cross-sectional anatomy.

    Implements the specific structure of gymnosperm needle leaves,
    including transfusion tissue and resin ducts.
    """

    def __init__(self, input_data: Any = None, seed: Optional[int] = None):
        """
        Initialize needle anatomy.

        Args:
            input_data: Parameter data (OrganInputData, list of dicts, or None for defaults).
            seed:       Integer seed for reproducible anatomy generation.
        """
        super().__init__(seed=seed)
        if isinstance(input_data, OrganInputData):
            self.params = input_data.to_dict_list()
        elif isinstance(input_data, list):
            self.params = input_data
        else:
            self.params = OrganInputData.for_needle().to_dict_list()

        self._initialize_params()
        self._initialize_default_layers()

    def _initialize_params(self) -> None:
        """Initialize central layers."""
        # 1. Global params
        self.global_params = next(p for p in self.params if p["name"] == "planttype")
        # 2. Central cylinder params
        self.central_cylinder_params = next(p for p in self.params if p["name"] == "central_cylinder")
        # 3. Transfusion tissue params
        self.transfusion_params = next(p for p in self.params if p["name"] == "transfusion_tissue")

        # 3. Intercellular spaces / aerenchyma — store raw config dicts directly
        self.intercellular_spaces_params = [p for p in self.params if p["name"] == "inter_cellular_spaces"]
        self.aerenchyma_params = self._get_param("aerenchyma")

        # 4. Extract layer definitions (any param with 'order' that is not a vascular zone)
        self.layers = [param for param in self.params if "order" in param]
        self.layers = sorted(self.layers, key=lambda x: x["order"])        
    
    def _initialize_default_layers(self) -> None:
        """Initialize default needle layers."""
        for param in self.layers:
            self.layer_manager.add_layer(Layer.from_dict(param))
    
    @staticmethod
    def _angular_multiplier(profile: List[List[float]], angle_deg: float) -> float:
        """
        Circularly interpolate a list of ``(angle_degrees, multiplier)``
        control points at ``angle_deg`` (wrapping at 360 degrees).

        Consumed by ``_offset_layer_polygon`` for any layer carrying a
        ``"thickness_profile"`` entry, e.g. the extra abaxial-only mesophyll
        ring or the corner-thickened hypodermis ring: the ring's nominal
        radial offset is scaled by this multiplier as a function of position
        around the needle cross-section. The actual pole/corner angles
        depend on the base shape's aspect ratio -- see example/needle/
        pinus_pinaster.py's ``_pole_and_corner_angles`` for the derivation
        (for this needle: adaxial pole ~270 deg, abaxial pole ~90 deg,
        corners ~0/~180 deg-ish).
        """
        if not profile:
            return 1.0
        pts = sorted((float(a) % 360.0, float(m)) for a, m in profile)
        angles = np.array([a for a, _ in pts])
        mults = np.array([m for _, m in pts])
        # Extend circularly so np.interp sees a monotonically increasing
        # x-range that safely covers any angle in [0, 360).
        ext_angles = np.concatenate(([angles[-1] - 360.0], angles, [angles[0] + 360.0]))
        ext_mults = np.concatenate(([mults[-1]], mults, [mults[0]]))
        return float(np.interp(angle_deg % 360.0, ext_angles, ext_mults))

    def _offset_layer_polygon(self, polygon: Polygon, distance: float, layer: Dict[str, Any],
                              smooth_factor: float, center: Optional[Point] = None) -> Polygon:
        """
        Needle override: dispatch to an angle-varying offset when the layer
        dict carries a ``"thickness_profile"`` entry; otherwise fall back to
        the shared uniform-buffer behavior (every other layer, unchanged).
        """
        profile = layer.get("thickness_profile")
        if not profile:
            return super()._offset_layer_polygon(polygon, distance, layer, smooth_factor, center=center)

        offset_center = center if center is not None else polygon.centroid
        offset_fn = lambda angle: distance * self._angular_multiplier(profile, angle)
        return GeometryProcessor.variable_buffer_polygon(
            polygon, offset_center, offset_fn, smooth_factor=smooth_factor
        )

    def _resolved_dimensions(self) -> tuple:
        """Resolve the needle's actual (width, thickness), computing from the
        layer stack whichever of the two global params was left at 0.
        Shared by ``_create_base_shape`` and ``_pole_and_corner_angles``
        (both need the *real* extent, not just whatever's in global_params).
        """
        if self.global_params["width"] == 0 and self.global_params["thickness"] == 0:
            return self._calculate_needle_width(), self._calculate_needle_thickness()
        elif self.global_params["width"] == 0:
            return self._calculate_needle_width(), self.global_params["thickness"]
        elif self.global_params["thickness"] == 0:
            return self.global_params["width"], self._calculate_needle_thickness()
        else:
            return self.global_params["width"], self.global_params["thickness"]

    def _create_base_shape(self) -> Polygon:
        """
        Create the half-ellipse shape of a needle cross-section.

        Returns:
            Half-ellipse polygon
        """
        width, thickness = self._resolved_dimensions()
        return GeometryProcessor.half_ellipse_polygon(width, thickness)

    @staticmethod
    def pole_and_corner_angles(width: float, thickness: float) -> tuple:
        """Locate the needle cross-section's adaxial pole, abaxial pole, and
        two corners as polar angles (degrees) around the base shape's
        centroid -- the convention ``thickness_profile``/``zone_angles``
        entries use (see ``_angular_multiplier``/``_offset_layer_polygon``),
        and the single source of truth for every angle-based zone in this
        class, including directional stomata placement
        (``_directional_stomata_triplets``).

        The base half-ellipse (``GeometryProcessor.half_ellipse_polygon``) is
        flat at y=0 (adaxial edge, x in [-width/2, width/2]) and domed up to
        (0, thickness) (abaxial peak). Treated as a uniform lamina its
        centroid sits at y_c = k*thickness/(3*pi) above the flat edge (the
        half-disk-centroid formula has k=4; k=3.5 here tracks the actual
        padded "outside" polygon's centroid slightly better) -- a naive
        "0=adaxial, 180=abaxial, +-90=corners" guess is wrong for any but a
        specific aspect ratio, putting the poles at the corners instead for
        a needle this flat.
        """
        a = width / 2.0
        y_c = 3.5 * thickness / (3.0 * np.pi)
        adaxial_pole = np.degrees(np.arctan2(-y_c, 0.0)) % 360.0
        abaxial_pole = np.degrees(np.arctan2(thickness - y_c, 0.0)) % 360.0
        corner_pos = np.degrees(np.arctan2(-y_c, a)) % 360.0     # near 0/360 side
        corner_neg = np.degrees(np.arctan2(-y_c, -a)) % 360.0    # near 180 side
        return float(adaxial_pole), float(abaxial_pole), float(corner_pos), float(corner_neg)

    def reshape_layers(self, layers_polygons: List[LayerPolygon]) -> List[LayerPolygon]:
        """
        When "central_cylinder" has shape="ellipse", interpolate each layer
        polygon between the outer half-ellipse (t=0) and a full ellipse
        aligned with the endodermis layer (t=1).

        Layers from the outside down to the endodermis are gradually morphed.
        Layers inward from the endodermis (transfusion, parenchyma ...) are
        fully changed to fit inside the ellipse.
        """
        if self.central_cylinder_params.get("shape") != "ellipse":
            return layers_polygons

        if not layers_polygons:
            return layers_polygons

        # --- build the target ellipse ----------------------------------------
        # Use the layer_thickness and layer_length of the central cylinder as
        # the semi-axes of the target full ellipse.
        rx = self.central_cylinder_params["layer_length"] / 2
        ry = self.central_cylinder_params["layer_thickness"] / 2 
        n_pts = 120
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        ellipse_coords = [(rx * np.cos(a), ry * np.sin(a)) for a in angles]
        ellipse_coords = [(x, y + self.global_params["thickness"] / 2.2) for x, y in ellipse_coords]
        target_ellipse = GeometryProcessor.buffer_polygon(
            Polygon(ellipse_coords),
            0, smooth_factor=0.0
        )

        # --- find the index of the endodermis layer --------------------------
        layer_names = [lp["name"] for lp in layers_polygons]
        
        endo_idx = layer_names.index("endodermis")

        # outside polygon (index 0) is the reference half-ellipse shape; we
        # keep it as-is (t=0) and warp everything inward up to endo_idx (t=1).
        outer_poly = layers_polygons[0]["polygon"]

        # Pre-compute one interpolator between the outer shape and the ellipse.
        try:
            interp = PolygonInterpolator(outer_poly, target_ellipse)
        except Exception:
            # If PolygonInterpolator fails (degenerate geometry), skip reshape.
            return layers_polygons

        n_to_morph = endo_idx + 1  # indices 0 ... endo_idx inclusive
        
        for i in range(1, n_to_morph):          # skip index 0 (outside)
            t = i / max(n_to_morph - 1, 1)     # 0 < t <= 1
            try:
                new_poly = interp.fast_interpolate(t)
                if not new_poly.is_empty and new_poly.is_valid:
                    layers_polygons[i] = dataclasses.replace(layers_polygons[i], polygon=new_poly)
            except Exception:
                pass  # leave this layer polygon unchanged on error

        layers_polygons = layers_polygons[:endo_idx+1]  # remove layers after endodermis

        layers_polygons.extend(self._create_central_layers(target_ellipse, params= self.params))  # add new central layers

        return layers_polygons
    
    def _calculate_needle_width(self) -> float:
        """Calculate total needle width from layers."""
        # width of vascular cylinder
        width_vascular = self.central_cylinder_params["layer_length"]
        # width of all supplementary layers
        width_layer = 0
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, 'n_layers'):
                width_layer += layer.get_total_thickness()
            elif hasattr(layer, 'cell_diameter'):
                width_layer += layer.cell_diameter
        # thickness of vascular cylinder
        thickness_vascular = self.central_cylinder_params["layer_thickness"]
        # thickness of all supplementary layers which is equal to width_layer
        self.thickness_layer = width_layer
        thickness_total = (2 * self.thickness_layer) + thickness_vascular
        
        width = 2 * np.sqrt((width_vascular/2 + self.thickness_layer)**2 / 
                            (1-(self.thickness_layer/thickness_total)**2))

        return width
    
    def _calculate_needle_thickness(self) -> float:
        """Calculate total needle thickness from layers."""
        thickness = self.central_cylinder_params["layer_thickness"]
        
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, 'n_layers'):
                thickness += 2 * layer.get_total_thickness()
            elif hasattr(layer, 'cell_diameter'):
                thickness += 2 * layer.cell_diameter
        
        return thickness
    
    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[LayerPolygon]:
        """
        Create transfusion tissue and parenchyma layers.
        
        Args:
            current_polygon: Current inner polygon boundary
            params: Parameter dictionaries
        
        Returns:
            List of central layer polygon dictionaries
        """
        central_layers = []
        space_increment = self.central_cylinder_params["cell_diameter"] / 2
        transfusion_layers_remaining = self.transfusion_params["n_layers"]
        parenchyma_diameter = self.central_cylinder_params["cell_diameter"]

        self._transfusion_zone = None
        pack_circles = self.transfusion_params.get("pack_circles", False)
        if pack_circles:
            # Reserve the transfusion zone's depth here (so the parenchyma/
            # vascular region that follows starts in the right place), but
            # place the cells later via circle-packing (add_transfusion_tissue)
            # instead of one row of ring cells per nominal layer -- see that
            # method's docstring.
            nominal_diameter = self.transfusion_params.get("diameter_max", 0.05)
            transfusion_depth = transfusion_layers_remaining * nominal_diameter
            transfusion_outer = current_polygon
            if transfusion_depth > 0:
                shrunk = GeometryProcessor.buffer_polygon(
                    current_polygon, -space_increment - transfusion_depth, smooth_factor=0.6
                )
                if not shrunk.is_empty and shrunk.is_valid and shrunk.area > 0:
                    self._transfusion_zone = transfusion_outer.difference(shrunk)
                    current_polygon = shrunk
                    # Sized from the *upcoming* parenchyma ring's own (much
                    # smaller) cell scale, not the transfusion zone's own
                    # nominal_diameter -- using the latter left an oversized
                    # gap before the first parenchyma ring, whose seeds then
                    # had unusually large Voronoi territory (rendering as
                    # visibly bigger cells than every parenchyma ring after it).
                    space_increment = parenchyma_diameter / 2
            transfusion_layers_remaining = 0  # skip the ring-based branch below entirely

        tt_diameter = self.transfusion_params["tracheids_diameter"] if not pack_circles else 0.0
        tp_diameter = self.transfusion_params["parenchyma_diameter"] if not pack_circles else 0.0
        transfusion_type = self.transfusion_params.get("transfusion_type", False)
        tt_ratio = self.transfusion_params.get("transfusion_tracheids_ratio", 0.5)
        p_tt = tt_ratio / (1.0 + tt_ratio) if tt_ratio > 0 else 0.0

        i_layer = len(params)

        while current_polygon.area > (parenchyma_diameter / 2)**2 * np.pi:
            if transfusion_layers_remaining > 0:
                avg_diameter = (tp_diameter + tt_diameter) / 2
                transfusion_layers_remaining -= 1

                current_polygon = GeometryProcessor.buffer_polygon(
                    current_polygon,
                    -space_increment - avg_diameter / 2,
                    smooth_factor=0.6
                )
                if current_polygon.is_empty or not current_polygon.is_valid or current_polygon.area <= 0:
                    break  # the ring buffered down to nothing; stop before appending a degenerate polygon

                space_increment = avg_diameter / 2

                central_layers.append(LayerPolygon(
                    name="transfusion",
                    polygon=current_polygon,
                    cell_diameter=avg_diameter,
                    id_layer=i_layer + 1,
                    transfusion_type=transfusion_type,
                    tt_diameter=tt_diameter if transfusion_type else 0.0,
                    tp_diameter=tp_diameter if transfusion_type else 0.0,
                    p_tt=p_tt if transfusion_type else 0.0,
                ))
            else:
                # Parenchyma
                current_polygon = GeometryProcessor.buffer_polygon(
                    current_polygon,
                    -space_increment - parenchyma_diameter / 2,
                    smooth_factor=0.7
                )
                if current_polygon.is_empty or not current_polygon.is_valid or current_polygon.area <= 0:
                    break  # the ring buffered down to nothing; stop before appending a degenerate polygon

                space_increment = parenchyma_diameter / 2

                central_layers.append(LayerPolygon(
                    name="parenchyma",
                    polygon=current_polygon,
                    cell_diameter=parenchyma_diameter,
                    id_layer=i_layer + 1,
                ))
            
            i_layer += 1
        
        return central_layers
    
    def set_central_cylinder_params(self, **kwargs) -> None:
        """
        Update central cylinder parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        self.central_cylinder_params.update(kwargs)
        self._invalidate_geometry()
    
    def set_transfusion_params(self, **kwargs) -> None:
        """
        Update transfusion tissue parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        self.transfusion_params.update(kwargs)
        self._invalidate_geometry()

    def _which_layer_for_vascular(self, layers_polygons: List[LayerPolygon]):
        """
        Find the layer where vascular tissue will be allocated.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        """
        layer_for_vascular = [l["name"] for l in layers_polygons].index("parenchyma")
        polygon_for_vascular = layers_polygons[layer_for_vascular]["polygon"]
        return polygon_for_vascular
    
    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Needle vascular tissue as an inspectable recipe.

        Built and run by the shared ``Organ._create_vascular_tissue`` scaffold.

        The xylem / phloem / cambium / Strasburger grid packed into the two
        central ellipses is a **bespoke fill** — its "region" is two *oriented*
        ellipses carrying axis/angle metadata that the grid loop consumes, so it
        does not map onto the shape-first region+fill verbs.  It therefore stays a
        single ``special`` step, mirroring root's bespoke steps (metaxylem border,
        pizza-slice bundles).
        """
        recipe = TissueRecipe()
        recipe.special("vascular ellipse grid",
                       lambda: self.fit_vascular_elements(polygon),
                       produces=("xylem", "phloem", "cambium", "Strasburger cell", "Str. Interstitial cell"))
        return recipe

    def fit_vascular_elements(self, polygon):
        # from polygon, fit two ellipses
        rx = self.central_cylinder_params["vascular_width"]/2
        ry = self.central_cylinder_params["vascular_height"]/2
        angle = self.central_cylinder_params.get("vascular_angle")
        ellipses = GeometryProcessor.two_ellipses(polygon, rx, ry, angle=angle)
        self._parenchyma_pocket_zone = self._corner_parenchyma_pockets(polygon, ellipses)
        cells_in_ellipses, list_ellipses_polygons = self.vascular_elements_in_ellipses(ellipses)
        vascular_cm = CellManager()
        vascular_cm.cells = cells_in_ellipses
        self.vascular_cells = vascular_cm
        self.vascular_polygons = list_ellipses_polygons

    @staticmethod
    def _corner_parenchyma_pockets(polygon: Polygon, ellipses: List[Dict[str, Any]]) -> Optional[Polygon]:
        """The sliver of ``polygon`` left over beyond each ellipse's own
        outer (lateral, away-from-centre) side.

        Even after ``GeometryProcessor.push_ellipse_to_boundary`` slides an
        ellipse out to touch the region's real edge, its rounded outline
        can't perfectly fill a sharper corner -- the small crescent that's
        left over is exactly what should become Strasburger cell instead of
        parenchyma (see example/needle/Str place blue.png). Clipping to the
        half-plane on each ellipse's own outer side of its centre keeps just
        that corner sliver, excluding both the gap between the two ellipses
        and anything toward the needle's own centre.
        """
        minx, miny, maxx, maxy = polygon.bounds
        pad = max(maxx - minx, maxy - miny) + 1.0
        pockets = []
        for ellipse in ellipses:
            cx = ellipse["center"][0]
            if cx <= 0:
                half_plane = Polygon([(minx - pad, miny - pad), (cx, miny - pad),
                                      (cx, maxy + pad), (minx - pad, maxy + pad)])
            else:
                half_plane = Polygon([(cx, miny - pad), (maxx + pad, miny - pad),
                                      (maxx + pad, maxy + pad), (cx, maxy + pad)])
            pocket = polygon.difference(ellipse["polygon"]).intersection(half_plane)
            if not pocket.is_empty:
                pockets.append(pocket)
        return unary_union(pockets) if pockets else None

    def retag_corner_parenchyma(self) -> None:
        """Retag parenchyma cells in the corner pockets (see
        ``_corner_parenchyma_pockets``) to "Strasburger cell" -- a plain
        rename, existing cells kept at their existing size/position.
        """
        zone = getattr(self, "_parenchyma_pocket_zone", None)
        if zone is None or zone.is_empty:
            return
        for c in self.all_cells.get_cells_by_type("parenchyma"):
            if zone.contains(Point(c.x, c.y)):
                c.type = "Strasburger cell"


    def vascular_elements_in_ellipses(self, ellipses, debug=False):
        """Seed the rectangular xylem/phloem grid (with a central cambium row
        and interstitial Strasburger-lineage files) into each of the two
        vascular ellipses.

        Each grid coordinate is built in the ellipse's local frame, then rotated
        by the ellipse angle and translated to its centre (:func:`place`); cells
        that land inside the ellipse are kept.  A bespoke fill — there is no
        region to pack, the layout is an explicit grid.

        ``"Str. Interstitial cell"`` columns run through both the xylem
        *and* phloem rows of the grid (retyped from ordinary xylem/phloem),
        matching example/needle/vascular_ellipse.png. The other Strasburger-
        lineage cells -- the corner cluster outside the ellipse -- are
        handled separately, by retagging real parenchyma cells in place
        (see ``retag_corner_parenchyma``/``_corner_parenchyma_pockets``)
        rather than seeded here.
        """
        list_ellipses_polygons: List[Polygon] = []
        cells_in_ellipses: List[Cell] = []

        params_xylem       = next(p for p in self.params if p["name"] == "xylem")
        params_phloem      = next(p for p in self.params if p["name"] == "phloem")
        params_cambium     = next(p for p in self.params if p["name"] == "cambium")
        params_strasburger = next((p for p in self.params if p["name"] == "Strasburger cells"), None)
        xylem_rows       = params_xylem["n_files"]
        phloem_rows      = params_phloem["n_files"]
        xylem_cell_width = params_xylem["cell_diameter"]
        xylem_cluster_n  = int(params_xylem["n_clusters"])

        id_cell = 0
        for idx, ellipse in enumerate(ellipses):
            center = ellipse["polygon"].centroid
            rx, ry = ellipse["axes"]
            # `two_ellipses` mirrors the *outline* angle for the second
            # (right) ellipse across the vertical midline (180-angle) --
            # correct for the ellipse shape itself (symmetric under 180 deg
            # rotation), but the content grid below is built from an
            # explicit rotation, not a reflection, so reusing that mirrored
            # angle here rotates the right ellipse's internal xylem/phloem
            # layout 180 deg out of step with the left one's. Adding 180 deg
            # back for the first (left) ellipse only realigns the two,
            # independent of whatever angle central_cylinder.vascular_angle
            # is actually set to.
            content_angle_deg = ellipse["angle"] + (180.0 if idx == 0 else 0.0)
            angle  = np.deg2rad(content_angle_deg) - np.pi / 2
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # `angle` is built so that local_y always ends up aligned with
            # the ellipse's *actual* rx (major/width) direction (and local_x
            # with its ry/minor direction) once `place()` rotates+translates
            # it -- see `place()` below. The xylem/cambium/phloem stack is
            # radial (spans the bundle's short, ry axis) with repeated
            # column files spread along the long, rx axis, so rows are built
            # on local_x (sized by ry) and columns on local_y (sized by rx):
            # the reverse of a naive rx-for-rows/ry-for-columns assignment.
            xylem_cell_height  = (ry - params_cambium["cell_diameter"]) / xylem_rows
            phloem_cell_height = (ry - params_cambium["cell_diameter"]) / phloem_rows
            xylem_cell_diameter  = (xylem_cell_width + xylem_cell_height) / 2
            phloem_cell_diameter = (xylem_cell_width + phloem_cell_height) / 2
            strasburger_diameter = params_strasburger["cell_diameter"] if params_strasburger else xylem_cell_diameter

            n_xylem_width      = int(np.ceil(rx * 2 / xylem_cell_width))
            xylem_cluster_size = int(np.ceil(
                (rx * 2 - xylem_cell_width * (xylem_cluster_n - 1)) / (xylem_cell_width * xylem_cluster_n)
            ))
            temp_cluster_id    = xylem_cluster_size

            def place(local_x, local_y, cell_type, cell_diameter):
                """Tilt + translate a local grid coord; seed a cell if it lands in the ellipse."""
                nonlocal id_cell
                id_cell += 1
                tx = local_x * cos_a - local_y * sin_a + center.x
                ty = local_x * sin_a + local_y * cos_a + center.y
                if ellipse["polygon"].contains(Point(tx, ty)):
                    cells_in_ellipses.append(
                        Cell.radial(cell_type, tx, ty, cell_diameter, id_cell, center)
                    )

            for i in range(n_xylem_width + 1):
                col_y = i * xylem_cell_width - rx + xylem_cell_width / 2

                # Interstitial cluster files are the Strasburger lineage
                # running straight through the grid -- both the xylem *and*
                # phloem rows of that column are retyped (from ordinary
                # xylem/phloem), sized like the corner cluster below.
                is_interstitial = (temp_cluster_id == 0)
                xylem_type  = "Str. Interstitial cell" if is_interstitial else "xylem"
                phloem_type = "Str. Interstitial cell" if is_interstitial else "phloem"
                for j in range(xylem_rows + 1):
                    place(j * xylem_cell_height - ry + xylem_cell_height / 2, col_y,
                          xylem_type, strasburger_diameter if is_interstitial else xylem_cell_diameter)

                for j in range(1, phloem_rows + 1):
                    place(j * phloem_cell_height + phloem_cell_height / 2, col_y,
                          phloem_type, strasburger_diameter if is_interstitial else phloem_cell_diameter)

                if temp_cluster_id == 0:
                    temp_cluster_id = xylem_cluster_size + 1
                temp_cluster_id -= 1

                # cambium cell on the ellipse mid-line
                place(0, col_y, "cambium", xylem_cell_diameter)

            # The corner-pocket parenchyma (beyond the ellipse's own outer
            # edge) is retagged to "Strasburger cell" separately, once real
            # parenchyma cells exist to relabel -- see
            # NeedleAnatomy.retag_corner_parenchyma / _corner_parenchyma_pockets.
            list_ellipses_polygons.append(ellipse["polygon"])

            if debug:
                color_map = {"Strasburger cell": "red", "Str. Interstitial cell": "orange",
                            "xylem": "blue", "phloem": "green", "cambium": "yellow"}
                plt.plot(ellipse["polygon"].exterior.xy[0], ellipse["polygon"].exterior.xy[1])
                for cell in cells_in_ellipses:
                    plt.plot(cell.x, cell.y, "o", color=color_map[cell.type])
                plt.show()

        return cells_in_ellipses, list_ellipses_polygons

    def _organ_recipe(self) -> TissueRecipe:
        """Needle organ-specific tissues as a recipe of P2 special-tissue steps.

        Resin ducts and stomata are cell-relative post-fill placements
        (carved into existing cells); transfusion tissue (when
        ``transfusion_tissue.pack_circles`` is set) is a zone fill via
        circle-packing rather than a ring seed.

        ``add_stomata`` calls ``self.all_cells.recenter_cells()``, which
        shifts every *existing* cell's coordinates to the population mean but
        cannot retroactively shift ``self._transfusion_zone`` (a shapely
        polygon computed earlier, during layer-polygon construction, in the
        original un-recentred frame). "transfusion tissue" therefore has to
        run *before* "stomata": placing its cells while everything is still
        in that same original frame keeps them consistent with the zone, and
        the subsequent recenter shifts them correctly along with every other
        cell. Running it after "stomata" (as before) placed transfusion
        cells relative to the stale, un-shifted zone while the rest of the
        organ had already moved -- the two point clouds no longer shared an
        origin, and the downstream global Voronoi tessellation stretched
        cells across that gap into huge, wildly misplaced polygons.
        ``"layer-count zoning"`` reads ``cell.angle`` as set by
        ``recenter_cells``, so it must stay *after* "stomata".
        """
        recipe = TissueRecipe()
        recipe.special("resin ducts", self.add_canal,
                       produces=("resin duct", "duct"))
        recipe.special("transfusion tissue", self.add_transfusion_tissue,
                       produces=("transfusion parenchyma", "transfusion tracheid"))
        recipe.special("corner parenchyma to strasburger", self.retag_corner_parenchyma)
        recipe.special("stomata", self.add_stomata,
                       produces=("guard cell", "air space", "pore"))
        recipe.special("layer-count zoning", self._restrict_zoned_layers)
        return recipe

    # ------------------------------------------------------------------
    # Geometry helpers — pure computation, no cell placement
    # ------------------------------------------------------------------

    def _duct_zone_data(self, layers_polygons):
        """
        Compute resin duct geometry from layer polygons without placing cells.

        Returns (duct_data, rdp) where duct_data is a list of per-duct dicts:
          - "outer":        mask polygon used to remove existing cells
          - "ring":         parenchyma ring polygon (for resin-duct cell placement)
          - "canal":        inner lumen polygon (for duct cell placement)
          - "ring_center":  centroid of the fitted inner ellipse (angle reference)
        rdp is the resin_duct parameter dict.
        Returns ([], None) when there are no resin_duct params or no mesophyll layer.
        """
        rdp_list = [p for p in self.params if p["name"] == "resin_duct"]
        if not rdp_list:
            return [], None
        rdp = rdp_list[0]

        layer_names = [l["name"] for l in layers_polygons]
        # Match every mesophyll-family layer (plain "mesophyll" plus any
        # "mesophyll_*" variant, e.g. the adaxial-only extra ring), not just
        # the single literal "mesophyll" entry -- ducts should sit inside the
        # full combined mesophyll region. layers_polygons is ordered
        # outer-to-inner, so the first match bounds the combined zone and the
        # last match is its inner edge.
        mesophyll_idx = [i for i, n in enumerate(layer_names) if n == "mesophyll" or n.startswith("mesophyll_")]
        if not mesophyll_idx:
            return [], None

        mesophyll_polys = [layers_polygons[i]["polygon"] for i in mesophyll_idx]
        polygon_for_duct = mesophyll_polys[0].difference(mesophyll_polys[-1]) if len(mesophyll_polys) > 1 else mesophyll_polys[0]
        polygon_for_duct = polygon_for_duct.difference(
            GeometryProcessor.buffer_polygon(polygon_for_duct, -rdp["diameter"] * _DUCT_RING_BUFFER_FACTOR, 0)
        )

        n_canal = rdp["n_files"]
        if n_canal < 7:
            n_regions = 7
            add_duct = _DUCT_PLACEMENT_ORDER[:n_canal]
        else:
            n_regions = n_canal
            add_duct = list(range(n_regions))

        duct_data = []
        for slice_id, slice_polygon in enumerate(GeometryProcessor.pizza_slice(polygon_for_duct, n_regions)):
            if slice_id not in add_duct:
                continue
            duct_poly       = GeometryProcessor.fit_inner_ellipse(slice_polygon, rdp["diameter"] / 2)
            outer           = GeometryProcessor.buffer_polygon(duct_poly["polygon"],  rdp["cell_diameter"] / 2, 0)
            ring            = GeometryProcessor.buffer_polygon(duct_poly["polygon"], -(rdp["cell_diameter"] / 2) * _DUCT_RING_INNER_SHRINK)
            canal           = GeometryProcessor.buffer_polygon(ring,                 -rdp["cell_diameter"])
            duct_data.append({
                "outer":       outer,
                "ring":        ring,
                "canal":       canal,
                "ring_center": duct_poly["polygon"].centroid,
            })

        return duct_data, rdp

    @staticmethod
    def _stomata_carve_polygons(triplet_centers, sp, cell_diam):
        """
        Compute stomata geometry from triplet positions without placing cells.

        triplet_centers: list of ((px,py), (cx,cy), (nx,ny)) — prev/curr/next
                         epidermis seed positions for each stoma.
        sp:             stomata parameter dict.
        cell_diam:      epidermis cell diameter.

        Returns a list of (carve_poly, gc1, gc2, chamber, pore) tuples,
        one per successfully computed stoma.
        """
        results = []
        for k, (prev_xy, curr_xy, next_xy) in enumerate(triplet_centers):
            mock_cells = [
                Cell(x=prev_xy[0], y=prev_xy[1], diameter=cell_diam,
                     id_group=3 * k,     id_cell=3 * k,     type="epidermis"),
                Cell(x=curr_xy[0], y=curr_xy[1], diameter=cell_diam,
                     id_group=3 * k + 1, id_cell=3 * k + 1, type="epidermis"),
                Cell(x=next_xy[0], y=next_xy[1], diameter=cell_diam,
                     id_group=3 * k + 2, id_cell=3 * k + 2, type="epidermis"),
            ]
            try:
                results.append(CellGenerator.create_stomata(mock_cells, stomata_setting=sp))
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------
    # Intercellular air spaces (mesophyll-specific geometry)
    # ------------------------------------------------------------------

    def _apply_intercellular(self, ics: dict) -> None:
        """Needle override of the intercellular-space geometry.

        In the needle mesophyll the intercellular air spaces are small rhombic
        (diamond-shaped) lacunae seated *on the walls* between adjacent mesophyll 
        cells. This override intercepts the ``mesophyll`` tissue and builds those 
        wall-centred rhombi (see :meth:`_apply_mesophyll_wall_rhombi`); 
        any other tissue is delegated to the shared base implementation unchanged.
        """
        tissues = ics.get("tissue", [])
        if isinstance(tissues, str):
            tissues = [tissues]
        if "mesophyll" not in tissues:
            super()._apply_intercellular(ics)
            return

        smoothness = ics.get("smoothness", 0)
        if isinstance(smoothness, (int, float)):
            smoothness_per_tissue = [float(smoothness)] * len(tissues)
        else:
            smoothness_per_tissue = [float(s) for s in smoothness]
        smoothness_by_tissue = dict(zip(tissues, smoothness_per_tissue))

        # Build the wall-centred rhombic lacunae for the mesophyll only.
        self._apply_mesophyll_wall_rhombi(smoothness_by_tissue.get("mesophyll", 0.0))

        # Delegate the remaining tissues (if any) to the generic implementation.
        other_tissues = [t for t in tissues if t != "mesophyll"]
        if other_tissues:
            remaining = dict(ics)
            remaining["tissue"] = other_tissues
            remaining["smoothness"] = [smoothness_by_tissue[t] for t in other_tissues]
            super()._apply_intercellular(remaining)

    def _apply_mesophyll_wall_rhombi(self, smoothness: float) -> None:
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
            protect_topology=True,
        )

    @staticmethod
    def _iter_wall_segments(shared) -> List[LineString]:
        """Yield the 1-D wall segments from a cell/cell intersection geometry.

        Two adjacent Voronoi cells share their boundary as a ``LineString`` (or a
        ``MultiLineString``); point-only touches and degenerate parts are ignored.
        """
        walls: List[LineString] = []
        if isinstance(shared, LineString):
            if shared.length > 0:
                walls.append(shared)
        elif isinstance(shared, MultiLineString):
            walls.extend(g for g in shared.geoms if g.length > 0)
        elif isinstance(shared, GeometryCollection):
            for g in shared.geoms:
                if isinstance(g, (LineString, MultiLineString)):
                    walls.extend(NeedleAnatomy._iter_wall_segments(g))
        return walls

    @staticmethod
    def _wall_rhombus(wall: LineString, major_fraction: float, axis_ratio: float) -> Optional[Polygon]:
        """Build a small rhombus centred on ``wall`` and aligned with it.

        The rhombus is a four-vertex diamond whose principal diagonal is parallel
        to the wall and spans ``major_fraction`` of the wall length; the secondary
        (perpendicular) diagonal is the principal divided by ``axis_ratio``. 
        Returns ``None`` for walls too short.
        """
        length = wall.length
        if length <= 1e-9:
            return None

        # Wall midpoint (by arc length) and local orientation.
        mid = wall.interpolate(0.5, normalized=True)
        p0 = np.asarray(wall.coords[0])
        p1 = np.asarray(wall.coords[-1])
        direction = p1 - p0
        norm = np.hypot(direction[0], direction[1])
        if norm <= 1e-12:
            return None
        unit = direction / norm                       # along the wall
        perp = np.array([-unit[1], unit[0]])          # perpendicular to the wall

        half_major = 0.5 * length * major_fraction    # along the wall
        half_minor = half_major / axis_ratio          # across the wall
        if half_major <= 1e-9 or half_minor <= 1e-9:
            return None

        c = np.array([mid.x, mid.y])
        verts = [
            c + half_major * unit,    # tip along the wall (+)
            c + half_minor * perp,    # tip across the wall (+)
            c - half_major * unit,    # tip along the wall (-)
            c - half_minor * perp,    # tip across the wall (-)
        ]
        return Polygon([tuple(v) for v in verts])

    # ------------------------------------------------------------------
    # Cell-placement methods — call geometry helpers then place cells
    # ------------------------------------------------------------------

    def add_canal(self):
        """Add resin ducts (parenchyma ring + inner lumen) to the mesophyll.

        Geometry is computed here (organ-specific); the cell placement is the
        shared :func:`special_tissues.place_resin_duct`.
        """
        duct_data, rdp = self._duct_zone_data(self._layers_polygons)
        if not duct_data:
            return

        layer_for_duct = [l["name"] for l in self._layers_polygons].index("mesophyll")
        place_resin_duct(self.all_cells, duct_data, rdp, layer_for_duct)

    def add_transfusion_tissue(self):
        """Fill the reserved transfusion-tissue zone by circle-packing.

        Only runs when ``transfusion_tissue.pack_circles`` is set (see
        ``_create_central_layers``, which reserves ``self._transfusion_zone``
        instead of emitting one-row-per-ring "transfusion" layer polygons in
        that case). Transfusion cells are irregular and densely packed
        rather than a uniform ring, so :func:`tissue_class.fill_by_packing`
        (Apollonian circle packing, already used this way for vessels/rays
        elsewhere in the engine) is a better fit than the ring seeder.

        Two structural passes, tracheids first: transfusion tracheids are
        packed into the *full* zone, then transfusion parenchyma is packed
        into whatever's left over (``zone`` minus the tracheids' own
        footprint) -- parenchyma filling the gaps around an already-placed
        tracheid structure, rather than both types being packed as one
        uniform batch and relabeled after the fact. ``transfusion_tracheids
        _ratio`` (tracheid:parenchyma, default 1.0 = "same quantities" per
        the docstring) splits the overall ``proportion`` occupancy target
        between the two passes; the second pass's ``proportion`` is rescaled
        against the *remaining* region's own (usually smaller) area so the
        combined tissue still lands close to the original total occupancy.
        """
        zone = getattr(self, "_transfusion_zone", None)
        if zone is None or zone.is_empty:
            return

        tp = self.transfusion_params
        diameter_max = tp.get("diameter_max", 0.05)
        proportion = tp.get("proportion", 0.6)
        ratio = tp.get("transfusion_tracheids_ratio", 1.0)
        p_tt = ratio / (1.0 + ratio) if ratio > 0 else 0.0

        zone_area = zone.area
        remaining_zone = zone
        if p_tt > 0:
            tracheid_placed = fill_by_packing(
                self.all_cells, zone, "transfusion tracheid",
                rng=self.rng, diameter_max=diameter_max,
                proportion=proportion * p_tt, allow_ellipse=True,
            )
            if tracheid_placed:
                tracheid_union = unary_union([p for p, _, _ in tracheid_placed])
                remaining_zone = zone.difference(tracheid_union)

        if remaining_zone.is_empty or remaining_zone.area <= 0 or p_tt >= 1.0:
            return

        remaining_target_area = zone_area * proportion * (1.0 - p_tt)
        parenchyma_proportion = min(0.95, remaining_target_area / remaining_zone.area)
        if parenchyma_proportion <= 0:
            return
        fill_by_packing(
            self.all_cells, remaining_zone, "transfusion parenchyma",
            rng=self.rng, diameter_max=diameter_max,
            proportion=parenchyma_proportion, allow_ellipse=True,
        )

    def _restrict_zoned_layers(self) -> None:
        """Prune any layer carrying a ``zone_angles`` entry down to only the
        cells whose recentred polar angle falls inside the configured zone.

        Layers like ``hypodermis_corner`` or ``mesophyll_abaxial`` need a
        nonzero ``thickness_profile`` floor everywhere (see
        ``_offset_layer_polygon``) purely to keep ``CellGenerator``'s
        next-layer bleed clip from cropping the *neighboring* ring -- that
        floor was never meant to also seed a visible row of cells outside the
        layer's real zone (a corner wedge, or one half of the cross-section).
        This mirrors how ``DicotLeafAnatomy`` varies its palisade layer count
        by region (some rows genuinely don't exist outside their zone,
        leaf_class.py:862-965) without needing that mechanism's per-column
        ``fill_along`` seeding -- the ring geometry here is left completely
        untouched; only cell *presence* is restricted, via the same
        ``CellManager.remove_cells``/``cell.angle`` (set by
        ``recenter_cells``, already called earlier by ``add_stomata``) used
        for directional stomata placement.
        """
        for layer in self.layers:
            zone = layer.get("zone_angles")
            if not zone:
                continue
            cells = self.all_cells.get_cells_by_type(layer["name"])
            if not cells:
                continue
            to_remove = [c for c in cells
                        if not self._angle_in_zone(np.degrees(c.angle) % 360.0, zone)]
            if to_remove:
                self.all_cells.remove_cells(to_remove)

    @staticmethod
    def _angle_in_zone(angle_deg: float, zone: Dict[str, Any]) -> bool:
        if zone.get("mode") == "half":
            pole = float(zone["pole"]) % 360.0
            return NeedleAnatomy._circular_diff(angle_deg, pole) < 90.0
        half_width = float(zone.get("half_width", 15.0))
        return any(NeedleAnatomy._circular_diff(angle_deg, c) <= half_width
                   for c in zone.get("centers", []))

    @staticmethod
    def _circular_diff(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def _aerenchyma_target_denominator(self, n_files: int) -> float:
        return float(n_files ** 1.12 + 1)

    def add_stomata(self):
        """Add stomata to the needle epidermis.

        With ``n_files`` only, stomata are spread evenly around the whole
        epidermis ring (previous behavior, unchanged). With ``n_adaxial`` and
        ``n_abaxial`` both set, placement instead goes through
        :meth:`_directional_stomata_triplets` for a directional,
        corner-excluded layout.
        """
        self.all_cells.recenter_cells()
        stomata_params_list = [p for p in self.params if p["name"] == "stomata"]
        if not stomata_params_list:
            return

        sp = stomata_params_list[0]
        epidermis_cells = self.all_cells.get_cells_by_type("epidermis")
        if not epidermis_cells:
            return

        cell_diam = epidermis_cells[0].diameter
        n_adaxial = sp.get("n_adaxial")
        n_abaxial = sp.get("n_abaxial")

        if n_adaxial is not None and n_abaxial is not None:
            triplet_centers = self._directional_stomata_triplets(epidermis_cells, sp, n_adaxial, n_abaxial)
        else:
            n_stomata = sp["n_files"]
            # Sample n_stomata evenly spaced groups, avoiding the very ends
            indices = np.linspace(
                _STOMATA_SKIP_BORDER_PTS,
                len(epidermis_cells) - np.round(len(epidermis_cells) / n_stomata),
                n_stomata, dtype=int
            )

            # Build triplet centroids from placed epidermis cell groups
            triplet_centers = []
            for i in indices:
                g = epidermis_cells[i].id_group
                try:
                    triplet_centers.append((
                        self.all_cells.get_centroid_of_group(g - 1),
                        self.all_cells.get_centroid_of_group(g),
                        self.all_cells.get_centroid_of_group(g + 1),
                    ))
                except KeyError:
                    pass  # adjacent group was removed; skip this stomata position

        stomata_geoms = self._stomata_carve_polygons(triplet_centers, sp, cell_diam)

        # Cell placement (guard cells + chamber + pore, carved into the
        # epidermis) is the shared special-tissue function.
        place_stomata(self.all_cells, stomata_geoms, sp, cell_diam)

    def _directional_stomata_triplets(self, epidermis_cells: List[Cell], sp: Dict[str, Any],
                                      n_adaxial: int, n_abaxial: int) -> List[tuple]:
        """Split epidermis groups into an adaxial (flat) and abaxial (domed)
        angular run around the needle centre, excluding a corner wedge from
        both, and pick ``n_adaxial``/``n_abaxial`` evenly spaced triplets.

        ``recenter_cells`` (called by ``add_stomata`` before this) already
        centres every cell on the organ's centroid and sets ``cell.angle =
        atan2(y, x)``, in the same frame ``pole_and_corner_angles`` measures
        -- used here as the single source of truth for where the adaxial/
        abaxial poles and the two corners actually sit (a fixed 0/90/180/270
        guess is only exactly right for one specific aspect ratio; this
        needle's actual corners sit well off 0/180, see
        ``pole_and_corner_angles``'s docstring).
        """
        width, thickness = self._resolved_dimensions()
        adaxial_pole, abaxial_pole, corner_pos, corner_neg = self.pole_and_corner_angles(width, thickness)

        group_angle: Dict[int, float] = {}
        for c in epidermis_cells:
            group_angle.setdefault(c.id_group, float(np.degrees(c.angle) % 360.0))

        centroid: Dict[int, Any] = {}
        for g in group_angle:
            try:
                centroid[g] = self.all_cells.get_centroid_of_group(g)
            except KeyError:
                pass  # adjacent group was removed

        # edge_margin (0-0.5, default 0.12) as a fraction of a quarter-turn:
        # how many degrees of wedge to exclude on each side of each corner.
        corner_half_width = float(sp.get("edge_margin", 0.12)) * 90.0
        corner_zone = {"mode": "wedge", "centers": [corner_pos, corner_neg], "half_width": corner_half_width}
        adaxial_zone = {"mode": "half", "pole": adaxial_pole}

        non_corner = [(g, a) for g, a in group_angle.items() if not self._angle_in_zone(a, corner_zone)]
        adaxial_run = sorted(g for g, a in non_corner if self._angle_in_zone(a, adaxial_zone))
        abaxial_run = sorted(g for g, a in non_corner if not self._angle_in_zone(a, adaxial_zone))

        return (self._pick_stomata_triplets(adaxial_run, centroid, n_adaxial)
                + self._pick_stomata_triplets(abaxial_run, centroid, n_abaxial))

    @staticmethod
    def _pick_stomata_triplets(run: List[int], centroid: Dict[int, Any], n: int) -> List[tuple]:
        """``n`` evenly spaced prev/curr/next triplets along one epidermis
        angular run (an adaxial or abaxial arc, corners already excluded)."""
        if n <= 0 or not run:
            return []
        picks = np.unique(np.linspace(0, len(run) - 1, n).astype(int))
        out = []
        for i in picks:
            g = run[i]
            if (g - 1) in centroid and (g + 1) in centroid:
                out.append((centroid[g - 1], centroid[g], centroid[g + 1]))
        return out

    # ------------------------------------------------------------------
    # Visualization hook
    # ------------------------------------------------------------------

    def _extra_tissue_polygons(self, layers_polygons):
        """
        Return resin-duct and stomata polygons for plot_tissues visualization,
        without placing any cell.

        Resin ducts are exact (same geometry as add_canal).
        Stomata positions are approximated from epidermis seed positions using
        the same index formula as add_stomata, so count and placement are
        consistent with generate_cells output.
        """
        extra = {}

        # Resin ducts — delegate entirely to the shared geometry helper
        duct_data, _ = self._duct_zone_data(layers_polygons)
        if duct_data:
            extra["resin_duct"]  = [d["outer"] for d in duct_data]
            extra["resin_canal"] = [d["canal"] for d in duct_data]

        # Stomata — approximate seed positions from the epidermis layer polygon
        stomata_params_list = [p for p in self.params if p["name"] == "stomata"]
        if stomata_params_list and layers_polygons:
            sp        = stomata_params_list[0]
            n_stomata = sp["n_files"]
            layer_names = [l["name"] for l in layers_polygons]
            if "epidermis" in layer_names:
                epi_layer  = layers_polygons[layer_names.index("epidermis")]
                cell_diam  = epi_layer.get("cell_diameter", 0.015)
                cell_width = epi_layer.get("cell_width", 0)
                shift      = epi_layer.get("shift", 0)

                seeds   = CellGenerator.cells_on_layer(epi_layer["polygon"], cell_diam, cell_width, shift, rng=self.rng)
                # n_border matches cell_border: 14 pts if rectangular, 9 if circular
                n_border    = 14 if cell_width != 0 else 9
                n_epi_cells = max(1, (len(seeds) - 1) * n_border)

                # Mirror the index selection from add_stomata
                end_idx = n_epi_cells - int(np.round(n_epi_cells / n_stomata))
                indices = np.linspace(_STOMATA_SKIP_BORDER_PTS, end_idx, n_stomata, dtype=int)
                seed_indices = np.clip(indices // n_border, 0, len(seeds) - 2)

                triplet_centers = [
                    (
                        tuple(seeds[max(0, si - 1)]),
                        tuple(seeds[si]),
                        tuple(seeds[min(len(seeds) - 2, si + 1)]),
                    )
                    for si in seed_indices
                ]

                stomata_geoms = self._stomata_carve_polygons(triplet_centers, sp, cell_diam)
                extra["stomata"] = [geom[0] for geom in stomata_geoms]

        return extra
