"""Leaf cross-section anatomy.

A leaf is a flat *slab*, not a radial organ.  ``LeafAnatomy(input_data)`` is a
transparent factory: it returns a ``MonocotLeafAnatomy`` (planttype 1) or a
``DicotLeafAnatomy`` (planttype 2), both ``isinstance(obj, LeafAnatomy)``.

The base holds the shared slab geometry (ribbed lamina outline, the transverse
vein row, stomata) and reuses the whole cell pipeline (seeding, Voronoi,
intercellular spaces, gap fusing).  The subclasses supply only the **mesophyll
stack**:

* **monocot** — ``spongy / palisade / spongy``: concentric peeling of the
  wide-thin outline already stacks epidermis / spongy / palisade / spongy /
  epidermis top-to-bottom, so the palisade is just the central fill;
* **dicot** — dorsiventral ``palisade (adaxial) / spongy (abaxial)``: the core is
  split at the mid-plane, palisade filling the top half and spongy the bottom.

Veins are the idealized all-transverse view: every vein is a proper transverse
bundle (xylem adaxial, phloem abaxial), and each thickens the leaf into a rib.
See ``LEAF_PLAN.md``.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from shapely.geometry import Polygon, box, LineString
from shapely.ops import unary_union

from openalea.granap.organ_class import Organ
from openalea.granap.layer_class import Layer, LayerPolygon
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.input_data import OrganInputData
from openalea.granap.cell_class import Cell
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.tissue_class import TissueRecipe, fill_by_rings
from openalea.granap.vascular_bundle import build_bundle, _place_region_cell, _largest
from openalea.granap.special_tissues import place_stomata


def _leaf_planttype(input_data) -> int:
    """planttype (1 = monocot, 2 = dicot) from any accepted input form."""
    if isinstance(input_data, OrganInputData):
        params = input_data.to_dict_list()
    elif isinstance(input_data, list):
        params = input_data
    else:
        return 1
    pt = next((p for p in params if p["name"] == "planttype"), {})
    return int(pt.get("value", 1))


# ---------------------------------------------------------------------------
# Base class — shared slab geometry, acts as factory via __new__
# ---------------------------------------------------------------------------

class LeafAnatomy(Organ):
    """Leaf cross-section (base + factory).  Calling ``LeafAnatomy(input_data)``
    returns a ``MonocotLeafAnatomy`` or ``DicotLeafAnatomy`` per ``planttype``."""

    # Keep each peeled band's thickness exact (no ring smoothing), like the stem.
    LAYER_SMOOTH_FACTOR: float = 0.0

    def __new__(cls, input_data: Any = None, seed: Optional[int] = None):
        if cls is LeafAnatomy:
            actual = (DicotLeafAnatomy if _leaf_planttype(input_data) == 2
                      else MonocotLeafAnatomy)
            return super().__new__(actual)
        return super().__new__(cls)

    def __init__(self, input_data: Any = None, seed: Optional[int] = None):
        super().__init__(seed=seed)
        if isinstance(input_data, OrganInputData):
            self.params = input_data.to_dict_list()
        elif isinstance(input_data, list):
            self.params = input_data
        else:
            self.params = self._default_params()

        self._parse_params()
        self._initialize_default_layers()

    def _default_params(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _parse_params(self) -> None:
        self.global_params = self._get_param("planttype")
        self._palisade = self._get_param("palisade")
        self._spongy = self._get_param("spongy")
        self._mesophyll = self._get_param("mesophyll")
        # Reused by add_intercellular_spaces / fuse_gaps (empty here = no-ops).
        self.intercellular_spaces_params = [p for p in self.params
                                            if p["name"] == "inter_cellular_spaces"]
        self.aerenchyma_params = self._get_param("aerenchyma")
        # Peeled bands = ordered params (a central fill has no 'order').
        self.layers = sorted(
            [p for p in self.params if "order" in p and p["name"] != "palisade"],
            key=lambda x: float(x["order"]),
        )

    def _initialize_default_layers(self) -> None:
        for param in self.layers:
            self.layer_manager.add_layer(Layer.from_dict(param))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _bundle_specs(self) -> List[Dict[str, Any]]:
        """All ``vascular_bundle`` specs (vein size-classes), widest first so a big
        vein (midrib) is placed before the smaller ones yield around it."""
        specs = [p for p in self.params
                 if p["name"] == "vascular_bundle"
                 and (int(p.get("n_bundles", 0)) > 0 or len(p.get("x_positions", [])) > 0)]
        return sorted(specs, key=lambda s: -float(s.get("width", 0.12)))

    def _vein_layout(self) -> List[tuple]:
        """``(x, spec)`` for every vein along the mid-plane — the single source for
        the outline ribs and the vein placement.

        Each size-class spec places ``n_bundles`` by its ``placement``:
        ``center`` (a midrib at x=0), ``scatter`` (an interleaved even row, offset
        half a step), else ``even`` (endpoints included).  A candidate is dropped if
        it comes within a bundle-width of a wider vein already placed, so classes
        don't collide.
        """
        width = float(self.global_params.get("width", 4.0))
        placed: List[tuple] = []
        for spec in self._bundle_specs():
            n = int(spec.get("n_bundles", 0))
            span = float(spec.get("span_fraction", 0.75)) * width
            placement = spec.get("placement", "even")
            if placement == "explicit":
                xs = np.asarray(spec.get("x_positions", []), dtype=float)
            elif placement == "center":
                xs = np.linspace(-span / 2.0, span / 2.0, n) if n > 1 else np.array([0.0])
            elif placement == "scatter":
                step = span / n
                xs = -span / 2.0 + step * (np.arange(n) + 0.5)
            else:
                xs = np.linspace(-span / 2.0, span / 2.0, n) if n > 1 else np.array([0.0])
            w = float(spec.get("width", 0.12))
            for x in np.atleast_1d(xs):
                clear = all(abs(x - px) > 0.5 * (w + float(ps.get("width", 0.12))) + 0.02
                            for px, ps in placed)
                if clear:
                    placed.append((float(x), spec))
        return placed

    def _vein_x_positions(self) -> List[float]:
        """x of each vein (any size-class)."""
        return [x for x, _ in self._vein_layout()]

    def _mid_line(self, xg):
        """The lamina mid-line y(x): the centre of the leaf slab at each ``x``.

        Two optional displacements ride on it (both 0 by default = a flat, straight
        leaf on ``y=0``):

        * **twist** — a gentle sine (``twist_amplitude`` mm depth, ``twist_waves``
          half-bends) so the leaf is not perfectly straight;
        * **fold** — a smooth central *sag* (``fold_sag`` mm) that drops the whole
          slab below the tip-to-tip chord, deepest at the centre and easing to 0 at
          ``±fold_width/2`` (default the whole width).  With the thickness centred on
          the mid-line this makes the adaxial surface hug the chord while the abaxial
          swings out into a keel — a folded, keeled leaf (e.g. a snowdrop).
        """
        g = self.global_params
        xg = np.asarray(xg, dtype=float)
        width = float(g.get("width", 4.0))
        y = np.zeros_like(xg)
        amp = float(g.get("twist_amplitude", 0.0))
        if amp != 0.0:
            waves = float(g.get("twist_waves", 1.0))
            y = y + amp * np.sin(waves * np.pi * xg / (width / 2.0))
        sag = float(g.get("fold_sag", 0.0))
        if sag != 0.0:
            fw = float(g.get("fold_width", width)) or width   # 0 / unset => full width
            t = xg / (fw / 2.0)
            y = y - np.where(np.abs(t) < 1.0, sag * 0.5 * (1.0 + np.cos(np.pi * t)), 0.0)
        return y

    def _mid_at(self, x: float) -> float:
        return float(self._mid_line(np.array([x]))[0])

    def _half_profile(self, xg):
        """Half the lamina thickness at each ``xg``.

        Default is the ellipse profile.  ``thickness_profile`` (a list of ``[x,
        thickness]`` control points from the centre outward, mm) overrides it with a
        custom, interpolated profile (e.g. a thick-keeled snowdrop leaf); thickness is
        symmetric in ``x`` and 0 beyond the last point (the tip)."""
        g = self.global_params
        xg = np.asarray(xg, dtype=float)
        prof = g.get("thickness_profile")
        if prof:
            px = np.array([p[0] for p in prof], dtype=float)
            pt = np.array([p[1] for p in prof], dtype=float)
            return 0.5 * np.interp(np.abs(xg), px, pt, left=float(pt[0]), right=0.0)
        width = float(g.get("width", 4.0))
        thickness = float(g.get("thickness", 0.7))
        return (thickness / 2.0) * np.sqrt(np.clip(1.0 - (2.0 * xg / width) ** 2, 0.0, 1.0))

    def _half_thickness_at(self, x: float) -> float:
        """Half the lamina thickness at ``x``."""
        return float(self._half_profile(np.array([x]))[0])

    def _create_base_shape(self) -> Polygon:
        """The lamina outline: a wide, thin lens, ribbed where each vein thickens it.

        The base is an ellipse (width along x, thickness along y); each vein adds an
        **independent** raised-cosine bump to the top (adaxial) and bottom (abaxial)
        profile, using **its own size-class's** height and full width
        (``rib_adaxial_height`` / ``rib_adaxial_width`` and the abaxial pair).  The
        bump has compact support (exactly 0 beyond ±width/2), so a rib can be
        one-sided (a height of 0), the two faces can differ, a big midrib and small
        minor veins each get their own rib, and adjacent ribs never overlap.  No rib
        (or no veins) -> a plain lens.
        """
        g = self.global_params
        width = float(g.get("width", 4.0))
        thickness = float(g.get("thickness", 0.7))
        layout = self._vein_layout()

        xg = np.linspace(-width / 2.0, width / 2.0, 600)
        half = self._half_profile(xg)                  # ellipse or custom thickness profile
        mid = self._mid_line(xg)                       # twist: curved mid-line

        def cos_bump(x0, height, full_width):
            """Raised-cosine bump at ``x0``: ``height`` at the vein, 0 beyond ±width/2."""
            if height <= 0.0 or full_width <= 0.0:
                return np.zeros_like(xg)
            t = (xg - x0) / (full_width / 2.0)
            return np.where(np.abs(t) < 1.0, height * 0.5 * (1.0 + np.cos(np.pi * t)), 0.0)

        top, bot, any_rib = mid + half, mid - half, False
        for x, spec in layout:
            ad = cos_bump(x, float(spec.get("rib_adaxial_height", 0.0)),
                          float(spec.get("rib_adaxial_width", 0.25)))
            ab = cos_bump(x, float(spec.get("rib_abaxial_height", 0.0)),
                          float(spec.get("rib_abaxial_width", 0.30)))
            top = top + ad
            bot = bot - ab
            any_rib = any_rib or bool(np.any(ad)) or bool(np.any(ab))

        # A folded leaf sags via the mid-line (see ``_mid_line``), so ``mid`` is
        # already non-zero and the ellipse shortcut below is skipped for it.
        if not any_rib and not bool(np.any(mid)) \
                and not self.global_params.get("thickness_profile"):
            # angle_deg=90 maps width->x, thickness->y (GeometryProcessor.oriented_ellipse).
            return self._round_edges(GeometryProcessor.oriented_ellipse(0.0, 0.0, width, thickness, 90.0))
        coords = np.vstack([np.column_stack([xg, top]),
                            np.column_stack([xg[::-1], bot[::-1]])])
        return self._round_edges(Polygon(coords).buffer(0))

    def _round_edges(self, poly: Polygon) -> Polygon:
        """Round sharp convex features (the pointed margins/tips and a narrow rib
        crest) with a morphological *opening* of radius ``edge_radius`` (mm), so the
        thin leaf edges don't render as spikes.  ``edge_radius`` 0 (default) = no
        change; a small value (~half an epidermis cell) blunts the tips and the keel
        point while leaving the broad profile intact."""
        r = float(self.global_params.get("edge_radius", 0.0))
        if r <= 0.0 or poly.is_empty:
            return poly
        opened = poly.buffer(-r).buffer(r)
        if opened.is_empty or opened.area <= 0:
            return poly
        if opened.geom_type == "MultiPolygon":
            opened = max(opened.geoms, key=lambda p: p.area)
        return opened

    def reshape_layers(self, layers_polygons: List[LayerPolygon]) -> List[LayerPolygon]:
        return layers_polygons

    def generate_layer_polygons(self) -> List[LayerPolygon]:
        """Same as the base, but keep each layer a single Polygon.

        Peeling the ribbed (scalloped) outline — or the mid-plane split — can pinch a
        layer into a main piece plus tiny slivers; the cell seeder traces a single
        exterior, so we keep the largest piece and let ``fuse_gaps`` absorb the
        dropped slivers into their neighbours.
        """
        lps = super().generate_layer_polygons()
        for lp in lps:
            g = lp.polygon
            if g is not None and g.geom_type == "MultiPolygon":
                lp.polygon = max(g.geoms, key=lambda p: p.area)
        return lps

    def _which_layer_for_vascular(self, layers_polygons: List[LayerPolygon]):
        """Region veins occupy — the innermost mesophyll band at the mid-plane."""
        real = [l for l in layers_polygons if l["name"] != "outside"]
        return real[-1]["polygon"] if real else self.generate_base_shape()

    def _peel_region(self, region, name: str, pparams: dict,
                     out: List[LayerPolygon], i_layer: int) -> int:
        """Peel ``region`` inward into ``name`` rings (appended to ``out``); returns
        the next layer index.  In the wide-thin lamina the concentric rings collapse
        into ~horizontal rows."""
        if region is None or region.is_empty:
            return i_layer
        pparams = pparams or {}
        cell_d = float(pparams.get("cell_diameter", 0.035))
        cell_w = float(pparams.get("cell_width", cell_d))
        cur = region
        space = cell_d / 2.0
        while not cur.is_empty and cur.area > (cell_d / 2.0) ** 2 * np.pi:
            cur = GeometryProcessor.buffer_polygon(cur, -space - cell_d / 2.0, smooth_factor=0.0)
            if cur.is_empty or cur.area <= 0:
                break
            space = cell_d / 2.0
            out.append(LayerPolygon(name=name, polygon=cur, cell_diameter=cell_d,
                                    cell_width=cell_w, id_layer=i_layer + 1))
            i_layer += 1
        return i_layer

    def _create_central_layers(self, current_polygon, params):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Veins (idealized all-transverse: an even row along the mid-plane)
    # ------------------------------------------------------------------

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """One step that lays the row of transverse veins (empty when none)."""
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        if not self._bundle_specs():
            return recipe
        recipe.special(
            "leaf veins", lambda: self._build_veins(),
            produces=("xylem", "phloem", "sieve element", "companion cell",
                      "cambium", "sclerenchyma", "bundle sheath", "parenchyma",
                      "air space"))
        return recipe

    def _build_veins(self) -> None:
        """Place every vein transversely along the mid-plane (y=0), each built from
        its own size-class spec.

        ``theta`` points to the abaxial (lower) surface, so with ``phloem_outward``
        the phloem faces the lower surface and the xylem the adaxial (upper) one —
        the idealized leaf view.
        """
        xylem, phloem, cambium = (self._get_param("xylem"),
                                  self._get_param("phloem"),
                                  self._get_param("cambium"))
        theta = -np.pi / 2.0            # local +y (phloem pole) -> abaxial (down)
        outline = self.generate_base_shape()
        placed = []                    # (x, y, spec, envelope) for the girders
        for x, spec in self._vein_layout():
            y = self._vein_y(float(x), spec)
            res = build_bundle(self.vascular_cells, self.rng, float(x), y, theta,
                               spec, xylem, phloem, cambium,
                               ground_cell_size=None, sheath_outline=outline)
            self._register_bundle(res)
            placed.append((float(x), y, spec, res.envelope))
        self._build_inter_bundle_aerenchyma()
        self._build_bundle_girders(placed, outline)

    def _vein_y(self, x: float, spec: dict) -> float:
        """The vein's y, shifted by its ``relative_distance`` toward a face: 0.5 =
        mid-plane, 1 = adaxial (up), 0 = abaxial (down).  Measured against the **local
        ribbed** thickness (the outline includes each vein's own rib/keel), so the
        relative position accounts for the bump; the vein stays inside the lamina."""
        rel = float(spec.get("relative_distance", 0.5))
        vh = float(spec.get("height", 0.2))
        surf = self._surface_y(x, self.generate_base_shape())
        if surf is None:                                  # off the lamina: fall back
            room = max(2.0 * self._half_thickness_at(x) - vh, 0.0)
            return self._mid_at(x) + (rel - 0.5) * room
        bot_y, top_y = surf
        mid_local = 0.5 * (bot_y + top_y)                 # centre of the ribbed slab here
        room = max((top_y - bot_y) - vh, 0.0)             # travel that keeps the vein inside
        return mid_local + (rel - 0.5) * room

    def _build_inter_bundle_aerenchyma(self) -> None:
        """Compute the gap region **between** each pair of adjacent veins (the monocot
        ``inter_bundle_aerenchyma`` param; a no-op when absent) and stash it for the
        post-Voronoi conversion pass.

        Unlike the bundle lacunae (drawn as one cell up front), the inter-bundle
        aerenchyma follows the **root** model: the mesophyll tessellates the gap
        normally, then :meth:`_convert_inter_bundle_aerenchyma` turns the mesophyll
        cells inside the region into ``air space`` and fuses them, so the lacuna
        takes an organic outline from the cell walls instead of a hard box.  Here we
        only record the regions; nothing is removed yet.
        """
        self._inter_bundle_regions = []
        p = self._get_param("inter_bundle_aerenchyma")
        if not p:
            return
        bundles = sorted((e for e in self.vascular_tissue_polygons.get("bundle", [])
                          if e is not None and not e.is_empty), key=lambda e: e.centroid.x)
        if len(bundles) < 2:
            return
        outline = self.generate_base_shape()
        env_union = unary_union(bundles)
        # Margin mode: a lacuna filling the gap between two bundles, held ``side_margin``
        # off each bundle and ``adaxial_margin`` / ``abaxial_margin`` off the two faces.
        # Ellipse mode (``width`` / ``height``): a fixed lacuna on the mid-line.  Typed
        # params always carry every key, so pick the mode by whether any margin is set.
        margins = any(float(p.get(k, 0.0)) > 0.0
                      for k in ("side_margin", "adaxial_margin", "abaxial_margin"))
        for e0, e1 in zip(bundles[:-1], bundles[1:]):
            xm = 0.5 * (e0.centroid.x + e1.centroid.x)
            if margins:
                side = float(p.get("side_margin", 0.05))
                x_lo, x_hi = e0.bounds[2] + side, e1.bounds[0] - side
                surf = self._surface_y(xm, outline)
                if surf is None or x_hi <= x_lo:
                    continue
                bot_y, top_y = surf
                y_lo = bot_y + float(p.get("abaxial_margin", 0.08))
                y_hi = top_y - float(p.get("adaxial_margin", 0.10))
                if y_hi <= y_lo:
                    continue
                # An ellipse inscribed in the margin box, so the fused lacuna is
                # rounded (root-style) rather than a hard rectangle.  The margins
                # only set roughly how many mesophyll cells hug the bundle/faces.
                region = GeometryProcessor.oriented_ellipse(
                    0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi),
                    x_hi - x_lo, y_hi - y_lo, 90.0).intersection(outline)
            else:
                region = GeometryProcessor.oriented_ellipse(
                    xm, self._mid_at(xm), float(p.get("width", 0.10)),
                    float(p.get("height", 0.24)), 90.0).intersection(outline)
            region = _largest(region.difference(env_union))
            if region is None or region.is_empty or region.area <= 0:
                continue
            self._inter_bundle_regions.append(region)

    def _mesophyll_tissue_name(self) -> str:
        """The ground tissue the inter-bundle aerenchyma is carved from."""
        p = self._get_param("inter_bundle_aerenchyma") or {}
        return str(p.get("tissue", "mesophyll"))

    def add_intercellular_spaces(self):
        """Run the shared intercellular / aerenchyma pass, then convert the
        inter-bundle mesophyll into fused air lacunae (root-style)."""
        super().add_intercellular_spaces()
        self._convert_inter_bundle_aerenchyma()

    def _convert_inter_bundle_aerenchyma(self) -> None:
        """Turn the mesophyll cells inside each stored inter-bundle region into a
        single fused ``air space`` cell — the root aerenchyma model (convert existing
        cells + fuse touching air), so the lacuna outline follows the cell walls."""
        regions = getattr(self, "_inter_bundle_regions", None)
        if not regions:
            return
        meso = self._mesophyll_tissue_name()
        for region in regions:
            group = [c for c in self.all_cells.cells
                     if c.type == meso and c.polygon is not None
                     and region.contains(c.polygon.representative_point())]
            if not group:
                continue
            fused = unary_union([c.polygon for c in group])
            if fused.is_empty or fused.area <= 0:
                continue
            parts = list(fused.geoms) if fused.geom_type == "MultiPolygon" else [fused]
            self.all_cells.remove_cells(group)
            id_cell = min(c.id_cell for c in group)
            id_layer = min(c.id_layer for c in group)
            id_group = min(c.id_group for c in group)
            for part in parts:                       # one air cell per connected piece
                if part.is_empty or part.area <= 0:
                    continue
                self.all_cells.cells.append(Cell(
                    x=part.centroid.x, y=part.centroid.y,
                    diameter=np.sqrt(part.area / np.pi) * 2,
                    id_cell=id_cell, id_layer=id_layer, id_group=id_group,
                    type="air space", polygon=part,
                ))

    def _surface_y(self, x: float, outline: Polygon):
        """(bottom_y, top_y) of the lamina outline at abscissa ``x`` (a vertical cut)."""
        miny, maxy = outline.bounds[1], outline.bounds[3]
        seg = LineString([(x, miny - 1.0), (x, maxy + 1.0)]).intersection(outline)
        if seg.is_empty:
            return None
        return seg.bounds[1], seg.bounds[3]

    def _build_bundle_girders(self, placed, outline: Polygon) -> None:
        """Sclerenchyma girder rays from a bundle to the epidermis.

        For a bundle whose spec sets ``girder_adaxial`` / ``girder_abaxial`` (True),
        a **triangle** is filled with sclerenchyma between the epidermis and the vein:
        its base (width ``girder_base_width``) sits against the inner edge of that
        face's epidermis, its apex touches the bundle pole.  A pronounced rib raises
        the surface, so the same mechanism fills the rib — a girder can equally sit
        on a bump.  The triangle region is registered so the mesophyll under it is
        cleared.  A no-op for bundles without a girder flag.
        """
        epi = self._get_param("epidermis")
        epi_th = float(epi.get("cell_diameter", 0.03)) * float(epi.get("n_layers", 1)) if epi else 0.03
        for x, y, spec, env in placed:
            if env is None or env.is_empty:
                continue
            if not (spec.get("girder_adaxial", False) or spec.get("girder_abaxial", False)):
                continue
            surf = self._surface_y(x, outline)
            if surf is None:
                continue
            bot_y, top_y = surf
            _, e_lo, _, e_hi = env.bounds
            bw = float(spec.get("girder_base_width", 0.10))
            cell_d = float(spec.get("girder_cell_diameter",
                                    spec.get("sclerenchyma_cell_diameter", 0.012)))
            cell_w = float(spec.get("girder_cell_width", cell_d))
            for side in ("adaxial", "abaxial"):
                if not spec.get("girder_" + side, False):
                    continue
                if side == "adaxial":
                    apex, base_y = (x, e_hi), top_y - epi_th        # up toward upper face
                    if base_y <= e_hi:
                        continue
                else:
                    apex, base_y = (x, e_lo), bot_y + epi_th        # down toward lower face
                    if base_y >= e_lo:
                        continue
                tri = _largest(Polygon([apex, (x - bw / 2.0, base_y),
                                        (x + bw / 2.0, base_y)]).buffer(0).intersection(outline))
                if tri is None or tri.is_empty or tri.area <= 0:
                    continue
                fill_by_rings(self.vascular_cells, tri, cell_d, cell_w, "sclerenchyma",
                              x, y, self.vascular_cells.next_group_id(), erosion_polygon=tri)
                self.vascular_tissue_polygons.setdefault("sclerenchyma", []).append(tri)

    def _register_bundle(self, res) -> None:
        """Record one built vein for the removal mask and the tissue view."""
        if res.envelope is not None and not res.envelope.is_empty:
            self.vascular_tissue_polygons.setdefault("bundle", []).append(res.envelope)
        for role, geom in res.zone_polygons:
            if geom is not None and not geom.is_empty:
                self.vascular_tissue_polygons.setdefault(role, []).append(geom)
        self.vascular_polygons.extend(res.vessel_polygons)

    # ------------------------------------------------------------------
    # Stomata (reuses the needle stomata machinery)
    # ------------------------------------------------------------------

    def _organ_recipe(self) -> TissueRecipe:
        recipe = TissueRecipe()
        if self._get_param("stomata"):
            recipe.special("stomata", self.add_stomata,
                           produces=("guard cell", "air space", "pore"))
        return recipe

    def add_stomata(self) -> None:
        """Carve stomata into both epidermes.

        Epidermis cells are seeded along the outline in ``id_group`` order, so
        consecutive groups are boundary neighbours.  We split the boundary into the
        adaxial (upper) and abaxial (lower) runs and place ``n_adaxial`` /
        ``n_abaxial`` stomata evenly along each — so a monocot leaf is amphistomatous
        and a dicot carries fewer on the adaxial side.  Each stoma is a
        prev/curr/next triplet, the same input the needle feeds to
        ``create_stomata`` / ``place_stomata``.
        """
        sp = self._get_param("stomata")
        if not sp:
            return
        epi = self.all_cells.get_cells_by_type("epidermis")
        if not epi:
            return
        cell_diam = epi[0].diameter

        groups = defaultdict(list)
        for c in epi:
            groups[c.id_group].append(c)
        centroid = {g: (float(np.mean([c.x for c in cs])),
                        float(np.mean([c.y for c in cs]))) for g, cs in groups.items()}
        ordered = sorted(centroid)                          # ~ boundary order
        # Split by the local mid-line, not y=0: a folded leaf sags entirely below the
        # chord, so "upper surface" means above the mid-line, not above y=0.
        adaxial = [g for g in ordered if centroid[g][1] > self._mid_at(centroid[g][0])]
        abaxial = [g for g in ordered if centroid[g][1] < self._mid_at(centroid[g][0])]

        default_n = int(sp.get("n_files", 10))
        margin = float(sp.get("edge_margin", 0.12))
        triplets = (self._pick_triplets(adaxial, centroid, int(sp.get("n_adaxial", default_n)), margin)
                    + self._pick_triplets(abaxial, centroid, int(sp.get("n_abaxial", default_n)), margin))

        geoms = self._stomata_geoms(triplets, sp, cell_diam)
        place_stomata(self.all_cells, geoms, sp, cell_diam)

    @staticmethod
    def _pick_triplets(run, centroid, n, margin=0.12):
        """``n`` evenly spaced prev/curr/next triplets along one epidermis run.

        ``margin`` is the fraction of the run skipped at *each* end, so no stoma sits
        near the leaf tips — where the adaxial and abaxial runs meet, which is what
        put two stomata right next to each other at the margin.
        """
        if n <= 0 or len(run) < 3:
            return []
        inset = int(round(margin * len(run)))
        lo = max(1, inset)
        hi = min(len(run) - 2, len(run) - 1 - inset)
        if hi < lo:
            return []
        picks = np.unique(np.linspace(lo, hi, min(n, hi - lo + 1)).astype(int))
        out = []
        for i in picks:
            g = run[i]
            if (g - 1) in centroid and (g + 1) in centroid:
                out.append((centroid[g - 1], centroid[g], centroid[g + 1]))
        return out

    @staticmethod
    def _stomata_geoms(triplets, sp, cell_diam):
        """(carve, gc1, gc2, chamber, pore) per stoma from prev/curr/next centroids."""
        results = []
        for k, (prev_xy, curr_xy, next_xy) in enumerate(triplets):
            mock = [
                Cell(x=prev_xy[0], y=prev_xy[1], diameter=cell_diam,
                     id_group=3 * k, id_cell=3 * k, type="epidermis"),
                Cell(x=curr_xy[0], y=curr_xy[1], diameter=cell_diam,
                     id_group=3 * k + 1, id_cell=3 * k + 1, type="epidermis"),
                Cell(x=next_xy[0], y=next_xy[1], diameter=cell_diam,
                     id_group=3 * k + 2, id_cell=3 * k + 2, type="epidermis"),
            ]
            try:
                results.append(CellGenerator.create_stomata(mock, stomata_setting=sp))
            except Exception:
                pass
        return results


# ---------------------------------------------------------------------------
# Monocot — spongy / palisade / spongy (palisade is the central fill)
# ---------------------------------------------------------------------------

class MonocotLeafAnatomy(LeafAnatomy):
    """Monocot leaf: one uniform mesophyll tissue fills the core (no palisade/spongy
    split), amphistomatous."""

    def _default_params(self) -> List[Dict[str, Any]]:
        return OrganInputData.for_monocot_leaf().to_dict_list()

    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[LayerPolygon]:
        """A single ``mesophyll`` tissue fills the whole core."""
        out: List[LayerPolygon] = []
        self._peel_region(current_polygon, "mesophyll", self._mesophyll, out, len(params))
        return out


# ---------------------------------------------------------------------------
# Dicot — dorsiventral palisade (adaxial) / spongy (abaxial)
# ---------------------------------------------------------------------------

class DicotLeafAnatomy(LeafAnatomy):
    """Dicot leaf: palisade under the upper (adaxial) epidermis, spongy above the
    lower (abaxial) one — the core split at the mid-plane."""

    def _default_params(self) -> List[Dict[str, Any]]:
        return OrganInputData.for_dicot_leaf().to_dict_list()

    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[LayerPolygon]:
        """Split the mesophyll core at the mid-line (where the veins sit — a straight
        line, or the twist curve) and fill the adaxial half with palisade, the abaxial
        half with spongy."""
        minx, miny, maxx, maxy = current_polygon.bounds
        xg = np.linspace(minx, maxx, 200)
        mid = self._mid_line(xg)
        # Region above the mid-line: the curve, then closed off along the top.
        above = Polygon(list(zip(xg, mid))
                        + [(maxx, maxy + 1.0), (minx, maxy + 1.0)]).buffer(0)
        top = current_polygon.intersection(above)          # adaxial half
        bot = current_polygon.difference(above)            # abaxial half
        out: List[LayerPolygon] = []
        i = len(params)
        i = self._peel_region(top, "palisade", self._palisade, out, i)
        i = self._peel_region(bot, "spongy", self._spongy, out, i)
        return out
