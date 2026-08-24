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
from openalea.granap.tissue_class import TissueRecipe, fill_by_rings, fill_along
from openalea.granap.vascular_bundle import (
    build_bundle, build_arc_bundle, _place_region_cell, _largest,
)
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

    # The lamina outline is built from a piecewise-linear thickness profile + cosine
    # ribs, which leave corners/kinks that force the boundary layer into stretched
    # cells.  Gaussian-smooth the two surface profiles over this many epidermis cells
    # before the outline is used to seed/limit cells.  A ``outline_smooth`` planttype
    # param overrides it with an absolute mm length; 0 disables (sharp corners kept).
    OUTLINE_SMOOTH_CELLS: float = 3.0

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

    def _mid_slope(self, x: float) -> float:
        """dy/dx of the lamina mid-line at ``x`` (central finite difference).

        Zero on a flat leaf; non-zero wherever the twist or the fold tilts the slab,
        which is exactly where a vein has to lean to stay square to the leaf plane."""
        h = max(float(self.global_params.get("width", 4.0)) * 1e-3, 1e-4)
        y = self._mid_line(np.array([x - h, x + h]))
        return float((y[1] - y[0]) / (2.0 * h))

    def _normal_theta(self, x: float) -> float:
        """World angle of a vein's abaxial (outward, lower-surface) pole at ``x``.

        A bundle's local +y points to the abaxial surface; :meth:`build_bundle` then
        orients the whole vein — xylem->phloem axis, ribs and all — about it.  On a
        flat leaf that pole is straight down (``-pi/2``).  On a folded or twisted leaf
        the slab tilts, so the pole follows the **outward normal of the mid-line**:
        perpendicular to the local tangent ``(1, dy/dx)`` and pointing to the abaxial
        side.  This swings each vein to stay perpendicular to the local leaf plane
        instead of every vein pointing top-to-bottom regardless of the fold."""
        m = self._mid_slope(x)
        return float(np.arctan2(-1.0, m))

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
            """Raised-cosine bump at ``x0``: ``height`` at the vein, 0 beyond ±width/2.

            A **negative** height dips the surface *inward* instead of out — an
            adaxial groove / sunken channel over a major vein (paired with a positive
            abaxial rib it makes the classic keeled, top-grooved midrib)."""
            if height == 0.0 or full_width <= 0.0:
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
                and not self.global_params.get("thickness_profile") \
                and self._outline_smooth_len() <= 0.0:
            # angle_deg=90 maps width->x, thickness->y (GeometryProcessor.oriented_ellipse).
            return self._round_edges(GeometryProcessor.oriented_ellipse(0.0, 0.0, width, thickness, 90.0))
        # The upper/lower profiles are assembled from a piecewise-linear thickness
        # profile plus cosine ribs, so they carry corners (at each profile control
        # point) and sharp rib shoulders.  Those become high-curvature kinks in the
        # outline, and since the outline is what limits cell growth (via the ring of
        # "outside" seeds sampled along it), a kink forces the boundary layer to
        # stretch one cell there.  Gaussian-smooth the two surfaces over ~a cell so the
        # corners become gentle curves the epidermis can tile — without moving the
        # broad shape.
        top, bot = self._smooth_surfaces(top, bot, width)
        coords = np.vstack([np.column_stack([xg, top]),
                            np.column_stack([xg[::-1], bot[::-1]])])
        return self._round_edges(Polygon(coords).buffer(0))

    def _outline_smooth_len(self) -> float:
        """Along-surface smoothing length (mm) for the outline: the ``outline_smooth``
        planttype param if given, else ``OUTLINE_SMOOTH_CELLS`` epidermis cells."""
        v = self.global_params.get("outline_smooth")
        if v is not None:
            return float(v)
        epi = self._get_param("epidermis") or {}
        return self.OUTLINE_SMOOTH_CELLS * float(epi.get("cell_diameter", 0.02))

    def _smooth_surfaces(self, top, bot, width: float):
        """Gaussian-smooth the adaxial/abaxial surface profiles along x, rounding the
        piecewise-linear corners and rib shoulders (see ``_create_base_shape``)."""
        smooth_len = self._outline_smooth_len()
        if smooth_len <= 0.0 or len(top) < 5:
            return top, bot
        from scipy.ndimage import gaussian_filter1d
        dx = float(width) / (len(top) - 1)
        sigma = smooth_len / dx if dx > 0 else 0.0
        if sigma < 0.3:
            return top, bot
        return (gaussian_filter1d(top, sigma, mode="nearest"),
                gaussian_filter1d(bot, sigma, mode="nearest"))

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
        """Same as the base, but split each multi-piece layer into one seedable
        Polygon per piece.

        A peeled band routinely pinches into several pieces: the wide-thin lamina is
        notched by the palisade band (so a spongy ring splits into a left arm and a
        right arm) or scalloped by the ribs.  The cell seeder traces a *single*
        exterior per layer, so a ``MultiPolygon`` layer has to be resolved into
        single Polygons — but keeping only the largest piece **deletes a whole side**
        of the leaf (e.g. the smaller spongy arm), which is what made the two halves
        carry a different number of layers.  Instead, every piece down to about one
        cell is emitted as its own layer (same tissue, same cell sizes), so both
        arms are seeded; only true sub-cell slivers are dropped for ``fuse_gaps`` to
        absorb into their neighbours."""
        import dataclasses
        lps = super().generate_layer_polygons()
        out: List[LayerPolygon] = []
        for lp in lps:
            g = lp.polygon
            if g is None or g.geom_type != "MultiPolygon":
                out.append(lp)
                continue
            # Drop only genuine slivers: pieces smaller than ~one of this layer's cells.
            min_area = np.pi * (float(lp.cell_diameter) / 2.0) ** 2
            pieces = sorted((p for p in g.geoms if not p.is_empty),
                            key=lambda p: -p.area)
            kept = [p for p in pieces if p.area >= min_area] or pieces[:1]
            for piece in kept:
                out.append(dataclasses.replace(lp, polygon=piece))
        return out

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
        """Lay the transverse vein row, then (dicot) the columnar palisade files."""
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        if self._bundle_specs():
            recipe.special(
                "leaf veins", lambda: self._build_veins(),
                produces=("xylem", "phloem", "sieve element", "companion cell",
                          "cambium", "sclerenchyma", "bundle sheath", "parenchyma",
                          "air space"))
        # Palisade is seeded as horizontal files (see _build_palisade), after the veins
        # so their envelopes can be carved out.  Only present when the subclass stashed
        # file lines in _create_central_layers (the dicot).
        if getattr(self, "_palisade_files", None):
            recipe.special("leaf palisade", lambda: self._build_palisade(),
                           produces=("palisade",))
        return recipe

    def _build_palisade(self) -> None:
        """Seed the columnar palisade as ``n_layers`` horizontal cell files hugging the
        adaxial surface (the lines stashed by the dicot ``_create_central_layers``).

        Each file is a row of tangentially-oriented cells (:func:`fill_along`): along a
        horizontal line the tangent is horizontal, so the cells stand up columnar
        (tall ``cell_diameter`` × narrow ``cell_width``).  Seeding independent rows —
        rather than peeling concentric rings (which shrink inward, so deeper files come
        out shorter) or adjacent slabs (whose shared edges collide into ballooning
        Voronoi cells) — gives files that are all the same length and stay bounded.  The
        vein envelopes are carved out so a file never runs through a bundle."""
        data = getattr(self, "_palisade_files", None)
        if not data:
            return
        lines, d, w = data
        envs = self.vascular_tissue_polygons.get("bundle", [])
        veinmask = unary_union(envs) if envs else None
        for line in lines:
            seg = line.difference(veinmask) if veinmask is not None else line
            if seg.is_empty:
                continue
            fill_along(self.vascular_cells, seg, "palisade", d, w,
                       line.centroid.x, line.centroid.y)

    def _build_veins(self) -> None:
        """Place every vein transversely along the mid-plane, each built from its own
        size-class spec.

        Each vein's ``theta`` points to the abaxial (lower) surface along the local
        outward normal of the mid-line (see :meth:`_normal_theta`), so on a folded or
        twisted leaf the vein leans with the slab instead of always pointing straight
        down; with ``phloem_outward`` the phloem then faces the abaxial surface and the
        xylem the adaxial one, square to the local leaf plane.
        """
        xylem, phloem, cambium = (self._get_param("xylem"),
                                  self._get_param("phloem"),
                                  self._get_param("cambium"))
        outline = self.generate_base_shape()
        ground = self._ground_cell_size_for_veins()
        placed = []                    # (x, y, spec, envelope) for the girders
        for x, spec in self._vein_layout():
            y = self._vein_y(float(x), spec)
            theta = self._normal_theta(float(x))   # abaxial pole = local mid-line normal
            if float(spec.get("arc_degrees", 0.0)) > 0.0:
                # A slice of a vascular cylinder (concentric xylem/cambium/phloem arcs).
                res = build_arc_bundle(self.vascular_cells, self.rng, float(x), y, theta,
                                       spec, xylem, phloem, cambium,
                                       ground_cell_size=ground, sheath_outline=outline)
            else:
                res = build_bundle(self.vascular_cells, self.rng, float(x), y, theta,
                                   spec, xylem, phloem, cambium,
                                   ground_cell_size=ground, sheath_outline=outline)
            self._register_bundle(res)
            placed.append((float(x), y, spec, res.envelope))
        self._build_inter_bundle_aerenchyma()
        self._build_bundle_girders(placed, outline)

    def _ground_cell_size_for_veins(self):
        """Ground (mesophyll) cell diameter the veins sit among — turns on the outer
        bundle sheath so a ring of intermediate-sized cells wraps each vein, instead
        of the small bundle parenchyma stretching out into the coarse mesophyll as a
        radial sunburst.  Uses the spongy size (the veins sit at the mid-plane in the
        spongy layer); falls back to the uniform mesophyll of a monocot leaf."""
        for p in (self._spongy, self._mesophyll):
            if p and float(p.get("cell_diameter", 0.0)) > 0:
                return float(p["cell_diameter"])
        return None

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

    def _add_aerenchyma_to_proportion(self, tol: float = 5e-3, max_iter: int = 20):
        """Leaf (slab) aerenchyma — replaces the root/stem radial mechanism.

        The base :meth:`Organ._add_aerenchyma_to_proportion` grows aerenchyma in
        *angular sectors* around the origin, which is meaningless on a flat lamina.
        A leaf's aerenchyma is instead **scattered**: irregular air lacunae spread
        through the ground tissue (dense in a dicot — "almost everywhere" — but
        never the monocot's tidy one-lacuna-per-vein-gap).  We convert a target
        area fraction of the ground tissue into air as random blobs, leaving thin
        tissue walls between them, then fuse touching air into organic lacunae.

        With no ``aerenchyma`` param this is just the shared fuse/simplify pass, so
        the plain intercellular-air behaviour is unchanged.
        """
        requested = float(self.aerenchyma_params.get("aerenchyma_proportion", 0.0)) \
            if self.aerenchyma_params else 0.0
        if requested > 0.0:
            self._scatter_leaf_aerenchyma(requested)
        # Fuse all touching air into lacunae (n_files=1 -> the whole slab is one
        # sector, so fusing follows geometry, not a radial grid).
        self._aerenchyma_n_files = 1
        self._aerenchyma_start_angle = 0.0
        self.merge_intercellular_aerenchyma()

    def _scatter_leaf_aerenchyma(self, proportion: float) -> None:
        """Convert **random** ground-tissue cells into air until their combined area
        reaches ``proportion`` of the tissue, then let the fuse pass merge whichever
        happen to touch.  Picking cells at random (rather than growing blobs) leaves
        the un-picked cells as scattered walls, so the air reads as many irregular
        lacunae instead of one solid void."""
        tissue = str(self.aerenchyma_params.get("tissue", "spongy"))
        cells = [c for c in self.all_cells.cells
                 if c.type == tissue and c.polygon is not None]
        if len(cells) < 2:
            return
        areas = np.array([c.polygon.area for c in cells])
        target = float(proportion) * float(areas.sum())

        order = self.rng.permutation(len(cells))
        cumulative = 0.0
        for i in order:
            if cumulative >= target:
                break
            cells[i].type = "air space"
            cumulative += float(areas[i])

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

        # Keep stomata off the undifferentiated-mesophyll bands over a major vein —
        # they belong over palisade / spongy tissue, not the vein's mesophyll region.
        # A one-cell margin keeps even a boundary stoma's guard cells off the band.
        meso_x = []
        for vx, spec in self._vein_layout():
            wr = float(spec.get("mesophyll_region_width", 0.0))
            if wr > 0.0:
                meso_x.append((vx - wr / 2.0 - cell_diam, vx + wr / 2.0 + cell_diam))
        if meso_x:
            def _off_meso(g):
                gx = centroid[g][0]
                return not any(lo <= gx <= hi for lo, hi in meso_x)
            adaxial = [g for g in adaxial if _off_meso(g)]
            abaxial = [g for g in abaxial if _off_meso(g)]

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
    """Dicot leaf: ``palisade.n_layers`` columnar palisade files hugging the adaxial
    (upper) surface, with spongy mesophyll filling the whole rest of the core."""

    def _default_params(self) -> List[Dict[str, Any]]:
        return OrganInputData.for_dicot_leaf().to_dict_list()

    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[LayerPolygon]:
        """Reserve an adaxial palisade band ``n_layers`` cells deep, fill the whole rest
        of the core with spongy, and stash the palisade file centre-lines for
        :meth:`_build_palisade` to seed.

        The palisade is *only* the band ``n_layers × cell_diameter`` deep measured down
        from the adaxial surface (per abscissa, so it rides the twist/fold); everything
        below it — the rest of the adaxial side and the whole abaxial side — is spongy.
        The palisade cells themselves are seeded later as horizontal files (see
        :meth:`_build_palisade`); here we only carve the band out of the spongy region
        and record one centre-line per file across the thick central span (the tapering
        margins are left to spongy, which fills them without the files pinching)."""
        minx, miny, maxx, maxy = current_polygon.bounds
        n_pal = max(int(self._palisade.get("n_layers", 2)), 0)
        out: List[LayerPolygon] = []
        i = len(params)
        self._palisade_files = None

        # A vein may replace the palisade/spongy around it with undifferentiated
        # mesophyll: a full-thickness tangential band (``mesophyll_region_width``)
        # centred on the vein, filled with mesophyll instead.  Carve these bands out of
        # the core up front so neither the palisade band nor the spongy peel enters
        # them, then fill each with its own mesophyll cell size.
        meso_region, meso_x = self._mesophyll_regions(current_polygon)
        core = current_polygon.difference(meso_region) if meso_region is not None else current_polygon
        for strip, spec in self._mesophyll_strips(current_polygon):
            i = self._peel_region(
                strip, "mesophyll",
                {"cell_diameter": float(spec.get("mesophyll_cell_diameter", 0.03)),
                 "cell_width": float(spec.get("mesophyll_cell_width", 0.03))},
                out, i)

        if n_pal <= 0:
            self._peel_region(core, "spongy", self._spongy, out, i)
            return out

        d = float(self._palisade.get("cell_diameter", 0.075))
        w = float(self._palisade.get("cell_width", d))

        xg = np.linspace(minx, maxx, 400)
        surf = [self._surface_y(float(x), current_polygon) for x in xg]
        top = np.array([(s or (miny, miny))[1] for s in surf])
        bot = np.array([(s or (miny, miny))[0] for s in surf])
        thickness = np.array([((s[1] - s[0]) if s else 0.0) for s in surf])

        # Palisade layers TAPER toward the margins instead of stopping abruptly: at each
        # abscissa we keep as many palisade files as fit under the adaxial surface (up to
        # ``n_pal``), so a thin margin gets 3, then 2, then 1 layer — but the adaxial side
        # stays palisade all the way out, never spongy.  Spongy fills only what is left
        # *below* the (tapering) band.
        n_col = np.clip(np.floor_divide(thickness, d).astype(int), 0, n_pal)
        # Never inside a mesophyll band (that whole column is undifferentiated mesophyll).
        in_meso = np.array([any(lo <= x <= hi for lo, hi in meso_x) for x in xg])
        n_col[in_meso] = 0
        if not (n_col >= 1).any():
            self._peel_region(core, "spongy", self._spongy, out, i)
            return out

        # The band's lower edge rides ``n_col`` cells below the surface (never past the
        # abaxial face), so it steps down 4->3->2->1 layers deep across the taper.
        band_lower = np.maximum(top - n_col * d, bot)
        band_polys: List[Polygon] = []
        run_idx: List[int] = []

        def _flush(idxs: List[int]) -> None:
            if len(idxs) < 2:
                return
            xs = xg[idxs]
            poly = Polygon(list(zip(xs, top[idxs]))
                           + list(zip(xs[::-1], band_lower[idxs][::-1]))).buffer(0)
            if not poly.is_empty:
                band_polys.append(poly)

        for j, ok in enumerate(n_col >= 1):
            if ok:
                run_idx.append(j)
            else:
                _flush(run_idx); run_idx = []
        _flush(run_idx)

        band = unary_union(band_polys).intersection(core) if band_polys else Polygon()
        spongy_region = core.difference(band)
        self._peel_region(spongy_region, "spongy", self._spongy, out, i)

        # One centre-line per palisade file, at k+0.5 cell-diameters below the surface;
        # file ``k`` runs only across the columns deep enough to hold that many layers
        # (``n_col > k``), so the deeper files are shorter and the taper emerges.
        lines = []
        for k in range(n_pal):
            cy = top - (k + 0.5) * d
            present = n_col > k
            run: List[tuple] = []
            for x, y, ok in zip(xg, cy, present):
                if ok:
                    run.append((float(x), float(y)))
                elif len(run) >= 2:
                    lines.append(LineString(run)); run = []
                else:
                    run = []
            if len(run) >= 2:
                lines.append(LineString(run))
        self._palisade_files = (lines, d, w) if lines else None
        return out

    def _mesophyll_strips(self, current_polygon: Polygon):
        """``(strip, spec)`` for each vein that requests an undifferentiated-mesophyll
        region: a full-thickness tangential band ``mesophyll_region_width`` wide centred
        on the vein, clipped to the lamina."""
        minx, miny, maxx, maxy = current_polygon.bounds
        strips = []
        for x, spec in self._vein_layout():
            wr = float(spec.get("mesophyll_region_width", 0.0))
            if wr <= 0.0:
                continue
            strip = box(x - wr / 2.0, miny - 1.0, x + wr / 2.0, maxy + 1.0) \
                .intersection(current_polygon)
            if not strip.is_empty and strip.area > 0:
                strips.append((strip, spec))
        return strips

    def _mesophyll_regions(self, current_polygon: Polygon):
        """Union of the mesophyll bands (or ``None``) and their ``(x_lo, x_hi)`` spans."""
        strips = self._mesophyll_strips(current_polygon)
        if not strips:
            return None, []
        region = unary_union([s for s, _ in strips])
        spans = [tuple(s.bounds[i] for i in (0, 2)) for s, _ in strips]
        return region, spans
