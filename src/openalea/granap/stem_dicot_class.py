"""Dicot stem anatomy (eustele).

``DicotStemAnatomy`` builds a dicot stem: a central pith ringed by a single ring
of discrete *collateral* vascular bundles — xylem on the inner (pith) face,
phloem on the outer (cortex) face, with a strip of fascicular cambium between —
then a cortex and epidermis outside.  Instantiate via ``StemAnatomy(input_data)``
— the factory in :mod:`openalea.granap.stem_class` dispatches here when
``planttype == 2``.

The bundles are placed on an evenly-spaced ring at the pith/cortex boundary
(``_bundle_ring_positions``).  The requested ``n_bundles`` is clamped to the
count that fits without adjacent bundle envelopes overlapping (a warning is
issued when it has to be reduced), which keeps the ring symmetric.
"""

import logging
import warnings
from typing import List, Tuple

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.tissue_class import TissueRecipe, fill_along
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.vascular_bundle import (
    build_bundle, bundle_cambium_anchor, outer_sheath_mask_pad,
)
from openalea.granap import secondary_growth as sg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dicot stem subclass
# ---------------------------------------------------------------------------

class DicotStemAnatomy(StemAnatomy):
    """Dicot stem: a ring of discrete collateral bundles (xylem in / phloem out /
    cambium between) around a central pith (eustele)."""

    # ------------------------------------------------------------------
    # Vascular tissue
    # ------------------------------------------------------------------
    #
    # No _parse_vascular_params override: build_bundle reads the raw xylem /
    # phloem / cambium param dicts directly, and the bundle count is the
    # vascular_bundle.n_bundles field — so there is nothing to pre-parse.

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Declarative description of how the eustele bundle ring is assembled.

        Built and run by the shared ``Organ._create_vascular_tissue`` scaffold;
        the remove-mask + extend step runs later in ``Organ.generate_cells``.
        The build order is data, inspectable via ``recipe.describe()``.

        The single ``collateral bundle ring`` step defers to
        :meth:`_build_bundle_ring`, which builds one collateral bundle per ring
        slot.
        """
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        bp = self._get_param("vascular_bundle")
        if not bp or int(bp.get("n_bundles", 0)) == 0:
            return recipe                       # no bundles -> empty
        recipe.special(
            "collateral bundle ring",
            lambda: self._build_bundle_ring(polygon),
            produces=("xylem", "cambium", "phloem", "bundle sheath"),
        )
        return recipe

    # ------------------------------------------------------------------
    # Cambium-ring contour (the shape the bundles are placed on)
    # ------------------------------------------------------------------

    def _secondary_growth(self) -> bool:
        """True when secondary growth is requested."""
        sec = self._get_param("secondary_growth")
        return bool(sec.get("value", False)) if sec else False

    def _secondary_cambium_radius(self) -> float:
        """Outer radius (mm from the centre) the secondary cambium sits at.

        The cambium is described by its radius (like the root): ``radius_valley_side``
        for a circle / ellipse / focus_ellipse ring, or the larger of the star's
        peak / valley radii for a ``star`` contour."""
        sc = self._get_param("secondary_cambium") or {}
        r = float(sc.get("radius_valley_side", 0.0))
        if sc.get("shape") == "star":
            r = max(r, float(sc.get("radius_peak_side", 0.0)))
        return r

    def _secondary_annulus_thickness(self) -> float:
        """Radial thickness the stem gains under secondary growth: the secondary
        xylem annulus (the secondary cambium radius minus the primary bundle-ring
        radius) + the secondary phloem band (``secondary_phloem.height``) + the
        displaced primary-phloem remnant pushed just outside it.  Zero under primary
        growth.

        The primary ring radius under secondary growth is a known constant, the pith
        half-thickness (see :meth:`_primary_ring_radius`), so this is computable
        before the base-shape polygon exists."""
        if not self._secondary_growth():
            return 0.0
        sp = self._get_param("secondary_phloem") or {}
        r_prim = float(self.vascular_params["thickness"]) / 2.0
        growth = max(self._secondary_cambium_radius() - r_prim, 0.0)
        return growth + float(sp.get("height", 0.0)) + self._primary_phloem_thickness()

    def _primary_phloem_thickness(self) -> float:
        """Radial thickness reserved outside the secondary phloem for the displaced
        primary-phloem remnant — the bundle's phloem-fraction share of its height.

        Zero unless ``secondary_phloem.keep_primary`` is set: secondary growth
        crushes the primary phloem, so it is not rendered (nor room reserved for it)
        by default."""
        sp = self._get_param("secondary_phloem") or {}
        if not sp.get("keep_primary", False):
            return 0.0
        bp = self._get_param("vascular_bundle") or {}
        return float(bp.get("height", 0.0)) * float(bp.get("phloem_fraction", 0.35))

    def _calculate_radius(self) -> float:
        """Stem outer radius, grown outward to make room for secondary growth."""
        return super()._calculate_radius() + self._secondary_annulus_thickness()

    def _primary_ring_radius(self, polygon: Polygon) -> float:
        """Radius of the primary bundle ring (where the bundles / fascicular cambium
        sit).

        Under secondary growth the central region is enlarged to make room for the
        secondary-xylem annulus, so the primary ring is pinned to the pith radius
        (``thickness/2``) rather than the enlarged region's radius; primary growth
        keeps the historical ``sqrt(area/pi)``.
        """
        if self._secondary_growth():
            return float(self.vascular_params["thickness"]) / 2.0
        return float(np.sqrt(polygon.area / np.pi))

    def _ring_contour(self, cx: float, cy: float, r_default: float, spec: dict) -> Polygon:
        """A ring contour in the ``ring_shape`` family from a bundle / cylinder spec.

        Shared by the eustele bundle ring and the continuous cylinder.  ``circle`` /
        ``ellipse`` sit at ``r_default`` (the auto-derived pith-edge radius, flattened
        by ``ring_ellipse_ratio`` for the ellipse).  ``star`` uses the **same absolute
        peak/valley parameterisation as the root** — ``radius_peak_side`` /
        ``radius_valley_side`` (mm) + ``arc_peak_side`` / ``arc_valley_side`` +
        ``n_peaks`` — so there is a single way to describe a star contour across
        organs (``r_default`` is not used for a star)."""
        shape = spec.get("ring_shape", "circle")
        if shape == "star":
            return GeometryProcessor.contour_polygon(
                "star", cx=cx, cy=cy,
                n_branches=int(spec.get("n_peaks", 5)),
                radius_peak_side=float(spec.get("radius_peak_side", r_default)),
                radius_valley_side=float(spec.get("radius_valley_side", r_default)),
                arc_peak_side=float(spec.get("arc_peak_side", 0.12)),
                arc_valley_side=float(spec.get("arc_valley_side", 0.10)))
        return GeometryProcessor.contour_polygon(
            shape, cx=cx, cy=cy, radius=r_default,
            ellipse_ratio=float(spec.get("ring_ellipse_ratio", 0.75)))

    def _cambium_ring_contour(self, polygon: Polygon) -> Polygon:
        """The drawn contour the eustele bundles are placed on — the (primary)
        cambium ring — in the ``ring_shape`` family."""
        bp = self._get_param("vascular_bundle")
        cx, cy = polygon.centroid.x, polygon.centroid.y
        return self._ring_contour(cx, cy, self._primary_ring_radius(polygon), bp)

    def _contour_slots(self, contour: Polygon, cx: float, cy: float,
                       n: int) -> List[Tuple[float, float, float]]:
        """``n`` evenly-spaced ``(px, py, theta)`` slots along ``contour``.

        Points are sampled by arc length around the contour; ``theta`` is the
        outward normal at each (from the local tangent of neighbouring samples),
        so every bundle points away from the organ centre wherever it sits on a
        circle, ellipse or star arm.
        """
        if n <= 0:
            return []
        coords = np.asarray(contour.exterior.coords)
        pts = GeometryProcessor.resample_coords(coords, target_n_points=n + 1)
        slots = []
        for i in range(n):
            px, py = pts[i]
            nx_pt, ny_pt = pts[(i + 1) % n]
            # (i - 1) % n, not i - 1: the resampled ring is closed (pts[n] == pts[0]),
            # so a bare pts[-1] is pts[0] itself — the current point — which flattens
            # the tangent for slot 0 and mis-orients that bundle.
            px_pt, py_pt = pts[(i - 1) % n]
            tx, ty = nx_pt - px_pt, ny_pt - py_pt          # local tangent
            nx, ny = ty, -tx                                # rotate -90 -> a normal
            if nx * (px - cx) + ny * (py - cy) < 0:         # make it point outward
                nx, ny = -nx, -ny
            slots.append((float(px), float(py), float(np.arctan2(ny, nx))))
        return slots

    def _ring_slots(self, polygon: Polygon) -> List[Tuple[float, float, float]]:
        """Bundle slots along the cambium-ring contour, clamped to non-overlap.

        The requested ``n_bundles`` is reduced until no two adjacent bundle
        envelopes touch (a warning is issued when it must drop); because
        resampling re-spaces the slots, the whole slot set is recomputed at each
        candidate count so the ring stays evenly spaced.
        """
        bp = self._bundle_params()
        n_req = int(bp.get("n_bundles", 0))
        if n_req <= 0:
            return []
        contour = self._cambium_ring_contour(polygon)
        cx, cy = polygon.centroid.x, polygon.centroid.y
        r_pith = np.sqrt(polygon.area / np.pi)
        anchor = bundle_cambium_anchor(bp)
        gap = self._bundle_clearance(bp)
        ground = self._pith_cell_diameter_at(r_pith, r_pith)

        for n in range(n_req, 0, -1):
            slots = self._contour_slots(contour, cx, cy, n)
            envs = [self._placed_bundle_envelope(px, py, th, bp, ground, anchor)
                    for px, py, th in slots]
            if n == 1 or all(
                not self._bundle_overlaps(envs[i], [envs[(i + 1) % n]], gap)
                for i in range(n)
            ):
                if n < n_req:
                    warnings.warn(
                        f"DicotStemAnatomy: {n_req} bundles overlap on the "
                        f"{bp.get('ring_shape', 'circle')} ring; placing {n} "
                        f"evenly-spaced non-overlapping bundles instead (reduce "
                        f"vascular_bundle.width/height or n_bundles to fit more).",
                        stacklevel=2,
                    )
                return slots
        return []

    def _cambium_band_thickness(self, cambium: dict) -> float:
        """Radial thickness of the cambium band = ``n_layers`` cell files."""
        n = max(int(cambium.get("n_layers", 2)), 1)
        return n * float(cambium.get("cell_diameter", 0.01))

    def _bundle_params(self, spec: dict = None) -> dict:
        """Bundle params with the fascicular cambium gap sized to the ring.

        The cambium — the fascicular strip *and* the secondary ring — is one band
        of ``n_layers`` cell files laid on the ring contour.  For the fascicular
        strip to hold the same ``n_layers`` as the interfascicular ring, the bundle
        must reserve a gap at least that thick between xylem and phloem; so this
        returns a *copy* of the given spec (default the single ``vascular_bundle``
        param) with ``cambium_fraction`` raised to fit (banded open bundles only),
        leaving everything else untouched.  ``spec`` lets each bundle *kind* in a
        mixed-kind eustele pattern get its own adjusted copy.
        """
        bp = dict((spec if spec is not None else self._get_param("vascular_bundle")) or {})
        if not bp:
            return bp
        if bp.get("has_cambium", True) and bp.get("bundle_type", "collateral") != "concentric":
            cambium = self._get_param("cambium") or {}
            t = self._cambium_band_thickness(cambium)
            h = float(bp.get("height", 0.18)) or 0.18
            bp["cambium_fraction"] = max(float(bp.get("cambium_fraction", 0.08)),
                                         1.2 * t / h)
        return bp

    # ------------------------------------------------------------------
    # Mixed-kind bundle pattern (dicot eustele)
    # ------------------------------------------------------------------

    def _bundle_spec_by_kind(self, kind: str) -> dict:
        """The ``vascular_bundle`` spec whose ``kind`` label matches (raises if none)."""
        for p in self.params:
            if p.get("name") == "vascular_bundle" and str(p.get("kind", "")) == kind:
                return p
        raise ValueError(
            f"bundle_pattern references kind {kind!r} but no vascular_bundle spec "
            f"has kind={kind!r}."
        )

    @staticmethod
    def _contour_sampler(contour: Polygon, cx: float, cy: float, n_dense: int = 1440):
        """Return ``(pose, perimeter)`` where ``pose(s)`` maps an arc-length position
        ``s`` (mm along the contour) to a ``(px, py, theta)`` slot — the point at that
        arc length and the outward normal there.

        The contour is densely, uniformly resampled once so a bundle can be dropped
        at *any* arc position (the mixed-kind pattern needs uneven spacing), unlike
        :meth:`_contour_slots` which only emits ``n`` even slots."""
        pts = GeometryProcessor.resample_coords(
            np.asarray(contour.exterior.coords), target_n_points=n_dense + 1)
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        perimeter = float(seg.sum())

        def pose(s: float):
            f = (s % perimeter) / perimeter * n_dense
            i = int(f) % n_dense
            px, py = pts[i]
            ax, ay = pts[(i - 1) % n_dense]
            bx, by = pts[(i + 1) % n_dense]
            nx, ny = (by - ay), -(bx - ax)                  # rotate tangent -90 -> normal
            if nx * (px - cx) + ny * (py - cy) < 0:         # point outward
                nx, ny = -nx, -ny
            return float(px), float(py), float(np.arctan2(ny, nx))

        return pose, perimeter

    def _slot_footprint(self, bp: dict, ground: float) -> float:
        """Tangential footprint of a bundle at the ring (envelope width + outer
        sheath), used to check that equally-spaced bundles don't overlap."""
        return float(bp.get("width", 0.12)) + 2.0 * outer_sheath_mask_pad(bp, ground)

    def _star_arm_phase(self, polygon: Polygon, contour: Polygon,
                        repeats: int, pat: dict) -> float:
        """Arc-length offset so the first bundle of each repeat lands on a star arm.

        Zero (no phasing) unless the ring is a ``star`` with as many arms as repeats
        and ``align_to_arms`` is on: then it puts slot 0 on arm 0's peak (at angle 0);
        the other first-of-repeat bundles follow at the equal arm spacing."""
        bp0 = self._get_param("vascular_bundle") or {}
        if not pat.get("align_to_arms", True) or bp0.get("ring_shape") != "star":
            return 0.0
        if int(bp0.get("n_peaks", 5)) != repeats:
            return 0.0
        cx, cy = polygon.centroid.x, polygon.centroid.y
        # Arm 0 peaks along +x (angle 0); find where that ray meets the contour.
        minx, miny, maxx, maxy = contour.bounds
        r_far = 2.0 * max(maxx - cx, cx - minx, maxy - cy, cy - miny, 1e-6)
        frame = sg.cambium_local_frame(contour.exterior, cx, cy, 0.0, r_far)
        if frame is None:
            return 0.0
        (px, py), _t, _n = frame
        return float(contour.exterior.project(Point(px, py)))   # arm-0 peak arc pos

    def _pattern_slots(self, polygon: Polygon):
        """Mixed-kind eustele slots, or ``None`` when no ``bundle_pattern`` is set.

        Returns ``[(px, py, theta, bp_kind), ...]``.  The full tiled sequence
        (``sequence`` x ``repeats``) is placed at **equal spacing** around the ring,
        so a given kind is equidistant from the next occurrence of itself (e.g. the
        ``big`` bundles land equally spaced) and the intervening bundles sit evenly
        between them.  ``spacing`` chooses the metric: ``"distance"`` = equal
        arc-length along the ring, ``"angle"`` = equal angular step from the centre
        (identical on a circle; different on an ellipse / star).  ``align_to_arms``
        phases the first bundle of each repeat onto a star arm and ``angle`` rotates
        the whole pattern.  ``repeats`` is reduced (with a warning) if the spacing
        would make neighbouring envelopes overlap."""
        pat = self._get_param("bundle_pattern")
        seq = list(pat.get("sequence") or []) if pat else []
        if not seq:
            return None

        contour = self._cambium_ring_contour(polygon)
        cx, cy = polygon.centroid.x, polygon.centroid.y
        r_pith = np.sqrt(polygon.area / np.pi)
        ground = self._pith_cell_diameter_at(r_pith, r_pith)
        pose, L = self._contour_sampler(contour, cx, cy)

        kinds = {k: self._bundle_params(self._bundle_spec_by_kind(k)) for k in set(seq)}
        foot = {k: self._slot_footprint(kinds[k], ground) for k in kinds}
        clearance = max(self._bundle_clearance(kinds[k]) for k in kinds)
        m = len(seq)
        n_req = int(pat.get("repeats", 1))
        angle_off = float(pat.get("angle", 0.0))
        spacing = str(pat.get("spacing", "distance"))
        by_angle = spacing == "angle"
        grouped = spacing == "grouped"

        ext = contour.exterior
        minx, miny, maxx, maxy = contour.bounds
        r_far = 2.0 * max(maxx - cx, cx - minx, maxy - cy, cy - miny, 1e-6)

        def _place(repeats):
            """(px, py, theta, kind) list for a candidate repeat count, or None if a
            ray misses the contour (angle mode)."""
            n = m * repeats
            out = []
            if grouped:
                # Each repeat's sequence is one cluster centred in its equal share of
                # the ring, members spaced by their footprints (+ clearance); the
                # leftover of the share is the empty valley to the next cluster.
                seg = L / repeats
                widths = [foot[seq[i]] for i in range(m)]
                group = sum(widths) + clearance * (m - 1)
                arm = self._star_arm_phase(polygon, contour, repeats, pat)
                phase = arm - seg / 2.0 + (angle_off / 360.0) * L   # centre group on arm
                for rp in range(repeats):
                    s = phase + rp * seg + (seg - group) / 2.0
                    for i in range(m):
                        px, py, th = pose(s + widths[i] / 2.0)
                        out.append((px, py, th, seq[i]))
                        s += widths[i] + clearance
            elif by_angle:
                base = np.radians(angle_off)      # slot 0 at angle 0 (+ offset); on a
                for i in range(n):                # star with repeats arms that IS arm 0
                    frame = sg.cambium_local_frame(ext, cx, cy, base + 2.0 * np.pi * i / n, r_far)
                    if frame is None:
                        return None
                    (px, py), _t, (nx, ny) = frame
                    out.append((px, py, float(np.arctan2(ny, nx)), seq[i % m]))
            else:
                step = L / n
                phase = self._star_arm_phase(polygon, contour, repeats, pat)
                phase += (angle_off / 360.0) * L
                for i in range(n):
                    px, py, th = pose(phase + i * step)
                    out.append((px, py, th, seq[i % m]))
            return out

        def _fits(slots):
            """No adjacent pair's envelopes overlap (centre distance vs footprints)."""
            p = len(slots)
            for i in range(p):
                x0, y0, _, k0 = slots[i]
                x1, y1, _, k1 = slots[(i + 1) % p]
                if np.hypot(x1 - x0, y1 - y0) < (foot[k0] + foot[k1]) / 2.0 + clearance:
                    return False
            return True

        def _ok(repeats, slots):
            if grouped:
                # Arc-based: the whole cluster must fit inside one repeat's share of
                # the ring (with a little valley left over).  The Euclidean pair test
                # is unreliable here because members are packed by arc length.
                group = sum(foot[k] for k in seq) + clearance * (m - 1)
                return group <= L / repeats
            return _fits(slots)

        for repeats in range(n_req, 0, -1):
            slots = _place(repeats)
            if slots is None:
                continue
            if repeats == 1 or _ok(repeats, slots):
                if repeats < n_req:
                    warnings.warn(
                        f"DicotStemAnatomy: bundle_pattern {seq} x{n_req} does not fit the "
                        f"{self._get_param('vascular_bundle').get('ring_shape', 'circle')} ring "
                        f"at '{spacing}' spacing; placing "
                        f"{repeats} repeat(s) instead (shrink the bundles or the sequence).",
                        stacklevel=2,
                    )
                return [(px, py, th, kinds[k]) for (px, py, th, k) in slots]
        return None

    def _build_bundle_ring(self, polygon: Polygon) -> None:
        """Build the eustele: bundles on a shared cambium contour, then the cambium.

        The cambium *shape* is defined once — the ring contour
        (:meth:`_cambium_ring_contour`) and its ``n_layers`` cell files — and each
        bundle is placed on it, anchored on its cambium band
        (``bundle_cambium_anchor``) so every fascicular strip sits on the contour.
        Bundles are built *without* their cambium (``fill_cambium=False``); the
        cambium is then laid down in one pass by :meth:`_build_cambium`, which
        materialises either just the in-bundle arcs (primary growth) or the entire
        ring (secondary growth).  Building it in one place is what makes the
        fascicular and interfascicular cambium share the same number of layers.

        Each bundle envelope is registered in ``vascular_tissue_polygons`` so
        ``generate_cells`` clears the pith/cortex seeds underneath it.
        """
        bp = self._bundle_params()
        xylem = self._get_param("xylem")
        phloem = self._get_param("phloem")
        cambium = self._get_param("cambium")
        if not bp:
            return
        if self._secondary_growth():
            # Full secondary growth replaces the discrete-bundle path.
            self._build_secondary_growth(polygon)
            return
        cx0, cy0 = polygon.centroid.x, polygon.centroid.y
        r_pith = np.sqrt(polygon.area / np.pi)
        contour = self._cambium_ring_contour(polygon)
        # A mixed-kind pattern gives each slot its own bundle spec; otherwise every
        # slot is the single spec (legacy uniform ring).
        pattern = self._pattern_slots(polygon)
        if pattern is not None:
            slots = pattern
        else:
            slots = [(px, py, th, bp) for (px, py, th) in self._ring_slots(polygon)]

        conducting: List[Polygon] = []      # xylem / phloem zones — the ring avoids these
        fascicular: List[Polygon] = []      # per-bundle cambium zones — the primary clip region
        sheath: List[Polygon] = []          # bundle-sheath zones — the ring must not eat these
        inner_cambium: List[Polygon] = []   # bicollateral inner-phloem-side cambium bands
        for cx, cy, theta, slot_bp in slots:
            ground = self._pith_cell_diameter_at(np.hypot(cx - cx0, cy - cy0), r_pith)
            res = build_bundle(self.vascular_cells, self.rng, cx, cy, theta,
                               slot_bp, xylem, phloem, cambium,
                               ground_cell_size=ground,
                               anchor=bundle_cambium_anchor(slot_bp),
                               fill_cambium=False)
            self._register_bundle(res)
            cambia_here: List[Polygon] = []
            for role, g in res.zone_polygons:
                if g is None or g.is_empty:
                    continue
                if role in ("xylem", "phloem"):
                    conducting.append(g)
                elif role == "cambium":
                    cambia_here.append(g)
                elif role in ("bundle sheath", "sclerenchyma", "parenchyma"):
                    sheath.append(g)
            # The bundle is anchored on its outermost cambium band, so that one sits
            # on the shared ring contour (laid by _build_cambium).  A bicollateral
            # bundle also has an inner cambium band on the inner-phloem side — off the
            # ring — which _build_cambium never reaches, so fill it directly.
            if cambia_here:
                cambia_here.sort(key=lambda gg: np.hypot(gg.centroid.x - cx0,
                                                         gg.centroid.y - cy0))
                fascicular.append(cambia_here[-1])          # outermost -> on contour
                inner_cambium.extend(cambia_here[:-1])      # inner (bicollateral)

        # Primary growth: fascicular cambium only (secondary growth branched off
        # earlier into _build_secondary_growth).
        self._build_cambium(contour, fascicular, conducting, cambium,
                            secondary=False, sheath=sheath)

        # Inner (bicollateral) cambium bands sit on their own inner ring, off the main
        # contour.  Lay them with the *same* n_layers-file renderer as the outer
        # cambium — along an inner contour at their mean radius, clipped to the inner
        # zones — so both faces read as the same number of layers.
        if inner_cambium:
            r_in = float(np.mean([np.hypot(z.centroid.x - cx0, z.centroid.y - cy0)
                                  for z in inner_cambium]))
            # A plain circle through the inner bands; the file loop is clipped to the
            # inner cambium zones, so the contour's exact shape doesn't matter here.
            inner_contour = GeometryProcessor.contour_polygon(
                "circle", cx=cx0, cy=cy0, radius=r_in)
            self._build_cambium(inner_contour, inner_cambium, conducting, cambium,
                                secondary=False, sheath=sheath)

    def _build_cambium(self, contour: Polygon, fascicular: List[Polygon],
                       conducting: List[Polygon], cambium: dict,
                       secondary: bool, sheath: List[Polygon] = None) -> None:
        """Lay the vascular cambium as ``n_layers`` cell files along the contour.

        The cambium shape is fixed (the contour and its ``n_layers`` files); what
        gets *materialised* depends on the growth stage:

        * **primary** (``secondary`` False): only the arcs that fall **inside a
          bundle** — the fascicular cambium strips (``keep_union`` = the bundle
          cambium zones).  These zones are already registered per bundle.
        * **secondary** (``secondary`` True): the **entire ring**.  The bundle
          sheaths (and the parenchyma bundle-sheath file) lying on the band are
          removed first, so the ring replaces them and runs unbroken through every
          bundle; the whole band is then registered as ``cambium``.

        Either way the files are blocked from the conducting tissues (xylem /
        phloem), so the cambium never overwrites a vessel or sieve element.  Because
        both stages use the *same* file loop, the fascicular and interfascicular
        cambium always carry the same number of layers.
        """
        cx, cy = contour.centroid.x, contour.centroid.y
        n_layers = max(int(cambium.get("n_layers", 2)), 1)
        cell_d = float(cambium.get("cell_diameter", 0.01))
        cell_w = float(cambium.get("cell_width", cell_d))
        conducting_union = unary_union(conducting) if conducting else None
        block_union = conducting_union

        if secondary:
            half = max(n_layers * cell_d / 2.0, cell_d / 2.0)
            band = contour.buffer(half).difference(contour.buffer(-half))
            if conducting_union is not None:
                band = band.difference(conducting_union)
            if band.is_empty:
                return
            # Remove the bundle sheaths on the band so the ring replaces them
            # instead of overlapping surviving sheath cells.
            self._clear_vascular_cells(band, ("bundle sheath", "parenchyma", "sclerenchyma"))
            keep_union = None
            register = band
        else:
            if not fascicular:
                return
            # Confine the cambium to the bundle cambium zones (buffered a touch so
            # points on the strip edge are not dropped by the point-in test), but
            # hold it a cell-width clear of the bundle sheath: a seed centre may pass
            # the sheath block yet still have its border cells reach into the sheath,
            # so keep the whole seed a margin away from it.
            sheath_union = unary_union(sheath) if sheath else None
            keep_union = unary_union(fascicular).buffer(0.25 * cell_d)
            if sheath_union is not None:
                keep_union = keep_union.difference(sheath_union.buffer(0.5 * cell_d))
            register = None      # the fascicular zones are registered per bundle
            # Also block the fascicular files from the sheath (and the conducting
            # tissues) so their files never seed on the sheath ring.
            blockers = list(conducting) + list(sheath or [])
            block_union = unary_union(blockers) if blockers else None

        for k in range(n_layers):
            off = (k - (n_layers - 1) / 2.0) * cell_d
            ring = contour.buffer(off) if off else contour
            if ring.is_empty:
                continue
            fill_along(self.vascular_cells, ring.exterior, "cambium",
                       cell_d, cell_w, cx, cy,
                       xylem_union=block_union, keep_union=keep_union)

        # Primary: a band-edge seed can pass the sheath block yet still have a border
        # cell reach into the sheath.  Drop any cambium group that touches the sheath
        # so the sheath keeps its ring unbroken (the secondary path clears the sheath
        # the other way — the ring replaces it).
        if not secondary and sheath:
            self._clear_vascular_cells(unary_union(sheath), ("cambium",))

        if register is not None and not register.is_empty:
            self.vascular_tissue_polygons.setdefault("cambium", []).append(register)

    def _clear_vascular_cells(self, region: Polygon, types: Tuple[str, ...]) -> None:
        """Drop whole vascular-cell groups of ``types`` whose seed lies in ``region``.

        Seed cells are grouped by ``id_group`` (one Voronoi cell per group); a group
        is removed entirely if any of its seeds of a matching type falls inside
        ``region``, so no half-cells are left behind.
        """
        if region is None or region.is_empty:
            return
        from shapely.geometry import Point
        drop = {c.id_group for c in self.vascular_cells.cells
                if c.type in types and region.contains(Point(c.x, c.y))}
        if not drop:
            return
        self.vascular_cells.cells = [
            c for c in self.vascular_cells.cells
            if not (c.id_group in drop and c.type in types)
        ]

    # ------------------------------------------------------------------
    # Secondary growth (the cambium has closed and moved outward)
    # ------------------------------------------------------------------

    def _build_secondary_growth(self, polygon: Polygon) -> None:
        """Secondary growth that *grows* the primary bundle instead of replacing it.

        The transition is only partially destructive: the **primary xylem** is
        endarch and cannot move, so it stays pinned at the pith edge and the
        secondary xylem is built radially on top of it; the **primary phloem** is
        crushed outward ahead of the new secondary phloem.  Radial stack, pith->cortex:

        * **primary xylem** — a thin remnant per bundle pinned just inside the primary
          ring (:meth:`_build_primary_xylem`), reusing ``build_bundle``;
        * **secondary xylem** — the annulus between the primary ring and the secondary
          cambium (minus the cambial band shell), split into a graded-vessel sector
          behind each bundle (``prop_stele`` wide, so ``prop_stele`` -> 1 merges the
          sectors into a continuous cylinder) with **parenchyma rays** in the gaps;
        * **secondary cambium** — a closed contour at radius
          ``secondary_cambium.radius_valley_side`` (the shared circle / ellipse /
          star / focus_ellipse family), the stem having grown outward to hold it;
        * **secondary phloem** — a band just outside the cambium (root-style trapezes,
          alive/dead split), one arm behind each bundle;
        * **primary phloem** — a thin remnant per bundle pushed just outside the
          secondary phloem band (:meth:`_build_primary_phloem`).
        """
        cx, cy = polygon.centroid.x, polygon.centroid.y
        bp = self._get_param("vascular_bundle")
        sc = self._get_param("secondary_cambium")
        sx = self._get_param("secondary_xylem")
        xylem = self._get_param("xylem")
        phloem = self._get_param("phloem")
        cambium = self._get_param("cambium")
        n = int(bp.get("n_bundles", 0)) if bp else 0
        if n <= 0 or not sc or not sx:
            return

        r_prim = self._primary_ring_radius(polygon)
        r_sec = self._secondary_cambium_radius()
        primary_contour = self._ring_contour(cx, cy, r_prim, bp)
        # Inner radius of the secondary-xylem annulus.  On a non-circular primary ring
        # (star / ellipse) the bundles sit at varying radii, so the sectors and the
        # interfascicular fills must start at the *minimum* contour radius (the star
        # valleys) and be clipped back to the annulus — otherwise the valley bundles
        # are left with an empty band out to the peak radius.  Equals r_prim for a
        # circle, so the uniform eustele is unchanged.
        _pc = np.asarray(primary_contour.exterior.coords)
        r_inner = float(np.hypot(_pc[:, 0] - cx, _pc[:, 1] - cy).min())
        # The secondary cambium is described by radius (like the root) from the
        # shared contour family: circle / ellipse sized by radius_valley_side, a
        # peak/valley star, or a focus_ellipse.  It is not clipped — the stem grows
        # outward to hold it (see _secondary_annulus_thickness).
        secondary_contour = GeometryProcessor.contour_polygon(
            sc.get("shape", "circle"), cx=cx, cy=cy,
            radius=float(sc.get("radius_valley_side", r_sec)),
            ellipse_ratio=float(sc.get("ellipse_ratio", 0.75)),
            n_branches=int(sc.get("n_peaks", 0)) or n,
            radius_peak_side=float(sc.get("radius_peak_side", r_sec)),
            radius_valley_side=float(sc.get("radius_valley_side", r_sec)),
            arc_peak_side=float(sc.get("arc_peak_side", 0.20)),
            arc_valley_side=float(sc.get("arc_valley_side", 0.10)),
            profile=sc.get("profile"), exponent=float(sc.get("exponent", 4.0)))

        band_depth = int(sc["n_layers"]) * float(sc["cell_diameter"])
        xylem_boundary = secondary_contour.buffer(-band_depth)
        if xylem_boundary.is_empty:
            xylem_boundary = secondary_contour
        annulus = xylem_boundary.difference(primary_contour)
        if annulus.is_empty:
            return

        # --- slots: angular position + angular share, per bundle -----------
        # A mixed-kind bundle_pattern gives each slot its own spec, half-width and
        # angular share; the plain eustele is n uniform slots of the single spec.
        # The uniform branch keeps the exact legacy scalars, so single-kind
        # secondary growth is byte-identical (a list of equal values feeds the same
        # arithmetic to the slot-driven helpers).
        prop = float(sx["prop_stele"])
        flare = np.radians(float(sx.get("flare_angle", 30.0)))
        r_outer_wedge = r_sec * 2.0
        pattern = self._pattern_slots(polygon)
        if pattern is None:
            phis = [2.0 * np.pi * k / n for k in range(n)]
            slot_bps = [bp] * n
            half = (np.pi / n) * prop
            caps = [half] * n
            half_widths = [float(bp.get("width", 0.1)) / 2.0] * n
            gap_centers = [th + np.pi / n for th in phis]
            gap_halves = [max(np.pi / n - half, 0.0)] * n
        else:
            items = sorted(
                ((float(np.arctan2(py - cy, px - cx) % (2.0 * np.pi)), sb)
                 for (px, py, _th, sb) in pattern), key=lambda t: t[0])
            phis = [a for a, _ in items]
            slot_bps = [b for _, b in items]
            m = len(phis)
            dnext = [((phis[(i + 1) % m] - phis[i]) % (2.0 * np.pi)) for i in range(m)]
            dprev = [dnext[(i - 1) % m] for i in range(m)]
            # cap each sector's flare to half the distance to its nearer neighbour
            # (x prop_stele) so sectors never overlap; the leftover is the gap.
            caps = [min(dprev[i], dnext[i]) / 2.0 * prop for i in range(m)]
            half_widths = [float(slot_bps[i].get("width", 0.1)) / 2.0 for i in range(m)]
            gap_centers = [phis[i] + dnext[i] / 2.0 for i in range(m)]
            gap_halves = [max(dnext[i] / 2.0 * (1.0 - prop), 0.0) for i in range(m)]

        # --- preserved primary xylem (pinned at the pith edge) -------------
        self._build_primary_xylem(primary_contour, phis, slot_bps, xylem, phloem,
                                   cambium, cx, cy, r_outer_wedge)

        # --- secondary cambium ring ----------------------------------------
        sg.render_cambium_files(self.vascular_cells, secondary_contour,
                                int(sc["n_layers"]), float(sc["cell_diameter"]),
                                float(sc["cell_width"]), cx, cy)
        cambial_shell = secondary_contour.difference(secondary_contour.buffer(-band_depth))
        if not cambial_shell.is_empty:
            self.vascular_tissue_polygons.setdefault("cambium", []).append(cambial_shell)

        # --- secondary xylem sectors (behind each bundle) ------------------
        # Flared sectors: start at the bundle width against the primary xylem and
        # widen outward (see flared_wedge), so they only merge once their capped
        # edges meet further out.
        sectors = []
        for th, hw, cap in zip(phis, half_widths, caps):
            zone = sg.flared_wedge(cx, cy, th, r_inner, r_outer_wedge,
                                   hw, flare, cap).intersection(annulus)
            if not zone.is_empty:
                sectors.append(zone)

        # Real outer xylem radius (inner edge of the cambium band) — the span over
        # which rate-driven medullar rays are initiated and vessels graded.
        r_outer = r_sec - band_depth

        # Medullar rays are built BEFORE vessel packing so they cut the sectors, and
        # the growth-ring bands re-grade the vessels per ring (n_ring).
        mr = self._get_param("medullar_rays") or {}
        # Start the rays just clear of the primary xylem (half a cell) so they run
        # through the whole secondary xylem yet their inner tips still sit in dense
        # parenchyma rather than ballooning against the sparse primary-xylem edge.
        ray_inner_min = r_inner + 0.5 * float(sx["cell_diameter"])
        medullar_polys, medullar_union = sg.prepare_medullar_rays(
            self.vascular_cells, self.rng, annulus, sectors, primary_contour,
            cx, cy, r_outer_wedge, r_outer, phis, caps, prop,
            float(sc["cell_diameter"]), mr, r_inner_min=ray_inner_min)
        annual_bands = sg.build_annual_bands(secondary_contour, primary_contour,
                                             int(sx.get("n_ring", 1)))

        next_id = self.vascular_cells.next_group_id()
        vessels = []
        for zone in sectors:
            vs, next_id = sg.fill_secondary_xylem_sector(
                self.vascular_cells, self.rng, zone, sx, cx, cy, next_id,
                annual_bands=annual_bands, medullar_union=medullar_union)
            vessels.extend(vs)
        if sectors:
            self.vascular_tissue_polygons.setdefault("xylem", []).append(unary_union(sectors))
        self.vascular_polygons.extend(vessels)

        # --- parenchyma rays (the interfascicular gaps between bundles) -----
        # Lane-splitting radial fill in each gap wedge (centre midway between adjacent
        # sectors, half-width the leftover after the sectors' caps).  With a mixed
        # pattern the between-group gaps are wider than the within-group ones.
        next_id = sg.fill_ray_parenchyma_split(
            self.vascular_cells, self.rng, sectors, annulus, cx, cy, sx,
            r_outer, gap_centers, gap_halves, r_inner, next_id)
        # Register ONLY the wedge footprint actually filled with ray parenchyma, not
        # the whole interfascicular gap: the sectors flare from the bundle width, so
        # near the pith the gap is far wider than the fixed +/-gap_half wedge.  Clearing
        # that whole gap of pith seeds while the ray fill reaches only the wedge would
        # leave an empty band the neighbouring Voronoi cells balloon into — so the
        # pith is left in place there (as it grades into the interfascicular rays).
        if sectors and max(gap_halves, default=0.0) > 0.0:
            wedges = unary_union([sg.angular_wedge(cx, cy, gc, gh, r_outer_wedge)
                                  for gc, gh in zip(gap_centers, gap_halves) if gh > 0.0])
            ray_zone = annulus.intersection(wedges).difference(unary_union(sectors))
            if not ray_zone.is_empty:
                self.vascular_tissue_polygons.setdefault("parenchyma", []).append(ray_zone)

        # --- medullar rays --------------------------------------------------
        for poly, theta_c in medullar_polys:
            next_id = sg.fill_medullar_rays(self.vascular_cells, poly, theta_c,
                                            cx, cy, mr, next_id)
        if medullar_union is not None and not medullar_union.is_empty:
            self.vascular_tissue_polygons.setdefault("medullar_ray", []).append(medullar_union)

        # --- secondary phloem ----------------------------------------------
        self._build_secondary_phloem(secondary_contour, phis, caps, cx, cy)

        # --- displaced primary phloem (pushed outside the secondary phloem) -
        self._build_primary_phloem(secondary_contour, phis, slot_bps, xylem, phloem,
                                   cambium, cx, cy, r_outer_wedge)

    def _primary_tissue_bp(self, bp: dict, role: str, thickness: float) -> dict:
        """A copy of the ``vascular_bundle`` params reduced to a single-tissue banded
        bundle — xylem-only or phloem-only, no cambium, no sheath — of radial
        ``thickness``.

        Lets :func:`~openalea.granap.vascular_bundle.build_bundle` lay just the
        preserved primary xylem or the displaced primary phloem remnant under
        secondary growth, reusing the same conducting-cell fills as a primary bundle.
        The bundle sheath (for the primary xylem) is laid separately as a *partial*
        cup by :meth:`_build_primary_xylem`, so it is disabled here."""
        out = dict(bp)
        out["bundle_type"] = "collateral"
        out["has_cambium"] = False
        out["inner_phloem_fraction"] = 0.0
        out["outer_sheath"] = False
        out["sheath"] = "none"
        out["sheath_thickness"] = 0.0
        out["height"] = float(thickness)
        out["xylem_fraction"] = 1.0 if role == "xylem" else 0.0
        out["phloem_fraction"] = 1.0 if role == "phloem" else 0.0
        return out

    def _place_primary_remnant(self, contour: Polygon, angles, remnant_bps,
                               xylem: dict, phloem: dict, cambium: dict,
                               cx: float, cy: float, r_far: float,
                               anchors) -> list:
        """Build one single-tissue primary remnant bundle per ``(angle, remnant_bp,
        anchor)`` slot on ``contour`` and return the ``(BundleResult, remnant_bp)``
        pairs.

        A ray from the centre at each angle meets ``contour`` at ``P``; the bundle is
        placed there oriented along the local outward normal and shifted by its
        ``anchor`` so the requested edge (outer for xylem, inner for phloem) lands on
        ``P``.  A ``None`` remnant_bp (a kind with no xylem/phloem share) is skipped.
        """
        results = []
        for th, rbp, anch in zip(angles, remnant_bps, anchors):
            if rbp is None:
                continue
            frame = sg.cambium_local_frame(contour.exterior, cx, cy, th, r_far)
            if frame is None:
                continue
            (px, py), _tangent, (nx, ny) = frame
            theta = float(np.arctan2(ny, nx))
            res = build_bundle(self.vascular_cells, self.rng, px, py, theta,
                               rbp, xylem, phloem, cambium,
                               ground_cell_size=None, anchor=anch,
                               fill_cambium=False)
            self._register_bundle(res)
            results.append((res, rbp))
        return results

    def _build_primary_xylem(self, primary_contour: Polygon, angles, slot_bps,
                             xylem: dict, phloem: dict, cambium: dict,
                             cx: float, cy: float, r_far: float) -> None:
        """Lay the preserved primary xylem as a thin remnant pinned at the pith edge,
        one per bundle position, then cup each with a **partial bundle sheath**.

        The primary xylem is endarch and immovable, so it stays put while the
        secondary xylem is built radially on top of it.  A xylem-only remnant bundle
        (:meth:`_primary_tissue_bp`) is anchored so its **outer** edge sits on the
        primary ring — i.e. it occupies the outer pith, flush against where the
        secondary-xylem annulus begins.  A single file of ``bundle sheath`` cells is
        then laid along the **pith-facing** part of each envelope only (the inner
        half, radius below the envelope centre), leaving the outer side open so the
        primary xylem joins straight onto the secondary xylem instead of being
        wrapped all the way round.
        """
        remnant_bps, anchors = [], []
        for bpk in slot_bps:
            h_px = float(bpk.get("height", 0.0)) * float(bpk.get("xylem_fraction", 0.5))
            if h_px <= 0.0:
                remnant_bps.append(None); anchors.append(0.0)
            else:
                remnant_bps.append(self._primary_tissue_bp(bpk, "xylem", h_px))
                anchors.append(h_px / 2.0)
        if not any(r is not None for r in remnant_bps):
            return
        results = self._place_primary_remnant(primary_contour, angles, remnant_bps,
                                              xylem, phloem, cambium, cx, cy, r_far,
                                              anchors)
        # Partial sheath: a file of cells hugging each envelope's inner (pith-facing)
        # arc, cut off at the envelope centre radius so it never crosses onto the
        # outer (secondary-xylem) side.  Sizes come from the slot's own spec.
        for res, rbp in results:
            env = res.envelope
            if env is None or env.is_empty:
                continue
            scd = float(rbp.get("parenchyma_diameter", 0.012))
            scw = float(rbp.get("parenchyma_width", 0.012))
            r_center = np.hypot(env.centroid.x - cx, env.centroid.y - cy)
            inner_disc = Point(cx, cy).buffer(max(r_center, 0.0))
            inner_arc = env.exterior.intersection(inner_disc)
            geoms = inner_arc.geoms if hasattr(inner_arc, "geoms") else [inner_arc]
            for line in geoms:
                if line.geom_type in ("LineString", "LinearRing") and line.length > 0:
                    fill_along(self.vascular_cells, line, "bundle sheath",
                               scd, scw, cx, cy)

    def _build_primary_phloem(self, secondary_contour: Polygon, angles, slot_bps,
                              xylem: dict, phloem: dict, cambium: dict,
                              cx: float, cy: float, r_far: float) -> None:
        """Lay the displaced primary phloem as a thin remnant pushed just outside the
        secondary phloem band, one per bundle position.

        The transition crushes the primary phloem outward ahead of the new secondary
        phloem.  A phloem-only remnant bundle (:meth:`_primary_tissue_bp`) is anchored
        so its **inner** edge sits on the outer edge of the secondary phloem band
        (the secondary cambium contour buffered out by ``secondary_phloem.height``).
        """
        h_pp = self._primary_phloem_thickness()
        if h_pp <= 0.0:
            return
        sp = self._get_param("secondary_phloem")
        sp_height = float(sp.get("height", 0.0)) if sp else 0.0
        contour = secondary_contour.buffer(sp_height) if sp_height > 0.0 else secondary_contour
        remnant_bps = [self._primary_tissue_bp(bpk, "phloem", h_pp) for bpk in slot_bps]
        anchors = [-h_pp / 2.0] * len(slot_bps)
        self._place_primary_remnant(contour, angles, remnant_bps, xylem, phloem,
                                    cambium, cx, cy, r_far, anchors)

    def _build_secondary_phloem(self, secondary_contour: Polygon, angles, halves,
                                cx: float, cy: float) -> None:
        """Root-style secondary phloem: a band just outside the secondary cambium,
        carved into one tapering trapeze per bundle and split radially into an alive
        (sieve + companion + parenchyma) and a dead (sieve + parenchyma) sub-zone."""
        sp = self._get_param("secondary_phloem")
        if not sp:
            return
        band = secondary_contour.buffer(float(sp["height"])).difference(secondary_contour)
        if band.is_empty:
            return
        minx, miny, maxx, maxy = band.bounds
        r_outer = max(maxx - cx, cx - minx, maxy - cy, cy - miny) * 1.5
        cam_ext = secondary_contour.exterior

        halves = (list(halves) if isinstance(halves, (list, tuple))
                  else [halves] * len(angles))
        masks = []
        for th, h in zip(angles, halves):
            frame = sg.cambium_local_frame(cam_ext, cx, cy, th, r_outer)
            if frame is None:
                continue
            P, tangent, normal = frame
            r_P = np.hypot(P[0] - cx, P[1] - cy)
            base_hw = h * r_P
            top_w = min(float(sp["top_width"]), 2.0 * base_hw)
            masks.append(sg.phloem_trapeze_curved(
                cam_ext, cx, cy, P, tangent, normal, base_hw, top_w, float(sp["height"])))
        arms = band.intersection(unary_union(masks)) if masks else band
        if arms.is_empty:
            return
        self.vascular_tissue_polygons.setdefault("phloem", []).append(arms)

        alive_annulus = secondary_contour.buffer(float(sp["alive_distance"]))
        alive_zone = arms.intersection(alive_annulus)
        dead_zone = arms.difference(alive_annulus)
        next_id = self.vascular_cells.next_group_id()
        next_id = sg.fill_phloem_zone(self.vascular_cells, self.rng, alive_zone, True, cx, cy, sp, next_id)
        next_id = sg.fill_phloem_zone(self.vascular_cells, self.rng, dead_zone, False, cx, cy, sp, next_id)
