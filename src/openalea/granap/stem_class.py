"""
Stem anatomy implementation.

``StemAnatomy`` acts as a transparent factory: calling ``StemAnatomy(input_data)``
returns either a ``MonocotStemAnatomy`` or a ``DicotStemAnatomy`` instance
depending on the ``planttype`` value in the input.  Both subclasses are
``isinstance(obj, StemAnatomy)`` == True, so callers can stay type-agnostic.

This mirrors the root package (:mod:`openalea.granap.root_class` +
``root_monocot_class`` / ``root_dicot_class``): the base holds the shared
geometry (base shape, central *pith* rings, layer setup) and the two concrete
subclasses supply only the vascular layout.

Botanical model (canonical textbook):

* **Monocot stem** — an *atactostele*: vascular bundles scattered throughout the
  ground tissue, no cambium (built by :class:`MonocotStemAnatomy`).
* **Dicot stem** — a *eustele*: a ring of discrete collateral bundles (xylem
  inner, phloem outer, cambium between) around a central pith, with cortex and
  epidermis outside (built by :class:`DicotStemAnatomy`).

``generate_cells()`` runs end-to-end: the shared geometry here builds the
epidermis + cortex + pith ground tissue, and each subclass places its vascular
bundles (a scattered atactostele or an evenly-spaced eustele ring) via
:func:`vascular_bundle.build_bundle`.  Bundle placement is constrained so no two
bundle envelopes overlap (see ``_placed_bundle_envelope`` /
``_bundle_overlaps`` and the per-subclass placement routines).
"""

import warnings
import numpy as np
from typing import List, Dict, Any, Tuple

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from openalea.granap.organ_class import Organ
from openalea.granap.layer_class import Layer, LayerPolygon
from openalea.granap.cell_manager import CellManager
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.input_data import OrganInputData
from openalea.granap.math_functions import GRADIENT_FUNCTIONS, rescale
from openalea.granap.special_tissues import consider_as_cell


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _input_params(input_data):
    """The raw param-dict list from any accepted input form (or None)."""
    if isinstance(input_data, OrganInputData):
        return input_data.to_dict_list()
    if isinstance(input_data, list):
        return input_data
    return None


def _get_planttype(input_data) -> int:
    """Extract the planttype integer (1 = monocot, 2 = dicot) from raw input."""
    params = _input_params(input_data)
    if params is None:
        return 1
    pt = next((p for p in params if p["name"] == "planttype"), {})
    return int(pt.get("value", 1))


def _get_arrangement(input_data) -> str:
    """Dicot stele arrangement: 'fascicular' (discrete bundle ring) or
    'continuous' (a non-fascicular vascular cylinder), read from the
    ``vascular_bundle`` spec.  Defaults to 'fascicular'."""
    params = _input_params(input_data)
    if params is None:
        return "fascicular"
    vb = next((p for p in params if p["name"] == "vascular_bundle"), {})
    return str(vb.get("arrangement", "fascicular"))


# ---------------------------------------------------------------------------
# Base class — shared geometry, acts as factory via __new__
# ---------------------------------------------------------------------------

class StemAnatomy(Organ):
    """
    Stem cross-sectional anatomy.

    Calling ``StemAnatomy(input_data)`` transparently returns a
    ``MonocotStemAnatomy`` (planttype=1) or ``DicotStemAnatomy`` (planttype=2)
    instance.  Subclass directly if you always know the type.
    """

    # Stems: like roots, keep each peeled layer's thickness exact so the central
    # pith is not shrunk below its nominal size (no ring smoothing).
    LAYER_SMOOTH_FACTOR: float = 0.0

    # ------------------------------------------------------------------
    # Factory dispatch
    # ------------------------------------------------------------------

    def __new__(cls, input_data=None, seed=None):
        if cls is StemAnatomy:
            # Lazy imports: the subclasses live in their own modules and import
            # StemAnatomy from here, so importing them at module load time would
            # be circular.  Resolving them at call time avoids that.
            from openalea.granap.stem_monocot_class import MonocotStemAnatomy
            from openalea.granap.stem_dicot_class import DicotStemAnatomy
            planttype = _get_planttype(input_data)
            if planttype == 2:
                if _get_arrangement(input_data) == "continuous":
                    from openalea.granap.stem_dicot_continuous_class import (
                        ContinuousDicotStemAnatomy,
                    )
                    actual_cls = ContinuousDicotStemAnatomy
                else:
                    actual_cls = DicotStemAnatomy
            else:
                actual_cls = MonocotStemAnatomy
            return super().__new__(actual_cls)
        return super().__new__(cls)

    # ------------------------------------------------------------------
    # Initialisation — shared across both plant types
    # ------------------------------------------------------------------

    def __init__(self, input_data=None, seed=None):
        super().__init__(seed=seed)
        if isinstance(input_data, OrganInputData):
            # Surface known cross-field footguns up front as warnings rather than
            # a silently broken render.
            for issue in input_data.validate():
                warnings.warn(f"[anatomy config] {issue}", stacklevel=2)
            self.params = input_data.to_dict_list()
        elif isinstance(input_data, list):
            self.params = input_data
        else:
            self.params = OrganInputData.for_monocot_stem().to_dict_list()

        # Containers for vascular tissue building (populated by _create_vascular_tissue)
        self.vascular_cells: CellManager = CellManager()
        self.vascular_polygons: list = []

        # Set by _create_central_layers when the pith is hollow (cavity_radius > 0);
        # a true void registered into the removal mask by allocate_vascular_tissue.
        self.pith_cavity_polygon = None

        self._parse_shared_params()
        self._parse_vascular_params()   # optional per-subclass hook (see below)
        self._initialize_default_layers()

    @property
    def planttype(self) -> int:
        """Plant type: 1 = monocot, 2 = dicot."""
        return int(self.global_params.get("value", 1))

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------

    def _parse_shared_params(self) -> None:
        """Parse the parameters common to all stem types.

        The central ground tissue is the ``pith`` param (the stem analogue of the
        root ``stele``); it drives the central-region thickness + cell-size
        gradient the same way.
        """
        self.global_params = self._get_param("planttype")

        pith = self._get_param("pith")
        self.vascular_params = {
            "thickness":                pith["thickness"],
            "cell_diameter":            pith["cell_diameter"],
            "cell_diameter_center":     pith.get("cell_diameter_center", pith["cell_diameter"]),
            "size_gradient_function":   pith.get("size_gradient_function",   "five_pl"),
            "size_gradient_inflection": pith.get("size_gradient_inflection", 0.5),
            "size_gradient_steepness":  pith.get("size_gradient_steepness",  3.0),
            "size_gradient_asymmetry":  pith.get("size_gradient_asymmetry",  1.0),
        }
        # Hollow / fistular pith: 0 = solid parenchyma; >0 = central medullary cavity.
        self.cavity_radius = float(pith.get("cavity_radius", 0.0))

        self.intercellular_spaces_params = [p for p in self.params if p["name"] == "inter_cellular_spaces"]
        self.aerenchyma_params = self._get_param("aerenchyma")

        # Peeled layers are any ordered param that is not the central pith or a
        # vascular tissue (xylem / phloem / cambium) or aerenchyma.
        self.layers = [
            p for p in self.params
            if "order" in p and p["name"] not in ("parenchyma", "xylem", "phloem", "cambium", "aerenchyma")
        ]
        self.layers = sorted(self.layers, key=lambda x: float(x["order"]))

    def _parse_vascular_params(self) -> None:
        """Optional per-subclass hook for extra vascular parsing.

        Not needed by the stem subclasses: the bundle cell-level sizes are read
        straight from the raw ``xylem`` / ``phloem`` / ``cambium`` param dicts by
        :func:`vascular_bundle.build_bundle`, and the bundle *count* is the
        ``vascular_bundle.n_bundles`` field — so there is a single source of truth
        for each and nothing to pre-parse here.
        """
        pass

    def _initialize_default_layers(self) -> None:
        """Initialise stem layers from parsed params."""
        for param in self.layers:
            self.layer_manager.add_layer(Layer.from_dict(param))

    # ------------------------------------------------------------------
    # Geometry — shared
    # ------------------------------------------------------------------

    def _create_base_shape(self) -> Polygon:
        # Same shape family as the root (GeometryProcessor.contour_polygon):
        # circle / ellipse / square / rectangle / triangle / star / focus_ellipse.
        sp_ = self._get_param("base_shape")
        return GeometryProcessor.contour_polygon(
            sp_.get("shape", "circle"),
            radius=self._calculate_radius(),
            width=float(sp_.get("width", 0.0)), height=float(sp_.get("height", 0.0)),
            n_branches=int(sp_.get("n_peaks", 5)),
            radius_peak_side=float(sp_.get("radius_peak_side", 0.6)),
            radius_valley_side=float(sp_.get("radius_valley_side", 0.4)),
            arc_peak_side=float(sp_.get("arc_peak_side", 0.05)),
            arc_valley_side=float(sp_.get("arc_valley_side", 0.10)),
            profile=sp_.get("profile"), exponent=float(sp_.get("exponent", 4.0)),
        )

    def _calculate_radius(self) -> float:
        radius = self.vascular_params["thickness"] / 2
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, "n_layers"):
                radius += layer.get_total_thickness()
            elif hasattr(layer, "cell_diameter"):
                radius += layer.cell_diameter
        return radius

    def _pith_diameter_fn(self):
        """Pith radial size gradient: ``r_norm`` (0 = centre .. 1 = pith edge) ->
        ground-tissue cell diameter.  The single source for both the central pith
        rings and the outer-sheath sizing against the surrounding ground cells."""
        return rescale(
            GRADIENT_FUNCTIONS[self.vascular_params["size_gradient_function"]],
            lo=self.vascular_params["cell_diameter"],
            hi=self.vascular_params["cell_diameter_center"],
            c=self.vascular_params["size_gradient_inflection"],
            b=self.vascular_params["size_gradient_steepness"],
            m=self.vascular_params["size_gradient_asymmetry"],
        )

    def _pith_cell_diameter_at(self, r: float, r_pith: float) -> float:
        """Ground-tissue (pith) cell diameter at radius ``r`` mm from the organ
        centre, evaluated from the same gradient the central rings use — so a
        bundle's outer sheath is sized against the ground cells it sits among."""
        r_norm = float(np.clip(r / r_pith, 0.0, 1.0)) if r_pith > 0 else 1.0
        return float(self._pith_diameter_fn()(r_norm))

    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[LayerPolygon]:
        """Create pith parenchyma rings from the pith edge toward the centre.

        Identical machinery to the root centre (``stele``), but the cells are
        tagged ``pith`` — the stem's central ground tissue.
        """
        central_layers = []

        diameter_fn = self._pith_diameter_fn()

        pith_radius = np.sqrt(current_polygon.area / np.pi)
        cx, cy = current_polygon.centroid.x, current_polygon.centroid.y
        min_order = min(l.order for l in self.layer_manager.get_layers() if l.order > 0)
        space_increment = self.layer_manager.get_layer_by_order(min_order).cell_diameter / 2
        i_layer = len(params)

        while not current_polygon.is_empty and current_polygon.area > 0:
            # Hollow pith: stop building rings once we reach the medullary cavity,
            # leaving the centre free of cells (a true void, like a big lacuna).
            if self.cavity_radius > 0.0 and np.sqrt(current_polygon.area / np.pi) <= self.cavity_radius:
                break

            r_norm = np.clip(np.sqrt(current_polygon.area / np.pi) / pith_radius, 0.0, 1.0)
            cell_diameter = diameter_fn(r_norm)

            if current_polygon.area <= (cell_diameter / 2) ** 2 * np.pi:
                break

            current_polygon = GeometryProcessor.buffer_polygon(
                current_polygon, -space_increment - cell_diameter / 2, smooth_factor=0.0,
            )
            space_increment = cell_diameter / 2

            central_layers.append(LayerPolygon(
                name="parenchyma",
                polygon=current_polygon,
                cell_diameter=cell_diameter,
                id_layer=i_layer + 1,
            ))
            i_layer += 1

        if self.cavity_radius > 0.0:
            self.pith_cavity_polygon = Point(cx, cy).buffer(self.cavity_radius)

        return central_layers

    def _create_vascular_tissue(self, polygon):
        """Build the vascular tissue, then register the medullary cavity (if any).

        The cavity is a *void*: no cells are placed in it, and its polygon is
        added to ``vascular_tissue_polygons`` so ``Organ.generate_cells`` clears
        any ground-tissue seed inside it (a clean hollow centre) and
        ``plot_tissues`` draws it as empty.  Hooked here — not in
        ``allocate_vascular_tissue`` — because the tissue-view dry run calls
        ``_create_vascular_tissue`` directly, so both paths register the cavity.
        """
        super()._create_vascular_tissue(polygon)
        if self.pith_cavity_polygon is not None and not self.pith_cavity_polygon.is_empty:
            self.vascular_tissue_polygons.setdefault("medullary cavity", []).append(
                self.pith_cavity_polygon
            )

    def _register_bundle(self, res) -> None:
        """Record one built bundle for the removal mask and the tissue view.

        The envelope goes in first (drawn underneath), then each sub-zone under
        its role key so ``plot_tissues`` shows the xylem / phloem / cambium /
        sclerenchyma arrangement.  Sub-zones are subsets of the envelope, so the
        removal-mask union — hence the cell census — is unchanged by adding them.
        """
        if res.envelope is not None and not res.envelope.is_empty:
            self.vascular_tissue_polygons.setdefault("bundle", []).append(res.envelope)
        for role, geom in res.zone_polygons:
            if geom is not None and not geom.is_empty:
                self.vascular_tissue_polygons.setdefault(role, []).append(geom)
        self.vascular_polygons.extend(res.vessel_polygons)

    def add_intercellular_spaces(self):
        """Add cortex air spaces, then tidy the hollow cavity + the lacunae.

        The pith itself is left exactly as the plain stele-style parenchyma (same
        machinery as the root stele — no special central cell).  Run after the
        normal intercellular/aerenchyma pass so it has the final say on the centre.
        """
        super().add_intercellular_spaces()
        self._finalize_pith_and_lacunae()

    def _pith_is_aerenchyma(self) -> bool:
        """True only when aerenchyma is explicitly requested for the pith.

        (An ``aerenchyma`` param targeting ``pith`` with a non-zero proportion.)
        Aerenchyma is never the silent default of a plain stem.
        """
        aer = self.aerenchyma_params or {}
        tissue = aer.get("tissue")
        tissues = list(tissue) if isinstance(tissue, (list, tuple)) else [tissue]
        return "parenchyma" in tissues and float(aer.get("aerenchyma_proportion", 0) or 0) > 0.0

    @staticmethod
    def _largest_piece(geom):
        """Largest Polygon piece of a (possibly Multi)Polygon, or None."""
        if geom is None or geom.is_empty:
            return None
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        parts = [g for g in parts if g.geom_type == "Polygon" and not g.is_empty]
        return max(parts, key=lambda g: g.area) if parts else None

    def _finalize_pith_and_lacunae(self) -> None:
        """Post-Voronoi tidy-up of the pith centre and the protoxylem lacunae.

        The solid pith is left untouched — plain stele-style parenchyma, like the
        root stele.  Two exceptions to that:

        * a hollow (fistular) pith: pith cells are clipped out of the medullary
          cavity so it stays a clean void instead of a fan of slivers spilling in;
        * aerenchyma explicitly requested for the pith: the whole pith collapses
          into one ``air space`` zone.

        Each protoxylem lacuna is a gas cavity: the cells under it are removed and
        it becomes one ``air space`` cell (the tag the root/organ aerenchyma path
        uses too).
        """
        pith_cells = [c for c in self.all_cells.get_cells_by_type("parenchyma")
                      if c.polygon is not None and not c.polygon.is_empty]
        cavity = self.pith_cavity_polygon
        cavity = cavity if (cavity is not None and not cavity.is_empty) else None

        if pith_cells and self._pith_is_aerenchyma():
            # Opt-in: whole pith -> one air-space zone.
            region = unary_union([c.polygon for c in pith_cells])
            if cavity is not None:
                region = region.difference(cavity)
            id_layer = pith_cells[0].id_layer
            self.all_cells.cells = [c for c in self.all_cells.cells if c.type != "parenchyma"]
            consider_as_cell(self.all_cells, region, "air space",
                             id_layer=id_layer, replace=False)

        elif pith_cells and cavity is not None:
            # Hollow culm: clip cells out of the cavity so it stays a void.  Any
            # type can spill in — pith cells bordering it, or a bundle's outer
            # sheath cell whose Voronoi region balloons across the empty cavity —
            # so clip by overlap, not by tissue type.
            kept = []
            for c in self.all_cells.cells:
                if c.polygon is not None and c.polygon.intersects(cavity):
                    piece = self._largest_piece(c.polygon.difference(cavity))
                    if piece is None:
                        continue                    # cell was entirely inside the cavity
                    c.polygon = piece
                kept.append(c)
            self.all_cells.cells = kept

        # (Protoxylem lacunae are seeded as ordinary 'air space' cells by the
        # bundle builder, so they need no post-Voronoi carving here.)

    # ------------------------------------------------------------------
    # Bundle placement geometry (shared by both plant types)
    # ------------------------------------------------------------------

    def _bundle_clearance(self, bp: dict) -> float:
        """Minimum ground-tissue gap required between two bundle envelopes.

        Bundles are separated by ground tissue in a real stem, so two envelopes
        must never touch: we keep at least one parenchyma cell's worth of clear
        space between them.
        """
        return float(bp.get("parenchyma_diameter", 0.012))

    def _placed_bundle_envelope(self, cx: float, cy: float, theta: float, bp: dict,
                                ground_cell_size: float = None,
                                anchor: float = 0.0) -> Polygon:
        """The bundle footprint placed at ``(cx, cy)`` oriented radially at ``theta``.

        Reproduces exactly what :func:`vascular_bundle.build_bundle` sets as
        ``res.envelope`` — but without seeding any cells — so a placement routine
        can test a *candidate* footprint for overlap before committing to it.
        ``ground_cell_size`` (the local pith cell size) grows the footprint by the
        outer bundle sheath, so overlap tests spend the same clearance the built
        bundle will actually occupy.  ``anchor`` mirrors ``build_bundle``'s cambium
        anchoring so the tested footprint sits exactly where the built one will.
        """
        from shapely.affinity import translate
        from openalea.granap.vascular_bundle import _local_envelope, outer_sheath_mask_pad
        env_local = _local_envelope(bp["width"], bp["height"], bp.get("shape", "ellipse"))
        pad = outer_sheath_mask_pad(bp, ground_cell_size)
        if pad > 0:
            env_local = env_local.buffer(pad)
        if anchor:
            env_local = translate(env_local, 0.0, -anchor)
        return GeometryProcessor.place_local([env_local], cx, cy, np.degrees(theta))[0]

    def _bundle_overlaps(self, env: Polygon, others: List[Polygon], gap: float) -> bool:
        """True if ``env`` comes within ``gap`` of any already-placed envelope."""
        if not others:
            return False
        probe = env.buffer(gap) if gap > 0 else env
        return any(probe.intersects(o) for o in others)

    def _bundle_specs(self) -> List[dict]:
        """All ``vascular_bundle`` param specs, sorted by inner band radius.

        A stem may carry more than one ``vascular_bundle`` entry, each owning a
        radial band (``radius_min`` .. ``radius_max`` mm from the organ centre).
        The monocot atactostele places each spec's ``n_bundles`` within its own
        annulus, so bundle *kind* can vary with radius; with a single spec (the
        default) that one kind fills the whole ground tissue.  ``_get_param``
        still returns just the first entry — this is the multi-spec view.
        """
        return sorted(
            (p for p in self.params if p["name"] == "vascular_bundle"),
            key=lambda b: float(b.get("radius_min", 0.0)),
        )

    @staticmethod
    def _spec_band(bp: dict, r_pith: float, r_section: float = None) -> Tuple[float, float]:
        """Radial band ``(r_lo, r_hi)`` of a bundle spec, in mm.

        ``radius_max <= 0`` means the band runs out to the pith edge (the default
        ground tissue), so ``r_hi`` falls back to ``r_pith``.  An explicit
        ``radius_max`` may reach *past* the pith into the cortex/rind — it is clamped
        only to ``r_section`` (the whole cross-section out to the epidermis) so a band
        can place bundles in any tissue.  ``r_section`` defaults to ``r_pith`` (the
        historical pith-only behaviour) when the caller does not supply it.
        """
        if r_section is None:
            r_section = r_pith
        r_lo = float(bp.get("radius_min", 0.0))
        r_hi = float(bp.get("radius_max", 0.0))
        return r_lo, (min(r_hi, r_section) if r_hi > 0.0 else r_pith)

    def reshape_layers(self, layers_polygons: List[LayerPolygon]) -> List[LayerPolygon]:
        return layers_polygons

    def set_vascular_params(self, **kwargs) -> None:
        self.vascular_params.update(kwargs)
        self._invalidate_geometry()

    def _which_layer_for_vascular(self, layers_polygons: List[LayerPolygon]):
        """Region the vascular recipe operates within — the (outermost) pith ring.

        For the dicot eustele the bundles form a ring at this region's periphery;
        for the monocot atactostele they scatter across it.  Both subclasses key
        their placement off this (pith) polygon, which also fixes the ground-tissue
        radius a full-span bundle band is capped to.

        A bundle band may reach *past* the pith into the cortex/rind, though (a
        ``vascular_bundle`` spec whose ``radius_max`` exceeds the pith radius) — for
        that the whole cross-section (everything inside the epidermis) is stashed on
        ``self._vascular_section_polygon`` for the placement code to test against.

        Falls back to the innermost layer polygon if the pith was fully hollowed
        out (``cavity_radius`` >= the pith radius, so no ``pith`` rings exist).
        """
        names = [l["name"] for l in layers_polygons]
        section = [l["polygon"] for l in layers_polygons if l["name"] != "outside"]
        self._vascular_section_polygon = unary_union(section) if section else None
        # Radius -> local tissue cell size and name, so a bundle placed out in the
        # cortex/rind is sized (and its sheath cells named) against the cells it
        # actually sits among (not the pith gradient).
        # (r_outer, cell_diameter, tissue_name) per real layer, sorted outward.
        sizes = []
        for l in layers_polygons:
            if l["name"] == "outside" or l["polygon"] is None or l["polygon"].is_empty:
                continue
            xs, ys = l["polygon"].exterior.coords.xy
            r_out = float(np.max(np.hypot(np.asarray(xs), np.asarray(ys))))
            layer = self.get_layer(l["name"])
            if layer is not None and layer.cell_diameter > 0:
                sizes.append((r_out, float(layer.cell_diameter), l["name"]))
        self._vascular_layer_sizes = sorted(sizes)
        idx = names.index("parenchyma") if "parenchyma" in names else len(layers_polygons) - 1
        return layers_polygons[idx]["polygon"]

    def _local_ground_cell_size(self, x: float, y: float, r_pith: float) -> float:
        """Cell diameter of the tissue at ``(x, y)`` — the ground a bundle sits in.

        Inside the pith (r <= ``r_pith``) this follows the pith gradient (as before);
        past the pith it is the cell size of the layer (cortex, hypodermis, ...) whose
        ring contains the point, so a peripheral bundle's sheath is sized against the
        rind cells around it rather than the much larger pith cells.
        """
        r = float(np.hypot(x, y))
        if r <= r_pith or not getattr(self, "_vascular_layer_sizes", None):
            return self._pith_cell_diameter_at(r, r_pith)
        for r_out, cd, _name in self._vascular_layer_sizes:
            if r <= r_out:
                return cd
        return self._vascular_layer_sizes[-1][1]

    def _local_ground_tissue_name(self, x: float, y: float, r_pith: float) -> str:
        """Name of the tissue at ``(x, y)`` — the ground a bundle sheath cell sits in.

        The stem analogue of :meth:`_local_ground_cell_size`: inside the pith
        (r <= ``r_pith``) the ground tissue is the pith parenchyma (tagged
        ``parenchyma``); past the pith it is the layer (cortex, hypodermis, ...)
        whose ring contains the point.  Used to name each outer bundle-sheath cell
        after the tissue it is fitted into instead of a generic ``bundle sheath``.
        """
        r = float(np.hypot(x, y))
        if r <= r_pith or not getattr(self, "_vascular_layer_sizes", None):
            return "parenchyma"
        for r_out, _cd, name in self._vascular_layer_sizes:
            if r <= r_out:
                return name
        return self._vascular_layer_sizes[-1][2]

    def _neighbour_ground_cell_size(self, cx: float, cy: float, bp: dict,
                                    r_pith: float, k: int = 12) -> float:
        """Representative ground-cell size of the bundle's *neighbours*.

        Samples :meth:`_local_ground_cell_size` at ``k`` points just outside the
        bundle footprint (a ring of radius ~half its larger dimension) and returns
        the median.  Sizing the outer sheath and the removal mask from the
        neighbouring cells — not only the tissue the bundle *centre* falls in —
        keeps a bundle that straddles two tissues from reserving the coarse-tissue
        clearance everywhere: a small bundle in fine tissue no longer clears the
        space a coarse-celled centre would demand, so it stops crowding out its
        smaller neighbours.
        """
        w, h = float(bp.get("width", 0.12)), float(bp.get("height", 0.18))
        rr = 0.5 * max(w, h) + 0.5 * float(bp.get("sclerenchyma_cell_diameter", 0.008))
        sizes = [self._local_ground_cell_size(cx + rr * np.cos(a), cy + rr * np.sin(a), r_pith)
                 for a in np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)]
        sizes = [s for s in sizes if s and s > 0]
        return float(np.median(sizes)) if sizes else self._local_ground_cell_size(cx, cy, r_pith)


# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# ---------------------------------------------------------------------------
# The concrete subclasses live in their own modules (``stem_monocot_class`` /
# ``stem_dicot_class``).  Expose them here lazily (PEP 562 module __getattr__)
# so ``from openalea.granap.stem_class import DicotStemAnatomy`` works without a
# circular import at load time.

def __getattr__(name):
    if name == "MonocotStemAnatomy":
        from openalea.granap.stem_monocot_class import MonocotStemAnatomy
        return MonocotStemAnatomy
    if name == "DicotStemAnatomy":
        from openalea.granap.stem_dicot_class import DicotStemAnatomy
        return DicotStemAnatomy
    if name == "ContinuousDicotStemAnatomy":
        from openalea.granap.stem_dicot_continuous_class import ContinuousDicotStemAnatomy
        return ContinuousDicotStemAnatomy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
