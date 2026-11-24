"""Resin-duct placement in ``NeedleAnatomy._duct_zone_data``.

Two placement modes share one geometry pipeline (see that method's docstring):
the historical fixed pizza-slice positions, and explicit bearings via
``"angles"`` on a ``resin_duct`` param dict (used by
``example/needle/pinus_nigra.py`` to put large ducts at the corners and the
abaxial pole and a small one on one flank).

These tests work on the *geometry* stage only -- ``_duct_zone_data`` takes
layer polygons and places no cells -- so they stay fast and independent of the
Voronoi census guarded by ``test_vascular_regression.py``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(".."))

from openalea.granap.needle_class import NeedleAnatomy, _DUCT_PLACEMENT_ORDER


WIDTH, THICKNESS = 1.396, 0.955

# Sizes taken from example/needle/pinus_nigra.py, whose two duct classes exist
# precisely to exercise per-duct sizing.
LARGE = {"lumen_diameter": 0.037, "cell_diameter": 0.006, "cell_width": 0.013,
         "sheath_cell_diameter": 0.018, "sheath_cell_width": 0.023}
SMALL = {"lumen_diameter": 0.022, "cell_diameter": 0.004, "cell_width": 0.008,
         "sheath_cell_diameter": 0.011, "sheath_cell_width": 0.014}


def _base_params(*resin_ducts, mesophyll_rings=2):
    """A minimal needle config; the duct blocks are appended as given.

    ``mesophyll_rings=2`` mirrors the real examples (pinus_nigra.py has three,
    pinus_pinaster.py two), which is the geometry the duct annulus is built
    for. Pass 1 to exercise the single-ring fallback.
    """
    params = [
        {"name": "planttype", "value": 3, "organ": "needle",
         "width": WIDTH, "thickness": THICKNESS},
        {"name": "central_cylinder", "vascular_width": 0.22, "vascular_height": 0.19,
         "vascular_angle": 25, "cell_diameter": 0.02},
        {"name": "endodermis", "cell_diameter": 0.03, "cell_width": 0.055, "order": 3},
        {"name": "mesophyll", "cell_diameter": 0.052, "cell_width": 0.044, "order": 4},
        {"name": "palisade", "cell_diameter": 0.055, "cell_width": 0.038, "order": 4.5},
        {"name": "hypodermis", "cell_diameter": 0.025, "cell_width": 0.03,
         "n_layers": 2, "order": 5},
        {"name": "epidermis", "cell_diameter": 0.02, "cell_width": 0.015, "order": 6},
        # Required by NeedleAnatomy._initialize_params / the vascular stage.
        # Most tests here stop at the geometry stage and never place a cell,
        # but the adjacency test below runs a full generate_cells().
        {"name": "transfusion_tissue", "n_layers": 2, "pack_circles": True,
         "diameter_max": 0.055, "proportion": 0.85,
         "parenchyma_diameter": 0.0505, "tracheids_diameter": 0.025,
         "transfusion_tracheids_ratio": 3.0},
        {"name": "xylem", "cell_diameter": 0.008, "n_files": 3,
         "n_clusters": 4, "n_per_cluster": 3},
        {"name": "phloem", "cell_diameter": 0.005, "n_files": 3},
        {"name": "cambium", "cell_diameter": 0.006},
        {"name": "Strasburger cells", "cell_diameter": 0.02},
    ]
    if mesophyll_rings > 1:
        params.append({"name": "mesophyll_second", "cell_diameter": 0.052,
                       "cell_width": 0.044, "order": 4.1})
    params.extend(resin_ducts)
    return params


def _duct_data(*resin_ducts, mesophyll_rings=2):
    """Build the layer polygons and return ``_duct_zone_data``'s duct list."""
    organ = NeedleAnatomy(_base_params(*resin_ducts, mesophyll_rings=mesophyll_rings), seed=0)
    layers_polygons = organ.generate_layer_polygons()
    duct_data, _rdp = organ._duct_zone_data(layers_polygons)
    return organ, layers_polygons, duct_data


def _bearing(organ, layers_polygons, duct):
    """Polar angle (deg) of a duct centre, in pole_and_corner_angles' frame."""
    y_c = 3.5 * THICKNESS / (3.0 * np.pi)
    return float(np.degrees(np.arctan2(duct["center"].y - y_c, duct["center"].x)) % 360.0)


# ---------------------------------------------------------------------------
# Explicit placement
# ---------------------------------------------------------------------------

def test_angles_place_one_duct_per_bearing():
    angles = [192.4, 346.0, 85.7]
    _organ, _lps, ducts = _duct_data({"name": "resin_duct", "angles": angles, **LARGE})
    assert len(ducts) == len(angles)


@pytest.mark.parametrize("angle", [192.4, 346.0, 85.7, 15.5])
def test_duct_lands_on_its_requested_bearing(angle):
    """Each duct sits on the requested bearing, within its own wedge.

    The radius is whatever the annulus has room for (fit_inner_ellipse's
    Chebyshev-centre search), so only the bearing is asserted -- but it must
    stay inside the half-width the wedge was cut with, otherwise the wedge is
    not what actually positioned the duct.
    """
    organ, lps, ducts = _duct_data({"name": "resin_duct", "angles": [angle], **LARGE})
    assert len(ducts) == 1
    got = _bearing(organ, lps, ducts[0])
    assert NeedleAnatomy._circular_diff(got, angle) <= 180.0 / 7.0


def test_wedge_overrides_the_default_half_width():
    """A narrower wedge still yields a duct, and keeps it nearer the bearing."""
    _o, _l, wide = _duct_data({"name": "resin_duct", "angles": [15.5], **LARGE})
    _o2, _l2, narrow = _duct_data({"name": "resin_duct", "angles": [15.5], "wedge": 8.0, **LARGE})
    assert len(wide) == 1 and len(narrow) == 1
    # Both are placed; the narrow wedge cannot drift further off-bearing than
    # its own half-width.
    organ = NeedleAnatomy(_base_params(), seed=0)
    y_c = 3.5 * THICKNESS / (3.0 * np.pi)
    got = float(np.degrees(np.arctan2(narrow[0]["center"].y - y_c, narrow[0]["center"].x)) % 360.0)
    assert NeedleAnatomy._circular_diff(got, 15.5) <= 8.0


def test_several_blocks_give_ducts_of_different_sizes():
    """Two resin_duct blocks -> per-duct sizing, not the first block's for all."""
    _o, _l, ducts = _duct_data(
        {"name": "resin_duct", "angles": [192.4, 346.0, 85.7], **LARGE},
        {"name": "resin_duct", "angles": [15.5], **SMALL},
    )
    assert len(ducts) == 4

    # Absolute sizes are not comparable across ducts -- scale-to-fit shrinks
    # each one independently by however much its own wedge is short of the
    # requested radius. What must hold is that every duct carries the
    # *proportions* of the block it came from, which is scale-invariant.
    def profile(sizes):
        return (sizes["lumen_diameter"] / sizes["cell_diameter"],
                sizes["lumen_diameter"] / sizes["sheath_cell_diameter"])

    want_large, want_small = profile(LARGE), profile(SMALL)
    assert want_large != want_small, "the two blocks must be distinguishable"

    n_large = n_small = 0
    for d in ducts:
        got = profile(d)
        if got == pytest.approx(want_large, rel=1e-6):
            n_large += 1
        elif got == pytest.approx(want_small, rel=1e-6):
            n_small += 1
        else:
            raise AssertionError(f"duct matches neither block's proportions: {got}")
        assert d["carve"].area > 0
    assert (n_large, n_small) == (3, 1), \
        f"expected 3 ducts from the large block and 1 from the small, got {n_large}/{n_small}"


def test_small_block_keeps_its_own_proportions():
    _o, _l, ducts = _duct_data({"name": "resin_duct", "angles": [15.5], **SMALL})
    d = ducts[0]
    # Inside-out and additive: epithelium outer edge = canal grown by
    # cell_diameter, and the sheath sits strictly outside that.
    assert d["epithelium_outer"].area > d["canal"].area
    assert d["carve"].area > d["epithelium_outer"].area
    ratio = d["lumen_diameter"] / SMALL["lumen_diameter"]
    assert d["cell_diameter"] == pytest.approx(SMALL["cell_diameter"] * ratio, rel=1e-6), \
        "a scaled-to-fit duct scales every band by the same factor"


# ---------------------------------------------------------------------------
# Slice placement — unchanged behaviour
# ---------------------------------------------------------------------------

def test_n_files_without_angles_uses_fixed_slices():
    _o, _l, ducts = _duct_data({"name": "resin_duct", "n_files": 2, **LARGE})
    assert len(ducts) == 2


def test_slice_placement_matches_the_documented_algorithm():
    """Re-derive slice placement independently and require an exact match.

    Recomputes the whole chain here -- the home zone (palisade + mesophyll
    family, buffered off the hypodermis and endodermis by their own
    cell_diameter), the subhypodermal band, ``pizza_slice`` and
    ``_DUCT_PLACEMENT_ORDER`` -- rather than trusting the code to agree with
    itself. 3 ducts stay on the legacy ``_DUCT_PLACEMENT_ORDER`` fallback
    (only exactly 2 slice-placed ducts are corner-anchored -- see
    ``test_two_slice_placed_ducts_land_on_the_true_corners`` below).

    The actual per-slice fit is delegated to ``organ._build_duct`` rather
    than re-deriving its assembly-radius formula a second time here -- that
    formula (built core + transition ring + carve margin, from unscaled
    sizes) is exercised directly by ``test_resin_duct_placement.py``'s other
    assembly-fit tests; duplicating it here would just be two copies of the
    same nontrivial arithmetic drifting apart.
    """
    from openalea.granap.geometry_collection import GeometryProcessor
    from openalea.granap.needle_class import _DUCT_RING_BUFFER_FACTOR

    rdp = {"name": "resin_duct", "n_files": 3, **LARGE}
    organ, lps, ducts = _duct_data(rdp)
    assert len(ducts) == 3

    names = [l["name"] for l in lps]
    home_idx = [i for i, n in enumerate(names)
                if n == "palisade" or n == "mesophyll" or n.startswith("mesophyll_")]
    outer_idx = home_idx[0] - 1
    home_outer = lps[outer_idx]["polygon"] if outer_idx >= 0 else lps[home_idx[0]]["polygon"]
    home_inner = lps[home_idx[-1]]["polygon"]

    hyp = next(p for p in _base_params() if p["name"] == "hypodermis")["cell_diameter"]
    endo = next(p for p in _base_params() if p["name"] == "endodermis")["cell_diameter"]
    zone = GeometryProcessor.buffer_polygon(home_outer, -hyp, 0).difference(
        GeometryProcessor.buffer_polygon(home_inner, endo, 0))

    built = LARGE["lumen_diameter"] + 2 * LARGE["cell_diameter"] + 2 * LARGE["sheath_cell_diameter"]
    shell = home_outer.difference(
        GeometryProcessor.buffer_polygon(home_outer, -built * _DUCT_RING_BUFFER_FACTOR, 0))
    band = shell.intersection(zone)
    band_slices = GeometryProcessor.pizza_slice(band, 7)
    zone_slices = GeometryProcessor.pizza_slice(zone, 7)

    mesophyll_cell_diameter = next(p for p in _base_params() if p["name"] == "mesophyll")["cell_diameter"]
    expected = []
    for slice_id in sorted(_DUCT_PLACEMENT_ORDER[:3]):
        duct = organ._build_duct(band_slices[slice_id], LARGE, built, mesophyll_cell_diameter)
        if duct["cell_diameter"] < LARGE["cell_diameter"] * 0.999:
            zone_duct = organ._build_duct(zone_slices[slice_id], LARGE, built, mesophyll_cell_diameter)
            if zone_duct["cell_diameter"] > duct["cell_diameter"]:
                duct = zone_duct
        expected.append(duct["center"])

    got = [d["center"] for d in ducts]
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g.distance(e) < 1e-9


def test_ducts_never_touch_hypodermis_or_endodermis():
    """A resin duct lives in the home zone -- palisade included -- and must
    never touch the hypodermis or the endodermis. Palisade encroachment is
    now *allowed*, not required: since palisade is part of the home zone (it
    *is* palisade mesophyll), a duct that fits entirely within the ordinary
    mesophyll ring(s) is just as valid as one that also carves palisade. The
    hard requirement is the two forbidden boundaries, with the buffer
    (that neighbour's own cell_diameter) each carve mask must clear."""
    _o, lps, ducts = _duct_data(
        {"name": "resin_duct", "angles": [192.4, 346.0, 85.7], **LARGE},
        {"name": "resin_duct", "angles": [15.5], **SMALL},
    )
    assert len(ducts) == 4

    names = [l["name"] for l in lps]
    endo = lps[names.index("endodermis")]["polygon"]
    hyp_i = names.index("hypodermis")
    hypodermis_band = lps[hyp_i - 1]["polygon"].difference(lps[hyp_i]["polygon"])

    hyp_clearance = next(p for p in _base_params() if p["name"] == "hypodermis")["cell_diameter"]
    endo_clearance = next(p for p in _base_params() if p["name"] == "endodermis")["cell_diameter"]

    for d in ducts:
        assert not d["carve"].intersects(endo), "duct carve reaches the endodermis"
        assert not d["carve"].intersects(hypodermis_band), "duct carve reaches the hypodermis"
        # Not just non-intersection -- the full required clearance. A small
        # negative slack (1% of the buffer) absorbs polygon-buffer/pizza_slice
        # discretisation, not a real shortfall.
        assert d["carve"].distance(endo) >= endo_clearance * 0.99, \
            "duct carve is closer to the endodermis than its required buffer"
        assert d["carve"].distance(hypodermis_band) >= hyp_clearance * 0.99, \
            "duct carve is closer to the hypodermis than its required buffer"


def test_no_resin_duct_params_yields_nothing():
    _o, _l, ducts = _duct_data()
    assert ducts == []


# ---------------------------------------------------------------------------
# Every duct is surrounded by mesophyll — the default, not a size-ratio
# coincidence
# ---------------------------------------------------------------------------

def test_transition_ring_is_unconditional():
    """A duct gets its host-tissue ring even when sheath and host sizes are
    close (here 0.052/0.018 = 2.9, which the old _DUCT_SHEATH_MIN_RATIO = 4
    guard rejected, leaving the sheath bordering palisade directly)."""
    _o, _l, ducts = _duct_data({"name": "resin_duct", "angles": [192.4, 85.7], **LARGE})
    assert len(ducts) == 2
    for d in ducts:
        assert d["transition_ring"] is not None, "every duct needs its mesophyll ring"
        assert d["transition_cell_size"] > 0
        # Geometric mean of sheath cell and host (mesophyll) cell.
        assert d["transition_cell_size"] == pytest.approx(
            np.sqrt(d["sheath_cell_diameter"] * 0.052), rel=1e-6)
        # The ring lies strictly outside the sheath, and the carve mask
        # covers it so the host seeds under it are replaced by its own cells.
        assert d["transition_ring"].area > d["epithelium_outer"].area
        assert d["carve"].area > d["transition_ring"].area


def test_duct_sheath_is_never_adjacent_to_palisade_or_hypodermis():
    """The placed cells, not just the geometry: walk each duct sheath cell's
    touching neighbours and require mesophyll (or duct tissue) all round.

    This is the actual guarantee -- a duct embedded in mesophyll -- and it
    needs a full generate_cells() because it is about Voronoi adjacency.
    """
    from openalea.granap.needle_class import NeedleAnatomy as NA

    params = _base_params(
        {"name": "resin_duct", "angles": [192.4, 346.0, 85.7], **LARGE},
        {"name": "resin_duct", "angles": [15.5], **SMALL},
    )
    organ = NA(params, seed=0)
    organ.generate_cells()

    sheath = [c for c in organ.all_cells.cells
              if c.type == "resin duct sheath" and c.polygon is not None]
    assert sheath, "no sheath cells were placed"

    others = [c for c in organ.all_cells.cells
              if c.polygon is not None and c.type != "resin duct sheath"]
    allowed = {"mesophyll", "resin duct epithelium", "duct", "air space"}
    forbidden = set()
    for s in sheath:
        for o in others:
            if o.type in allowed or o.type.startswith("mesophyll_"):
                continue
            # touching, not merely nearby: shared boundary of real length
            shared = s.polygon.intersection(o.polygon.buffer(1e-9))
            if not shared.is_empty and shared.length > 1e-6:
                forbidden.add(o.type)
    assert not forbidden, f"duct sheath borders non-mesophyll tissue: {sorted(forbidden)}"


# ---------------------------------------------------------------------------
# Real-config coverage: nigra, pinaster, gallery -- not just the synthetic
# harness. These import the actual example scripts (added to sys.path
# above) rather than re-typing their parameters, so a change to a real
# config's numbers can't silently drift out of sync with what this file
# tests.
# ---------------------------------------------------------------------------

def _assert_real_config_ducts(organ, require_full_size=True, min_scale=0.0):
    """Shared checks for a real config's ``NeedleAnatomy`` instance.

    Always hard-required: every duct's carve mask clears the hypodermis and
    the endodermis (the home-zone buffer, non-negotiable per the approved
    plan -- ducts shrink before buffers do).

    ``require_full_size``/``min_scale`` govern the *size* check, which is
    NOT uniformly true across every real config: once ``_build_duct`` fits
    the whole assembly (built core + transition ring + carve margin) rather
    than just the built core, nigra's tightest duct (its abaxial one, on the
    side the approved plan's headroom table already flagged as "just" -- 0.148
    usable vs 0.146 assembly, before this tightening) no longer reaches
    scale 1.0. That is the anatomical clearance rule doing its job, not a
    bug -- see the module comment on ``_build_duct``'s assembly_radius fit.
    Callers that know their config's ducts should all reach full size pass
    the default; nigra's test passes ``min_scale`` instead, matching the
    achieved scale observed when this test was written.
    """
    from openalea.granap.needle_class import _is_duct_home_layer

    lps = organ.generate_layer_polygons()
    ducts, _rdp = organ._duct_zone_data(lps)
    assert ducts, "no ducts were placed for this config"

    names = [l["name"] for l in lps]
    home_idx = sorted(i for i, n in enumerate(names) if _is_duct_home_layer(n))
    outer_idx = home_idx[0] - 1
    endo = lps[names.index("endodermis")]["polygon"]
    # The hypodermis band: between its own outer edge (the previous entry's
    # polygon) and its inner edge (its own polygon) -- same convention as
    # _duct_zone_data's home-zone construction.
    hypodermis_band = lps[outer_idx - 1]["polygon"].difference(lps[outer_idx]["polygon"])

    expected_cell_diameters = [p["cell_diameter"] for p in organ.params if p["name"] == "resin_duct"]

    for d in ducts:
        assert not d["carve"].intersects(endo), \
            f"duct at {d['center']} carve reaches the endodermis"
        assert not d["carve"].intersects(hypodermis_band), \
            f"duct at {d['center']} carve reaches the hypodermis"
        # Achieved scale against whichever resin_duct block's raw
        # cell_diameter is closest (a config may mix duct sizes, e.g.
        # pinus_nigra.py's large/small pair) -- scale-invariant matching
        # would require re-deriving lumen:cell_diameter proportions per
        # block; closest-raw-value is enough here since scale is bounded to
        # (0, 1] and blocks are chosen with distinct cell_diameters.
        closest = min(expected_cell_diameters, key=lambda want: abs(d["cell_diameter"] - want))
        scale = d["cell_diameter"] / closest if closest else 1.0
        if require_full_size:
            assert scale == pytest.approx(1.0, rel=0.02), \
                f"duct at {d['center']} did not keep its full size: " \
                f"cell_diameter={d['cell_diameter']}, closest raw size={closest}, scale={scale:.3f}"
        else:
            assert scale >= min_scale, \
                f"duct at {d['center']} scale {scale:.3f} dropped below the documented floor {min_scale}"
    return ducts


def test_pinaster_ducts_clear_boundaries_and_keep_full_size():
    from needle_configs import build_pinaster
    organ = NeedleAnatomy(build_pinaster(), seed=0)
    ducts = _assert_real_config_ducts(organ)
    assert len(ducts) == 2


def test_nigra_ducts_clear_boundaries_and_stay_near_full_size():
    """Nigra's abaxial duct sits in the config's tightest headroom (see the
    approved plan's headroom table) and, once the fit target became the
    whole assembly rather than just the built core, no longer reaches
    scale 1.0 on that one duct -- an anatomical consequence of the
    hypodermis/endodermis buffers, not a bug (do not loosen the buffers to
    force it back to 1.0). 0.95 is a regression floor a few points under the
    ~0.96 achieved when this test was written -- report the exact achieved
    scale rather than tightening this further if it moves.
    """
    from needle_configs import build_nigra
    organ = NeedleAnatomy(build_nigra(), seed=0)
    ducts = _assert_real_config_ducts(organ, require_full_size=False, min_scale=0.95)
    assert len(ducts) == 4


def test_gallery_ducts_clear_boundaries_and_keep_full_size():
    from needle_configs import build_gallery_needle_data
    organ = NeedleAnatomy(build_gallery_needle_data(), seed=0)
    ducts = _assert_real_config_ducts(organ)
    assert len(ducts) == 2


def test_two_slice_placed_ducts_land_on_the_true_corners():
    """Pinaster's n_files=2 resin_duct block (no ``angles``) must anchor to
    the needle's real corners (``NeedleAnatomy.pole_and_corner_angles``),
    not the historical pizza-slice positions 3/6 -- see the module comment
    on ``_DUCT_PLACEMENT_ORDER``. Tolerance is the escalating search's own
    widest wedge cap (180/7 deg): a duct can drift at most that far off its
    requested bearing before ``_build_duct`` would shrink it instead of
    seating it further off-bearing.
    """
    from needle_configs import build_pinaster
    organ = NeedleAnatomy(build_pinaster(), seed=0)
    lps = organ.generate_layer_polygons()
    ducts, _rdp = organ._duct_zone_data(lps)
    assert len(ducts) == 2

    width, thickness = organ._resolved_dimensions()
    _adax, _abax, corner_pos, corner_neg = NeedleAnatomy.pole_and_corner_angles(width, thickness)
    y_c = 3.5 * thickness / (3.0 * np.pi)
    got_bearings = sorted(
        float(np.degrees(np.arctan2(d["center"].y - y_c, d["center"].x)) % 360.0) for d in ducts
    )
    want_bearings = sorted([corner_pos, corner_neg])
    for got, want in zip(got_bearings, want_bearings):
        assert NeedleAnatomy._circular_diff(got, want) <= 180.0 / 7.0, \
            f"duct at {got:.1f} deg is not within one legacy slice's width of corner {want:.1f} deg"


def test_single_mesophyll_ring_keeps_ducts_inside_the_mesophyll():
    """With one mesophyll ring there is no inner sibling to subtract, and the
    zone used to fall back to the whole disc inside that ring -- placing
    ducts among the endodermis/transfusion tissue of the central cylinder.
    The ring's own band is used instead."""
    _o, lps, ducts = _duct_data({"name": "resin_duct", "n_files": 2, **LARGE},
                                mesophyll_rings=1)
    assert ducts, "a single-ring config should still get its ducts"

    names = [l["name"] for l in lps]
    endo = lps[names.index("endodermis")]["polygon"]
    for d in ducts:
        assert not endo.contains(d["center"]), \
            "duct was placed inside the endodermis/central cylinder"
