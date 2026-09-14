"""Tests for the parameter-metadata layer (grouping / tiering / dead-knob lint).

These guard the sidecar ``FIELD_META`` registry and the ``describe_params`` /
``OrganInputData.lint`` helpers that tame ``VascularBundleParams`` (69 flat fields).
The registry is metadata only — it must not change the emitted params (that
invariant is covered by ``test_param_schema_equivalence``); here we check the
*views* and the *lint* behave.
"""

from openalea.granap.input_data import (
    OrganInputData, VascularBundleParams, FIELD_META, describe_params,
    RENDERING_KNOBS, _field_tier, _field_kind, BUNDLE_GROUP_MODELS,
)


def test_registry_covers_every_bundle_field():
    """Every VascularBundleParams field (except name) is grouped + tiered."""
    meta = FIELD_META["VascularBundleParams"]
    fields = set(VascularBundleParams.model_fields) - {"name"}
    assert set(meta) == fields, (
        f"registry/field mismatch: only-in-registry={set(meta) - fields}, "
        f"only-in-model={fields - set(meta)}"
    )
    for f, m in meta.items():
        assert m["tier"] in ("primary", "advanced")
        assert m["group"]


def test_primary_tier_is_small():
    """The point of the tiering: a handful of primary knobs, not 69."""
    meta = FIELD_META["VascularBundleParams"]
    primary = [f for f, m in meta.items() if m["tier"] == "primary"]
    assert 5 <= len(primary) <= 15          # a curated few, well under the 69 total
    # the fields presets actually move must all be primary
    for f in ("bundle_type", "width", "height", "xylem_layout", "n_bundles", "sheath"):
        assert meta[f]["tier"] == "primary"


def test_describe_active_only_hides_off_mode_fields():
    """A 'files' bundle's description omits the face-only fields, and vice-versa."""
    files = VascularBundleParams(xylem_layout="files")
    face = VascularBundleParams(xylem_layout="face")
    assert "metaxylem_gap" not in describe_params(files, active_only=True)
    assert "metaxylem_gap" in describe_params(face, active_only=True)


def test_lint_flags_dead_knob():
    """A mode-gated field set while its mode is off is reported (and the value is
    otherwise silently ignored downstream)."""
    d = OrganInputData(params=[
        VascularBundleParams(xylem_layout="files", metaxylem_gap=0.09)
    ])
    warns = d.lint()
    assert any("metaxylem_gap" in w and "xylem_layout" in w for w in warns)


def test_lint_clean_when_modes_match():
    d = OrganInputData(params=[
        VascularBundleParams(xylem_layout="face", metaxylem_gap=0.09)  # active -> fine
    ])
    assert d.lint() == []


def test_lint_ignores_unregistered_params():
    """Params with no registered metadata (everything but the bundle today) are
    skipped, not errored."""
    d = OrganInputData.for_root()
    assert isinstance(d.lint(), list)      # no crash on stele/xylem/etc.


# ---------------------------------------------------------------------------
# B1: package-wide auto-tiering (primary = preset-moved)
# ---------------------------------------------------------------------------

def test_auto_tier_primary_is_preset_moved():
    """A field a preset moves is primary; a field no preset touches is advanced —
    even for classes with no explicit FIELD_META entry (root stele)."""
    # root preset moves stele.thickness (SteleParams) -> primary
    assert _field_tier("SteleParams", "thickness") == "primary"
    # a never-moved gradient tuning knob on the root xylem -> advanced
    assert _field_tier("RootXylemParams", "gradient_asymmetry") == "advanced"


def test_organ_describe_is_curated_not_the_full_wall():
    """OrganInputData.describe(primary, active) shows far fewer lines than the raw
    flat surface — the whole point of the tiering."""
    d = OrganInputData.for_dicot_stem()
    primary = d.describe(tier="primary", active_only=True)
    full_fields = sum(len(b) for b in d.to_dict_list())
    # curated view lists well under half the raw field count
    assert primary.count(" = ") < full_fields / 2


# ---------------------------------------------------------------------------
# B2: rendering (algorithm-tuning) knobs are quarantined
# ---------------------------------------------------------------------------

def test_rendering_knobs_are_classified_and_advanced():
    """The algorithm-tuning knobs are tagged 'rendering' and never primary."""
    for k in ("packing_strategy", "xylem_file_jitter", "first_vessel_shift", "smoothness"):
        assert k in RENDERING_KNOBS
        assert _field_kind(k) == "rendering"
    # and hide_rendering actually removes them from a bundle view
    b = VascularBundleParams(xylem_layout="files")
    shown = describe_params(b, hide_rendering=True)
    assert "xylem_file_jitter" not in shown
    assert "packing_strategy" not in shown


# ---------------------------------------------------------------------------
# v2: nested sub-model construction with flat emission
# ---------------------------------------------------------------------------

def test_group_models_cover_every_field():
    """The derived group sub-models partition the bundle's flat fields exactly."""
    flat = set(VascularBundleParams.model_fields) - {"name"}
    grouped = set()
    for m in BUNDLE_GROUP_MODELS.values():
        grouped |= (set(m.model_fields) - {"name"})
    assert grouped == flat


def test_nested_construction_equals_flat():
    """Authoring a bundle via group sub-models yields the identical flat model."""
    G = BUNDLE_GROUP_MODELS
    nested = VascularBundleParams(
        bundle_type="collateral",
        envelope=G["envelope"](width=0.5, height=0.3),
        xylem_face=G["xylem_face"](xylem_layout="face", n_metaxylem=1),
        placement=G["placement"](n_bundles=12),
    )
    flat = VascularBundleParams(bundle_type="collateral", width=0.5, height=0.3,
                                xylem_layout="face", n_metaxylem=1, n_bundles=12)
    assert nested.model_dump() == flat.model_dump()


def test_scalar_fields_that_share_a_group_name_stay_flat():
    """sheath / placement / ring_shape are both group names and flat fields; a scalar
    value must be kept as the flat field, not eaten as a (empty) group block."""
    b = VascularBundleParams(sheath="caps", placement="even", ring_shape="star")
    assert b.sheath == "caps" and b.placement == "even" and b.ring_shape == "star"


def test_explicit_flat_kwarg_wins_over_group_block():
    G = BUNDLE_GROUP_MODELS
    b = VascularBundleParams(envelope=G["envelope"](width=0.5), width=0.9)
    assert b.width == 0.9
