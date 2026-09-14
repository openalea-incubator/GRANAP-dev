# Vascular-bundle sclerenchyma caps — per-pole layer control

This note proposes how to let a vascular bundle carry **sclerenchyma (fibre)
caps** whose thickness is controlled **independently on the two radial poles**,
measured in **cell layers** rather than millimetres.

## What already exists

The bundle builder (`openalea.granap.vascular_bundle`) already places a fibre
sheath. It is peeled off the envelope by `_sheath_zones` before the tissue bands
are laid, and driven by these `vascular_bundle` params:

- `sheath`: `"none"` | `"ring"` | `"caps"` | `"both"`
  - `"ring"` — a full fibre ring around the whole envelope;
  - `"caps"` — fibre caps at the **two radial poles** of the bundle
    (inner = xylem/pith side, outer = phloem/cortex side);
  - `"both"` — caps + a thin ring;
  - `"none"` — no fibres (a thin *parenchyma* bundle-sheath ring is still laid,
    so every bundle has a sheath of some kind).
- `sheath_thickness` (mm) — one depth used for the ring **and both caps**.
- `sclerenchyma_cell_diameter` / `sclerenchyma_cell_width` — the fibre cell size.

So "add sclerenchyma at the bundle" is already possible with `sheath="caps"`.
The gap is control: **both caps share a single `sheath_thickness` in mm**, so you
cannot make the phloem-side cap thick and the xylem-side cap thin or absent.

Local-frame convention (see `_local_envelope`): **local +y is radial-outward**.
So in `_sheath_zones` the `maxy` cap is the **outer** pole (phloem / organ
surface) and the `miny` cap is the **inner** pole (xylem / organ centre).

## Proposal

Keep the `caps` mechanism; express each cap as a **cell-layer count per pole**.
Add two integer params to `VascularBundleParams`:

- `sheath_cap_layers_outer` — layers on the **top** (outer, phloem/cortex) pole;
- `sheath_cap_layers_inner` — layers on the **bottom** (inner, xylem/pith) pole.

Semantics:

- Each cap's depth becomes `layers × sclerenchyma_cell_diameter`, so the ring-fill
  lays approximately that many fibre rows.
- `0` on a pole ⇒ **no cap** there (e.g. a fibre cap only on the phloem side —
  the common stem arrangement).
- `-1` (default) ⇒ **derive from `sheath_thickness`** (old behaviour), so existing
  configs are untouched.
- The `"ring"` part of the sheath keeps using `sheath_thickness`; only the caps
  become layer-driven.

### Why this is the right shape

- **Layer count is the natural, mesh-independent unit.** The request is "number
  of layers"; a layer count stays correct if the fibre cell size changes, whereas
  a fixed mm thickness would silently change the row count.
- **Per-pole independence matches real bundles.** The typical arrangement is a
  thick fibre cap on the phloem side and a thin/absent one on the xylem side;
  today that asymmetry is inexpressible.
- **Small, localized, organ-agnostic.** Only `_sheath_zones` changes (use a
  per-pole depth instead of one `t`), plus the two params. It works for every
  bundle (dicot eustele, monocot atactostele) with no organ-class changes, and is
  fully backward compatible via the `-1` sentinel.

### Implementation sketch

`_sheath_zones` currently builds both caps with one thickness `t`:

```python
if sheath in ("caps", "both"):
    minx, miny, maxx, maxy = working.bounds
    cap_in  = working.intersection(box(minx - 1, miny,     maxx + 1, miny + t))
    cap_out = working.intersection(box(minx - 1, maxy - t,  maxx + 1, maxy + 1))
    ...
    working = working.intersection(box(minx - 1, miny + t, maxx + 1, maxy - t))
```

Per-pole version (derive each depth, allow 0 to skip a pole):

```python
scl   = bp.get("sclerenchyma_cell_diameter", 0.008)
def _cap_depth(layers_key):
    layers = int(bp.get(layers_key, -1))
    return layers * scl if layers >= 0 else t   # -1 -> fall back to sheath_thickness

if sheath in ("caps", "both"):
    minx, miny, maxx, maxy = working.bounds
    t_in  = _cap_depth("sheath_cap_layers_inner")    # bottom / xylem pole
    t_out = _cap_depth("sheath_cap_layers_outer")    # top / phloem pole
    if t_in > 0:
        cap_in = working.intersection(box(minx - 1, miny, maxx + 1, miny + t_in))
        if not cap_in.is_empty:
            zones.append(("sclerenchyma", cap_in, scl, scl_w))
    if t_out > 0:
        cap_out = working.intersection(box(minx - 1, maxy - t_out, maxx + 1, maxy + 1))
        if not cap_out.is_empty:
            zones.append(("sclerenchyma", cap_out, scl, scl_w))
    working = working.intersection(box(minx - 1, miny + max(t_in, 0.0),
                                       maxx + 1, maxy - max(t_out, 0.0)))
```

The cap zone is filled by `build_bundle`'s existing sclerenchyma branch
(`_fill_parenchyma(... "sclerenchyma" ...)` = concentric ring fill), so a cap of
`n × cell_diameter` renders as ~`n` fibre rows. No new tissue plumbing.

### New params (`VascularBundleParams`)

```python
sheath_cap_layers_outer : int = Field(default=-1, ge=-1, title="Outer Cap Layers",
    description="caps/both only. Number of sclerenchyma fibre cell layers on the "
                "OUTER (phloem/cortex) pole of the bundle. 0 = no outer cap; "
                "-1 = derive the depth from sheath_thickness (legacy behaviour).")
sheath_cap_layers_inner : int = Field(default=-1, ge=-1, title="Inner Cap Layers",
    description="caps/both only. Number of sclerenchyma fibre cell layers on the "
                "INNER (xylem/pith) pole of the bundle. 0 = no inner cap; "
                "-1 = derive the depth from sheath_thickness (legacy behaviour).")
```

### Suggested dicot-stem preset

To show it immediately, the dicot-stem preset (`OrganInputData.for_dicot_stem`)
could switch its bundles to a phloem-side fibre cap:

```python
VascularBundleParams(
    ...,
    sheath="caps",
    sheath_cap_layers_outer=2,   # a 2-layer fibre cap on the phloem side
    sheath_cap_layers_inner=0,   # none on the xylem side
)
```

## Alternatives considered

- **Separate `_outer`/`_inner` thickness in mm** — rejected: not the requested
  unit, and not mesh-independent.
- **A general per-pole config object / list** — rejected as overkill for a
  two-pole feature.

## Status

Proposed. Not yet implemented — this note is the design to review before wiring
`_sheath_zones` and the two params.
