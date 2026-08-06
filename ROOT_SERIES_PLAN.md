# Root developmental series with tracked xylem — plan

## Goal
Produce a **longitudinal series** of root cross-sections: the anatomy at the apex,
at the collet, and X user-defined intermediate levels (physical positions between
length y and Z). Each level generates its own anatomy (root diameter, stele
diameter, metaxylem count, vessel sizes… all change level to level). The **one hard
requirement**: a xylem vessel keeps its identity up the series, so you can point at
a vessel low down and find the same one higher up, let it **grow**, and have the
tissue **refit** around it.

## Confirmed design decisions (from the user)
- **Output**: X+2 independent 2D root sections, each a normal GRANAP render, all
  sharing a `vessel_id` column + a track table (id → per-level radius/position).
- **Mechanism = forward-carry ("apex then grow")**: each level's xylem is
  *inherited* from the level below — same ids, positions rescaled + radii grown +
  new vessels topped up — never redrawn. Identity is exact by construction (no
  post-hoc matching). The apex is generated **normally** (packed once) and *becomes*
  the seed vessel set (not specified by hand).
- **Tracked scope**: xylem vessels only. Cortex / phloem / parenchyma / epidermis
  are regenerated fresh at each level from that level's params ("refit").
- **Interpolation**: per-parameter schedules — each evolving parameter has its own
  `value(stage)` (linear / curve / step), not one global blend.
- **Xylem changes apex→collet**: monocot metaxylem **count grows** (new appear, old
  persist); dicot vessels near the cambium **enlarge**; dicot cambium **births** new
  vessels.
- Drop the "birth stage" framing — it over-formalized. Think: carry vessels up,
  grow them, add new ones when the count rises, re-tessellate the rest.

## Current code reality (grounding)
- Root xylem today = **stochastic circle-packing** inside a star region:
  `_xylem_star_region()` builds the star geometry, then `pack_circles(...)` (via
  `_xylem_pack_kwargs()`) fills it → `placed = [(polygon, type, gid)]`;
  `_record_xylem_vessels()` records the wide "xylem" ones into `vascular_polygons`.
- `Cell` (cell_class.py) has `id_cell / id_group / id_layer` but **no persistent
  tracking id**, and `cell_to_dict()` doesn't emit one.
- Pipeline (organ_class.generate_cells): seed cells → vascular mask → Voronoi →
  groups/simplify → intercellular/fuse → layer populate → gdf export.

## Three new pieces (everything else reused)
1. **`Cell.track_id`** — optional persistent id (default None), survives Voronoi
   grouping/simplify/export, becomes a `vessel_id` gdf column. Additive change to
   `Cell.__init__` + `cell_to_dict`. Non-xylem cells stay None. (Must make sure
   grouping/simplify carry it through — check `process_voronoi_groups`/`simplify_cells`.)
2. **Vessel prescription in `RootAnatomy`** — a mode where instead of packing the
   star, the root consumes an explicit vessel list `[{id, x, y, r}]`, turns each into
   a fixed vascular polygon (circle/ellipse at (x,y) radius r) tagged with its
   track_id; star region + stele + cortex + phloem + epidermis generate **around**
   them as today. One new branch in the xylem step that bypasses the packer. THE
   core new mechanism.
3. **`RootSeries` driver** (new class) — owns: the levels (positions apex→collet),
   the per-parameter schedules, the carried vessel set, and the advance rule.

## The forward-carry loop (heart)
```
vessels = generate_apex()                    # normal pack once; assign track_ids 0..k
emit section(level_0, prescribe=vessels)     # apex render + vessel_id column
for level in levels[1:]:
    vessels = advance(vessels, cfg(level))
    emit section(level, prescribe=vessels)   # refit tissue around carried vessels
```
`advance(vessels, cfg)`:
1. **rescale** positions as the stele grows (keep relative radial position:
   (x,y) *= R_new/R_old);
2. **grow** each radius toward this level's target (dicot: cambium-adjacent grow
   more — gain weighted by proximity to the cambium radius);
3. **top up** count when the schedule says more vessels exist now — insert new ids
   (monocot: widest angular gaps of the pole ring; dicot: fresh annulus just inside
   the cambium);
4. return updated set (all old ids preserved).

## Output
X+2 GeoDataFrames (one/level) each with `vessel_id`; a track table
`vessel_id → {level: (x, y, radius, present?)}`.

## Phasing (build the risky core first, small)
- **Phase 1 — prescription plumbing** ✅ DONE (2026-08-03). What landed:
  - `Cell.track_id` (default None) in `cell_class.py` — added to `__init__`,
    `Cell.radial(track_id=...)`, and emitted by `cell_to_dict` → `track_id` gdf column.
  - Survives the pipeline: `generate_cell.process_voronoi_groups` now copies
    `track_id=r.track_id` onto the grouped cell (the one rebuild point; simplify /
    remove_nested MUTATE cells so they keep it; fuse_gaps only rebuilds gap slivers,
    not solid vessels).
  - `tissue_class.place_packed_group(..., track_ids=[...])` stamps a track_id per
    circle onto its seeds.
  - `RootAnatomy.prescribe_vessels([(x,y,r,track_id),...])` + `_place_prescribed_xylem`
    (in root_class.py) place exactly those vessels via place_packed_group and feed
    `vascular_polygons` (mask). Monocot `_vascular_recipe` intercepts at the top: if
    `_prescribed_vessels` set → one `recipe.special("prescribed xylem", ...)` step and
    return (Phase-1 = xylem only, no phloem yet).
  - Verified: 3 vessels land within <0.02 of their prescribed (x,y), tagged xylem
    with ids 10/20/30 in the gdf, tissue refits around them, deterministic across
    runs. Tests: `test/test_root_series.py` (3). No regression (39 root/leaf/param
    tests pass; gdf extra column harmless — anatomy_writer uses iterrows).
- **Phase 2 — monocot series**: `RootSeries` + schedules + advance rule (rescale,
  grow, top-up in gaps) for a monocot root. Deliver apex→collet with tracked,
  growing, multiplying metaxylem.
- **Phase 3 — dicot series**: cambium-proximity growth + newborns in the cambium
  annulus, hooking the existing secondary-growth machinery.
- **Phase 4 — polish**: track table, gallery drawing the series side by side with
  one vessel colored consistently across levels.

## Known-uncertain (tune when visible, not now)
- New-vessel placement: Phase 2 uses "largest angular gap" (monocot) / "cambium
  annulus" (dicot); revisit if positions look wrong.
- Position rescaling under dicot secondary growth: proportional rescale is clean for
  primary xylem; secondary growth expands radius a lot + adds an annulus, so Phase 3
  may need the cambium radius on its own schedule.

## Open threads / integration risks
- `track_id` survival through `process_voronoi_groups` + `simplify_cells` +
  `remove_nested_cells` + `fuse_gaps` (fused cells: keep the min/representative id?).
- Prescribed vessels must feed the vascular removal mask like packed ones do
  (`vascular_polygons` + `vascular_tissue_polygons`).
- Reuse, don't fork, the existing dicot secondary-growth machinery in Phase 3.
