# Tissue refactor — the shape-first model

This note records the architecture we are converging on for GRANAP's anatomy
generation, why it is shaped that way, and what is done vs. still pending.

## The core idea: a tissue **is a region**

A *tissue* is a tagged anatomical **region** — a shapely shape plus the tag its
cells will take. You manipulate the region as pure geometry, and **only then**
fill it with cells. Cells are the terminal product of filling; the only thing
you ever do to a cell afterwards is **retag** it.

```
define region  →  transform / combine region (pure shapely)  →  fill with cells  →  (cells: retag-only)
```

This replaces an earlier *cells-first* framing (a "tissue" = a bag of tagged
cells, transformed after placement). Cells-first forced a tangle of
pre-/post-Voronoi timing questions and "place seeds everywhere then delete the
wrong ones" masking. Shape-first removes all of that: at transform time there
are no cells, so there is nothing to keep in sync.

## Why shape-first — and its one boundary

Shape-first is the better abstraction for the **layout layer**:

- it matches how anatomy is described ("the xylem region, minus the pith");
- overlaps are resolved **up front** by region algebra
  (`stele.difference(xylem)`) instead of by deleting mis-placed seeds — exact,
  order-independent, nothing to desync;
- transforms are plain geometry, so the seed/polygon/Voronoi timing confusion
  disappears;
- a region is just a polygon: trivially composable and testable in isolation.

It does **not** replace the whole pipeline. It sits on top of a **cells-first
engine** that stays, because two things are inherently global / post-fill:

1. **Voronoi is global.** Final cell boundaries depend on *all* seeds together —
   neighbouring tissues interlock along shared edges. So regions decide *where
   seeds go and what tag they get*; the actual tessellation still pools every
   seed and runs **once**. Shape-first governs the *input* to the engine, not
   the engine.
2. **Some operations are genuinely post-fill / cell-relative** — intercellular
   spaces and aerenchyma are carved from cell polygons after they exist; the
   metaxylem *sheath* is defined relative to already-placed vessels; resin ducts
   and stomata are carved into existing needle cells. These stay cells-first,
   below the tissue abstraction, and live in **`special_tissues.py`** (the
   organ-agnostic "special function" vocabulary — `carve_and_insert`,
   `place_resin_duct`, `place_stomata`, `consider_as_cell`). An organ's recipe
   invokes them via `recipe.special(...)`; the geometry of *where* the feature
   goes stays with the organ, the shared function does the cell placement.

So: **shape-first for layout, cells-first engine (+ special_tissues) beneath it.**

## The vocabulary (`openalea/granap/tissue_builder.py`)

**`Tissue(tag, shape)`** — the region. All transforms mutate `self.shape` and
return `self` (chainable):

- `.rotate(angle, origin)`, `.translate(dx, dy)`, `.smooth(factor)`
- `.difference(other)`, `.intersection(other)`, `.union(other)`
  (`other` may be a `Tissue` or a raw shapely geometry)
- `.is_empty`, `.area`

**Fill primitives** — region → cells, appended to a `CellManager`:

- `place_packed_group(target, packed, tag, ...)` — seed every circle of a
  pre-computed packing.
- `fill_by_packing(target, zone, tag, *, rng, **pack_kwargs)` — pack a zone with
  circles, then seed each. The natural "fill a region" verb.
- `fill_along(target, geometry, tag, ...)` — seed cells along an edge / line.
- `fill_by_rings(target, zone, ...)` — seed concentric inward rings.

**Edit verb (terminal, cell-level)**

- `retag_tissue(cell_manager, old_tag, new_tag)` — the only verb valid on a
  filled cell.

**Composition**

- `TissueStep(name, fn, produces=(...))` and `TissueRecipe` — an ordered,
  inspectable list of build steps. An organ's stele is assembled by a recipe
  whose order is *data* (`recipe.describe()` returns `[(name, produces), ...]`),
  not control flow.
- Recipes are written **declaratively**: `recipe.bind(cells, rng)` (cells may be
  a callable, resolved at build time) then —
  - `recipe.fill(name, tissue, strategy="packing"|"rings"|"line", record=…, **kw)`
    — one region → cells via the matching `fill_*` primitive;
  - `recipe.fill_each(name, tissues_or_callable, …)` — many regions (the iterable
    may be a zero-arg callable, deferred to build time when the regions depend on
    an earlier step, e.g. phloem valleys carved from the xylem star);
  - `recipe.cleanup(name, fn)` — a cell/group-level cleanup (produces nothing);
  - `recipe.special(name, fn, produces=…)` — a bespoke placement that isn't a
    plain region+fill (border / pizza-slice / rings / sheath).
  - `recipe.add(name, fn, produces=…)` — the low-level escape hatch (wrap any
    callable). The `record` hook receives whatever the fill primitive returns,
    for mask bookkeeping (e.g. recording placed vessels in `vascular_polygons`).
- Every step carries a `kind` (fill / fill_each / cleanup / special / add).
  `recipe.describe()` → `[(name, produces), …]` (back-compat); `recipe.plan()` →
  `[(name, kind, produces), …]`; `recipe.format_plan()` renders the plan as text.
- **Shared scaffold (`Organ`):** `Organ._create_vascular_tissue` /
  `_organ_specific_tissues` are concrete (`self._vascular_recipe(...).build()` /
  `self._organ_recipe().build()`); each organ overrides only `_vascular_recipe` /
  `_organ_recipe` (both default to an empty recipe). Vascular guards live inside
  each `_vascular_recipe`.

## How an organ uses it

`MonocotRootAnatomy` / `DicotRootAnatomy` expose `_vascular_recipe(polygon)`
returning a `TissueRecipe`; `_create_vascular_tissue` just guards then
`recipe.build()`. The two genuine region+fill steps are now declarative — the
monocot star recipe reads:

```python
recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
recipe.fill("xylem star", self._xylem_star_region(polygon), strategy="packing",
            record=self._record_xylem_vessels, **self._xylem_pack_kwargs())
recipe.cleanup("clear stele under xylem", self._remove_stele_seeds_near_xylem)
self._add_phloem_step(recipe, polygon, type="monocot")   # fill_each over valley regions
```

where the region builders are pure geometry, e.g. a phloem valley is
`Tissue("phloem", raw_ellipse).intersection(stele).difference(xylem_star)`.
Bespoke placements (metaxylem border, pizza-slice bundles, sheath, secondary
growth, primary cambium) stay `recipe.special(...)`.

**All three organs are now on the recipe model.** `NeedleAnatomy` too exposes
`_vascular_recipe(polygon)` (one `special` step: the bespoke xylem/phloem/
cambium/Strasburger ellipse grid) and `_organ_recipe()` (`special` steps for
resin ducts + stomata, placed via `special_tissues`); `_create_vascular_tissue`
and `_organ_specific_tissues` are just `recipe.build()`.

## The cells-first engine (unchanged, lives in `Organ.generate_cells`)

1. build layer ring polygons → seed them (`generate_cells_info`);
2. run the vascular recipe → fills `vascular_cells`, records mask regions in
   `vascular_tissue_polygons` (and `vascular_polygons` for xylem);
3. remove layer seeds inside the vascular mask, then add the vascular seeds;
4. **one global Voronoi** → seeds become cell polygons (grouped by `id_group`);
5. post-fill cell ops: intercellular spaces, aerenchyma;
6. populate layers, recompute properties, export GeoDataFrame.

Step 3's mask is the legacy "place then delete" mechanism; under shape-first it
should migrate toward region algebra (subtract the xylem region from the stele
region before seeding stele).

## Invariants for safe refactoring

- The pipeline is **deterministic given a seed** (`RootAnatomy(input, seed=0)`).
  Unseeded construction is RNG-dependent — never use it as a regression anchor.
  (This was only made *fully* true on 2026-06-24: `Cell.jitter` — called on every
  seed by `CellGenerator.voronoi_diagram` just before tessellation — used the
  global `np.random` instead of the organ's `self.rng`. Monocot was unaffected
  because the sub-micron jitter never flips its Voronoi topology, but dicot
  secondary growth drifted run-to-run. Both now thread `self.rng`.)
- `id_group` values are arbitrary; only *which seeds share an id* matters
  (Voronoi grouping). Ids are reassigned downstream (`extend_cells` offsets,
  `recalculate_cell_properties` sets `id_cell=i`). So id-numbering changes during
  a refactor are safe as long as grouping is preserved.
- Regression anchors (seed=0): star `xylem 37 / stele 202 / phloem 5`;
  star+pith `xylem 34 / stele 226 / phloem 5`; default `stele 341 / metaxylem 5 /
  protoxylem 10 / phloem 10`. (Star/star+pith were rebaselined by the chebyshev
  speed-up; see `test/test_vascular_regression.py` for the authoritative census.)

## Test harness

conda env `granap` (`mamba activate granap`); **no pytest** — run test modules as
scripts (call `test_*` functions directly, `MPLBACKEND=Agg`). Shape-first
vocabulary tests: `test/test_recipe_vocabulary.py`. Special-tissue vocabulary:
`test/test_special_tissues.py`. Golden census regression
(the safety net for all of the above): `test/test_vascular_regression.py` — pins
the exact `seed=0` cell-type counts for monocot default/star/star+pith, dicot
primary/secondary, **and needle default/features** (added 2026-06-24, P0). The
golden builders now return a constructed organ (root **or** needle). Treat a
golden change as a deliberate update, not a fix.

Run the env via `mamba run -n granap python <test>.py` (bash `conda run` is not
configured in this checkout).

## The seeding idiom: `Cell.radial`

Almost every seed is built the same way — a cell at `(x, y)` whose polar
attributes are taken relative to a tissue centre, with `area` a disc of its
diameter. `Cell.radial(type, x, y, diameter, id_group, center, *, id_cell=None,
id_layer=-1)` (in `cell_class.py`) captures exactly that: `angle =
arctan2(y-cy, x-cx)`, `radius = hypot`, `area = π(d/2)²`, `id_cell` defaulting to
`id_group`. It replaced ~40 hand-written 8-line `Cell(...)` blocks across
`tissue_builder`, `special_tissues`, `root_class` and `needle_class` (byte-identical;
the few sites with *shared* (non per-point) angle/radius or a non-disc area —
medullar rays — keep the explicit form). Tested in `test_special_tissues.py`.

## Status

**Done**

- `tissue_builder.py`: fill trio + `place_packed_group`, `retag_tissue`,
  `Tissue` (shape-first), `TissueStep`/`TissueRecipe` + **declarative recipe
  ergonomics** (`bind`/`fill`/`fill_each`/`cleanup`/`special`, strategy dispatch;
  roadmap P1, 2026-06-24).
- **File simplification (2026-06-24):** `Cell.radial` factory applied everywhere;
  needle `vascular_elements_in_ellipses` grid rewritten (3 duplicated tilt/build
  blocks → one `place()` closure, dead vars dropped) ~140→75 lines. All byte-identical.
- **Secondary-growth de-duplication (2026-06-24):** the angular-wedge construction
  (built identically for vessel slices and medullar-ray corridors) extracted to
  `DicotRootAnatomy._angular_wedge(...)`; the secondary-xylem vessel pack-and-seed
  loop replaced by the shared `place_packed_group` verb. Census-identical
  (id_group numbering shifts, but grouping/positions are preserved — golden is a
  census). The secondary-phloem **sieve** loop is left inline because its
  companion-cell placement is interleaved per sieve (companion extraction deferred).
- Monocot & dicot vascular construction expressed as inspectable recipes; the two
  region+fill steps (xylem star, phloem valleys) are now declarative
  `recipe.fill`/`fill_each`, the rest `recipe.special` (P1).
- Needle golden regression anchors added (P0).
- Converted to shape-first `Tissue` regions (region-build + fill, all
  byte-identical against the seed=0 anchors):
  - `fit_phloem_elements` (`_phloem_valley_zones` → `Tissue` via region algebra);
  - `fit_star_shapped_xylem` (star placed/clipped/pith-subtracted by algebra,
    filled with a diameter split into "xylem"/"stele");
  - `protoxylem_elements_in_slice`, `phloem_elements_in_slice`
    (ellipse region → `fill_by_packing`);
  - `vascular_elements_in_slice` (metaxylem inner-ellipse region; bespoke
    border fill kept inline);
  - `fit_primary_cambium_elements` (cambium star region; line-fill along its
    visible exterior via `_render_layer`/`fill_along`).
- Cells-first detour removed (cells-first `Tissue`, `transform_tissue`,
  `_post_voronoi_edits`, `tissue_smoothing`, cell-level rotate/translate/smooth).

**Deliberately left as-is**

- **Metaxylem sheath** (`fit_metaxylem_sheath`) — a ring seeded *around each
  already-placed vessel*: cell-relative, stays cells-first by nature.
- **Secondary growth** (`fit_secondary_xylem` / `fit_secondary_phloem` /
  medullar rays) — already decomposed into region-builders (`_build_*_polygon`)
  + the extracted fills (`fill_by_rings`, `_render_layer`). Wrapping those
  builders' return type in `Tissue` would only add `.shape` accesses through the
  most complex/fragile consumers — cosmetic churn with real regression risk and
  little readability gain. Convert only alongside a genuine simplification of
  those consumers (e.g. replacing their manual unions/masks with region algebra).

**Mask migration (done 2026-06-24 — outcome was mostly "already done / must stay")**

Investigated the seed-removal masks. Findings:

- The **unified vascular mask** in `Organ.generate_cells` is *already* region
  algebra: it unions every vascular region (`vascular_polygons` +
  `vascular_tissue_polygons`) and drops layer seeds inside the union. Nothing to
  migrate — this is the model.
- The per-slice stele removal inside `protoxylem_elements_in_slice` was **pure
  redundancy** (its ellipse is appended to `vascular_polygons`, so the unified
  mask already removed exactly those seeds). Deleted — consolidated onto the one
  region mask. Golden unchanged.
- `_remove_stele_seeds_near_xylem` and the primary-cambium removal are **not**
  redundant and **must stay**: they delete *whole `id_group`s* (a parenchyma cell
  is dropped if any of its border seeds is engulfed). The point-level region mask
  cannot express this — it would leave the cell's other border seeds in place and
  produce a partial, distorted Voronoi cell. Verified: stubbing
  `_remove_stele_seeds_near_xylem` changes the star root stele 217 -> 307.
  Documented on the method.

**Pending**

- Express post-fill ops (intercellular/aerenchyma) — stay cells-first by nature.
- (Optional, high-risk) make the central stele "subtract-then-fill" instead of
  "fill-then-mask" — needs the pipeline reordered so vascular regions are known
  before the stele parenchyma is seeded; low payoff over the current mask.
