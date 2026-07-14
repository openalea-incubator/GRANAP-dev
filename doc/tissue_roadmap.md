# Roadmap — toward simplified recipes for monocot, dicot, needle

Companion to `doc/tissue_refactor.md` (the model and what's already done). This
file is the forward plan: the goal is that **each organ's anatomy reads as a
short, declarative recipe** composed from the shape-first vocabulary — region
builders + fills + a small set of named "special functions" — instead of long
bespoke `fit_*` methods.

## Target: what a "simplified recipe" should look like

Today a recipe step is an opaque wrapper, e.g. `lambda: self.fit_phloem_elements(polygon)`.
The target is a step that *declares* a region and how it is filled:

```python
def _vascular_recipe(self, stele):
    r = TissueRecipe()
    if self.is_star:
        r.fill("xylem star", self.xylem_star_region(stele),       # region builder -> Tissue
               strategy="packing", split=("xylem", "stele"),       # fill verb + params
               **self.xylem_pack)
        r.cleanup("clear stele under xylem", self._remove_stele_seeds_near_xylem)
        r.fill_each("phloem valleys", self.phloem_valley_regions(stele),
                    strategy="packing", **self.phloem_pack)
    else:
        r.fill_each("metaxylem ring", self.metaxylem_regions(stele), strategy="border")
        r.special("metaxylem sheath", self.sheath_around_vessels)
        r.fill_each("proto + phloem", self.bundle_regions(stele), strategy="packing")
    return r
```

Three building blocks make that possible: **region builders** (`*_region(s)` →
`Tissue`), **fill strategies** (packing / rings / border / line), and **special
functions** (sheath, stomata, resin duct, companion cells, intercellular space).
The phases below build those out, organ by organ, behind the golden tests.

## Guardrails

- Every phase keeps `test/test_vascular_regression.py` green (byte-identical
  census), or makes a *deliberate, reviewed* golden update.
- **Prerequisite for any needle work:** add a needle golden first (P0). Needle
  currently has no census regression anchor.
- Determinism is now real (seed RNG threaded through jitter). Keep it: every new
  random draw must use `self.rng`, never global `np.random`.

---

## P0 — Safety net for needle  *(DONE 2026-06-24)*

Needle is about to be refactored but had no golden. **Done:** added two needle
profiles to `test_vascular_regression.py`, pinning the seed=0 census, both
verified reproducible (two builds identical):

- `needle_default` — `OrganInputData.for_needle()`. Census: Strasburger 38 /
  air 312 / cambium 58 / duct 3 / endodermis 49 / epidermis 231 / guard 8 /
  hypodermis 387 / mesophyll 228 / parenchyma 244 / phloem 310 / pore 4 /
  resin duct 42 / transfusion 103 / xylem 270.
- `needle_features` — resin_duct `n_files=2`, stomata `n_files=10` (matches
  `test_needle`). Census differs in air 327 / duct 2 / epidermis 219 / guard 20 /
  hypodermis 366 / mesophyll 230 / pore 10 / resin duct 28.

Note: `n_files=0` is rejected by the pydantic params (`ge=1`), so "ducts/stomata
off" variants aren't expressible; the two profiles above (default vs. denser
ducts+stomata) are the meaningful anchors instead. The regression suite's golden
builders now return a *constructed organ* (root **or** needle), not just an
`OrganInputData`, so it spans both organ classes.

Deliverable: needle goldens. Risk: none (test only). **Complete.**

---

## P1 — Recipe ergonomics: region→fill as a first-class step  *(DONE 2026-06-24)*

Made `TissueRecipe` express "build region(s), fill them" directly so recipes stop
wrapping `fit_*`. All byte-identical against the goldens (incl. the new needle
ones) + full battery green.

**Done in `tissue_builder.py`:**

- `recipe.fill(name, tissue, *, strategy="packing", produces=None, record=None, **kw)`,
  `recipe.fill_each(name, tissues, ...)` (tissues may be an iterable **or** a
  zero-arg callable, deferred to build time when the regions depend on an earlier
  step — e.g. phloem valleys carved from the not-yet-built xylem star),
  `recipe.cleanup(name, fn)`, `recipe.special(name, fn, produces=...)`.
- `recipe.bind(cells, rng)` sets the fill target; `cells` may be a `CellManager`
  **or a zero-arg callable** resolving the current manager at build time (the
  star path replaces `self.vascular_cells` mid-build — `bind(lambda: self.vascular_cells, self.rng)`).
- `_dispatch_fill(target, tissue, strategy, rng, **kw)` maps
  `"packing"|"rings"|"line"` onto `fill_by_packing|fill_by_rings|fill_along`;
  whatever the primitive returns is handed to the step's `record` hook.
  (No `"border"` strategy: the single-ellipse metaxylem seeding stays a `special`
  — it isn't a region+fill.)
- `fill_by_packing` now no-ops on an empty/None zone (makes `recipe.fill` safe for
  a star that didn't fit; previously the caller guarded).

**Done in `root_class.py`:**

- **Xylem star** split into a pure region builder `_xylem_star_region(stele) ->
  Tissue` (sets `xylem_star`/`pith_polygon`) + `_xylem_pack_kwargs()` +
  `_record_xylem_vessels()`. `fit_star_shapped_xylem` is now a thin shim
  (region + fill) kept for direct/test use; both recipes drive it declaratively
  via `recipe.fill("xylem star", self._xylem_star_region(polygon),
  strategy="packing", record=self._record_xylem_vessels, **self._xylem_pack_kwargs())`.
- **Phloem** `fit_phloem_elements` removed; replaced by `_add_phloem_step(recipe,
  stele, type)` using `recipe.fill_each` over a *deferred* `_phloem_valley_zones`
  callable, with a `record` hook that buffers each region by `_phloem_adjustment`
  into `vascular_tissue_polygons`. Shared by monocot star + dicot primary.
- Bespoke steps now use `recipe.special(...)` (semantic marker, identical
  behaviour): monocot **metaxylem ring** (border fill), **metaxylem sheath**
  (cell-relative), **phloem + protoxylem bundles** (pizza-slice); dicot
  **secondary xylem/phloem** (ring fills) and **primary cambium** (line fill).
- `clear stele under xylem` is now a `recipe.cleanup(...)`.

Tests added: `test_recipe_fill_packs_a_region`,
`test_recipe_fill_each_and_lazy_target`, `test_recipe_empty_region_is_safe`,
`test_recipe_special_and_cleanup_order` in `test_recipe_vocabulary.py`.

**Not converted (deliberate, matches doc/tissue_refactor.md):** the `special`
steps above are bespoke fills (border / pizza-slice / rings / line + companion
cells + sheath), not plain region+fill; converting them is P2/P3 work (special
vocabulary) or out of scope. The two genuine region+fill cases (xylem star,
phloem valleys) are now declarative — that's the P1 deliverable.

Deliverable: monocot/dicot recipes read as region+fill declarations; goldens
unchanged. Risk: low (mechanical, byte-identical). **Complete.**

---

## P2 — Special-function vocabulary  *(DONE 2026-06-24; companion cells deferred)*

Promote the organ-specific "special" placements into a shared, named library —
new **`special_tissues.py`**, parameterised and organ-agnostic. These are the
user's original "special functions". The geometry of *where* a feature goes
stays with the organ (it is organ-specific); the shared functions take that
precomputed geometry and do the cell placement.

**Done (byte-identical; needle goldens + `test_special_tissues.py` green):**

- `carve_and_insert(cell_manager, carve_polygons, new_cells, *, buffer=0, recalc=True)`
  — the shared post-fill structural pattern (remove cells under a mask, insert
  new cells, recompute) behind both needle features.
- `place_resin_duct(cell_manager, duct_data, rdp, layer_index)` ← cell-placement
  half of `NeedleAnatomy.add_canal`. `add_canal` now just computes geometry
  (`_duct_zone_data`) then calls it. (`_CANAL_RESAMPLE_PTS` moved here.)
- `place_stomata(cell_manager, stomata_geoms, sp, cell_diam)` ← cell-placement
  half of `NeedleAnatomy.add_stomata`. `add_stomata` keeps the recenter +
  triplet-index math (organ geometry) then calls it.
- `consider_as_cell(cell_manager, region, tag, *, id_layer=0, replace=True)` —
  collapse a region into a single cell whose polygon **is** the region (a fresh
  polygon, not Voronoi-derived); `replace` first removes cells under it.
  Semantics **confirmed by Dilhan 2026-06-24**.

**Deferred (deliberate):**

- `add_companion_cells` — the companion placement is interleaved *inside*
  `_fill_phloem_zone`'s per-sieve packing loop (each companion is positioned
  tangent to the sieve just placed). Extracting it cleanly is a real rewrite of
  that loop with regression risk and little reuse — same call as secondary
  growth in `tissue_refactor.md`. Convert only alongside a genuine simplification
  of `_fill_phloem_zone`.
- `add_intercellular_space` — already organ-agnostic on the `Organ` base
  (`_apply_intercellular`); no extraction needed. Could be surfaced as a named
  recipe verb later, but it runs in the engine's post-Voronoi phase, not the
  vascular recipe.

Each placed feature is recipe-callable via `recipe.special(...)`. The
cell-relative ones (sheath, stomata, ducts) stay honest — they run after their
reference cells exist (see `tissue_refactor.md` on cells-first ops).

Deliverable: shared special-function API; needle's `add_canal`/`add_stomata`
re-expressed on top of it (behind needle goldens). **Largely complete** —
companion cells deferred; `consider_as_cell` pending semantics confirmation.

---

## P3 — Needle into the recipe model  *(DONE 2026-06-24)*

Brought `NeedleAnatomy` onto the same footing as root. Byte-identical (needle
goldens + needle network test green).

- `NeedleAnatomy._vascular_recipe(polygon)` added; `_create_vascular_tissue` is
  now just `recipe.build()`, matching root. The xylem / phloem / cambium /
  Strasburger grid stays a single `recipe.special("vascular ellipse grid", ...)`
  step — its "region" is two *oriented* ellipses carrying axis/angle metadata
  that the grid loop consumes, so it does **not** map onto the shape-first
  region+fill verbs (same call as root's metaxylem-border / pizza-bundle bespoke
  fills). `two_ellipses` deliberately **not** wrapped in `Tissue`: it would strip
  the axis/angle metadata for cosmetic churn + regression risk, the same lesson
  as secondary growth in `tissue_refactor.md`.
- `NeedleAnatomy._organ_recipe()` added; `_organ_specific_tissues` is now
  `recipe.build()` over two `special` steps — `"resin ducts"` (→ `add_canal` →
  `place_resin_duct`) and `"stomata"` (→ `add_stomata` → `place_stomata`), the P2
  vocabulary.
- Inspectable: `test_needle_recipes_are_inspectable` in `test_recipe_vocabulary.py`.

Deliverable: needle expressed as recipes; needle goldens hold. **Complete.**
(The ~140-line `vascular_elements_in_ellipses` grid itself is left as the bespoke
fill it is — decomposing it further is out of P3's scope and has no golden-safe
payoff.)

---

## P4 — Unify the three recipes  *(DONE 2026-06-24)*

With monocot, dicot, needle all on the recipe model, factored the shared shape.
Byte-identical (all goldens + recipe/special/ROI suites green).

**Common recipe scaffold on `Organ`:**

- `Organ._create_vascular_tissue(polygon)` is now concrete:
  `self._vascular_recipe(polygon).build()`. `Organ._organ_specific_tissues()` is
  concrete: `self._organ_recipe().build()`.
- `Organ._vascular_recipe(polygon)` and `Organ._organ_recipe()` are the single
  overridable contract, each defaulting to an **empty** `TissueRecipe` (no longer
  `@abstractmethod`). The per-organ files supply *only* their recipe.
- Removed the duplicated `_create_vascular_tissue` from `MonocotRootAnatomy`,
  `DicotRootAnatomy`, `NeedleAnatomy` and the `RootAnatomy` stub; removed
  `_organ_specific_tissues` from `RootAnatomy` (now Organ's no-op default) and
  `NeedleAnatomy`. The vascular guards (`n_vascular_bundles==0` /
  `n_vascular_peak==0`) moved *into* each `_vascular_recipe` (return empty).
  `RoiOrgan` keeps its explicit `pass` overrides (no vascular).

**Inspection / preview (`recipe.describe()` extended):**

- `TissueStep.kind` ∈ {fill, fill_each, cleanup, special, add}; set by each
  recipe verb.
- `recipe.plan()` → `[(name, kind, produces), ...]` (full plan; `describe()` kept
  name+produces for back-compat). `recipe.format_plan()` renders an indented,
  human-readable block (`[fill] xylem star -> xylem, stele` …); `TissueRecipe.__repr__`.
- Tests: `test_recipe_plan_reports_kinds_and_renders`,
  `test_organ_default_recipes_are_empty`, and `plan()` kind assertions added to
  `test_recipe_is_inspectable`.

**Deliberately NOT done — data-driven spec.** Expressing a recipe as a list of
tissue-spec *data* interpreted by an engine is the roadmap's *optional* end state;
skipped per "resist over-generalising before there's a concrete need." The three
recipes are already short, parallel and declarative, and `plan()` gives the
inspection a data spec would have fed. Revisit only when a non-programmer-editable
spec is actually required.

Deliverable: three short, parallel, declarative organ recipes on one shared
scaffold, fully inspectable. **Complete.**

---

## P5 — Optional layout/cleanup follow-ups  *(CLOSED — not pursued, 2026-06-24)*

Reviewed and **deliberately not done** (Dilhan's call): both items are explicitly
optional, and the central-stele rework is not a transparent refactor.

- Central stele **subtract-then-fill** instead of fill-then-mask — would need the
  pipeline reordered so vascular regions exist before stele seeding. Decided
  against because (a) it is **not byte-identical** — cutting the vascular regions
  out of the stele polygon before seeding shifts where the remaining stele seeds
  land, changing the seed=0 goldens (a behaviour change, not a cleanup); and
  (b) the unified vascular mask in `Organ.generate_cells` already *is* the
  region-algebra model (see `tissue_refactor.md`, mask-migration findings), so
  the payoff is marginal. Revisit only if a concrete need appears.
- Post-fill ops (intercellular / aerenchyma) stay cells-first by nature; left as
  engine steps below the tissue abstraction — no action needed.

**Roadmap status: P0–P4 complete; P5 closed.** The refactor's goal — each organ's
anatomy as a short, declarative, inspectable recipe on a shared scaffold — is met
for monocot, dicot and needle.

---

## Dependency order

```
P0 (needle golden)  ─┐
P1 (recipe ergonomics, monocot+dicot) ─┬─> P2 (special funcs) ─> P3 (needle) ─> P4 (unify) ─> P5 (optional)
                                       └ P1 unblocks the target recipe shape P3/P4 reuse
```

P0 and P1 are independent and low-risk; do them first. P2 is the prerequisite for
P3 (needle leans on the special-function library). P4 needs all three organs on
the model. Each phase ends green against the goldens.
