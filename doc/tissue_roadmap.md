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

## P2 — Special-function vocabulary  *(medium)*

Promote the organ-specific "special" placements into a shared, named library
(new `special_tissues.py`, or a section of `tissue_builder.py`), parameterised
and organ-agnostic where possible. These are the user's original "special
functions":

- `add_stomata(...)`            ← from `NeedleAnatomy.add_stomata` / `_stomata_carve_polygons`
- `add_resin_duct(...)`         ← from `NeedleAnatomy.add_canal` / `_duct_zone_data`
- `add_companion_cells(tag, diameter, width)` ← dicot secondary phloem companion logic
- `add_intercellular_space(...)`← generalise `Organ.add_intercellular`
- `consider_as_cell(region, tag)` ← collapse a region/group into one cell
  (the user's `consider_as_cell`; clarify exact semantics when implementing)

Each becomes a step a recipe can call via `recipe.special(...)`. Keep the
cell-relative ones (sheath, stomata) honest — they run after their reference
cells exist (see `tissue_refactor.md` on cells-first ops).

Deliverable: shared special-function API; needle's `add_canal`/`add_stomata`
re-expressed on top of it (behind needle goldens). Risk: medium (API design;
stomata index math is fiddly).

---

## P3 — Needle into the recipe model  *(medium–high)*

Bring `NeedleAnatomy` onto the same footing as root.

- **Vascular**: `vascular_elements_in_ellipses` (~140 lines) → region builders
  (the two ellipses from `two_ellipses` become `Tissue`s) + a fill. The xylem /
  phloem / cambium / Strasburger grid is a bespoke fill — keep it as a named
  fill strategy (it does not map onto packing); the *region* half still becomes
  `Tissue` + algebra.
- **Organ-specific**: a `NeedleAnatomy._organ_recipe` running `add_resin_duct`
  and `add_stomata` from the P2 vocabulary.
- Add `NeedleAnatomy._vascular_recipe` so `_create_vascular_tissue` is just
  `recipe.build()`, matching root.

Deliverable: needle expressed as recipes; needle goldens hold (or deliberate
update). Risk: medium–high (bespoke grid + stomata geometry).

---

## P4 — Unify the three recipes  *(medium)*

With monocot, dicot, needle all on the recipe model, factor the shared shape and
push toward a **declarative spec**:

- Common recipe scaffold on `Organ` (or a mixin): the per-organ files supply only
  region builders + the ordered list of `(region, fill, params)` / `special`
  steps.
- Optionally express a recipe as **data** (a list of tissue specs) that a
  non-programmer can read/edit, with the engine interpreting it — the end state
  of "simplified recipe."
- `recipe.describe()` already gives inspection; extend it to render the full
  region/fill/special plan for `plot_tissues` previews.

Deliverable: three short, parallel, declarative organ recipes. Risk: medium
(abstraction; resist over-generalising before all three are proven).

---

## P5 — Optional layout/cleanup follow-ups  *(low priority)*

- Central stele **subtract-then-fill** instead of fill-then-mask — needs the
  pipeline reordered so vascular regions exist before stele seeding; low payoff
  over the current unified mask (see `tissue_refactor.md`). Do only if a concrete
  need appears.
- Post-fill ops (intercellular / aerenchyma) stay cells-first by nature; leave as
  engine steps below the tissue abstraction.

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
