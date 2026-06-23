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

## P0 — Safety net for needle  *(small, do first)*

Needle is about to be refactored but has no golden. Add a `needle_*` profile (and
any meaningful variants: ducts on/off, stomata on/off) to
`test_vascular_regression.py`, pinning the seed=0 census. Confirm reproducibility
(two builds identical) — needle uses `recenter_cells` + stomata index math, so
verify it is stable before touching it.

Deliverable: needle goldens. Risk: none (test only).

---

## P1 — Recipe ergonomics: region→fill as a first-class step  *(low risk)*

Make `TissueRecipe` express "build region(s), fill them" directly so recipes stop
wrapping `fit_*`.

- Add to `tissue_builder.py`: `recipe.fill(name, tissue, strategy=..., **kw)`,
  `recipe.fill_each(name, tissues, ...)`, `recipe.cleanup(name, fn)`,
  `recipe.special(name, fn, ...)`. `strategy` dispatches to the existing
  `fill_by_packing` / `fill_by_rings` / `fill_along` (+ a `border` strategy for
  the single-ellipse metaxylem border seeding).
- Split each converted `fit_*` into a pure **region builder** (`*_region(s)`
  returning `Tissue`) and let the recipe own the fill. The `fit_*` methods either
  become thin shims or disappear.
- Rewrite the **monocot** and **dicot** `_vascular_recipe` in the target style.

Deliverable: monocot/dicot recipes read as region+fill declarations; goldens
unchanged. Risk: low (mechanical, byte-identical).

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
