# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`openalea.granap` (package name), a Python translation of GRANAR. It generates synthetic
**2-D plant-organ cross-section anatomy** — root, stem, leaf, needle — as a population of
tagged cells, via Voronoi tessellation of seeded points inside concentric layer rings plus a
bespoke central vascular zone. Output exports as geometry (GeoPandas/SVG/OBJ/GMSH/XML) or as a
hydraulic network (sparse adjacency matrix, for MECHA).

**`doc/TUTORIAL.md` is the authoritative deep-dive** — mental model, full public API
(`OrganInputData`, `Organ`), the exact generation pipeline, a worked "creating a new organ"
skeleton, and the tissue-design toolbox/cheat sheet. Read it before extending an organ or
adding a tissue; this file only covers what a session needs before opening it, plus gotchas
that aren't written down anywhere else.

## Setup & environment

```bash
mamba create -f ./conda/environment.yaml -y
mamba activate granap
pip install -e .            # editable install, from repo root
```

**The `geos`/`shapely`/`opencv`/`scipy` version pins in `pyproject.toml`'s
`[tool.conda.environment]` and `conda/environment.yaml` are load-bearing, not hygiene.**
`test/test_vascular_regression.py` pins an exact per-organ cell-type census, and that census
is decided by the geometry stack (`air space` cells are connected components of a GEOS boolean
difference, filtered by an absolute area cutoff). Pinning `shapely` alone does *not* pin GEOS —
one shapely version ships many conda builds linked against different GEOS releases — so a dev
env that drifts from these exact pins will disagree with CI on the golden tests even though
nothing in the diff looks geometry-related.

**Windows: invoking the `granap` env's `python.exe` directly (without full shell activation)
makes numpy hard-crash at import** — `Windows fatal exception: code 0xc06d007f` inside
`numpy.blas_fpe_check`, exit 127, no usable traceback. Under pytest this looks like a
collection crash, not an environment problem. Fix: prefix `PATH` with the env's own
directories before invoking `python`/`pytest` from a non-activated shell:

```powershell
$env:PATH = "<miniconda>\envs\granap;<miniconda>\envs\granap\Library\bin;<miniconda>\envs\granap\Library\mingw-w64\bin;<miniconda>\envs\granap\Library\usr\bin;<miniconda>\envs\granap\Scripts;" + $env:PATH
```

## Commands

```bash
pytest test/ -q                                          # full suite; ~10-15 min
pytest test/test_special_tissues.py -q                    # one file
pytest test/test_vascular_regression.py::test_needle_default_golden -q   # one test
python test/test_param_schema_equivalence.py --update     # deliberately regenerate golden/param_dicts.json after an intentional schema change
cd example && python custom_tissue_vessel_ring.py          # worked end-to-end example (custom tissue recipe)
cd example && python cellset_hybrid.py                     # CellSet real data + generated stele -> MECHA-ready XML
cd example && python brassicaceae_roots.py                 # batch template: many digitized species -> XML + contact sheet
cd example && python brassicaceae_kr_mecha.py              # radial conductivity of those XMLs -- RUN IN THE `mecha` ENV, not granap
python example/needle/pinus_pinaster.py                    # measured needle example, renders + prints cell census
python doc/generate_callgraph.py                           # needs pyan3/graphviz (mamba install -c conda-forge pyan3 graphviz python-graphviz)
```

Always pass a fixed `seed=` (e.g. `seed=0`) when constructing an organ for anything you intend
to compare or test — an unseeded organ uses fresh randomness every run, and the whole test
suite (golden regressions included) depends on that determinism.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Architecture

### The generation pipeline (`Organ.generate_cells`, `organ_class.py`)

One method drives every organ, in order: build layer polygons (peel rings inward from the base
shape) → scatter seed points per layer → `allocate_vascular_tissue` (your `_vascular_recipe`
places vascular seeds; the base class then removes every layer seed inside the vascular mask —
subclasses never do this removal themselves) → `_organ_specific_tissues` (your `_organ_recipe`,
e.g. needle's resin ducts/transfusion tissue/stomata — **pre-Voronoi**, seed level) → one global
`CellGenerator.voronoi_diagram` tessellation (seeds sharing an `id_group` fuse into one cell) →
`add_intercellular_spaces` (aerenchyma/air spaces — **post-Voronoi**, on real `cell.polygon`) →
`fuse_gaps` → layer population + `recalculate_cell_properties` → GeoDataFrame export. Knowing
which side of the Voronoi step a fix belongs on is usually the whole design question: deleting
a *seed* lets neighbours expand into the space for free; carving an *air space* has to slice a
finished polygon and reseat the host's boundary to match.

### Recipe-dict input model

An organ's configuration is always, ultimately, a flat `List[Dict]`, each dict identified by
its `"name"` key (e.g. `{"name": "stomata", "width": 0.025, ...}`). Two ways to produce one:

- **`OrganInputData`** (`input_data.py`) — the typed front door. Preset factories
  (`OrganInputData.for_root()`, `for_needle()`, `for_dicot_stem()`, …) build a list of *actual*
  pydantic model instances (`StomataParams()`, `RootXylemParams()`, …); `to_dict_list()` then
  calls `.model_dump()` on each. `set_value`/`set_values`/`validate()` give validated,
  cross-field-checked edits before that final flatten. `from_xml()` is a separate ingestion path
  that parses a legacy GRANAR-style XML config into the same flat shape (attribute renames +
  pydantic defaults merged in) — unrelated to `AnatomyWriter.write_to_xml`, which is a
  differently-shaped *output* format for generated cells, not organ config.
- **A raw `List[Dict]` written by hand** — every example script under `example/` does this
  directly (see `example/needle/pinus_pinaster.py`'s `build_pinaster()`). This works because
  **`Organ.__init__` never re-validates through pydantic** — every organ family does the same
  `isinstance(input_data, OrganInputData) / isinstance(input_data, list) / else-use-a-preset`
  branch, and everything downstream looks params up ad hoc by name
  (`next(p for p in self.params if p["name"] == "xylem")`), not by class. The one exception:
  any dict carrying an `"order"` key is routed through `Layer.from_dict()` into a concentric
  `Layer` — that's what makes a param block "a layer" versus an ad-hoc vascular/tissue dict a
  recipe looks up itself.

### Organ family tree

`Organ` (abstract; `organ_class.py`) → `RootAnatomy`, `StemAnatomy`, `LeafAnatomy`,
`NeedleAnatomy`, `RoiOrgan`. Root/stem/leaf are **transparent factories**: their `__new__`
inspects the `planttype` param and returns a monocot/dicot (/continuous) subclass instead of an
instance of the base class itself — `RootAnatomy(...)` can hand you back a
`MonocotRootAnatomy` or `DicotRootAnatomy`. `Organ.create_from_input(data)` is one level up:
it picks the organ *family* first (root vs. stem vs. leaf vs. needle), then delegates to that
family's own factory.

| Class | Represents |
| --- | --- |
| `MonocotRootAnatomy` | Root: ring of metaxylem bundles, or actinostele-style "arch" (metaxylem ring + graded protoxylem poles) |
| `DicotRootAnatomy` | Root: star-shaped xylem + cambium + phloem, optional secondary growth |
| `RootSeries` / `DicotRootSeries` (`root_series.py`) | Longitudinal series of root cross-sections along an axis with identity-tracked vessels; drives multiple `RootAnatomy` instances, not an `Organ` subclass itself |
| `MonocotStemAnatomy` | Stem: atactostele — bundles scattered through ground tissue, collateral, no cambium |
| `DicotStemAnatomy` | Stem: eustele — a ring of discrete collateral bundles around a pith |
| `ContinuousDicotStemAnatomy` (extends `DicotStemAnatomy`) | Dicot stem with a continuous vascular cylinder instead of discrete bundles |
| `MonocotLeafAnatomy` | Leaf: one uniform mesophyll, amphistomatous |
| `DicotLeafAnatomy` | Leaf: dorsiventral palisade/spongy split |
| `NeedleAnatomy` | Gymnosperm needle: no monocot/dicot split; transfusion tissue, resin ducts, bespoke central vascular ellipses |
| `RoiOrgan` | Bypasses procedural generation — loads a folder of ImageJ `.roi` files and populates cells directly from digitized real-image outlines |
| `CellSetOrgan` (`cellset_organ.py`) | Hybrid: imports a CellSet `<cellsetdata>` XML verbatim, then regenerates only the tissues MECHA can't consume (any annular cell, plus `regenerate=`) with a GRANAP donor organ grafted into the real outline |

### Shape-first tissues & `TissueRecipe` (`tissue_class.py`)

A shared, organ-agnostic vocabulary every organ family plugs into — not needle- or
root-specific. `Tissue(tag, shape)` couples a tag with a shapely region, transformed as pure
geometry (`.rotate`/`.translate`/`.smooth`/`.difference`/`.intersection`/`.union`) before any
cell exists. `TissueRecipe` is an ordered, inspectable list of build steps
(`.fill`/`.fill_each` = region → cells via a named strategy; `.cleanup` = cell-level fixup;
`.special` = bespoke placement; `.build()` runs it, `.describe()`/`.format_plan()` preview it
without placing cells). `Organ._create_vascular_tissue`/`_organ_specific_tissues` are concrete
on the base class and just call `self._vascular_recipe(...).build()` /
`self._organ_recipe().build()` — each organ subclass supplies only those two recipe methods.
`retag_tissue` (wrapped as `Organ.retag_cells`) is the only valid edit on an already-filled
cell: rename its tag.

Note: `doc/tissue_refactor.md`/`doc/tissue_roadmap.md` (the design docs for this refactor,
now complete) call this module `tissue_builder.py` — that file doesn't exist in this checkout;
the module is `tissue_class.py`.

### CellSet ingestion (`cellset_reader.py` + `cellset_organ.py`)

CellSet stores a section as a **shared-wall mesh** (cells reference walls by id; an interior
wall is referenced by exactly two cells), so adjacent cells share vertices verbatim and
`CellGenerator._build_topology` can pair their walls — the opposite of `RoiOrgan`, whose
independently-digitized, `smoothing_polygon`-resampled outlines share nothing and therefore
export a network with one private wall per cell and zero cell↔cell edges. **Never smooth or
resample an imported outline**, and set `protect_shape=True` on imported cells or
`CellGenerator.simplify_cells` rebuilds each ring from its junction vertices alone and
straightens every measured wall into a chord.

Grafting a fresh tessellation against existing cells has three traps, all of which fail
*silently* — the render looks perfect while the network is wrong. Each is enforced by
`CellSetOrgan.topology_report()` and `test/test_cellset_organ.py`:

- **A one-cell wall is not just a missing link.** MECHA's
  `NetworkBuilder.identify_border_walls_junctions` files a single-reference wall whose owner
  isn't epidermis under `border_aerenchyma`, i.e. it becomes a *gas-space boundary in the
  middle of the tissue*. So the region must be tiled exactly, and border walls must appear
  only on the section's real outer surface. Two things are needed for that: the donor is grown
  past the region rim and clipped back (a tessellation's outer ring is straightened into
  chords, cutting the corners of a wiggly real outline — measured 97.2 % coverage in 13 rim
  gaps without it), and `special_tissues.conform_boundaries` makes both sides carry a
  bit-identical vertex — outward across the rim, inward into the kept cells (where a vessel
  drawn with 9 vertices abuts 7 donor cells, so most junctions along it land mid-edge on the
  vessel's own coarse ring), **and donor↔donor**, because `absorb_residual` re-nodes the host
  it merges a patch into but not the other donor cells around that patch. Skipping the weld
  dropped shared endodermis↔pericycle walls from 40 to 5; conforming only donor→real left
  22-49 interior border walls at every seed but 0. (`weld_points_onto_rings` is the
  exactness-only variant and is **not** in this pipeline — don't go looking there for the
  behaviour.)
- **Don't use GEOS overlay noding to do that welding.** It would work as a side effect of
  `difference`, but the geos pins here are load-bearing (see above), so the insertion is
  explicit and deterministic instead.
- **Never set `protect_topology=True` on imported cells.** `NetworkExporter` derives
  `protected_shape_set` from `protect_topology` and folds it into the wall key as an *ordered*
  tuple; the two cells flanking a wall traverse that segment in opposite directions, so a
  non-empty signature reverses and the halves stop pairing. (`protect_shape` is the flag
  `simplify_cells` reads — they are not interchangeable.)

The vascular recipe is handed the innermost `"stele"` layer polygon, **not** the region
(`RootAnatomy._create_central_layers` has already peeled the pericycle and one stele ring off
it), so `default_stele_input` sizes the xylem/phloem/cambium radii against a `core_radius`
that `_build_donor` measures on a throwaway donor. Sizing them against the region instead puts
the phloem valleys outside the polygon and they produce no cells at all, with no error.

Units: the reader's default `scale=0.001` converts CellSet µm to GRANAP mm, which is what
MECHA's `GeometryData.im_scale` default of 1000 expects to read back from `write_to_xml`.

### Shared cross-organ modules

- **`vascular_bundle.py`** — bundle-building for **leaf and stem only** (needle and root build
  their vascular zones directly in their own files). `build_bundle()` = an oriented envelope
  polygon, partitioned banded (collateral/bicollateral) or concentric (core+ring), then filled
  per sub-zone. `_grow_bundle_sheath`/`_sheath_zones` size-guard the sheath against a "radial
  sunburst" Voronoi artifact when a small bundle sheath borders much coarser ground tissue —
  `needle_class.py` reimplements the same guard independently for resin-duct sheaths.
- **`secondary_growth.py`** — cambium secondary growth (annual rings, secondary xylem/phloem,
  medullary rays), opt-in via `secondary_growth.value` on `DicotRootAnatomy`/`DicotStemAnatomy`
  only; monocots, leaves and needles never call it.
- **`network_base.py` (`AbstractNetwork`) + `anatomy_writer.py`** — the hydraulic-network export
  (`export_to_adjencymatrix`, `plot_network`) is load-bearing, not a side feature: it's the
  intended MECHA integration point and is exercised directly by
  `test/test_root_needle_network.py`. `AnatomyWriter` also owns `write_to_xml`/
  `write_xml_geometry`/`write_to_obj`/`write_to_svg`/`write_to_geo`.
- **`shapes.py`** — `PolygonInterpolator` (vertex-correspondence polygon morphing), used solely
  by `needle_class.py`'s `reshape_layers()` to morph half-ellipse layer rings into a full
  ellipse around the central cylinder.
- **`math_functions.py`** — `five_pl`/`linear` normalized radial gradient functions +
  `rescale()`, used for stele/pith/vessel center-to-edge size gradients across
  `geometry_collection.py`, root and stem.

### Known stale/fragile things worth knowing before touching them

- `NeedleAnatomy._organ_recipe()` is an **append-only feature registry** — it grew from 2 steps
  to 5 (resin ducts, transfusion tissue, corner-parenchyma-to-Strasburger retagging, stomata,
  layer-count zoning) and will grow again. Its **order is load-bearing** and justified at length
  in that method's docstring: `"transfusion tissue"` must precede `"stomata"` (which calls
  `recenter_cells()`, and cannot retroactively shift the already-computed transfusion zone) and
  `"layer-count zoning"` must follow it (it reads `cell.angle` as `recenter_cells` leaves it).
  `test_recipe_vocabulary.py::test_needle_recipes_are_inspectable` therefore asserts the
  vocabulary contract plus that relative order, *not* an inventory — adding a sixth step is
  fine, reordering those three is not.
- **Exact-tie predicates are this codebase's recurring cross-platform bug**, and three have now
  been fixed (`_chebyshev_center`, `_grow_bundle_sheath`, `retag_corner_parenchyma` — see
  `test_vascular_regression.py`'s module docstring). The shape is always the same: a discrete
  test (`contains`, a `difference` severance, a branch-and-bound tie-break) evaluated on
  coordinates that are *mathematically exactly on* the boundary, because the predicate's region
  and its test points were derived from the same geometry through the same transform. glibc,
  UCRT and Apple/arm64 then disagree at the last bit — arm64 especially, since it contracts
  `a*b+c` into one FMA with a single rounding — and the anatomy follows the noise.
  **Before "fixing" one, measure the distance distribution**: in every case so far the true
  ties were separated from the genuine decisions by 6+ orders of magnitude (needle: ties within
  5.3e-17, nearest real decision beyond 1e-6), which is what makes a tolerance a *fix* rather
  than a fitted constant. If there is no such gap, a tolerance only moves the arbitrariness.
  Twelve more sites share the pattern (`grep -n 'contains(Point'`); none currently diverges, so
  they are latent and were deliberately left alone — touching them rebaselines goldens that
  agree on all three platforms today.
- `CellSetOrgan` is only verified sound for a **diarch** donor stele
  (`n_vascular_peak=2`) and its default options. Three or more xylem poles packs the donor's
  vessels tighter than a small stele can hold and leaves degenerate slivers — measured on the
  *Arabidopsis* section at `seed=0`: 2 poles → 0 interior border walls, 3/4/5 poles → 30-40.
  It is the donor's vessel packing upstream of the graft, not the interface welding (changing
  the weld barely moves the number). Measured on x86 at `seed=0` after the graft fixes, the
  other option sets that still break are `overshoot=0` (coverage 0.9941, 3 interior border
  walls) and `donor_xylem_tag=None` (2 multi-cell walls). `recenter=False`,
  `keep_inner_cells=False` and an explicit `star_orientation` now tile cleanly and are **no
  longer** known-bad — the older list here and in `topology_report`'s warning predated the
  graft fixes.
  **Don't write a test that pins how badly a bad option set breaks.** Two attempts were made
  and both had to be removed: `overshoot=0` gives coverage 0.9941 with 3 interior border walls
  on x86 but tiles *exactly* (coverage 1.0, no defect) on macOS/arm64, because where the
  straightened outer-ring chords fall depends on which vertices become junctions. Even rim
  coverage — a gross 0.6 % quantity, not a last-bit tie — is not portable. Assert instead that
  the **default** path tiles exactly, and that a defect *when present* is reported
  (`test_a_broken_interface_is_not_silent` does the latter on injected data, so it needs no
  misconfigured build at all).
  `topology_report()` warns loudly whenever a defect is present, and
  `example/brassicaceae_roots.py` gates on it — don't average a `TOPOLOGY`-flagged section
  into a comparison.
  The **seed** fragility that used to sit here is fixed: interior border walls were 49/40/39/
  34/22 at seeds 1-5 (0 at seed 0 by luck) and are now 0 at every seed tested, asserted by
  `test_no_interior_border_walls_at_any_seed`. Three causes, all silent — `absorb_residual`
  fed the zero-area slivers a boolean `difference` always returns, `_graft` conforming only
  donor→real, and `conform_boundaries` splicing into just the nearest ring.
  **One milder defect remains:** seeds 2 and 4 still produce exactly one wall with three
  flanking cells, because `carve_cells` keeps only the largest part when a real vessel splits
  a donor cell and the discarded part becomes a gap `absorb_residual` then hands to a
  neighbour. `topology_report()` warns, so it is not silent; `multi_cell_walls == 0` is
  asserted at seed 0 only. Fixing it means keeping both parts of a split cell, which changes
  `carve_cells` for `seat_air_spaces` too and would shift the golden censuses.
- `doc/performance_proposals.md` lists concrete, **not-yet-applied** speedups to
  `generate_cells` (e.g. reducing Voronoi border-seed density) that would shift the golden cell
  census. Don't apply them incidentally — they need a deliberate visual review and a rebaseline
  of `doc/perf_baseline.json` and the golden tests together.
- Per the current README: intercellular-space generation and general polygon creation have open
  known issues — treat anything in that area as more likely to need care, not as settled.

### Testing conventions

- Golden regression tests (`test/test_vascular_regression.py`) pin an exact per-organ,
  per-seed cell-type census. A deliberate anatomy change is expected to shift these; review the
  new render/census before updating the golden data, don't just silence the failure.
- `test/golden/param_dicts.json` (checked by `test_param_schema_equivalence.py`) freezes
  `to_dict_list()` output for the 11 named presets, so refactoring the pydantic param-class
  hierarchy never silently renames/reorders a consumer-visible field. Regenerate deliberately
  (`python test/test_param_schema_equivalence.py --update`) after an intentional schema change.
- PRs into `main` must originate from `develop` (enforced by a CI check in `.github/workflows`).
