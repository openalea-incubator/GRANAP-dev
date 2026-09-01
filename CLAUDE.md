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
python example/needle/pinus_pinaster.py                    # measured needle example, renders + prints cell census
python doc/generate_callgraph.py                           # needs pyan3/graphviz (mamba install -c conda-forge pyan3 graphviz python-graphviz)
```

Always pass a fixed `seed=` (e.g. `seed=0`) when constructing an organ for anything you intend
to compare or test — an unseeded organ uses fresh randomness every run, and the whole test
suite (golden regressions included) depends on that determinism.

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

- `test_recipe_vocabulary.py::test_needle_recipes_are_inspectable` asserts
  `NeedleAnatomy._organ_recipe()` has exactly 2 steps (`"resin ducts"`, `"stomata"`); the real
  recipe now has 5 (transfusion tissue, corner-parenchyma-to-Strasburger retagging, and
  layer-count zoning were added later without updating this assertion). This is a **pre-existing
  failure**, not a regression from unrelated work — don't chase it as a side effect of other
  changes; fix the assertion itself if you're the one touching this.
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
