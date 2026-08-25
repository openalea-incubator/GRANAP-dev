# GRANAP — Package Tutorial

GRANAP generates **2‑D cross‑sectional plant anatomy** (roots, needles, …) as a
population of cells, then exports it as geometry (GeoPandas / SVG / OBJ / XML /
GMSH) or as a hydraulic network (adjacency matrix). This tutorial explains how
the package is put together, how to drive it through the public API, how to add a
brand‑new organ, and which helper functions to reach for when designing a new
tissue.

- [1. Mental model](#1-mental-model)
- [2. Installation & quick start](#2-installation--quick-start)
- [3. The public API](#3-the-public-api)
  - [3.1 `OrganInputData` — configuration](#31-organinputdata--configuration)
  - [3.2 `Organ` — the anatomy object](#32-organ--the-anatomy-object)
  - [3.3 Plotting & inspection](#33-plotting--inspection)
  - [3.4 Export](#34-export)
- [4. The generation pipeline](#4-the-generation-pipeline)
- [5. Creating a new organ](#5-creating-a-new-organ)
- [6. Designing a new tissue — the useful functions](#6-designing-a-new-tissue--the-useful-functions)
- [7. Cheat sheet](#7-cheat-sheet)

---

## 1. Mental model

Think of an organ cross‑section as three nested things:

1. **A base shape** — the outline of the organ (a circle, half‑ellipse, star, …).
2. **Concentric layers ("tissues")** peeled inward from that outline —
   epidermis, cortex, endodermis, … each a ring of cells of a given size.
3. **A central vascular zone** — xylem / phloem / cambium, laid out with bespoke
   geometry per organ.

GRANAP builds this in a fixed pipeline: it computes **polygons** for every zone,
scatters **seed points** inside each zone, adds the **vascular seeds**, then runs
one global **Voronoi tessellation** to turn all seeds into touching cell
polygons. Post‑processing adds intercellular air spaces / aerenchyma and
recomputes per‑cell properties. See [§4](#4-the-generation-pipeline) for detail.

Two ideas run through the codebase and are worth internalising early:

- **Shape‑first tissues.** A tissue *is* a region (a tag + a shapely shape). You
  manipulate it as pure geometry — rotate, translate, smooth, boolean‑combine —
  *before* any cell exists. Filling the region with cells is a separate,
  terminal step. The `Tissue` class and the `fill_*` primitives in
  `tissue_class.py` embody this.
- **Recipes.** Each organ declares *how* it builds its vascular tissue as an
  ordered, inspectable list of steps (`TissueRecipe`), rather than burying the
  order in control flow. You can print a recipe's plan before running it.

Key modules:

| Module | Responsibility |
| --- | --- |
| `input_data.py` | `OrganInputData` + all the pydantic parameter schemas (defaults, validation, presets, XML/dict loading). |
| `organ_class.py` | `Organ` abstract base — the pipeline, plotting, export, intercellular/aerenchyma logic. |
| `root_class.py` / `root_monocot_class.py` / `root_dicot_class.py` | Root anatomy. `RootAnatomy(...)` is a factory that returns a monocot or dicot instance. |
| `needle_class.py` | Needle (gymnosperm leaf) anatomy. |
| `tissue_class.py` | The reusable tissue‑building **vocabulary**: `Tissue`, `fill_*`, `place_packed_group`, `retag_tissue`, `TissueRecipe`. |
| `special_tissues.py` | Post‑fill special structures (resin ducts, stomata, `consider_as_cell`, `carve_and_insert`). |
| `geometry_collection.py` | `GeometryProcessor` — all low‑level shapely geometry helpers (shapes, packing, ellipses, pizza slices…). |
| `cell_class.py` / `cell_manager.py` | `Cell` and `CellManager` (a tagged bag of cells with query/remove helpers). |
| `generate_cell.py` | `CellGenerator` — seed scattering, Voronoi, grouping, simplification. |
| `visualization.py` | `plot_tissues` / `build_anatomy_tissues` dry‑run previews. |

---

## 2. Installation & quick start

```bash
mamba env create -f ./conda/environment.yaml -y   # first time
mamba activate granap
pip install -e .                                   # editable install
```

Minimal end‑to‑end run — a default monocot root:

```python
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

data = OrganInputData.for_root()      # default monocot preset (planttype = 1)
root = RootAnatomy(data, seed=0)      # seed=0 → reproducible
root.generate_cells()                 # run the pipeline
root.plot_cells()                     # matplotlib cross‑section
```

> **Reproducibility.** An unseeded organ uses fresh randomness each run. Always
> pass `seed=0` (or any fixed int) when you need a stable result — the whole test
> suite relies on this.

The three built‑in organ presets:

```python
OrganInputData.for_root()            # monocot root
OrganInputData.for_dicot_root()      # dicot root (secondary growth off)
OrganInputData.for_dicot_secondary() # dicot root, secondary growth on
OrganInputData.for_needle()          # gymnosperm needle
```

You can also let the factory pick the class for you:

```python
from openalea.granap.organ_class import Organ
organ = Organ.create_from_input(OrganInputData.for_needle())
```

---

## 3. The public API

### 3.1 `OrganInputData` — configuration

Everything about an organ is configured through an `OrganInputData` object: a
list of typed parameter blocks (pydantic models such as `SteleParams`,
`RootXylemParams`, `EpidermisParams`, …). You rarely build this list by hand —
start from a preset and tweak.

**The golden rule: configure `data` fully, *then* build the organ once.**
Construction snapshots the params and parses the vascular geometry, so changes
made to `data` after `RootAnatomy(data)` are ignored.

```python
data = OrganInputData.for_dicot_root()

# read / set fields — three equivalent styles:
data["xylem"].n_vascular_peak = 6          # attribute on the live pydantic model
data.set_value("xylem", "outer_radius", 0.155)   # validated setter
data.set_values("stele", thickness=0.31, cell_diameter=0.01)  # several at once

# add a whole new tissue layer (a plain dict is fine):
data.params.append({
    "name": "inner_cortex",
    "cell_diameter": 0.065, "cell_width": 0.05,
    "n_layers": 1, "shift": 0.5, "order": 3.5,
})

# sanity‑check cross‑field geometry constraints before building:
issues = data.validate()          # returns a list of human‑readable problems
# data.validate(raise_on_error=True)  # or raise instead
```

Useful `OrganInputData` methods:

| Method | Purpose |
| --- | --- |
| `for_root` / `for_dicot_root` / `for_dicot_secondary` / `for_dicot_annual` / `for_needle` | Preset factories. |
| `from_dict_list(dicts)` | Build from raw dicts (no validation). |
| `from_xml(path)` | Parse a legacy GRANAR‑style XML (renames legacy attrs, fills defaults). |
| `get(name)` / `data[name]` / `data.name` | Fetch a param block by its `name`. |
| `names()` | List all param names present. |
| `set_value(name, field, value)` / `set_values(name, **fields)` | Validated updates. |
| `to_dict_list()` | Flatten to plain dicts (what the organ actually consumes). |
| `validate()` | Cross‑field geometry checks (star inner ≤ outer, secondary cambium encloses primary, …). |

**Layer ordering.** Any param block carrying an `order` key is treated as a
concentric layer. Higher `order` = more outward. That's how `epidermis` (order
6) ends up outside `cortex` (order 4) outside `endodermis` (order 3), and how
you slot a custom `inner_cortex` at `order: 3.5`.

### 3.2 `Organ` — the anatomy object

`Organ` is the abstract base; you always instantiate a concrete subclass
(`RootAnatomy`, `NeedleAnatomy`) or use `Organ.create_from_input(...)`.

```python
root = RootAnatomy(data, seed=0)
```

Construction parses params and installs the default layers, but does **not**
generate cells. The heavy lifting is lazy and cached:

| Call | What it returns / does |
| --- | --- |
| `generate_base_shape()` | The outline polygon. |
| `generate_layer_polygons()` | The list of `LayerPolygon` rings (cached). |
| `generate_cells()` | Runs the full pipeline; returns a `GeoDataFrame` of cells (cached). |
| `get_statistics()` | Dict: total cells, cell types, per‑type counts, areas, layer count. |
| `retag_cells(old, new)` | Rename every cell of one tag to another (updates cells *and* the cached GeoDataFrame). |
| `list_layers()` / `get_layer(name)` / `add_layer(layer, pos)` / `remove_layer(name)` | Layer management (invalidates cached geometry). |

Everything downstream (`plot_cells`, `export_*`, `get_statistics`,
`export_to_adjencymatrix`) calls `generate_cells()` for you, so an explicit call
is optional — but handy when you want to `retag_cells` or measure timing.

A common finishing touch is merging helper tags back together after the build:

```python
root.generate_cells()
root.retag_cells("inner_cortex", "cortex")
root.retag_cells("outer_cortex", "cortex")
```

### 3.3 Plotting & inspection

```python
root.plot_layers()          # just the layer boundary rings
root.plot_cells()           # the tessellated cells, coloured by type
root.plot_tissues()         # dry‑run preview of every tissue ZONE, no cells placed
```

`plot_tissues` is the design tool: it colours the layer rings and overlays the
vascular polygons **without running Voronoi**, so it's fast and shows you the
zones you're about to fill. All three accept `ax=`, `show=`, `title=`:

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
root.plot_tissues(ax=axes[0], show=False, labels=True, fuse=True)
root.plot_cells(ax=axes[1], show=False, title="cells")
plt.show()
```

To inspect the *plan* an organ will follow (see recipes in [§5](#5-creating-a-new-organ)):

```python
recipe = root._vascular_recipe(root.generate_layer_polygons()[-1]["polygon"])
print(recipe.format_plan())
# [special] vascular ellipse grid -> xylem, phloem, cambium, ...
```

### 3.4 Export

```python
gdf = root.export_to_geopandas()     # same as generate_cells()
root.export_to_csv("root.csv")       # cell table, no geometry
root.write_to_svg("root.svg")
root.write_to_obj("root.obj")
root.write_to_xml("root.xml")        # GRANAR‑style
root.write_xml_geometry("root_mecha.xml")   # for MECHA
root.write_to_geo("root.geo")        # GMSH
A = root.export_to_adjencymatrix()   # sparse hydraulic network (lil_matrix)
```

---

## 4. The generation pipeline

`Organ.generate_cells()` is the heart of the package. In order:

1. **Layer polygons** (`_build_layer_polygons`) — start from the base shape and
   buffer inward one layer at a time, each ring shrunk by its cell diameter.
   Subclasses add the central zone (`_create_central_layers`) and may morph the
   result (`reshape_layers`).
2. **Seed scattering** (`CellGenerator.generate_cells_info`) — scatter seed
   points inside every layer ring, spaced by the layer's cell diameter/width.
3. **Vascular tissue** (`allocate_vascular_tissue` → `_create_vascular_tissue` →
   your `_vascular_recipe(...).build()`) — populate `self.vascular_cells` (a
   `CellManager`) and `self.vascular_polygons`.
4. **Unified vascular mask** — every layer seed falling inside any vascular
   polygon is removed, then the vascular seeds are added. *You never do this
   removal yourself* — the base class does it once, so a recipe only has to
   place seeds and record its regions.
5. **Organ‑specific post‑fill tissues** (`_organ_specific_tissues` →
   `_organ_recipe().build()`) — e.g. needle resin ducts and stomata.
6. **Voronoi** (`voronoi_diagram` → `process_voronoi_groups` → `simplify_cells`)
   — one global tessellation of all seeds; seeds sharing an `id_group` fuse into
   a single cell.
7. **Intercellular spaces & aerenchyma** (`add_intercellular_spaces`).
8. **Layer population + property recompute + GeoDataFrame export.**

The two contracts a subclass fulfils are therefore small: **compute the zone
polygons** and **place the vascular seeds**. Grouping (`id_group`) is what
matters, not absolute id values — seeds sharing an id become one Voronoi cell.

---

## 5. Creating a new organ

A new organ is a subclass of `Organ`. There are only a handful of methods to
implement; the base class drives the pipeline.

### Required (abstract) methods

| Method | Contract |
| --- | --- |
| `_create_base_shape() -> Polygon` | Return the outline polygon. Use `GeometryProcessor` helpers. |
| `_create_central_layers(polygon, params) -> List[LayerPolygon]` | Peel the central zone inward from `polygon`, returning the inner rings (below the last concentric layer). |

### Overridable hooks (sensible defaults provided)

| Method | Default | Override when… |
| --- | --- | --- |
| `_which_layer_for_vascular(layers_polygons)` | — (abstract) | tell the base class which layer polygon hosts the vascular zone. |
| `_vascular_recipe(polygon) -> TissueRecipe` | empty | you have vascular tissue to place. **This is the main one.** |
| `_organ_recipe() -> TissueRecipe` | empty | you have post‑Voronoi special tissues (ducts, stomata…). |
| `reshape_layers(layers_polygons)` | identity | you want to morph the layer rings (e.g. needle's half‑ellipse → inner ellipse). |
| `_extra_tissue_polygons(layers_polygons)` | `{}` | you want `plot_tissues` to preview extra structures. |
| `LAYER_SMOOTH_FACTOR` (class attr) | `0.5` | roots use `0.0` (exact ring thickness); needles round corners. |

### Skeleton

```python
from shapely.geometry import Polygon
from openalea.granap.organ_class import Organ
from openalea.granap.layer_class import Layer, LayerPolygon
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.cell_manager import CellManager
from openalea.granap.tissue_class import Tissue, TissueRecipe
from openalea.granap.input_data import OrganInputData


class StemAnatomy(Organ):
    LAYER_SMOOTH_FACTOR = 0.0

    def __init__(self, input_data=None, seed=None):
        super().__init__(seed=seed)
        if isinstance(input_data, OrganInputData):
            self.params = input_data.to_dict_list()
        elif isinstance(input_data, list):
            self.params = input_data
        else:
            self.params = OrganInputData.for_root().to_dict_list()   # or your own preset

        # Parse the param blocks you care about:
        self.global_params = self._get_param("planttype")
        self.aerenchyma_params = self._get_param("aerenchyma")
        self.intercellular_spaces_params = [p for p in self.params
                                            if p["name"] == "inter_cellular_spaces"]

        # Vascular containers the base pipeline expects:
        self.vascular_cells = CellManager()
        self.vascular_polygons = []

        # Install concentric layers (any param with an 'order' key):
        for p in sorted((p for p in self.params if "order" in p),
                        key=lambda p: p["order"]):
            self.layer_manager.add_layer(Layer.from_dict(p))

    # --- required geometry ------------------------------------------------
    def _create_base_shape(self) -> Polygon:
        return GeometryProcessor.circle_polygon(radius=1.0)

    def _create_central_layers(self, current_polygon, params):
        layers, i = [], len(params)
        d = 0.03                                   # central cell diameter
        while current_polygon.area > (d / 2) ** 2 * 3.14159:
            current_polygon = GeometryProcessor.buffer_polygon(
                current_polygon, -d, smooth_factor=0.5)
            layers.append(LayerPolygon(name="pith", polygon=current_polygon,
                                       cell_diameter=d, id_layer=i + 1))
            i += 1
        return layers

    # --- which layer hosts the vascular zone ------------------------------
    def _which_layer_for_vascular(self, layers_polygons):
        names = [l["name"] for l in layers_polygons]
        return layers_polygons[names.index("pith")]["polygon"]

    # --- vascular tissue as a recipe --------------------------------------
    def _vascular_recipe(self, polygon) -> TissueRecipe:
        recipe = TissueRecipe(cells=self.vascular_cells, rng=self.rng)
        xylem = Tissue("xylem", polygon).smooth(0.3)          # a shape‑first region
        recipe.fill("central xylem", xylem, strategy="packing",
                    proportion=0.6,
                    record=lambda t, res: self.vascular_polygons.extend(
                        p for p, _tag, _gid in res))
        return recipe
```

That's a complete, if simple, organ. Register it in
`Organ.create_from_input` if you want the factory to dispatch to it, or just
instantiate `StemAnatomy(data, seed=0)` directly.

### Studying the real ones

- **`NeedleAnatomy`** is the cleanest full example: a `_create_base_shape`
  (half‑ellipse), a `reshape_layers` that morphs toward an inner ellipse, a
  `_vascular_recipe` with a single bespoke `special` step (the xylem/phloem grid
  packed into two ellipses), and an `_organ_recipe` with two `special` steps
  (resin ducts, stomata).
- **`MonocotRootAnatomy` / `DicotRootAnatomy`** show declarative recipes that mix
  `fill` / `fill_each` (region → cells), `cleanup` (cell‑level fixups), and
  `special` (bespoke placements like the metaxylem sheath or secondary‑growth
  pizza slices).

---

## 6. Designing a new tissue — the useful functions

This is the toolbox you compose inside a `_vascular_recipe`. The model is
**region first, then fill**.

### 6.1 Build the region — `GeometryProcessor` (`geometry_collection.py`)

All static methods. The ones you'll reach for most:

| Function | Use |
| --- | --- |
| `circle_polygon`, `rectangle_polygon`, `triangle_polygon`, `half_ellipse_polygon`, `star_polygon` | Base outlines / regions. |
| `ellipse_to_polygon(cx, cy, rx, ry, angle)` | An arbitrary ellipse. |
| `focus_ellipse_polygon(...)` / `fit_focus_ellipse(profile)` | Superellipse ("focus ellipse") and least‑squares fit to a measured contour. |
| `egg_polygon(...)` | Asymmetric teardrop oval. |
| `buffer_polygon(poly, distance, smooth_factor=0)` | Grow (+) / shrink (−) a region, with optional corner smoothing. The workhorse for peeling rings and insetting. |
| `union_polygons(list)` | Merge regions. |
| `pizza_slice(polygon, n_slices)` | Split a region into `n` radial wedges (e.g. one bundle per sector). |
| `fit_inner_ellipse(polygon, rx, ry)` | Largest ellipse fitting inside a region. |
| `two_ellipses(polygon, rx, ry)` | Split a region left/right and fit an oriented ellipse in each (needle vascular bundles). |
| `pack_circles(polygon, proportion=…, diameter_max=…, gradient_*=…, rng=…)` | Apollonian / gradient circle packing of a region — returns `(cx, cy, r)` (or ellipse) records. The basis of vessel packing. |
| `get_inscribed_circle(polygon)` | `(cx, cy, radius)` of the largest inscribed circle (pole of inaccessibility). |
| `resample_coords(coords, n)` | Even‑arclength resampling of a boundary. |

### 6.2 Wrap it as a `Tissue` (shape‑first) — `tissue_class.py`

`Tissue(tag, shape)` couples a region with the tag its cells will take. Every
transform mutates in place and returns `self`, so they chain. Nothing here
creates cells yet:

```python
from openalea.granap.tissue_class import Tissue

phloem = (Tissue("phloem", ellipse)
          .rotate(30, origin=(0, 0))
          .translate(0.1, 0.0)
          .smooth(0.2)
          .intersection(stele)      # clip to the stele
          .difference(xylem_star))  # carve the xylem out
if not phloem.is_empty:
    print(phloem.area)
```

Methods: `.rotate(angle, origin)`, `.translate(dx, dy)`, `.smooth(factor)`,
`.difference(other)`, `.intersection(other)`, `.union(other)` (each `other` may
be a `Tissue` or a raw shapely geometry), plus `.is_empty` / `.area`.

> **Resolve overlaps with region algebra up front**, not with post‑hoc seed
> removal. If two tissues would collide, `.difference()` one out of the other
> before filling.

### 6.3 Fill the region with cells

Four generative primitives turn a region into seed cells inside a
`CellManager`. In a recipe you usually name them via `strategy=`; you can also
call them directly.

| Primitive / strategy | What it does |
| --- | --- |
| `fill_by_packing` / `strategy="packing"` | Circle‑pack the region, seed a ring of border points per circle. For vessels, sieve elements, packed parenchyma. Forwards `**pack_kwargs` to `pack_circles`. |
| `fill_by_rings` / `strategy="rings"` | Seed concentric inward rings — for filling an annulus/zone with roughly radial files. |
| `fill_along` / `strategy="line"` | Seed cells along a polygon edge / line, oriented by the local tangent — for cambium files, sheaths. |
| `place_packed_group(target, packed, tag, …)` | The low‑level "seed every circle of a pre‑computed packing" used by `fill_by_packing`. Use directly when you packed the circles yourself, or to split one packing into two tags via `min_diameter` / `alt_type`. |

`Cell.radial(type, x, y, diameter, id_group, center)` is the seeding idiom for
placing an individual seed whose polar `angle`/`radius` are measured from a
tissue centre — handy inside a bespoke `special` step.

Remember: **seeds sharing an `id_group` fuse into one Voronoi cell.**
`CellManager.next_group_id()` gives you the next free id; the `fill_*` primitives
advance it for you.

### 6.4 Compose into a recipe — `TissueRecipe`

A recipe is an ordered, inspectable list of steps. Bind it to a target
`CellManager` + rng, then declare steps:

```python
recipe = TissueRecipe(cells=self.vascular_cells, rng=self.rng)

recipe.fill("xylem star", xylem_region, strategy="packing",
            proportion=0.7, record=self._record_xylem)     # one region
recipe.cleanup("clear stele under xylem", self._clear_stele)  # cell‑level fixup
recipe.fill_each("phloem valleys", self._phloem_zones,       # several regions
                 strategy="packing", proportion=0.5)         # (callable = lazy)
recipe.special("metaxylem sheath", self._fit_sheath,         # bespoke placement
               produces=("metaxylem",))
recipe.build()                                               # run every step
```

Step kinds: **`fill`** (one region), **`fill_each`** (many regions; the argument
may be a callable resolved at build time — use this when regions depend on an
earlier step), **`cleanup`** (produces nothing new — cell/group‑level fixups),
**`special`** (arbitrary bespoke placement), **`add`** (wrap any callable). The
optional `record(tissue, result)` hook lets you stash placed polygons into
`self.vascular_polygons` for the unified mask.

Inspection (no cells placed):

```python
recipe.describe()      # [(name, produces), ...]
recipe.plan()          # [(name, kind, produces), ...]
print(recipe.format_plan())
```

### 6.5 Special tissues & edit verbs

- `retag_tissue(cell_manager, old, new)` — the only valid edit on a *filled*
  cell: rename its tag. (`Organ.retag_cells` wraps this.)
- `special_tissues.consider_as_cell(cm, region, tag, …)` — collapse a whole
  region into a single cell whose polygon *is* the region.
- `special_tissues.carve_and_insert(cm, carve_polys, new_cells, …)` — the shared
  "remove cells under a mask, then insert new ones" pattern (behind resin ducts /
  stomata).
- `special_tissues.place_resin_duct(...)` / `place_stomata(...)` — cell‑placement
  halves of the needle's post‑fill structures.

### 6.6 The design loop

1. Sketch the zone with `GeometryProcessor` + `Tissue` algebra.
2. Preview with `organ.plot_tissues()` — fast, no Voronoi.
3. Pick a fill strategy and add a recipe step; `record` its polygons.
4. `generate_cells()` + `plot_cells()`; iterate.
5. Pin the result with `seed=0` and a cell‑type census
   (`get_statistics()["cells_per_type"]`) so later changes stay honest.

### 6.7 A complete worked example

`example/custom_tissue_vessel_ring.py` is a runnable end‑to‑end demonstration of
everything above. It subclasses `MonocotRootAnatomy` and overrides only
`_vascular_recipe` to build a **packed vessel ring**: an annulus region
(`Tissue(...).difference(pith)`) filled by `strategy="packing"` with a
centre→edge size gradient, its placed vessel polygons pushed into
`self.vascular_polygons` through the `record=` hook so the shared pipeline's
unified mask clears the layer seeds underneath. Run it with:

```bash
cd example && python custom_tissue_vessel_ring.py
```

It prints the cell‑type census and shows the `plot_tissues` zone preview beside
the final `plot_cells` output.

---

## 7. Cheat sheet

```python
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

# configure
data = OrganInputData.for_dicot_root()
data.set_values("xylem", n_vascular_peak=6, outer_radius=0.155)
data.validate(raise_on_error=True)

# build (once, after all config)
root = RootAnatomy(data, seed=0)
root.generate_cells()
root.retag_cells("inner_cortex", "cortex")

# inspect / plot / export
print(root.get_statistics()["cells_per_type"])
root.plot_tissues(show=False)   # zone preview
root.plot_cells()               # final cells
root.write_to_svg("root.svg")
```

**Design a tissue:** `GeometryProcessor` region → `Tissue(...).difference(...)` →
`recipe.fill(..., strategy="packing", record=...)` → `plot_tissues` →
`generate_cells`. Overlaps are resolved by region algebra before filling; cells
are retag‑only once placed.
