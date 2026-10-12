# Plan: GRANAP as a metafspm StructuralComponent on the MPG

Your request (2026-10-07):
- **Goal:** reimplement GRANAP on branch `to_metafspm` so that every module is backed by the MPG DataStructure, and whole plants can be populated with real anatomies.
- **What it should bring:**
  - robust indexing;
  - array-backed operations;
  - the extraction of a sparse hydraulic graph for solving, later passed to a MECHA functional component or an equivalent;
  - an optimised anatomy generation and/or a dynamic library reused per differentiation class;
  - unique xylem and phloem vessel ids, to wire successive anatomies explicitly through SubOrgan edges.
- **Strategy:**
  - progressively convert the sources in a separate `src/openalea/granap/<subpackage>`, so that GRANAP becomes a StructuralComponent;
  - anatomy nodes and edges map onto the Compartment and Connection scales;
  - other scales (e.g. cell layers) may sit between SubOrgan and Compartment.
- **Tests, two categories:**
  1. a rigorous comparison folder that reuses every existing test and replicates its results exactly, with one vertex per scale down to SubOrgan, carrying one cross-section;
  2. tests of the new capabilities: a real seedling populated with real anatomies, explicit wiring of successive anatomies with the information of the scales above, the dynamic library, the optimisation.

Already decided (metafspm `devplan_porting.md` §5, 2026-10-06):
- **Templates:** they are shared only so that the geometry is not regenerated. Each segment gets its own copy, i.e. its own Compartments and state variables, made as efficiently as possible.
- **Classes:** a class is defined by the diameter, the distance from the tip and the distance from the root–shoot junction. There are at most about a hundred classes.
- **Changing class:** a segment that changes class gets its anatomy replaced. Cell contents come from coarser components (e.g. root_carbon).
- **Axial junctions:** xylem and phloem only.
- **Graph:** every node is a Compartment, every edge a Connection.

## 1. Audit (2026-10-07, read only, nothing changed)

### 1.1 Size and stack

- **Sources:** `src/openalea/granap`, 30 modules, about 25 k lines. The largest are `needle_class` (2.5 k), `input_data` (1.9 k), `anatomy_writer` (1.6 k), `root_dicot_class` (1.5 k), `organ_class` (1.4 k), `geometry_collection` (1.4 k) and `vascular_bundle` (1.3 k).
- **Stack:** shapely 2.1.2 / GEOS 3.14.1 (pinned, because the goldens depend on it), scipy (Voronoi, cKDTree), numpy, geopandas, networkx, pydantic, opencv (one `fitEllipse`) and munkres (needle reshape).
- **Environments:**
  - the `granap` conda env: Python 3.14, GRANAP editable, no `openalea.mtg`;
  - the `metafspm` env: Python 3.13, no shapely.

  No environment holds both yet (Q1).

### 1.2 Data model today

- **`Cell`** (cell_class.py:5) is one mutable class used both as a Voronoi seed and as a finished cell.
  - Attributes: `x, y` (the seed, not the centroid), `diameter/width/height`, `type` (a free string; the tissue is only this string), `id_cell`, `id_layer`, `id_group`, `cgroup` (the MECHA int), `track_id`, `angle`, `radius`, `area`, `polygon` (shapely), `protect_topology/shape`.
  - The container is `CellManager`, a list with linear scans.
- **Identifiers:**
  - `id_group` is the real identity of a cell: about 10–25 Voronoi seeds share it, and their regions are unioned.
  - `id_cell` is overwritten with the list index at the end (cell_manager.py:131), and is not unique before that (special_tissues.py:111).
  - `id_layer` is an expanded-ring index. It is −1 for vascular and fill cells, 0 for air spaces, and a mean after fusions. It is not a reliable layer pointer.
  - Ids are deterministic for a given seed, but they depend on the order of the seed list.
- **Topology is not stored.** Adjacency is recomputed from the polygons by `CellGenerator._build_topology` (generate_cell.py:480): vertices are snapped with a cKDTree at 1 % of the 5th-percentile edge, then rounded to 6 decimals. Junctions are vertices shared by 3 or more cells, or by more than one cell pair.
- **Walls and junctions exist only in the exporter** (anatomy_writer.py:787–867), with throwaway ids.
  - A wall is the polyline between two consecutive junctions of a cell. Its key is the junction pair plus a shape signature, so two cells may share several walls.
  - A cell with fewer than two junctions has one closed wall.
  - A wall with one cell is a boundary wall.
- **The MECHA network** (`NetworkExporter.export`, anatomy_writer.py:705; networkx, network_base.py) has three node kinds:
  - walls `0..Nw−1`, junctions after them, then cells.
  - **Wall nodes** carry midpoint, length and thickness (by type, plus `outerwall`).
  - **Junction nodes** carry position.
  - **Cell nodes** carry centroid, area, cgroup, cell_type and bridge.

  Edge kinds (`path`):
  - `membrane` (cell–wall), `wall` (junction–wall, two per wall), `plasmodesmata` (cell–cell, for 2-cell walls), `wall_air` and `air_link` (air spaces, leaves), and needle bridges through virtual cells with no polygon.

  Edge attributes: `length`, `dist`, `d_vec`, `lateral_distance`, `wall_thickness`.
- **Not cells, or not plain cells:**
  - air spaces (`type="air space"`, sometimes `protect_topology`);
  - gaps and cavities (polygons only);
  - the "outside" ring (tessellated, then dropped);
  - virtual bridge cells;
  - multipart polygons (only the largest part is kept by the topology, the others silently lost);
  - holes (ignored).
- **Vessel identity:** none persists, except `track_id` on the prescribed metaxylem of the root series.
  - Vessels are distinguished only by `type` (xylem / metaxylem / protoxylem, collapsed to "xylem" in stems and leaves).
  - Bundles and poles exist only transiently (`BundleResult`, `vascular_tissue_polygons`).
- **Development:** there is no stage or class tag. It is driven by parameters: `secondary_growth`, `n_ring`, `alive_distance`, `arrangement`, `xylem_shape`, and the series' `length`.

### 1.3 Pipeline (`Organ.generate_cells`, organ_class.py:273)

1. the base shape;
2. the layer rings (negative buffers);
3. the seeds on each ring (ellipses of 10–15 seeds per cell);
4. the vascular recipe (`TissueRecipe`, packing, bundles);
5. layer seeds inside the vascular union are removed;
6. the organ recipe (ducts, stomata);
7. one global scipy Voronoi with a seeded jitter in list order;
8. the union by `id_group`, then `simplify_cells`;
9. intercellular spaces, and aerenchyma (a bisection of up to 21 rebuilds);
10. `fuse_gaps`;
11. `recalculate_cell_properties`, and a GeoDataFrame.

- **Randomness:** one `np.random.default_rng(seed)` per organ. `seed=None` by default, and `create_from_input` passes no seed.
- **Cost:** it scales with the Voronoi seeds, about 10–25 per cell. dicot_nettle is 504 k seeds for 36 k cells, about 53 s on Linux. `_build_topology` is pure-Python dicts and runs several times. The proposals in `doc/performance_proposals.md` that change the goldens (8/6 seed points per cell, a looser inscribed-circle tolerance) were not applied.

### 1.4 Organs and entry points

- **Factories:**
  - `RootAnatomy` → Monocot / Dicot;
  - `StemAnatomy` → Monocot / Dicot / ContinuousDicot;
  - `LeafAnatomy` → Monocot / Dicot;
  - `NeedleAnatomy`;
  - imported anatomies: `CellSetOrgan` (CellSet XML, with regenerated sub-regions grafted) and `RoiOrgan` (ImageJ ROIs, no shared walls).
- **Series:** `RootSeries` (monocot) and `DicotRootSeries` give the sections along one root.
  - Only the tracked metaxylem (`track_id`, with fusions in `members`) corresponds from section to section.
  - Everything else is regenerated per section with the same seed.
- **Parameters:** `OrganInputData` (pydantic v2), a list of named records with presets (`for_root`, `for_dicot_root`, …, `for_needle`), field metadata, `lint`, `validate` and `from_xml`. Lengths are in mm, implicitly.
- **Writers:** CellSet/GRANAR XML (the MECHA geometry), MECHA `Geometry.xml`, OBJ, SVG, gmsh `.geo` and CSV. `read_cellset` reads the XML back.

### 1.5 Tests

- **Files:** 22, with about 150 tests. They assert:
  - exact censuses per type at `seed=0` (`test_vascular_regression.py`, seven organs);
  - a frozen JSON of the presets (`golden/param_dicts.json`);
  - geometric properties (no overlaps, containment, counts per bundle, vessel tracking in the series, the transfusion network's edges, the CellSet round trip).
- **Platform:** the goldens are exact only on the pinned shapely/GEOS. The regression docstring documents the GEOS tie-breaks that were fixed.
- **Internals:** many tests read internal state: `all_cells.cells`, `Cell` attributes, `vascular_tissue_polygons`, recipe objects, `_layers_polygons`, the private needle tolerances.
- **Run (`granap` env, without the pytest cache):** 166 passed in 12 min 28 s, with about 100 k warnings (shapely deprecations, resin ducts scaled down). The slowest tests are `test_seed0_reproducible` (68 s), the medullar rays (48 s), `test_all_cases_reproducible` (43 s), the dicot star xylem (40 s) and the continuous stem (35 s). A comparison folder replaying everything on both implementations would take about 25 min: Q6 proposes a `slow` marker so a quick subset can run on each conversion step.

### 1.6 metafspm side

- **Scales:** the MPG already has **Layer (7)** and **Cell (8)** between SubOrgan (6) and Compartment (9), with tissue labels at Layer (Epidermis … MetaXylem, Phloem, Cambium, Aerenchyma, Mesophyll) and at Cell.
  - Compartment labels: Symplastic, Apoplastic.
  - Connection labels: Transmembrane, Symplastic, Apoplastic.
- **Anatomy mode (`MPGDataStructure(nodes="Compartment")`)** expects the Compartments directly under their SubOrgan: components of the Compartment anchor, with a topological link to the segment (`add_component_with_topo`, as in the SPAC example). A probe with Layer and Cell vertices in between fails: the SubOrgan is no longer a coarse location (§3, M1).
- **Already there:**
  - `add_components_bulk` (consecutive vids, one write per property);
  - `wire_junctions(rules)` with `match="nearest" | "equal" | "all"` on an ordering property, or a callable rule;
  - partial rewiring on differentiation;
  - `GraphView.edge_difference`;
  - per-vertex random streams (QPa).

## 2. Proposed design

**Subpackage:** `src/openalea/granap/mpg/` (name in Q2). The existing modules stay untouched and importable, so that the original tests keep running and the comparison can call both.

**The mapping onto the MPG**, for one cross-section under one SubOrgan:

| GRANAP | MPG scale | label / key properties |
|---|---|---|
| the section | SubOrgan | `anatomy_class`, `template_id`, diameter, organ type |
| a tissue (per type, and per bundle or pole for vascular tissues) | Layer | Layer tissue labels; `tissue`, `bundle_id`, `pole_id` |
| a biological cell (`id_group`) | Cell | Cell labels; polygon (shared from the template), area, centroid, `cgroup`, `type`, `vessel_id`, `track_id` |
| the cell's protoplast (MECHA cell node) | Compartment (Symplastic) | position, area; under its Cell |
| a wall (MECHA wall node) | Compartment (Apoplastic, `kind=wall`) | midpoint, length, thickness, `n_cells` |
| a wall junction (MECHA junction node) | Compartment (Apoplastic, `kind=junction`) | position |
| membrane, wall, plasmodesmata, wall_air, air_link, bridge | Connection | Transmembrane / Apoplastic / Symplastic labels; `path`, length, dist, lateral_distance, d_vec, wall_thickness |

- **Shared Compartments:** walls and junctions belong to two cells or more, so they cannot belong to one Cell. They hang from the SubOrgan (Q3).
- **Vessel ids:** unique xylem and phloem ids are given at seeding (`vessel_id`, plus `bundle_id` / `pole_id`). The root series' `track_id` carries over.
- **Axial junctions** between successive sections (SubOrgan edges): `wire_junctions` on the xylem and phloem Compartments, with `match="equal"` on `track_id` where it exists, else `"nearest"` on (angle, radius).
- **Library:** a template is generated once per class. It stores arrays: the Compartment and Connection tables, plus the shared geometry. Each segment instantiates it with `add_components_bulk` and an offset on the Connection endpoints, so it gets its own Compartments and Connections while the polygons are kept once per template.
- **Hydraulic graph:** the anatomy-mode GraphView of the Compartments and Connections. A `to_mecha_network()` rebuilds the current networkx graph, with MECHA's node order, from the MPG; it serves as the comparison oracle and as the bridge to MECHA until a MECHA FunctionalComponent reads the GraphView directly.

**Phases:**
- **P0, prerequisites in metafspm:** anatomy mode with Layer and Cell vertices between SubOrgan and Compartment (M1), and walls and junctions as Compartments shared under the SubOrgan.
- **P1, exact adapter:** the existing pipeline runs unchanged. Its final cells, the exporter's walls and junctions, and the network edges are written into a single-plant MPG, with one vertex per scale down to SubOrgan. The comparison tests (category 1) pass by construction. They pin what the MPG must carry.
- **P2, progressive conversion:**
  - module by module, the internal state moves into MPG arrays: cells and their properties first, then the topology (built once, stored as Connections, instead of `_build_topology` rerun), then the writers and the network exporter reading the MPG;
  - each step keeps the operation order and the rng stream, so the goldens stay exact;
  - the comparison folder guards every step.
- **P3, new capabilities (category 2):**
  - vessel ids;
  - explicit wiring of successive sections;
  - the library per class, and instantiation in a population MPG;
  - a real seedling (the SPAC example's architecture) populated with real anatomies;
  - the extraction of the hydraulic GraphView.
- **P4, optimisation:** the golden-changing levers (fewer seeds per cell, a vectorised topology, no repeated `simplify_cells` in the aerenchyma bisection), each opt-in and with its own goldens, measured against P1.

**Tests:**
- `test/mpg_comparison/`:
  - every existing test, run against the MPG-backed organs. The same assertions apply through a compatibility view that rebuilds `all_cells`, `Cell` attributes and the GeoDataFrame from the MPG (Q6);
  - a cross-check comparing both implementations entity by entity: cells (type, area, centroid, `cgroup`), walls, junctions, and the network's nodes, edges and attributes.
- `test/mpg_capabilities/`: the category-2 tests.

## 3. Questions

- **Q1 — environment.** The new code needs metafspm (openalea.mtg, numba) and GRANAP's pinned geometry stack in one environment. The `granap` env is Python 3.14 without openalea.mtg; the `metafspm` env is 3.13 without shapely.
  - Options: install metafspm (editable) and its dependencies into `granap`, or install GRANAP's pinned stack into `metafspm`, or create a new env.
  - **Recommendation:** add metafspm editable to `granap`, which keeps the pinned GEOS of the goldens. May I install it, once you have checked that openalea.mtg has a build for Python 3.14?
  - → answer: yes to recommendation
- **Q2 — subpackage name.** `src/openalea/granap/mpg/`? Or `structural/`, `component/`? It would hold the StructuralComponent, the MPG builders, the library and the compatibility view.
  - **Recommendation:** `mpg`.
  - → answer: mpg
- **Q3 — node scales.** MECHA's network has three node kinds; all three as Compartments (you said "all nodes are Compartments")? The cell protoplast would sit under its Cell vertex, and the walls and junctions under the SubOrgan, since each is shared by several cells.
  - Alternatives: a dedicated apoplast vertex per section at the Layer scale holding the walls and junctions, or walls hung from one of their cells (an arbitrary owner).
  - **Recommendation:** under the SubOrgan, with `kind` and label telling them apart. Aggregations per Cell or Layer then go through the Connections.
  - → answer: yes to recommendation
- **Q4 — the Layer scale.** In metafspm, Layer labels are tissues, while GRANAP's "layers" are peeled rings, and `id_layer` is not reliable. Should a Layer vertex be:
  - (a) one per tissue type in the section;
  - (b) one per tissue region (each bundle or pole its own Layer);
  - (c) one per GRANAP ring (`order` record × sub-ring)?

  **Recommendation:** (b). It keeps tissues and bundles as entities, for example for upscaling or for a phloem pole's own state. The ring index stays a Cell property.
  - → answer: yes
- **Q5 — metafspm prerequisite M1.** Anatomy mode today assumes Compartments linked directly to their SubOrgan, and a probe with Layer and Cell vertices in between fails. Making it support SubOrgan → Layer → Cell → Compartment is a change in metafspm: ownership chains, `owner("Cell")`, wiring and repartition through the intermediate scales.
  - **Recommendation:** do it first, in metafspm (P0), with its own plan and tests there.
  - → answer: yes to recommendation
- **Q6 — how the existing tests are reused.** Many tests read internal objects (`all_cells.cells`, `Cell` attributes, `vascular_tissue_polygons`, recipes, `_layers_polygons`). Options:
  - (a) copy each test into `test/mpg_comparison/`, parametrised over the old and the new organ classes, with the new ones offering a compatibility view (`all_cells`, `generate_cells()` GeoDataFrame, …) rebuilt from the MPG;
  - (b) rewrite them as old-vs-new output comparisons only.

  **Recommendation:** (a), plus the entity-by-entity cross-check, so every original assertion is replayed. A `slow` marker on the heaviest tests keeps a quick subset (a few minutes) for each conversion step, and the full replay (about 25 min) for the end of a phase. Tests that inspect pre-cell construction objects (recipes, tissue polygons, `plot_tissues`) stay on the original classes until P2 converts those modules; I list them in the folder as such.
  - → answer: yes to recommendation
- **Q7 — "exactly".** Exact means the same census, the same cells (type, `cgroup`, area and centroid to 1e-12, polygons equal), the same walls and junctions, and the same network (nodes, edges and attributes), on the pinned stack, at `seed=0`. Should the comparison also be byte-identical on the written files (XML, OBJ, SVG, geo)?
  - **Recommendation:** yes for XML and the network; the image-like formats by geometry equality.
  - → answer: yes to recommendation
- **Q8 — bugs met in the audit.** Should the comparison reproduce them first and fix them afterwards, with explicit golden updates? They are:
  - `cgroup` lost in `process_voronoi_groups` (generate_cell.py:459);
  - `id_layer=i_slice` stamped on monocot metaxylem (root_monocot_class.py:348);
  - multipart polygons silently reduced to their largest part by the topology;
  - the `air_link_radius` default being 1× the median wall, while the docstring says 3×;
  - `create_from_input` passing no seed.

  **Recommendation:** reproduce, then fix one at a time in P2, each with its golden change.
  - → answer: yes to recommendation
- **Q9 — organ scope and order.** Roots first (monocot, dicot, series, CellSet), since the seedling and MECHA need them; then stems, leaves and needles? Or all organs in P1?
  - **Recommendation:** P1 adapts every organ, since the adapter is generic and it makes the comparison folder complete. P2 and P3 go root first.
  - → answer: yes
- **Q10 — vessel ids and wiring.** Give `vessel_id` to every xylem and phloem cell (sieve elements included), with `bundle_id` / `pole_id` from the bundle, slice and arch loops? And between successive sections, pair them:
  - by `track_id` where the series prescribes vessels;
  - otherwise by nearest (angle, radius) within the same pole;
  - or by a rule you specify (e.g. count conservation, one-to-many for fusions)?

  **Recommendation:** the first two, with one-to-many links where the series fuses vessels.
  - → answer: yes to recommendation
- **Q11 — classes and templates.** A class is (diameter, distance from tip, distance from junction). Two things to settle:
  - How does a class become GRANAP parameters (`OrganInputData`)? By scaling the stele and cortex radii with the diameter, and switching tissues (aerenchyma proportion, barrier maturation, secondary growth) with the distances, through a rule you give?
  - What are the bin widths?

  **Recommendation:** a user-supplied function `class → OrganInputData` (a scenario entry), with a default for roots that scales the radii with the diameter and leaves the rest to presets. Bins are a parameter of the component.
  - → answer: yes, defaults for roots in the component
- **Q12 — growth and replacement.** Is the dynamic part in scope for the first capability tests: new segments getting their class's anatomy at emergence, and a segment changing class being replaced? Or only the initiation of a static plant?
  - **Recommendation:** initiation first (P3), then emergence and replacement with metafspm's partial rewiring.
  - → answer: yes to recommendation
- **Q13 — seedling scale.** A GRANAP root section is about 1.4–5 k cells, hence about 5–30 k Compartments with walls and junctions. A 48-segment seedling would hold about 0.3–1.4 M Compartments, far more than the MPG has held so far. What size should the capability test use?
  - **Recommendation:** a few segments in two or three classes for the test, and a separate benchmark for a whole seedling. Its timings would tell where metafspm's vertex creation needs work.
  - → answer: yes to recommendation
- **Q14 — your precise guidelines.** You mentioned in `devplan_porting.md` having precise guidelines for the GRANAP port. Do they change any of the above?
  - → answer: no, all was said in my previous prompt, the rest will be precised by questions if necessary.

## 4. Done

### P0 (2026-10-12, in metafspm: `devplan_anatomy_scales.md`, commit 99e1db7)

- Anatomy mode supports SubOrgan → Layer → Cell → Compartment: protoplasts under their Cell, walls and junctions anchored to the SubOrgan. Layer and Cell are locations; walls are skipped by aggregations and get defaults from broadcasts. Bulk creation takes one complex per component (0.17 s for a 45.7 k-vertex section).

### Environment (2026-10-08 to 10-12)

- `conda/environment.yaml`: openalea3 channel, `nodefaults` (the `defaults` channels of `~/miniconda3/.condarc` froze the solve), opencv 4.13 headless from pip (conda's builds need icu 78, plantgl's Qt 5 stack icu 75), essential pins only, flexible priority. It solves in about 5 min.
- GRANAP's suite passes in it on Python 3.13: 166 passed, so the goldens hold.

### P1, the exact adapter (2026-10-12)

- **`src/openalea/granap/mpg/`:**
  - `build.py`:
    - `single_plant_mpg()`, with one vertex per scale down to one SubOrgan;
    - `write_cells`: Layers, one per connected region of one tissue type (Q4 b, found with an STRtree); Cells holding every attribute of their GRANAP Cell, polygon included; one protoplast Compartment per Cell;
    - `write_network`: walls and junctions as Compartments anchored to the segment, each with its MECHA index and node attributes; virtual bridge cells as Cells in a Layer of their own; every edge as a Connection with its attributes and path label (membrane → Transmembrane, plasmodesmata → Symplastic, wall / air links → Apoplastic).
  - `views.py`: the Cells, the GeoDataFrame and the networkx network rebuilt from the MPG.
  - `organs.py`: every organ class backed by the MPG (a mixin), the factories `RootAnatomy`, `StemAnatomy`, `LeafAnatomy`, and `create_from_input`. After `generate_cells()` and after the network build, `all_cells`, the layers' cells, the GeoDataFrame, the graph, its counts and its wall map are read back from `organ.mpg`.
- **`test/mpg_comparison/`:**
  - 20 replay modules: each original test file's source is executed with its organ classes replaced by the MPG-backed ones, so no test is copied, and the library's own organ constructions are switched too (`root_series`, `create_from_input`).
  - `test_cross_check.py`: the eight golden organs, cell by cell (every attribute, polygons equal) and GeoDataFrame; the network of the monocot root and the needle (bridges), node by node and edge by edge, plus counts, wall map and adjacency matrix.
  - The `slow` marker covers six heavy replay files and seven cross-checks.
- **Results:**
  - 174 passed: 166 replayed and 8 cross-checks, in 33 min (`pytest test/mpg_comparison`, from `test/`).
  - The quick subset (`-m "not slow"`) is 131 tests in about 10 min.
  - On the monocot root, writing and reading the MPG adds about 0.8 s to a 10 s generation; the section is 30.5 k vertices.
- **As planned (Q6), still on the originals:** the CellSet donor anatomies (an intermediate whose cells are grafted into the MPG-backed CellSet organ). The pre-cell construction objects (recipes, tissue polygons) are inherited unchanged until P2.
- **Packaging, open:** `openalea.granap.mpg` imports metafspm, which is not a GRANAP dependency in `pyproject.toml` (and the conda recipe still pins conda's opencv). Should metafspm become an optional extra (`granap[mpg]`), or a hard dependency? **Recommendation:** an optional extra while P1–P3 are in progress, so the package and its CI stay installable without it.
  - → answer: metafspm will definitly become the main dependency for this version, so yes should be added to the requirements, but last version is not packaged yet so for now comment it in pyproject.toml and just mention manual clone and pip install . in the readme.

- **Packaging, done (your answer):** `openalea.metafspm` is listed, commented, in `pyproject.toml`; the README explains the install from source. The clone takes metafspm's `data_structure_api` branch, which holds the anatomy scales until it is merged. P0 (`99e1db7`) is not pushed yet.

## 5. P2: the pipeline writes into the MPG, stage by stage (your decision, 2026-10-12: option b)

The MPG becomes the record of every stage once cells exist. The GRANAP `Cell` objects remain the working copy that the algorithms edit, until P5 makes the MPG the storage itself.

- **Where the code goes.** `mpg/pipeline.py` reimplements `Organ.generate_cells` in the mixin as explicit stages: the same calls, in the same order, with the same rng stream, so that the goldens stay exact. The original modules are untouched.
- **Stage boundaries** (Q-P2a). From the first cells on, each stage's output is written to the section's MPG:
  1. Voronoi grouping (`process_voronoi_groups`, `simplify_cells`, `remove_nested_cells`): the first cells;
  2. the intercellular spaces;
  3. the aerenchyma;
  4. `fuse_gaps`;
  5. the layer population and the cell properties (the final cells).

  Before the cells exist (layer rings, seeds, recipes), the objects stay as they are; they are not cells, at about 10–25 seeds per cell.
- **Writing a stage.** The Cells keep their vertex when they survive a stage. Removed cells (aerenchyma fusions, nested cells, fused gaps) are removed from the MPG. New ones (air spaces) are added. The Layers are recomputed at the end, from the final types.
  - Each Cell keeps a stable id across the stages (`cell_uid`, set at grouping), which GRANAP's `id_cell` is not, since it is reset to the list index.
- **The topology, stored once** (Q-P2b). After the final cells, the walls and junctions are computed once (`_build_topology` and the exporter's wall walk) and written as Compartments and Connections at `generate_cells` time, not only at export. The exporter and the XML writer then read them from the MPG instead of recomputing them.
  - `simplify_cells` keeps its own topology inside the stages until P5.
- **Order, root first:**
  1. P2.1: the staged pipeline and the stage records, for monocot and dicot roots and the series;
  2. P2.2: the stored topology;
  3. P2.3: the network exporter and the writers (XML, OBJ, SVG, geo) read the MPG;
  4. P2.4: stems, leaves, needles, CellSet and ROI organs.
- **Tests:**
  - the comparison folder after each step;
  - plus per-stage cross-checks: the census and the cells of each stage, original against staged, through stage hooks on the original pipeline in the test only.

## 6. Later phases

- **P3 — new capabilities** (as in §2): vessel ids, wiring of successive sections, the library per class, the seedling, the hydraulic GraphView.
- **P4 — optimisation** (as in §2).
- **P5 — the MPG as the storage** (your request): the pipeline's algorithms read and write MPG arrays directly.
  - The `Cell` objects become views over the Cell vertices' properties, or disappear behind the compatibility view.
  - The topology is maintained in the MPG through the stages (`simplify_cells`, the aerenchyma fusions), instead of recomputed from the polygons.
  - Vectorised kernels replace the per-cell loops, where shapely allows.
  - The comparison folder still guards it. Exactness is kept, or each golden change is made explicit.

## 7. Questions on P2

- **Q-P2a — stage boundaries.** Write the MPG at the five boundaries above (grouping, intercellular spaces, aerenchyma, fused gaps, final)? Or only at grouping and at the end?
  - **Recommendation:** all five. Each is a point where the comparison can say which stage drifted, and the aerenchyma and gap fusions are where cells disappear.
  - → answer: yes to recommendation
- **Q-P2b — topology at `generate_cells`.** Compute and store the walls and junctions when the cells are generated, rather than only when the network is exported? It costs one `_build_topology` and one wall walk per section (about 1–2 s on a root section). In exchange, every section in an MPG carries its hydraulic graph, and the exporter and writers stop recomputing it.
  - **Recommendation:** yes. The hydraulic graph is the point of the port, and a population of sections would otherwise recompute it per export.
  - → answer: yes, if we need optimisation we'll consider it later.
- **Q-P2c — a stable cell id.** Add `cell_uid`, set once when a cell is grouped and kept through every stage, alongside GRANAP's `id_cell` and `id_group` (both unchanged)?
  - **Recommendation:** yes. It is what lets a stage write update the same vertex instead of rebuilding the section.
  - → answer: yes to recommendation


## 8. P2 progress

- **Your answers (2026-10-12):** Q-P2a, Q-P2b and Q-P2c, yes to the recommendations ("if we need optimisation we'll consider it later" for Q-P2b).
- **P2.1, done:** stage writes at grouping, intercellular, aerenchyma, fused_gaps and final.
  - **How:** instead of a copy of `generate_cells`, the mixin overrides the methods GRANAP's pipeline calls at those boundaries (`add_intercellular_spaces`, `_add_aerenchyma_to_proportion`, `fuse_gaps`). The calls, their order and the rng stream are GRANAP's own, so the stages are exact by construction, for every organ.
  - **Writing a stage:** `write_cells(…, written, stage)` keeps each surviving cell's vertex (by `cell_uid`, carried on the cells during the pipeline and stored as a vertex property, never on the rebuilt Cells). Gone cells are removed with their protoplast and properties, new cells are added, and Layers are rebuilt per stage, with Cells moved into them (metafspm `MPG.move_components`, `f606241`).
  - **Tests:** `test/mpg_comparison/test_stages.py` checks that at every stage the MPG equals the live cells; on an aerenchymatous root (proportion 0.2, where cells are fused away), that surviving cells keep their vertex; and that the final cells equal the original pipeline's. Full folder: **176 passed**.
- **P2.2, done:** the network is stored when the cells are generated.
  - GRANAP's exporter runs once into a scratch network, without bridges, and writes into the MPG.
  - `export_to_adjencymatrix` reads the stored network for a default export with no bridge to add. Otherwise (needle bridges, a custom air-link radius or wall thickness) it reruns the exporter and replaces the stored network.
  - Found: replacing it took 301 s, all in per-key deletions of array-backed properties. metafspm now deletes in bulk (`e0656f1`): 3.6 s.
  - Single checks pass: the monocot and needle cross-checks, and the replayed transfusion network (10).
- **P2.3, done:** the topology is stored once and read by GRANAP's own exporter and XML writer.
  - **Why:** the XML writer and the exporter number walls differently when cells are protected: the XML keys a wall by its junction pair, the exporter adds a shape signature. So the XML cannot be derived from the network's walls.
  - **What is stored:** the topology itself, the snapped ring of each cell on its protoplast (`boundary`). `mpg/topology.py` rebuilds GRANAP's five-tuple from it (phases 1–2 replayed, down to dict and set order), and `stored_topology(organ)` answers `CellGenerator._build_topology` from the MPG for this section's cells. It serves the stored network's export, the bridge/custom reruns and `organ.write_to_xml`.
  - **Result:** a section is snapped once at generation; the XML and the export read the MPG, with no new snap.
  - **The other writers:** OBJ, SVG, gmsh and `Geometry.xml` use only the cell polygons and types, which come from the MPG since P1.
  - **A direct `AnatomyWriter(organ).write_to_xml`** still snaps, exact on the MPG's polygons.
  - **Tests:** the cross-check compares the XML byte for byte (monocot root; needle, with protected air spaces): both pass.
- **P2, closed:** the full comparison folder passes, **176 tests, in 16 min** (36 min before P2.3: the export and the XML read the stored topology instead of snapping again).
  - **P2.4** needed no code of its own: the stage hooks and the stored topology are generic. Stems, leaves and needles go through them in the cross-checks and replays; the CellSet and ROI organs, with their own `generate_cells`, record the final stage and store their network.

## 9. P3: new capabilities — design (2026-10-12)

Your answers in §3: Q10 (vessel ids, matching), Q11 (classes, defaults for roots in the component), Q12 (initiation first), Q13 (a few segments in 2–3 classes; a separate benchmark).

- **The component.** `mpg/component.py`, `GranapAnatomy(StructuralComponent)`, for root segments. Its `initiate_plant(g, plant, parameters)` runs after the architecture's initiator, so the segments and their diameter and distances exist.
  - **Class:** each root segment's class is binned from its `radius`, distance from tip and distance from the root–shoot junction. The bins are component parameters.
  - **Library:** each class gets its anatomy from the library. A class's template is generated once, from `class → OrganInputData` (a function the user can give; a default for roots), and then instantiated on every segment of the class.
- **Templates.** A template is the section of one MPG-backed organ, kept as arrays: its Layers, Cells, protoplasts, walls, junctions and Connections, with their properties.
  - **Instantiation:** a segment gets its own copy, one bulk call per scale and Connection endpoints offset. Its Cells hold `template` and `template_row`, and share the template's polygons instead of copying them, so the geometry is stored once per class (your Q-A4 answer).
- **Vessels.** Every xylem and phloem conducting cell (metaxylem, protoxylem, xylem, phloem, sieve element) gets a `vessel_id`, unique in its section, and a `pole_id`. `track_id` is kept where the series prescribed vessels.
- **Junctions between successive sections.** `wire_junctions` with a callable rule pairs the conducting cells of the same family (xylem with xylem, phloem with phloem):
  - by equal `track_id` where both have one;
  - else each child vessel with its nearest parent vessel in (angle, relative radius), and each parent vessel with its nearest child vessel, so that a fusion or split gives one-to-many links.
- **The hydraulic graph.** `MPGDataStructure(g, from_scale=SubOrgan, nodes="Compartment", wiring=…)` over the plant: every Compartment a node, and every Connection, junctions included, an edge. GraphView and incidence come from metafspm.
- **Tests,** in `test/mpg_capabilities/`:
  - a small root system (a main root and two laterals, 2–3 classes, built by a test initiator);
  - templates shared per class, with copies per segment;
  - vessel ids and junctions;
  - the GraphView's sizes;
  - one graph-system solve on it (a steady flow, checked against a linear solve);
  - a benchmark script for a full seedling (Q13).

## 10. Questions on P3

- **Q-P3a — where pole and bundle ids come from.** GRANAP keeps no grouping on the cells. Changing its bundle, slice and arch loops would change the original modules, which P2 keeps untouched. Should the ids be recovered after generation instead?
  - stems and leaves: the bundle envelope (`vascular_tissue_polygons["bundle"]`) containing the cell;
  - dicot root phloem: the phloem-pole polygon containing it;
  - root xylem, and monocot phloem: the angular sector of the `n_vascular_bundles` poles.

  **Recommendation:** recover them now. Seeding them in the loops can come with P5, when the pipeline itself moves into the MPG.
  - → answer: yes to recommendation
- **Q-P3b — matching without `track_id`.** Nearest vessel of the same family, in both directions, in (angle, radius / section radius)? Or also restricted to the same pole?
  - **Recommendation:** nearest in both directions, without the pole restriction. Successive sections of different classes may have different pole counts, which makes poles hard to match.
  - → answer: yes to recommendation
- **Q-P3c — the default `class → parameters` for roots.** A thinner root could get either:
  - (a) every length scaled by the diameter ratio: the same cells, smaller;
  - (b) the cell sizes kept, with fewer cortex layers (`cortex.n_layers`) and a thinner stele (`stele.thickness`), both scaled with the diameter.

  **Recommendation:** (b). Thinner roots have fewer cortical layers and a smaller stele with cells of similar size. The distances from the tip and the junction change nothing by default; a user function can use them (e.g. aerenchyma, barrier maturation).
  - → answer: yes to (b), but don't we have the availability to also provide stele size and cell layers dimensions an number currently?
- **Q-P3d — geometry in the copies.** Should a segment's Cells point to their template's polygons (`template`, `template_row`), as in the design above? Or should they hold their own copy, which would be needed if a segment's cells later deform on their own (growth in girth)?
  - **Recommendation:** point to the template now. A segment that deforms can get its own polygons later.
  - → answer: yes to recommendation

## 11. P3 progress (2026-10-12; your answers: Q-P3a, b, d recommendations, Q-P3c (b))

- **Q-P3c, your point:** yes, GRANAP already takes the stele size (`stele.thickness`) and each layer's cell dimensions and number (`cell_diameter`, `cell_width`, `n_layers`). The default rule sets exactly those from the class's diameter, and `overrides` lets a class give its own values.
- **Done, the initiation of a static plant:**
  - `mpg/vessels.py`: `vessel_id`, `pole_id` (bundle envelope, phloem-pole polygon, or angular sector), `family`, angle and relative radius per conducting cell, recovered after generation (Q-P3a).
  - `mpg/library.py`:
    - `ClassBins` (mm);
    - `default_root_parameters`: the cell sizes kept; the stele thickness and the xylem lengths scaled; the cortex layers filling the radius;
    - `Template`: a generated section as tables;
    - `instantiate`: a segment's own copy in bulk, its Cells pointing to the template's polygons and rings (Q-P3d);
    - `AnatomyLibrary`: one template per class, generated at first use.
  - `mpg/component.py`:
    - `GranapAnatomy(StructuralComponent).initiate_plant` gives each root segment its class's anatomy;
    - `junction_rules` pairs xylem with xylem (apoplastic) and phloem with phloem (symplastic): by `track_id`, else the nearest in both directions (Q-P3b).
  - **Tests, `test/mpg_capabilities/` (5, 10 s):** a main root (4 segments, 0.55 mm) and a lateral (2 segments, 0.3 mm), so 2 classes.
    - one template per class, and disjoint copies per segment;
    - vessel ids unique per section;
    - every conducting cell of each pair of successive sections joined (5 links, the lateral included);
    - the graph view holds every Compartment and Connection;
    - a steady flow solved by a metafspm graph system over the whole root system, equal to a direct sparse solve.
  - **Benchmark,** `test/mpg_capabilities/bench_seedling.py`: 48 segments in 3 classes, 886 k vertices (241 k nodes, 594 k edges, 2,231 junctions).
    - The 3 templates take 6.6 s; the architecture and its 48 copies, 2.6 s.
    - `convert_properties_to_arraydict` takes **14.6 s**, the largest cost, on the metafspm side.
    - The DataStructure with its junctions, 0.7 s; the graph view, 0.09 s.
- **Found, to decide:**
  - **F-P3a — vessels in thinner roots.** The default rule scales the vessel diameters with the stele, so a thinner root gets *more*, smaller vessels: 61 conducting cells at 0.6 mm against 25 at 1.1 mm. Keep the vessel sizes fixed instead (the stele and xylem radii still scale), so that thinner roots get fewer vessels? **Recommendation:** yes. The number of metaxylem poles stays the preset's unless a class overrides it.
    - → answer: For now, I want the same scaling rules as the version of GRANAP we're reimplementing from.
  - **F-P3b — `convert_properties_to_arraydict` at seedling scale (14.6 s).** Speed it up in metafspm, e.g. a bulk conversion from dicts to arrays per property, or create the properties as arrays during the bulk instantiation? **Recommendation:** convert in bulk (one `np.fromiter` per property) in metafspm, the change being local to the MPG.
    - → answer: yes to recommendation
- **F-P3b, done (metafspm `ArrayDict.from_items`):** the seedling's conversion takes 3.6 s instead of 14.6 s.
- **F-P3a, your answer: "the same scaling rules as the version of GRANAP we're reimplementing from".** GRANAP has no rule from a root's diameter to its parameters. Its only rules for parameters varying along a root are the root series' schedules (`RootSeries` / `DicotRootSeries`): the stele radius, the vessel radius, the vessel counts (`n_fused`, terminators, creators) and any field (`param_schedules`), as functions of the position along the root. Which do you mean?
  - (a) **The series' schedules:** a class is evaluated through a `RootSeries` at its distance from the tip (the series' `length` axis), with GRANAP's own vessel tracking and fusions, and the diameter only selects the series (e.g. one per root order).
  - (b) **No scaling at all:** a class gets the preset's parameters unchanged (every root the same anatomy), unless a user function changes them.
  - (c) **Another rule,** in the R GRANAR or elsewhere: tell me where.

  **Recommendation:** (a). It is GRANAP's own model of how a root's anatomy varies, and its `track_id`s then give the vessel junctions between successive sections directly.
  - → answer: rather (b) to have a more flexible logic. If diameter is supposed to change anatomy, that's a differentiation we will introduce in the parametrization of this given class.
- **F-P3a, done (your answer b):** `default_root_parameters` returns the preset unchanged, plus the class's overrides; the provisional diameter rule is removed. A class's own parametrization comes from `AnatomyLibrary(class_parameters={key: {param: {field: value}}})` or a callable key → overrides.
  - Test: the lateral class, given a 2-layer cortex, has fewer cells; the same class without parametrization gets the preset. 6 capability tests pass.
- **Next (Q12):** emergence (new segments getting their class's anatomy) and replacement (a segment changing class), with metafspm's partial rewiring.

## 12. Q12: emergence and replacement — design

- **The step.** `GranapAnatomy` gets an MPG-style structural step, scheduled after the architecture's growth step. At each call it bins every root segment's class from its current radius and distances; the class is cheap to compute.
  - A segment with no anatomy (just emerged) gets its class's copy.
  - A segment whose class changed gets its anatomy **replaced**: its Layers, Cells, protoplasts, walls, junctions and anatomy Connections are removed, and the new class's copy is instantiated (your earlier answer: replaced, not edited).
  - A pruned segment's anatomy goes with it.
  - metafspm's `update_topology` then rewires only the junctions of the segments whose anatomy or link changed (signatures, partial rewiring).
- **What a replaced anatomy keeps.** Your earlier answer: cell contents (e.g. solute concentrations) are inputs from coarser components such as root_carbon, broadcast to the new Compartments. So the new Compartments start from their variables' defaults, and the coarse inputs reach them through the translator as for every other Compartment. No state is carried from the old cells.
- **What an emerged anatomy inherits.** metafspm's `on_grow="inherit"` takes the first same-label Compartment upstream. For GRANAP's copies, every protoplast has the same label, so it would take an arbitrary cell. Proposed instead: the Compartment of the same template and the same row in the nearest upstream segment of the same class (the same cell of the same anatomy). Otherwise, the same tissue and the nearest relative position. Otherwise, the default.
- **Tests:**
  - a root system grown over a few steps: new segments get their anatomy, and junctions are wired only for them;
  - a segment whose radius crosses a bin is replaced: its old vertices and properties are gone, the new ones are there, and only its own and its child's junctions are rebuilt;
  - inheritance by template row;
  - a pruned segment.

## 13. Questions on Q12

- **Q-G1 — when classes are re-evaluated.** At every call of the step, for every segment (replacing those whose class changed)? Or only on an explicit differentiation event raised by another component?
  - **Recommendation:** every call. Binning is cheap, and the bins decide how often a class changes. With distances binned finely, every step would replace many segments; the class parametrization (bin widths) is the knob.
  - → answer: yes to recommendation
- **Q-G2 — a replaced anatomy's state.** The new Compartments start from the defaults, with coarse inputs broadcast, and nothing is carried from the old cells (as I understood your earlier answer)?
  - → answer: yes
- **Q-G3 — inheritance at emergence.** The same template row upstream, then the same tissue at the nearest relative position, then the default (a GRANAP-specific inheritance), instead of metafspm's first same-label Compartment?
  - **Recommendation:** yes. It is a callable inheritance rule passed by the component, so metafspm's default stays for other anatomies.
  - → answer: yes

## 14. Q12 done (2026-10-12; your answers: Q-G1, Q-G2, Q-G3 yes)

- **`GranapAnatomy`'s structural step** (MPG-style `@postsegmentation`) bins every root segment's class at each call (Q-G1).
  - A segment without anatomy gets its class's copy; a segment whose class changed has its anatomy replaced (`remove_anatomy`, then a new copy).
  - The records of pruned segments are dropped. `last_update` gives {"emerged", "replaced"}.
  - `update_anatomies` is the same logic for `initiate_plant`.
- **Inheritance (Q-G3):** `granap_inheritance`, set as the DataStructure's `inheritance` by the component. A new Compartment inherits from the same template node (`template`, `template_node` on every copied Compartment) in the nearest upstream section; else from the same tissue at the nearest relative position; else the default. Replaced sections inherit nothing and start from the defaults (Q-G2).
- **metafspm changes it needed:**
  - inheritance in bulk with a model's own rule (`MPGDataStructure.inheritance`, `df6a956`): once per update for every variable, where it used to search per Compartment and per variable;
  - pruning removes a segment's whole anatomy (`61a33bf`): before, the MTG refused to remove a vertex with Layers.
- **Tests (`test/mpg_capabilities/test_growth.py`, 3):**
  - a new tip segment gets its class's copy; only its link is rewired; its cells carry their upstream namesakes' values;
  - a lateral segment thickening into the main class is replaced: its old vertices are gone, the new ones are at their defaults, and its junctions are rebuilt;
  - a pruned tip takes its anatomy with it.

  Capability folder: 9 passed, about 2 min.
- **What drives the topology update:** the anatomy step triggers the DataStructure's update only when it changes the MPG itself. A pruning done outside a step (as in the test) needs `ds.update_topology()`, which an architecture's MPG-style pruning step gets from the framework.
