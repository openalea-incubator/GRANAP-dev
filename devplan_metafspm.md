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

## 15. P4: optimisation — measurements and design (2026-10-12)

- **Example (your request):** `example/mpg/root_system.py` draws a root system's sections from the MPG: the main root and its lateral, cells by tissue, and the xylem and phloem junctions between successive sections.
- **Baseline** (seed 0, original = cells and network; MPG-backed = cells, stages, stored topology and network):

  | organ | cells | original | MPG-backed |
  |---|---|---|---|
  | monocot root | 1384 | 7.3 s | 7.7 s |
  | dicot root | 2834 | 16.0 s | 16.7 s |
  | dicot stem | 4800 | 5.5 s | 5.9 s |

- **Where the time goes** (dicot root, profiled, 32.5 s under the profiler):
  1. aerenchyma calibration: 12.1 s in 9 attempts, each one a deep copy of every cell (4.4 s in all), a merge (6.8 s) and a `simplify_cells`;
  2. `simplify_cells` / `_build_topology`: 9.9 s / 8.4 s over 11–12 calls, pure-Python dict and set loops, the topology recomputed at every simplification;
  3. the network export, run once: 4.8 s, per-row pandas access and list membership;
  4. seeding: 3.2 s; intercellular spaces: 2.8 s.
- **Two kinds of levers:**
  - **Exact** (outputs identical, bit for bit):
    - E1, `_build_topology` vectorised (numpy for phases 1–2, grouping instead of dicts of sets);
    - E2, the aerenchyma attempts copying only what an attempt changes (copy on write) instead of every cell;
    - E3, the exporter's per-row loops replaced by array operations.

    The comparison folder must stay green, unchanged.
  - **Approximate** (anatomy changed, opt-in, own goldens), from `doc/performance_proposals.md`:
    - A1, fewer Voronoi seeds per cell (8/6 instead of 15/10, about −28 % by that document);
    - A2, a looser inscribed-circle tolerance in the packing (1e-3);
    - A3, `coverage_union_all` for the grouping unions;
    - A4, a coarser aerenchyma calibration (fewer bisection steps).

## 16. Questions on P4

- **Q-P4a — which kinds.** Exact levers first, on by default for the MPG-backed organs, then the approximate ones as an opt-in fast profile?
  - **Recommendation:** yes. The exact levers cost no validation, and the fast profile gets its own goldens and a visual check (the side-by-side example, exact against fast).
  - → answer: Do only exact optimisations yes.
- **Q-P4b — where the exact levers live.** In `mpg/` (used during the MPG-backed organs' generation, as `stored_topology` is), with GRANAP's modules untouched? Or in GRANAP's own modules, so the original classes benefit too?
  - **Recommendation:** in `mpg/` now, consistent with P2. Once validated, they can move upstream in one step, since they are exact.
  - → answer: yes to recommendation
- **Q-P4c — the fast profile's contents.** All of A1–A4 under one opt-in profile (`fast=True`), or separate flags?
  - **Recommendation:** separate flags, grouped by a `fast` profile that turns them all on, so that each one's effect on the anatomy can be measured and pictured on its own.
  - → answer: yes to recommendation, how will it be turned on from the component's usage?

## 17. Stems and leaves (2026-10-12, your request: a seedling example with stems and leaves)

- **`GranapAnatomy.shoot`** maps a SubOrgan label to an `OrganAnatomy` (a library and the size property of its classes). Stems (`StemElement`) are classed by radius; leaves (`LeafElement`, size `None`) form one class. Root segments keep `library`.
  - Libraries take any preset (`preset_parameters("for_monocot_stem")`, `organ_class=mpg.StemAnatomy`), and template ids are unique across libraries (`find_template`).
  - Junctions join stems to stems, stems to leaves and stems to roots, by the same vessel rule.
- **`example/mpg/seedling.py`:** a stem of three elements, two leaves of two elements, and roots (a main root and a lateral), drawn at their real sizes from the MPG with their vessel junctions. 12 sections from 4 templates, 212 k Compartments, 526 k Connections, 1,555 junctions, built in 24 s.
- **Test `test/mpg_capabilities/test_shoot.py`:** each kind from its own library, and the stem–stem, stem–leaf and stem–root junctions. Capability folder: 10.

## 18. P4 (2026-10-12; your answers: exact optimisations only, in mpg/; Q-P4c moot)

- **Q-P4c, your question (how it is turned on from the component):** the exact speed-ups need no switch, since the outputs are identical. They are on for every MPG-backed organ, so also for the component's templates. `MPGBacked.exact_speedups = True` is a class flag, so the gain can be measured (False: the plain pipeline) and the comparison folder can run both ways.
- **Real times (no profiler), dicot root:** grouping 2.25 s, intercellular spaces 0.86 s, aerenchyma 2.09 s (topology 0.9 s, deep copies 0.42 s), fused gaps and final 0.2 s, network storage 1.11 s. The profiler had inflated the pure-Python parts several times over.
- **E1, done (`mpg/speedups.py`):** a topology cache, keyed by the polygons' WKB bytes, the cell ids and the protected ids. A repeated call on identical input returns the first call's result. 8 of the 11 calls are hits (the aerenchyma attempts simplify the same cells).
  - Generation: dicot root 6.49 → 5.71 s (−12 %); monocot root 3.02 → 2.59 s (−14 %).
  - The dicot cross-check and the stage tests pass.
- **E3 and E2, not done:**
  - **E3:** about 1 s per section goes to pandas row access inside GRANAP's exporter (`cells_gdf.loc[r]` per wall cell, `iterrows`). Speeding it up exactly means rewriting the exporter in a copy in `mpg/` (about 900 lines, bridges and air links included), kept in step with GRANAP's.
  - **E2:** the aerenchyma deep copies (0.42 s) would need a copy of the calibration loop. Since P3, both costs are paid once per template, not per segment.
  - **Q-P4d:** stop P4 at E1 and carry E2 and E3 into P5, where the pipeline moves into MPG arrays anyway? Or copy the exporter and the calibration now?
    - **Recommendation:** stop at E1. The copies would be rewritten again in P5, and their cost no longer grows with the plant.
    - → answer: stop at E1 (given in the conversation, 2026-10-12). E2 and E3 go to P5.

## 19. P4 closed

- E1's full check: the comparison folder passes, **176 tests, in 15 min** (16 before E1). P4 stops at E1 (your answer); E2 (aerenchyma deep copies) and E3 (the exporter's per-row access) move into P5.

## 20. P5: the MPG as the storage — plan (2026-10-12)

**Goal (§6):** the pipeline's algorithms read and write the MPG directly. Until now, GRANAP's `Cell` objects are the working copy, written into the MPG at stage boundaries (P2) and read back. In P5 the MPG holds the cells *during* the stages; the topology is kept in the MPG instead of recomputed from the polygons; the per-cell loops become array operations where shapely allows.

**Where the code goes.** Your original request: progressively convert the sources in the separate folder. P5 is that conversion: converted copies of GRANAP's modules under `mpg/pipeline/`, which the MPG-backed classes switch to one at a time. GRANAP's own modules stay untouched until every one has a converted copy; whether to retire them then is decided at that point. The comparison folder guards every step.

**What stays outside the MPG.** The stages before cells exist (layer rings, seeds, recipes) produce seeds, 10–25 per future cell, and polygons, not cells. They stay as arrays and polygons (numpy, shapely), not MPG vertices. The MPG starts at the grouping, as in P2.

**Steps, root first, each with its check:**

- **P5.1 — Cells as views of the MPG.**
  - A `CellView` reads and writes its attributes in its Cell vertex's properties: numbers in the arrays, the polygon as an object property.
  - The organ's `all_cells` becomes a manager of views in cell order, backed by the section's Cell vertices.
  - The stages after the grouping (intercellular spaces, aerenchyma, fused gaps, the final properties) then edit the MPG itself. P2's stage writes become a check that the MPG and the views agree, then disappear.
  - Exact by construction: the same values, through another store.
- **P5.2 — the exporter on the stored topology, on arrays (E3).** The network is built with array operations from the stored rings, walls and junctions instead of pandas row access. The bridges and air links follow. Exact: the same graph, node for node and edge for edge, and the XML byte for byte.
- **P5.3 — the aerenchyma calibration, copy on write (E2).** An attempt records the cells it changes, and only those are restored, instead of a deep copy of every cell per attempt. Exact.
- **P5.4 — the topology kept through the stages.** `simplify_cells`, the aerenchyma fusions and `fuse_gaps` update the stored rings, walls and junctions where cells change, instead of snapping the whole section again. Maintained and recomputed topologies may differ at snapping ties, so this step may change goldens (Q-P5c).
- **P5.5 — array kernels.** The per-cell loops that shapely's vectorised functions allow: the seed rings, the grouping unions, the cell properties, the intercellular buffers. Each is exact, or carries an explicit golden change.
- **P5.6 — the other organs:** stems, leaves, needles, CellSet and ROI.

**The audit's bugs (Q8, your answer: reproduce, then fix one at a time in P2, each with its golden change). Not done in P2.** The five are:
- `cgroup` dropped by `process_voronoi_groups`;
- `id_layer=i_slice` on monocot metaxylem;
- multipart polygons reduced to their largest part;
- the `air_link_radius` default (1× the median wall, documented as 3×);
- `create_from_input` passing no seed.

## 21. Questions on P5

- **Q-P5a — views or arrays.** Make the pipeline's cells views of the MPG first (P5.1), with the algorithms unchanged, and turn the algorithms into array operations afterwards (P5.5)? Or write array versions of the algorithms directly?
  - **Recommendation:** views first. Every algorithm then works on the MPG at once and exactly; the array versions replace them one at a time, each checked.
  - → answer: yes to recommendation
- **Q-P5b — converted copies.** Converted copies of GRANAP's modules under `mpg/pipeline/`, switched to one at a time, with the originals untouched until all are converted?
  - **Recommendation:** yes.
  - → answer: yes
- **Q-P5c — exactness.** Must every P5 step reproduce the goldens exactly? Or may P5.4 (maintained topology) and P5.5 change them, each change shown (census, the side-by-side example) and accepted by you?
  - **Recommendation:** exact for P5.1–P5.3. Explicit, reviewed golden changes allowed in P5.4–P5.5, where a maintained topology or a vectorised union may break a snapping tie differently.
  - → answer: yes to recommendation
- **Q-P5d — the Q8 bugs.** Fix them now, before P5 (each with its golden change, in GRANAP's modules and the converted copies alike)? Or in the converted copies only, as P5 reaches each module?
  - **Recommendation:** in the converted copies, as P5 reaches each module. Fixing them in GRANAP's modules changes the original goldens too, which you may want to do in GRANAP's own history rather than in this port.
  - → answer: in converted copies
- **Q-P5e — order of P5.1 and P5.3 (found while starting P5.1).** GRANAP's post-grouping stages replace cells wholesale. The aerenchyma calibration restores a deep copy of every cell before each of its up to 21 attempts (`organ_class.py:680`), and the air spaces and fusions create new `Cell` objects. With views and unchanged algorithms (Q-P5a), each replacement rewrites every cell into the MPG: exact, but about 15 % slower until P5.3, and the MPG would not hold the cells through the calibration. Options:
  - (a) P5.1 as planned, a full rewrite at each replacement;
  - (b) P5.1 and P5.3 together, attempts editing the MPG and rolling back to a section snapshot (rollback re-creating fused cells);
  - (c) P5.3 first, in a converted copy on plain cells: shallow copies instead of deep copies, exact since cell attributes are numbers or immutable shapely polygons. Then P5.1, where only the stages creating cells (air spaces, fusions) write into the MPG.

  **Recommendation:** (c).
  - → answer: yes, (c)

## 22. P5.3 done (2026-10-12; your answers: Q-P5a-d recommendations, Q-P5e c)

- **`mpg/pipeline/aerenchyma.py`:** GRANAP's `Organ._add_aerenchyma_to_proportion`, the snapshot and each attempt's restore made of shallow copies instead of deep copies. Exact: a Cell's attributes are numbers, strings, None or immutable shapely geometries, and the attempts set attributes rather than mutating them.
  - Used by the MPG-backed organs whose calibration is `Organ`'s; organs with their own (leaves) keep theirs.
  - `MPGBacked.converted_pipeline = True` (False: GRANAP's modules), like `exact_speedups`.
- **Times:** dicot root 5.81 → 5.60 s (−4 %); monocot root 2.77 → 2.53 s (−9 %). The dicot cross-check and the stage tests pass.
- **Next, P5.1:** cells as views of the MPG. The stages that create cells (air spaces, fusions, the calibration's attempts on copies) write them into the MPG.

## 23. P5.1 done (2026-10-13)

- **`mpg/cells.py`:** from the grouping on, the organ's cells are `CellView`s. A view is a `Cell` keeping nothing itself: its attributes are read from and written to its Cell vertex's properties. GRANAP's algorithms run on them unchanged, so `cell.type = …` edits the MPG.
  - **`MPGCells`** (a `CellManager`) and **`ViewList`** adopt the plain Cells GRANAP brings (an air space appended, fusions extended, the calibration's copies assigned). A cell of the section (same uid) updates only the attributes that differ; a new cell gets a vertex and a protoplast in a provisional Layer, dissolved by the next stage's regrouping.
  - Copying a view gives a detached plain Cell (same values and uid). The stage writes no longer rewrite the views' attributes; they only recompute Layers, rows and removals.
  - At the end the organ gets plain Cells back, rebuilt from the MPG, as since P1.
- **Time:** none added. Converted against GRANAP's modules: dicot root 6.12 against 6.40 s, monocot 2.80 against 3.02 s (absolute times vary from run to run on this machine).
- **Tests:** the stage test now also checks that the cells are views from the intercellular stage on. The stage tests, the monocot cross-check and the capability folder (10) pass; the full comparison folder runs at the end of the step.

## 24. P5.2 done (2026-10-13)

- **`mpg/pipeline/network.py`:** GRANAP's `NetworkExporter`, copied verbatim, with two edits. The cells' rows are built once (`_Rows`, the frame's columns as numpy arrays, values of the types pandas gives) and replace `cells_gdf.loc[r]` (one per wall cell) and `iterrows()`. The topology comes from the MPG (`stored_topology`). The MPG-backed organs use it for the stored network and the reruns (bridges, custom settings) when `converted_pipeline`.
- **Times** (machine loaded by a concurrent run, converted against GRANAP's modules): dicot root 7.02 against 8.02 s (−12 %), monocot root 3.26 against 3.92 s (−17 %).
- **Exactness:** the monocot and needle cross-checks (graph node for node and edge for edge, adjacency matrix, XML byte for byte; the needle with bridges) and the replayed transfusion network pass, 12 tests.

## 25. P5.1 checked; P5.4 and P5.5 reassessed (2026-10-13)

- **P5.1, full check:** the comparison folder passes, 176 tests. P5.2's full check is running.
- **What is left to gain** (dicot root, converted pipeline, machine loaded about 1.5×):

  | stage | time |
  |---|---|
  | grouping | 3.48 s |
  | intercellular spaces | 1.31 s |
  | aerenchyma | 1.30 s |
  | fused gaps | 0.20 s |
  | final | 0.11 s |
  | network stored | 0.86 s |
  | total | 7.25 s |

  Topology actually computed: 1.10 s in all (8 of 11 calls answered by E1). Most of it is the grouping's first snap, which nothing can reuse.
- **P5.4 (topology kept through the stages) — not worth it.** It could save at most the topology computed after the grouping, about 0.6 s, roughly 8 %. And it cannot be exact: GRANAP snaps globally (tolerance from the section's 5th-percentile edge, greedy clustering in vertex order), so an incremental update breaks ties differently and changes the goldens. That is a reviewed change for a small gain.
- **P5.5 (array kernels) — where the time is.** The grouping is half the time: seed rings, packing, the global Voronoi, a union per cell. Its cost is mostly GEOS and scipy, not Python. What can be vectorised exactly has to be measured first (a profile of the grouping alone).
- **The Q8 bugs (your answer: in the converted copies, as P5 reaches each module).** Two of their modules are converted now:
  - the `air_link_radius` default, in the converted exporter;
  - `create_from_input` passing no seed, in `mpg/organs.py`.

  The three others live in modules not converted (`process_voronoi_groups`, the monocot root's metaxylem loop, `_build_topology`).

## 26. Questions (2026-10-13)

- **Q-P5f — P5.4.** Drop it, given about 8 % at most, not exact?
  - **Recommendation:** drop it. The topology stays snapped once per section (P2.3), and the aerenchyma attempts reuse it (E1).
  - → answer: yes drop it
- **Q-P5g — P5.5.** Profile the grouping stage and convert only what proves exact and worthwhile (each with its measured gain), or stop P5 here?
  - **Recommendation:** profile first, then decide with the numbers.
  - → answer: yes to recommendation 
- **Q-P5h — the two Q8 fixes possible now.**
  - **`air_link_radius`:** the default is 1× the median wall, the docstring says 3×. Which is intended? Changing the code to 3× changes the leaves' air links, so the goldens of networks with protected air (leaves).
  - **`create_from_input`:** pass the organ's seed? It would take a `seed=None` argument passed to the class, which adds an argument and changes no golden.
  - **Recommendation:** for `create_from_input`, add the seed. For `air_link_radius`, tell me which is intended (code or docstring), and I fix the other.
  - → answer: keep 1x and follow recommendation

## 27. Your answers Q-P5f-h (2026-10-13)

- **Q-P5f:** P5.4 is dropped.
- **Q-P5g:** profile the grouping before converting anything in P5.5.
- **Q-P5h, done in the converted copies:**
  - `air_link_radius` keeps 1× the median wall, and the comment that said 3× is corrected (`mpg/pipeline/network.py`);
  - `mpg.create_from_input(input_data, seed=None)` passes the seed to the organ (a test checks it is reproducible).
- **Q8 still open:** `cgroup` dropped by `process_voronoi_groups`, `id_layer=i_slice` on monocot metaxylem, multipart polygons in `_build_topology`. Their modules are not converted.

## 28. N1 and N2: the section's network and cells built in the MPG — design (2026-10-13, your go-ahead)

**Why.** The solve already runs on the MPG (the GraphView over Compartments and Connections). What still goes through GRANAP's intermediate objects is each template's generation: the cell GeoDataFrame built twice (in GRANAP's `generate_cells`, then again from the MPG), and the stored network built by the exporter into a networkx graph, copied into the MPG, and rebuilt as networkx from the MPG for the matrix. Measured on the dicot root (2834 cells, 41964 edges): snap 0.25 s, exporter 0.49 s, MPG write 0.45 s, networkx rebuilt for the matrix 0.22 s.

**N1 — the network built in the MPG (`mpg/network.py`).**
- **Input:** the section's Cells (`cell__polygon`, `cell__type`, `cell__protect_topology`, `cell__cgroup`) and the rings stored on the protoplasts (`topology_from_mpg`). No GeoDataFrame, no networkx.
- **The exporter's rules, as array operations:**
  - the wall walk: every ring cut at its junctions in one pass, walls keyed and numbered in first-encounter order (`np.unique`), the cells of each wall in encounter order;
  - lengths, midpoints, thicknesses, distances and `d_vec` vectorised, with the exporter's rounding: sums in the same order, numpy's `mean` reproduced for segments under 8 vertices (sequential sum), numpy's own `mean` above;
  - junctions sorted as tuples (`lexsort`), cell centroids and areas from shapely's vectorised functions;
  - membrane, wall_air, wall and plasmodesmata edges as columns; the air links (both rules) on adjacency lists built from the walls, in the iteration order of GRANAP's sets and of networkx's neighbours, so that the `via_*` attributes are the same.
- **Output:** written straight into the MPG in bulk: walls and junctions as Compartments anchored to the segment, Connections with their `edge__` attributes, the protoplasts' `node__` attributes and `mecha_index`, as today.
- **The Connections' order** becomes the exporter's order of creation (it was networkx's iteration order). The graph is the same, node for node and edge for edge; only the Connections' numbering in the MPG differs.
- **Kept on the converted exporter (P5.2):** exports with other settings (bridges for needles, `air_link_radius`, wall thicknesses). They are rare, and run once per organ.
- **Views:** `organ.graph` is built from the MPG on first access (`network_from_mpg`). `export_to_adjencymatrix` reads the Connections' endpoints directly (COO, then LIL), without networkx.
- **Check:** the cross-checks (graph, matrix, XML) against GRANAP's originals, and a test comparing N1 with the converted exporter on every organ of the cross-check.

**N2 — `generate_cells` converted (`mpg/pipeline/generate.py`).**
- GRANAP's `Organ.generate_cells`, copied: the same calls in the same order, the stage writes called where the stages end (instead of the method overrides), and no GeoDataFrame built at the end.
- `_cells_gdf` becomes a view: built from the MPG's cells when first read, cached, and dropped when the cells change (stage writes, bridge cells, cells read back). GRANAP's code that reads it (XML writer, statistics, plots) or edits it in place (`retag_cells`) works unchanged.
- For organs whose `generate_cells` is `Organ`'s; the CellSet and ROI organs keep theirs (P5.6).
- **Check:** the comparison folder (176 tests).

## 29. N1 and N2 done (2026-10-13)

- **N1, `mpg/network.py`:** the stored network built from the MPG's cells and rings, as array operations, and written straight into Compartments and Connections. It is exact: the same graph as the exporter, node for node, edge for edge and every attribute (`via_*` of the air links included), on all 8 cross-check organs and both leaves.
  - **Exactness checks:**
    - numpy's `mean` reproduced bit for bit: its pairwise sum is emulated up to 128 values and checked against `np.mean` for every length;
    - `np.hypot` on arrays equals the scalar call.
  - **`organ.graph`** is a view, built from the MPG at its first read. **`export_to_adjencymatrix`** reads the Connections (COO, then LIL), without networkx.
  - **The converted exporter (P5.2)** remains for exports with other settings (needle bridges, `air_link_radius`, wall thicknesses).
  - **Flag:** `MPGBacked.native_network` (False: the exporter, copied into the MPG).
  - **Test:** `test_native_network.py` compares N1 with the exporter on 10 organs, plus the matrix.
- **N2, `mpg/pipeline/generate.py`:** GRANAP's `Organ.generate_cells`, copied.
  - **The stages are explicit** ("grouping", "fused_gaps") instead of method overrides. "intercellular" and "aerenchyma" stay on the calibration, which the organs' own methods call.
  - **No GeoDataFrame during generation:** `_cells_gdf` is a view, built from the cells when first read. `generate_section()` generates without building it (the library's templates, the XML writer, the matrix). The topology rows come from the cells.
- **metafspm, `ArrayDict.update` in bulk:**
  - new keys skip the per-key membership test;
  - the index of appended keys is built in one call;
  - the merge is one stable sort.

  A section's MPG writes go from 0.25 s to 0.07 s. The suite passes (data structures, 266).
- **Times** (generation plus matrix, median of 5 interleaved runs):

  | organ | before | after | change |
  |---|---|---|---|
  | dicot root | 5.71 s | 5.22 s | −9 % |
  | monocot root | 2.71 s | 2.42 s | −11 % |
  | dicot leaf | 3.98 s | 3.61 s | −9 % |

  - The stored-network stage goes from 0.65 to 0.24 s, the matrix from 0.21 to 0.02 s.
  - GeoDataFrames built: 0 (2 before).
- **Full check:** the comparison folder passes, 186 tests (176 plus the 10 of N1), 15 min. One test reads `inputs/` relative to `test/`, so the folder runs from there. The two examples run.

## 30. Overview: where a component-native logic would pay (2026-10-13, your request)

Dicot root, one template, with the stage times from GRANAP's log:

| stage | time |
|---|---|
| seeds | 0.64 s |
| vascular and organ tissues | 0.34 s |
| Voronoi | 0.17 s |
| grouping | 1.21 s |
| intercellular spaces and aerenchyma | 1.87 s |
| fused gaps | 0.15 s |
| network | 0.24 s |
| stage writes | 0.13 s |

Under the profiler, the shares within the stages are:
- the aerenchyma calibration: 9 attempts, each re-simplifying the whole section, about 28 %;
- the seeds: 68 k Polygons and 46 k Points created one by one, 16 %;
- the CellView attribute reads and adoptions, about 11 %;
- the GEOS unions of the grouping, 9 %.

**A. A population MPG: where the wall time is.** A population needs one template per anatomy class, and a seedling has tens of classes.
- **A1, templates generated in parallel.** Classes are independent and seeded deterministically, so a process pool can generate them. A `Template` is tables plus shapely geometries, which pickle once its back-reference to the organ is dropped. Generation time then divides by the number of cores. This fits your scene plan: parallel offline generation, then the simulation in one MPG and one process.
- **A2, a template cache on disk:** the library saved by (parameters, class key), and reloaded instead of regenerated. A rerun of a population then generates nothing.
- **A3, junctions and inheritance per pair of templates.** `vessel_pairs` and `granap_inheritance` run per pair of successive sections: Python loops, distance matrices, upstream chains through `linked_parent`. Yet their result depends only on the two templates. They could be computed once per (template, template) pair and applied in bulk through `template_node`. The cost then follows the number of distinct pairs, not the number of segments.
- **A4, classes and instantiation in bulk.**
  - `update_anatomies` loops over segments, and `segment_class` walks to the nearest junction for each one. The class keys can come from arrays (diameter, distance from the tip, and from the junction by an axis scan, DS14a in your scene plan).
  - Then one `instantiate` per template for all its new segments, and one removal for the replaced ones.

**B. Generation of one template, by measured weight:**
- **B1, the aerenchyma calibration on the MPG.** An attempt only changes which cells become aerenchyma, and the proportion is an area ratio. The target count can be found on the area arrays, and the geometry run once instead of once per attempt (up to 21). Whether this is exact depends on whether an attempt's geometry changes the areas it measures; to check before choosing.
- **B2, the post-grouping stages native** (intercellular spaces, fusions, gaps) on the MPG's arrays instead of through CellViews. This removes the view layer and its cost (about 11 %), and with it `cells.py` and the mixin's overrides.
- **B3, the seeds as arrays** (x, y, layer, group, type), with shapely's vectorised predicates instead of a Cell, Point and Polygon per seed. Likely exact (the same draws in the same order); to verify.
- **B4, the grouping unions as coverage unions:** Voronoi regions form a coverage, and GEOS's coverage union is faster than a general union. The vertex order of the result differs, so the goldens change (a reviewed change, Q-P5c).
- **B5, the snap** (`_build_topology`, 0.9 s of Python over 4 calls): vectorising it would have to keep GRANAP's greedy clustering order to stay exact. That is hard; a compiled loop is the fallback.

**C. Readability and the component design:**
- **C1, the network as DataStructure variables.** Today the attributes are `node__`/`edge__` properties holding tuples and small arrays (`position`, `d_vec`). As float variables registered with units (`x`, `y`, `dx`, `dy`, `length`, `wall_thickness`), the graph systems read them as they read any variable. networkx stays only as GRANAP's view.
- **C2, no process-wide patches.** `stored_topology` and `exact_speedups` replace `CellGenerator` methods for the whole process. That is hard to read and unsafe with threads. After N1, only the XML writer and the exports with bridges or other settings need them. Converting the XML writer (walls and junctions from the MPG) and adding bridges to N1 would retire both.
- **C3, one pipeline that reads as stages.** `generate.py` already lists the stages. As they turn native (B1, B2), the overrides retire, and the pipeline becomes a sequence of functions, each writing the section into the MPG.

**Recommended order:**
1. A1 and A2 (the most wall time for a population, and no exactness risk);
2. A3 and A4;
3. B1 with B2 (the largest stage, and the view layer goes);
4. C1 and C2;
5. B3;
6. B4 and B5 only with reviewed golden changes.

## 31. Questions (2026-10-13)

- **Q-N1 — next step.** Start with A1 and A2 (templates in parallel, cached on disk), or with B1 and B2 (the aerenchyma calibration and the post-grouping stages on the MPG)?
  - **Recommendation:** A1 and A2. They give the most for a population, are independent of the pipeline, and stay exact by construction.
  - → answer (2026-10-13): A1 and A2 are not the priority. New classes are computed progressively as they appear in a growing plant, so their cost is not paid at every time step. Otherwise the proposed order: A3, A4, then B1 with B2, C1, C2, B3.
- **Q-N2 — the cache's key.** Should the template cache be keyed on the full parameters (any change regenerates), and stored next to the run (e.g. `outputs/anatomy_cache/`) or in a user cache directory?
  - **Recommendation:** the full parameters plus GRANAP's and the geometry stack's versions, in a directory given to the library (default: none, i.e. no cache).
  - → answer: moot for now (A2 deferred).
- **Q-N3 — B1's exactness.** If the calibration found on area arrays differs from GRANAP's attempt loop in some cases, accept a reviewed golden change, or keep the attempt loop?
  - **Recommendation:** first measure on the cross-check organs whether it differs at all, then decide with the cases in hand.
  - → answer: the proposed order agreed; measure first (recommendation).

## 32. A3 re-scoped by measurement: a growth step's cost follows the change (2026-10-13)

**Measured.** The seedling has 48 segments and 1.46 M vertices; one growth step adds 9 segments. Under the profiler the step takes 5.1 s:

| part | time |
|---|---|
| instantiation of the 9 copies (2 M property values written) | 1.7 s |
| junction rewiring | 1.6 s |
| inheritance | 0.56 s |
| DataStructure update | the rest |

Most of it goes to passes over the whole MPG, made on every step to find what changed:
- the owner map of every Compartment, rebuilt 5 times (0.85 s);
- a Python tuple signature of every Compartment of every section (0.65 s);
- `wire_junctions` grouping the Compartments of all sections by label to rewire 9;
- `vessel_pairs` and `granap_inheritance`, per pair of sections, though their result depends only on the two templates.

**Changes, each exact (the same junctions, the same inherited values):**
- **metafspm:**
  - **M1:** `compartments_by_owner` cached on the MPG's structural signature (the one `topology_arrays` uses), and grouped with numpy;
  - **M2:** the anatomy signature as arrays (owner, vid, label, ordering), the changed sections found by comparing two tables;
  - **M3:** `_rewire_junctions`' alive set and stale junctions found with sorted arrays;
  - **M4:** `wire_junctions` groups only the sections it rewires and their parents, and creates the junction Connections in bulk.
- **GRANAP:**
  - **G1:** `vessel_pairs` computed once per (parent template, child template, family), in template-node space, then mapped to each pair's vids;
  - **G2:** `granap_inheritance` per (new template, upstream template), vectorised over a section's nodes.
- **Check:** a reference growth run before the change (junctions as (n_id_a, n_id_b, label), the inherited solute), compared after; the capability tests and metafspm's suite.
- **A4** follows: classes from arrays, one instantiation per template.
- **Done (2026-10-13).** Also found while measuring: replacing an anatomy (a class change) took **669 s** on the seedling.
  - **Cause:** metafspm's `remove_connections` removed vertex by vertex: one bulk property deletion per Cell's protoplast (20 k), and a list search in the Compartment anchor's list for each wall.
  - **Fix:** it now removes in one pass, with the same final state; the vertex-by-vertex path stays where `remove_vertex` would refuse a vertex.
  - **Results**, against a reference run of the old code (three growth steps; junctions, inherited values and rewired sections compared at each, identical):

    | step | before | after |
    |---|---|---|
    | 9 new segments | 5.3 s | 3.3 s |
    | class change | 669 s | 7.9 s (about 5 s of it generating the new class's template) |
    | pruning | 0.05 s | 0.04 s |

  - **Profiled emergence step:** 5.1 s → 4.3 s (rewiring 1.6 → 0.56 s, inheritance 0.56 → 0.27 s).
  - **What is left:** instantiation (2.2 s, writing about 2 M values through dicts), and the DataStructure's whole-MPG passes (owner scales 0.7 s, topology arrays 0.7 s).
  - **Tests:** metafspm 794 passed; GRANAP capabilities 11 passed (63 s → 44 s).

## 33. A4 (2026-10-13)

- **Instantiation as array appends:** a copy's values are appended to the array-backed properties in one copy (metafspm `ArrayDict.extend`, `MPG.set_values`), instead of a dict per property and `ArrayDict.update`. The same vids and values: identical to the growth reference.
  - **Emergence step** (9 segments): 3.3 s → 2.2 s;
  - **seedling build:** 27 s → 21 s;
  - **profiled step:** 4.3 s → 2.9 s;
  - **metafspm's suite:** 794 passed, 29 s → 17 s.
- **The classes from arrays: not done.**
  - Computing the class keys costs little (48 segments). What does cost is finding the segments (`components_at_scale`, 0.29 s profiled).
  - Reading them from arrays would change the order the new segments are processed in, so the vids they get, for little gain.
  - It is worth doing when the plants have thousands of segments, together with DS14a/b (the MPG's array mirrors in your scene plan).
- **One instantiation per template for all its new segments: not done either.** Since the writes are appends, it would save only per-call overhead, and it would also renumber the vids.
- **What remains of a growth step:** the DataStructure's whole-MPG passes:
  - owner scales and variable stores, 0.58 s;
  - topology arrays, 0.55 s;
  - rewiring, 0.45 s, mostly the linked parent of every segment.

  These belong with DS14b (array mirrors of parent, complex, scale).

## 34. B1 measured; B2 design (2026-10-13)

**B1, the aerenchyma calibration from areas: it cannot be exact (Q-N3, measured).**
- **Why:** the proportion an attempt realizes is measured after:
  - the air spaces are fused (`unary_union` per group);
  - the whole section is re-simplified (`simplify_cells`, global snapping);
  - the tissue is carved by the air (`difference`).

  GRANAP's docstring says so too ("the overshoot depends on the emergent geometry"). An area predictor would bisect through other values, and the goldens of aerenchymatous organs would change.
- **Done instead, exact:** the best attempt's cells are kept and restored at the end, instead of running that attempt again. Monocot root −3 %, dicot root −1 % (in the noise). The stage and aerenchyma tests pass.
- **What is left within an attempt:**
  - the fill order (`_connected_radial_order`) is the same for every attempt and could be computed once (about 0.15 s on the dicot root);
  - the attempt itself is mostly `simplify_cells` and the carving, which B2 makes cheaper.

**B2, the stages after the grouping on the MPG's arrays — design.**
- **Scope:** `Organ`'s stages after the grouping, converted copies in `mpg/pipeline/`, about 460 lines:
  - `_apply_intercellular` (84 lines) and `seat_air_spaces` (57);
  - `add_aerenchyma` (116) and `_connected_radial_order` (60);
  - `merge_intercellular_aerenchyma` (102);
  - `fuse_gaps` (87).

  The stems' and leaves' own versions (14 and 28 lines, they call these) follow.
- **How:**
  - each stage reads the section's columns once (polygons as a shapely array, type, id_layer, x, y, protect flags);
  - it computes with shapely's vectorised functions, which run the same GEOS operation per geometry, so the results are the same, and with numpy selections in GRANAP's order;
  - it writes the changed columns back into the Cell vertices.

  No CellView, no adoption. Where GRANAP's loop is sequential (the fill per quadrant, the union-find of the merge), the loop stays, over arrays.
- **What it retires:** `cells.py` (views, adoption), the overrides of the MPG-backed class, and the "plain cells back" step. The cells then exist only in the MPG during generation; GRANAP's `Cell` objects are built only for its API.
- **Order, each step checked by the comparison folder:**
  - **B2a:** `merge_intercellular_aerenchyma` (run in every calibration attempt);
  - **B2b:** `add_aerenchyma`, with its fill order computed once;
  - **B2c:** the intercellular spaces;
  - **B2d:** `fuse_gaps`;
  - then the stems' and leaves' versions.
- **Expected gain:** the view layer (about 11 % profiled), plus the per-cell carving and filling loops. To be measured after B2a before going on.

## 35. Questions (2026-10-13)

- **Q-B1 — the calibration's number of attempts.** Keep GRANAP's bisection (exact), or replace it with a secant search on the measured proportion (fewer attempts, about half; the goldens of aerenchymatous organs change, reviewed)?
  - **Recommendation:** keep the bisection for now, and revisit after B2 has made each attempt cheaper.
  - → answer: yes to recommendation
- **Q-B2 — B2 as designed (§34): converted stages working on the MPG's columns, in the order B2a–d, exact, checked step by step?**
  - **Recommendation:** yes, starting with B2a, with a measurement before going on.
  - → answer: yes to recommendation

## 36. B2a: the merge on the MPG's columns (2026-10-13; your answers Q-B1 and Q-B2: recommendations)

- **B2a done, `mpg/pipeline/merge.py`:**
  - `simplify_cells`: its polygons rebuilt and validated in bulk (`linearrings`, `polygons`, `is_valid`);
  - GRANAP's merge, copied, with that simplification and the carving in bulk (`intersects`, `difference`, `area`; one removal instead of one per cell);
  - polygons written straight into the MPG's column for the cells that are views.

  It is used for the grouping's simplification (`generate.py`) and, through the MPG-backed class, for organs whose merge is `Organ`'s. Exact: the stage tests and the cross-checks (monocot, dicot, needle) pass.
- **Measured** (dicot root, no profiler):
  - the merge, over 8 calibration attempts: 0.45 → 0.25 s (−45 %);
  - the grouping's simplification: unchanged (0.52 s; it is the section's first snap);
  - medians of 5 interleaved runs: monocot root 2.45 → 2.21 s (−10 %), dicot root −4 %, leaf unchanged (its own calibration).
- **What B2b–d can still give** (dicot root, 4.24 s in all):

  | part | time | exactly gainable |
  |---|---|---|
  | the intercellular spaces | 0.88 s | about 0.2 s (polygon rebuild, carving, perhaps the smoothing) |
  | the aerenchyma fill, 8 attempts | 0.19 s | about 0.1 s (fill order computed once) |
  | `fuse_gaps` | 0.11 s | little |
  | view adoption | 0.12 s | 0.12 s |

  Within the intercellular spaces, the rest is GEOS unions (0.32 s profiled) and a snap. In all, B2b–d give about 8–10 % for some 460 lines of converted code; retiring the view layer is mostly a readability gain.
- **What weighs more: the snap** (`CellGenerator._build_topology`). It is computed 4–5 times per section (grouping, air spaces, calibration, network): 0.90 s of the dicot root's 4.34 s (21 %), 0.44 of 2.29 s on the monocot root.
  - Its greedy clustering can be reproduced exactly: the same visiting order, means summed as numpy does, Python's `round`.
  - So can its maps (phases 1–2), with arrays.

  This is B5, which I had marked "hard to keep exact"; reading the code, it is not.

## 37. Question (2026-10-13)

- **Q-B3 — next step.** Continue B2b–d as agreed (about 8–10 %, retires the views), or do an exact B5 first (the snap: about 21 % of generation now, perhaps half of it saved, some 150 lines)?
  - **Recommendation:** B5 first, exact, checked by the comparison folder. Then B2b–d, whose gain is mostly readability.
  - → answer: yes to recommendation

## 38. B5: the snap converted (2026-10-13; your answer Q-B3: B5 first)

- **`mpg/pipeline/snap.py`:** GRANAP's `_build_topology`, converted. It is used by the topology cache (E1) and the stored topology for organs on the converted pipeline.
  - the rings and edge lengths are read in bulk;
  - the cluster means are summed as numpy does;
  - the edge keys come from one comparison;
  - the junction test reads each vertex's places in its rings, instead of scanning every ring around it (quadratic in the ring size).
- **Exact:** every snap of every organ's generation (75 calls, 14 organs) was computed both ways. The results, and the iteration order of every dict and set returned, are identical.
- **Found on the way:** since Python 3.12, `sum()` of exact floats is compensated (Neumaier), not a sum in order. A plain loop reproduces numpy. N1 is not affected: GRANAP's sums there run over `np.float64`, which take the plain path.
- **Times** (medians of 5 interleaved runs):

  | organ | before | after | change |
  |---|---|---|---|
  | dicot root | 4.90 s | 4.58 s | −7 % |
  | monocot root | 2.24 s | 2.01 s | −10 % |
  | dicot leaf | 3.61 s | 3.13 s | −13 % |

- **Full check:**
  - the comparison folder passes, 188 tests in 13 min (15 before);
  - the capability tests pass;
  - the 14 organs of `compare_sections.py` are identical.
- **Also this step:** `compare_sections.py` covers every organ kind, with a panel of differences. It found an N1 bug on sections without shared junctions (ROI), now fixed and tested.
- **Next:** B2b–d (your answer).

## 39. B2b: the aerenchyma fill planned once per calibration (2026-10-13, your go-ahead)

- **`mpg/pipeline/aerenchyma.py`:** GRANAP's `add_aerenchyma`, converted, in two parts.
  - **`fill_plan`:** the candidates, the areas and each file's fill order (connected radial channel or by distance), kept as positions in the cell list.
  - **The fill itself**, for a given target.
- **Every attempt of the calibration starts from the same snapshot copies,** so the plan is computed at the first attempt and reused by the others. Before, the whole preparation, the STRtree adjacency of `_connected_radial_order` included, was redone at each of the up to 21 attempts.
- **Scope:** used by the calibration for organs whose `add_aerenchyma` is `Organ`'s. A direct call to `organ.add_aerenchyma` stays GRANAP's.
- **Exact:** the stage tests, the dicot and needle cross-checks, and the replayed dicot-root and special-tissue tests pass (15). The aerenchymatous organs of `compare_sections.py` are identical.
- **Times** (medians of 5 interleaved runs): dicot root 4.47 → 4.24 s (−5 %), monocot root 2.00 → 1.89 s (−5 %).
- **Next:** B2c (the intercellular spaces) and B2d (`fuse_gaps`).

## 40. B2c: the intercellular spaces (2026-10-13, your go-ahead)

- **`mpg/pipeline/intercellular.py`:** GRANAP's `Organ._apply_intercellular` and `seat_air_spaces`, converted.
  - **The cells' smoothing in bulk:** one GEOS buffer for all the cells. Each ring is resampled by GRANAP's own `resample_coords`, then the 10 Laplacian passes run on all the rings at once, the same arithmetic element by element. A buffer that is not one Polygon, or an open ring, goes through GRANAP's function.
  - **The hosts carved in bulk**, then the converted simplification (B2a) on top of the converted snap (B5).
  - The unions and the pocket centering are GRANAP's.
- **Scope:** used for organs whose `_apply_intercellular` is `Organ`'s. The needle keeps its own (P5.6).
- **Exact:**
  - the stage, cross-check (monocot, dicot, needle, stems) and special-tissue tests pass (13);
  - the 14 organs of `compare_sections.py` are identical.
- **Times** (medians of 5 interleaved runs): dicot root 4.09 → 3.95 s (−3 %), monocot root 1.85 → 1.76 s (−5 %), dicot leaf unchanged (its own intercellular spaces).
- **Next:** B2d (`fuse_gaps`), then the full comparison folder.

## 41. B2d: the gap fusion; B2 closed (2026-10-13, your go-ahead)

- **`mpg/pipeline/gaps.py`:** GRANAP's `fuse_gaps` and `find_gaps`, converted.
  - **The unions computed once:** the union of the cells, and its filled mass, were each computed up to three times (`find_gaps`, `_filled_cell_mass` called by both, `fuse_gaps`). Being of the same polygons in the same order, they give the same result.
  - **The cells' properties recalculated in bulk:** the same GEOS centroid, distance and area; the angle stays `np.arctan2` cell by cell. This is also used for the final "cell properties" step of `generate.py`.
- **Exact:** the 14 organs of `compare_sections.py` are identical.
- **Full check:** the comparison folder passes, 188 tests in 12.6 min; the capability tests pass.
- **Times** (medians of 4 interleaved runs, machine loaded by the suite): monocot stem 8.94 → 8.39 s (−6 %), dicot root 5.12 → 4.86 s (−5 %).
- **From the start of the day** (`75bd5aa`, before N1) to now, generation plus matrix, medians of 3:

  | organ | start of day | now | change |
  |---|---|---|---|
  | dicot root | 5.66 s | 3.80 s | −33 % |
  | monocot root | 2.80 s | 1.75 s | −37 % |
  | dicot leaf | 3.84 s | 2.89 s | −25 % |

- **B2's end state:** the stages after the grouping run as converted copies on bulk operations.
  - **Still present, so not retired:**
    - the view layer (`cells.py`), because the stems', leaves' and needles' own methods still work on Cell objects;
    - the method overrides of the MPG-backed class, which now choose between the converted copies and GRANAP's.
  - **Retiring them** comes with P5.6, the organ-specific stages.
- **Open:** C1 (the network as DataStructure variables), C2 (the process-wide patches), B3 (the seeds as arrays), P5.6.

## 42. C1: the network as DataStructure variables — design (2026-10-13; your go-ahead for C1, C2, B3, P5.6 in order)

**Today:** the network's attributes are `node__`/`edge__` properties holding what networkx held: tuples (`position`), small arrays (`d_vec`), strings (`type`, `path`, `cell_type`), and keys that only some nodes or edges have. The DataStructure reads a property as a declared variable only if it is numeric and present on every node (or edge), so a graph system cannot use any of them.

**C1 — numeric columns, one per quantity, on every Compartment and every anatomy Connection:**

| location | variable | content |
|---|---|---|
| Compartment | `section_x`, `section_y` | its position in the section, mm |
| Compartment | `wall_length` | walls; 0 at junctions; NaN for protoplasts |
| Compartment | `wall_thickness` | walls; NaN otherwise |
| Compartment | `cell_area`, `cgroup`, `protect_topology`, `bridge` | protoplasts; NaN otherwise |
| Connection | `network_path` | a code: membrane 0, wall_air 1, wall 2, plasmodesmata 3, air_link 4 |
| Connection | `edge_length`, `edge_dx`, `edge_dy` | every anatomy Connection |
| Connection | `edge_dist` | NaN for wall edges |
| Connection | `edge_wall_thickness` | membrane, wall_air and wall edges |
| Connection | `lateral_distance`, `distnode_wall_cell` | wall edges |

- **Kept as they are:** the air links' `via_*` (node indices, on the air links only, for GRANAP's API), and the MECHA index.
- **Derived instead of stored:** the node `type` from the Compartment's kind; `cell_type` is the protoplast's `tissue`.
- **GRANAP's networkx graph,** now a view, is rebuilt from these columns with exactly the keys and values the exporter gives (a path's keys are fixed by the path). The cross-checks keep checking it node for node.
- **Templates copy the columns with the rest,** so every segment's anatomy carries them.
- **Units:** GRANAP's (mm, mm²). Converting to the MPG's length unit is left to the model that declares them, as with `mm_per_unit` for the classes.

**Q-C1a — values on the vessel junctions between sections.** The junction Connections (created by the wiring) have no in-section geometry. A declared edge variable needs a value on every edge.
- **Recommendation:**
  - `edge_dx`, `edge_dy`: the in-plane offset between the two vessels' positions;
  - `edge_length`: the distance between the two sections' centres (the mean of the two segments' lengths);
  - `network_path`: a code of its own (5, junction);
  - every other variable: NaN.

  Implemented only once you agree. Until then, junctions carry none of these variables.
- **C1 done (2026-10-13):** `mpg/columns.py`.
  - **Written:** N1 writes the columns as arrays; the exporter path converts its attribute dicts.
  - **Read back:** GRANAP's graph is rebuilt from the columns, exact (the cross-checks against GRANAP's original graph pass, the needle with bridges and air links included).
  - **New test, `test_network_variables.py`:** a component declares six of them at the Compartment and Connection scales; the DataStructure reads them on every node and edge, with the exporter's values.
  - **Examples:** they draw from `section_x`/`section_y`.
  - **Q-C1a (junction values):** still open; junctions carry none of these variables.

## 43. C2: no class method replaced for the converted pipeline (2026-10-13)

- **`MPGBacked.section_topology(polys, cell_ids, protect_ids)`:** the section's topology, asked of the organ. It returns the rings stored in the MPG for the stored network's cells; otherwise it computes, through the organ's E1 cache (`speedups.cached_topology`, now held by the organ) and the converted snap (B5).
- **Converted code calls it directly:**
  - the simplification (grouping, merge, air spaces);
  - the converted exporter;
  - a converted copy of GRANAP's XML writer (`mpg/pipeline/writer.py`), whose one edit is that call.
- **The two patches remain only where GRANAP's own modules still call `CellGenerator._build_topology`:**
  - `stored_topology`: GRANAP's exporter and writer, off the converted pipeline;
  - `exact_speedups`: during generation, where it now just routes the class method to `organ.section_topology`.

  With the class method forbidden and no patch, the 14 organs of `compare_sections.py` were generated and exported. Two still reach it, both through organ-specific code (P5.6):
  - the needle: `_apply_mesophyll_wall_rhombi` → GRANAP's `seat_air_spaces`;
  - the CellSet organ: its donor section runs GRANAP's own `Organ.generate_cells`.
- **Test, `test_no_class_patch.py`:** a converted root with the class method forbidden generates, exports and writes its XML, byte for byte GRANAP's.
- **Also:** C1's full check passed, 188 tests (12.5 min).

## 44. B3: the seeds' footprints in bulk (2026-10-13)

- **`mpg/pipeline/seeds.py`:** GRANAP's `generate_cells_info` and `resolve_cell_border_overlaps`, copied. The groups' convex-hull footprints are built in bulk (`multipoints`, `convex_hull`, `buffer`: the same GEOS operations per group), instead of a `MultiPoint`, a hull and a buffer per group, about 3,100 groups on a dicot root.
  - The buffer's resolution is passed explicitly: shapely's `buffer` function defaults to `quad_segs=8`, the geometry method GRANAP calls to 16.
- **The seeds stay Cell objects:** the grouping (`process_voronoi_groups`) reads them. Seeds as arrays would come with converting the grouping, which belongs to P5.5's grouping profile.
- **Exact:** the 14 organs of `compare_sections.py` are identical.
- **Times** (medians of 5 interleaved runs): dicot root 4.05 → 3.84 s (−5 %), monocot root 1.84 → 1.71 s (−7 %), dicot leaf −2 %.
- **Also, your request:** `example/mpg/monocot_root.py`, the simplest use of the component. One monocot root of three segments:
  - instantiated: plant, anatomies, DataStructure, component;
  - run: the root grows a segment, and the component gives it its anatomy;
  - plotted: the four sections from the MPG.

  About 100 lines, 4 s.

## 45. P5.6: the organ-specific stages — design (2026-10-13)

**Where the organs stand now:**
- **Roots:** every stage after the grouping runs through the converted copies.
- **Stems and leaves:** most stages too. Their own `add_intercellular_spaces` call `Organ`'s (converted), then an organ-specific finisher, which is GRANAP's code on Cell views:
  - stems: `_finalize_pith_and_lacunae`;
  - leaves: `_convert_inter_bundle_aerenchyma` and `_scatter_leaf_aerenchyma`.

  The leaves' merge is the converted one.
- **Needle:** its own `_apply_intercellular` (68 lines) and `_apply_mesophyll_wall_rhombi` (73 lines), plus the palisade slits, call GRANAP's `seat_air_spaces`. That is why it still needs the class-method patch (C2).
- **CellSet:** its `generate_cells` (62 lines) and `_build_donor` (62 lines) run GRANAP's own `Organ.generate_cells` for a donor section. That is the other user of the patch.
- **ROI:** no stage of its own beyond reading the files; nothing to convert.

**Two possible scopes:**
- **(a) Converted copies of the needle's and the CellSet organ's methods**, about 270 lines:
  - the needle's call the converted `seat_air_spaces` with the organ's topology;
  - the CellSet donor is generated by the converted pipeline.

  The class-method patches (`exact_speedups`, `stored_topology`) then serve no converted organ and are kept only for `converted_pipeline = False`. The stems' and leaves' finishers are measured first, and converted only if they weigh.
- **(b) (a), plus the view layer retired:** every stage after the grouping rewritten as column operations on the MPG, with no Cell object.
  - It is several thousand lines of GRANAP's logic to restate, organ by organ, each checked exact.
  - The measured gain is small: adoption and attribute reads are about 3 % of a dicot root.
  - It is mostly a readability gain: the pipeline would be MPG-native end to end.

## 46. Question (2026-10-13)

- **Q-P56 — the scope of P5.6:** (a) or (b)?
  - **Recommendation:** (a). It removes the last process-wide patch from the converted pipeline and keeps every step exact, for a few hundred lines. (b) is a rewrite of GRANAP's stage logic for a small measured gain; better decided after (a), with the finishers' measurements in hand.
  - → answer (2026-10-13): (a)

## 47. P5.6 (a) done (2026-10-13, your answer Q-P56: a)

- **Needle, `mpg/pipeline/needle.py`:** the needle's `_apply_intercellular`, `_apply_mesophyll_wall_rhombi` and `_apply_palisade_wall_slits`, copied. The remaining entries go to the converted intercellular spaces, and the lacunae are seated by the converted `seat_air_spaces` with the organ's topology.
- **CellSet organ, `mpg/pipeline/cellset.py`:** the CellSet organ's `_build_donor`, copied. The donor is generated by the MPG-backed root (the converted pipeline), without a stored network (`MPGBacked.stores_network = False`).
- **The class-method patch is no longer applied on the converted pipeline:** `exact_speedups` only for `converted_pipeline = False`. With the class method forbidden, all 14 organs generate and export.
- **The stems' and leaves' finishers, measured:** 6–18 ms (`_finalize_pith_and_lacunae`, `_convert_inter_bundle_aerenchyma`) on sections of 2.4–18.7 s. They stay GRANAP's.
- **Found:** the continuous dicot stem's 18.7 s are mostly its vascular packing (`pack_circles`): a `set_precision` and a `maximum_inscribed_circle` per packed circle, 1,315 each, 14 s profiled. This is the stage before cells; a candidate for later.
- **Full check:**
  - the comparison folder passes, 190 tests (12.5 min);
  - the 14 organs of `compare_sections.py` are identical;
  - the CellSet and needle tests pass.

## 48. One source: the port as GRANAP, the original as legacy — design (2026-10-13, your request)

**Your request:**
1. The online source moves to a legacy folder, and the MPG source becomes the main one.
2. The tests use the port; only a `comparison/` subfolder compares with the legacy.
3. The same for the examples.

**What the port is today.** It is not a standalone copy of GRANAP; it is GRANAP plus a layer:
- **About 20 of GRANAP's modules are building blocks it calls as they are:** layers, recipes, vascular bundles, stomata, organ parameters, readers.
- **About a dozen methods have converted copies** in `mpg/pipeline/`: the generation, the intercellular spaces, the aerenchyma, the merge, the gaps, the snap, the seeds, the exporter, the XML writer, the needle's and the CellSet's stages.
- **The MPG-backed classes are mixins over GRANAP's classes;** flags (`converted_pipeline`, `native_network`, `exact_speedups`) select the converted or the original path.

So "keeping only the port" means merging the two, not keeping `mpg/` alone.

**Proposed layout:**
- **`src/openalea/granap/legacy/`:** the online source, frozen and verbatim (`origin/to_metafspm`, 248adca; the original modules are unchanged since). Only its internal imports are rewritten (`openalea.granap.X` → `openalea.granap.legacy.X`), so that it runs on its own as the reference.
- **`src/openalea/granap/` (the main package) is the port:**
  - the organ classes under their own names and modules (`root_class.RootAnatomy` …) are the MPG-backed ones;
  - the converted methods replace the original methods in their modules (`generate_cell.py`, `organ_class.py`, `anatomy_writer.py`, `needle_class.py`, `cellset_organ.py`);
  - the building blocks stay where they are.
- **Removed from the main package:** the mixins, the flags and the patches (`exact_speedups`, `stored_topology`). `mpg/pipeline/` disappears into the modules.
- **`openalea.granap.mpg`** keeps what is MPG- and metafspm-specific: build, network, columns, topology, cells, views, component, library, vessels.
- **Tests:**
  - `test/` holds GRANAP's tests on the main package. They are today's replayed tests, the same goldens, since the port is exact;
  - the capability tests (component, growth, network variables) also go in `test/`;
  - `test/comparison/` holds the cross-checks, the stage tests, N1 against the exporter, and the class-method test, comparing the main package with `legacy`.
- **Examples:**
  - every example in `example/` on the main package;
  - `example/comparison/` holds `compare_sections.py`, now comparing with `legacy`.
- **Steps, each checked:**
  1. the legacy copy, with the comparison folder pointed at it;
  2. the merge of the port into the main modules, with every flag removed;
  3. the tests and examples moved.

  The full comparison runs at each step.

## 49. Questions (2026-10-13)

- **Q-R1 — where legacy goes.** You wrote `src/legacy/`. As a top-level package, `legacy` would be a generic import name, shared by anything else installed. `src/openalea/granap/legacy/` (importable as `openalea.granap.legacy`, shipped with GRANAP) keeps it inside GRANAP's namespace.
  - **Recommendation:** `src/openalea/granap/legacy/`.
  - → answer (2026-10-13): recommendation
- **Q-R2 — the main package's names.** Keep GRANAP's module and class names (`from openalea.granap.root_class import RootAnatomy` gives the port), or make `openalea.granap.mpg` the entry point?
  - **Recommendation:** GRANAP's names. Users' code and GRANAP's own tests keep working unchanged; `openalea.granap.mpg` stays the metafspm bridge.
  - → answer (2026-10-13): recommendation
- **Q-R3 — the original paths inside the main package.** With legacy holding the original, the main package drops the switches (`converted_pipeline=False`, `native_network=False`, `exact_speedups=False`) and the code they selected?
  - **Recommendation:** yes, drop them. The comparison tests compare with legacy instead.
  - → answer (2026-10-13): recommendation

## 50. Optimisations opened by an MPG-centred source (2026-10-13, your request: recorded as candidates)

Not scheduled. Each is a candidate step, to be designed when chosen. The second group changes results, so it is a modelling choice and needs reviewed golden changes.

**Exact (the same sections and networks):**
- **O1 — several sections in one pass:** the stages of every class that appears in one step run through the same bulk GEOS calls, spreading the Python cost across sections.
- **O2 — the solve per template:** every copy of a template has the same network structure.
  - The symbolic part of its matrix (sparsity, factorization pattern) is computed once and reused for all copies.
  - The full matrix is assembled from per-template index arrays; GRANAP's `fill_matrix` loops over networkx edges today.
- **O3 — compiled kernels:** the loops that must stay sequential (the snap's greedy clustering, the aerenchyma fill, the merge's union-find), compiled (numba) on the arrays, exact.
- **O4 — per-step DataStructure passes:** the array mirrors of the MPG (DS14b in metafspm's scene plan) remove most of the about 1.5 s of whole-MPG passes left in a growth step (owner scales, topology arrays, linked parents).

**Changing results:**
- **O5 — sections edited in place:** with the topology and network stored, a section changes as the plant develops (aerenchyma forming over time, a tissue differentiating), re-snapping and rewiring only the cells concerned, instead of being regenerated as a new class.
- **O6 — a class derived from a neighbouring template:** a new diameter class built by deforming an existing template's geometry, far cheaper than the whole pipeline for small steps between classes.
- **O7 — algorithmic replacements in the stages before the network:**
  - the vascular circle packing (14 s of the continuous dicot stem's 18.7 s) on a distance field;
  - the Voronoi grouping by coverage unions (B4);
  - a vectorised snap without GRANAP's greedy order.

## 51. One source: done (2026-10-13; your answers Q-R1–R3: recommendations; your request: `mpg` → `metafspm`)

- **`openalea.granap.legacy`:** the online source (`origin/to_metafspm`, 248adca), frozen and verbatim, its imports rewritten. It runs on its own; no main module is loaded.
- **The main package is the port, under GRANAP's names:**
  - **`Organ`** generates into the MPG: `generate_cells` → `generate_section` → `_generate_pipeline`, with the stages written, the cells as views, the stored network (`section_topology`, `graph` and the matrix as views of the MPG). The converted stages replace the originals: `_apply_intercellular`, `_add_aerenchyma_to_proportion` (with `_fill_plan`), `add_aerenchyma`, `merge_intercellular_aerenchyma`, `find_gaps`/`fuse_gaps`.
  - **`CellGenerator`:** the snap, the simplification and the seeds.
  - **`CellManager.recalculate_cell_properties`:** in bulk.
  - **`seat_air_spaces`:** carving in bulk, with the organ's snap.
  - **`anatomy_writer`:** the exporter and the XML writer ask the organ for the topology.
  - **The needle** seats its lacunae with the organ's snap; the **CellSet and ROI** organs give `_generate_pipeline`; the CellSet donor stores no network.
  - **`create_from_input`** takes a seed.
- **Removed:** the mixins, the flags (`converted_pipeline`, `native_network`, `exact_speedups`), the patches (`exact_speedups`, `stored_topology`), `mpg/pipeline/`, `mpg/organs.py`, `mpg/speedups.py`.
- **`openalea.granap.metafspm`** (was `openalea.granap.mpg`, your request): build, network, columns, topology, cells, views, component, library, vessels.
- **Tests:**
  - `test/`: GRANAP's own tests, now on the port, with the same goldens; the replay wrappers are gone;
  - `test/component/`: the capability tests;
  - `test/comparison/`: against legacy — the cross-checks, the stages, and the stored network on 13 organ kinds.
  - The full suite: 201 passed (13 min).
  - **CI** runs `pytest -v --ignore=comparison` (`conda/meta.yaml`), so the comparison against legacy runs only locally.
- **Examples:**
  - `example/comparison/compare_sections.py` compares legacy and GRANAP: the 14 organs are identical;
  - `example/mpg/` holds the component examples;
  - GRANAP's other examples use its names, so they run the port. All 36 were run on the port afterwards (2026-10-14; non-interactive plotting backend): 36 ran without error, the longest `root/dicot_oak.py` (227 s).
- **README and packaging:** metafspm is required (comments in `pyproject.toml` and `conda/environment.yaml`); `legacy` is described in the README.

## 52. O1–O4 under metafspm's logic — design and measurements (2026-10-14, your go-ahead and question)

**Your question: can the generation be declared as decorated steps of a component, with metafspm's scheduling?**
- **What already follows metafspm's logic:**
  - the anatomies of a population are a `StructuralComponent` (`GranapAnatomy`), updated in its `@postsegmentation` step;
  - its network is solved by `@graph_system`s over the DataStructure;
  - its quantities are DataStructure variables (C1).
- **The generation of a section is not a time-stepped process,** so its stages do not map onto metafspm's step kinds:
  - **Kinds are global rows run every time step.** A kind (`@stepinit`, `@rate`, `@state`, … `@postsegmentation`) runs for every component, in a fixed order, once per time step. A section is generated once, when its class first appears. There is no "once per new class" kind, and adding one would be a new scheduling concept for one model.
  - **The stages are strictly sequential within one section,** each one consuming the previous one's cells. Step kinds give no order finer than the row.
  - **The stages change the topology at every boundary** (cells created, fused, removed), with geometry (polygons) as their main data. metafspm steps compute fields on a fixed topology, which is then updated between steps (`update_topology`).
- **So the right fit is the one in place:** the generation is a service of the structural component (its library), called in its `@postsegmentation` step when a new class appears, and producing DataStructure-backed sections.
- **What can move further into metafspm's logic is the numeric part:** the quantities a stage computes on a fixed set of cells (areas, radii, fill targets) as DataStructure variables computed in bulk. That is the cell-view question (B2 option b), not needed for O1–O4.

**Measurements behind O1–O4.**
- **O2, the solve** (the seedling: 398 k nodes, 980 k edges; the capability test's hydraulic model; 60 s profiled):
  - the Jacobian is built by finite differences (no analytic one is declared): 35 s;
  - of these, 6 s are scipy's column grouping (`group_sparse`) and 9 s its perturbation vectors, recomputed at each solve although they depend only on the sparsity pattern, i.e. on the topology;
  - the LU solves (SuperLU) take 6 s.
- **O4, a growth step's whole-MPG passes** (§33): about 1.5 s on the seedling (owner scales 0.58 s, topology arrays 0.55 s, linked parents in the rewiring 0.45 s).
- **O3, one dicot root's generation** (11 s profiled):
  - the snap's own Python is 0.77 s;
  - the views' attribute reads and adoptions about 2 s (cumulative);
  - the grouping's unions 1.5 s, which is GEOS itself.
- **O1, several sections in one pass:** a section's work is GEOS operations proportional to its cells, plus per-cell Python. Batching sections removes only the per-call overhead of the bulk calls, which is small. No exact gain worth measuring is expected.

**Proposal, by where the code goes:**
- **O2 → metafspm's solver (`solve/`):** the Jacobian's sparsity pattern and its column grouping cached per topology version (the DataStructure's `topology_version`) and per equation set. The same groups give the same finite-difference Jacobian, so it is exact. An analytic Jacobian for linear edge laws would gain more, but it is a model-side declaration (`@graph_jacobian`), left to the models.
- **O4 → metafspm's MPG (DS14b in your scene plan):** array mirrors of `parent`, `complex` and `scale`, kept with the structural signature. The owner maps, `linked_parent` and the topology arrays read them instead of walking dicts. Exact (the same results), checked by metafspm's suite and the GRANAP growth reference.
- **O3 → GRANAP:** numba kernels for the snap's clustering and maps, with the same arithmetic. Python's `round` stays in Python, since numba's `round(x, 6)` is not guaranteed to round the same way. The views' cost goes with B2 (b), not O3.
- **O1:** not done, for the reason above.

## 53. Questions (2026-10-14)

- **Q-O1 — drop O1?**
  - **Recommendation:** yes. Its exact gain is negligible; the per-cell Python it would have amortized is O3's.
  - → answer (2026-10-14): yes, drop O1.
- **Q-O2/O4 — these two are changes in metafspm's core** (the solver, the MPG). Do them in metafspm now, in that order, each with metafspm's suite and the GRANAP checks?
  - **Recommendation:** yes. O2 is the largest measured gain (the solve), and O4 is step 5 of your scene plan (PA2: yes).
  - → answer (2026-10-14): drop O2 — the solve is the scope of the MECHA port that follows (package/MECHA-dev, branch to_metafspm), coupled to GRANAP the metafspm way. Yes to O4.
- **Q-O3 — numba as a GRANAP dependency** (it is in the env: 0.68)?
  - **Recommendation:** yes, for the snap first, with the comparison folder as the check.
  - → answer (2026-10-14): yes.

## 54. GRANAP's anatomy component with declared i/o — design, anticipating the MECHA coupling (2026-10-14, your request)

**Your concern:** `GranapAnatomy` has no declared fields (inputs, parameters, state variables) as a metafspm component should. So a coupled model (MECHA, next, in `package/MECHA-dev`, branch `to_metafspm`) cannot read GRANAP's outputs, pass it generation instructions, or be coupled to it explicitly. Today the configuration is class attributes (`library`, `shoot`, `bins`, `mm_per_unit`, the names `radius`/`length`/`distance_from_tip`), and the outputs are MPG properties nobody declares. Nothing is reimplemented for MECHA now; this only shapes GRANAP's side.

**What metafspm offers** (checked): a `StructuralComponent` declares fields like any component. Around an MPG-style step (`@postsegmentation _anatomies`), the framework:
1. writes the declared variables to the MPG;
2. updates the DataStructure's topology if the MPG changed;
3. re-reads the declared state variables from the MPG.

Per-plant parameters come from the scenario (`initiate_plant(g, plant, parameters)`, `parameter_values`).

**Proposed declarations:**
- **Inputs (the architecture's outputs that GRANAP reads),** at the SubOrgan scale: `radius`, `length`, `distance_from_tip`, `input_variable(by=<the architecture model>)`. The names become fixed declarations instead of class attributes.
- **Parameters (the instructions for generation),** per plant through the scenario:
  - the class bins (diameter, distance from the tip, from the junction);
  - `mm_per_unit`;
  - the anatomy's parametrization, as overrides of GRANAP's input sections (`{section: {field: value}}`, e.g. the cortex's number of layers), per organ kind and per class. The library is then keyed by (plant's parameters, class), so two genotypes in one population get their own templates.
- **State variables (the outputs MECHA couples to),** all `state_variable_type="descriptor"`, written by the component and re-read by the framework:
  - **SubOrgan:** the class (three numeric descriptors: diameter, tip and junction bins) and the template index;
  - **Cell:** tissue (an object variable at its own scale), vessel id, family, area;
  - **Compartment:** C1's columns (`section_x`, `section_y`, `wall_length`, `wall_thickness`, `cell_area`, `cgroup`, `protect_topology`, `bridge`), and the kind (protoplast, wall, junction);
  - **Connection:** C1's columns (`network_path`, `edge_length`, `edge_dx`, `edge_dy`, `edge_dist`, `edge_wall_thickness`, `lateral_distance`, `distnode_wall_cell`).
- **How MECHA would couple** (later, nothing done now): MECHA's component declares `input_variable(by="GranapAnatomy", scale=Compartment/Connection)` for what it needs (tissue, cgroup, wall lengths and thicknesses, areas, edge types and lengths), computes its conductances on them, and solves with `@graph_system`. Its own parameters (wall and membrane conductivities per tissue) are its own.

**This needs Q-C1a decided:** declared edge variables need a value on every edge, the vessel junctions included. The recommendation of §42 stands: offsets between the vessels, length between the sections' centres, a path code of its own, NaN otherwise.

## 55. Questions (2026-10-14)

- **Q-D1 — the declarations above** (inputs, parameters per plant through the scenario, descriptor state variables at the SubOrgan, Cell, Compartment and Connection scales)?
  - **Recommendation:** yes, implemented after O4 and O3, and checked by a test component declaring `input_variable(by="GranapAnatomy")` for the network, as MECHA will.
  - → answer:
- **Q-C1a — the junctions' values** (§42), needed for the Connection-scale declarations.
  - **Recommendation:** as in §42.
  - → answer:
- **Q-D2 — the anatomy's parametrization per plant:** overrides of GRANAP's input sections (`{section: {field: value}}`, per organ kind and class) as one object-valued parameter? Or flattened into numeric parameters (`cortex__n_layers`, …), one declaration per field?
  - **Recommendation:** the overrides object. GRANAP has hundreds of fields across organ kinds. Declaring each one would freeze a large API, while the overrides follow GRANAP's own `OrganInputData` sections and its validation.
  - → answer:

## 56. O4 done; O3 measured again (2026-10-14)

- **O4 (metafspm 26bb644, DS14b):**
  - `add_components_bulk` records its vertices' parents and complexes, and the incremental topology arrays read these records instead of the MTG's dicts, vertex by vertex;
  - `replace_parent` counts as a structural edit, so the cached arrays are rebuilt (they used to stay stale after a reparenting);
  - `_valid_vids_at` is cached per structure;
  - the anatomy signature reads its columns in bulk.

  The same arrays, as a new metafspm test checks against a full rebuild. metafspm's suite: 796 passed; GRANAP's component tests pass; the GRANAP growth reference is identical. On the seedling:

  | | before | after |
  |---|---|---|
  | build | 21 s | 13.6 s |
  | 9-segment emergence | 2.2 s | 1.6 s |
  | class change (with the new template's generation) | 7.8 s | 6.4 s |

- **O3 (numba for the snap), measured without the profiler: not worth it.**
  - The snap now takes 0.06–0.13 s per call, 2,834 cells for the largest. That is about 0.25 s per dicot root over its 4 calls, 6 % of its 3.8 s. The 0.77 s of §52 was cProfile's, which inflates pure-Python loops several times.
  - By phase, for the largest call: rings and tolerance 0.015 s, KD-tree query 0.007 s, clustering 0.011 s, vertex map 0.011 s, edge map 0.008 s, junctions 0.005 s.
  - A kernel could take half of that at best (about 3 %): the dicts and sets of tuples it returns, which the exporter, the writer and the simplification read, must still be built in Python.
  - numba also compiles at the first call in each process (about a second per kernel, unless cached on disk). For a simulation generating a few templates per process, that cancels the gain.
- **Q-O3b — drop O3?**
  - **Recommendation:** yes, not implemented. The measured cost of what remains is GEOS (the grouping's unions, the intercellular unions) and the cell views (the B2 b question), not Python loops a kernel would take.
  - → answer:
