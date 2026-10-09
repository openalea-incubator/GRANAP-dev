# Graph Report - GRANAP  (2026-09-11)

## Corpus Check
- 109 files · ~794,993 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2372 nodes · 4568 edges · 179 communities (133 shown, 44 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 477 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `07aa2788`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Tissue
- DicotRootAnatomy
- OrganInputData
- Cell
- Batch Cell.jitter / seed creation
- TissueRecipe
- generate_callgraph.py
- RootAnatomy
- granap.layer_class.Layer
- CellManager
- AnatomyWriter
- input_data.py
- PolygonInterpolator
- LayerPolygon
- AbstractNetwork
- MonocotRootAnatomy
- Layer
- Organ
- granap.geometry_collection.GeometryProcessor
- granap.organ_class.Organ
- .pack_circles
- test_cellset_organ.py
- test_resin_duct_placement.py
- .generate_cells
- granap.cell_manager.CellManager
- CellGenerator
- GeometryProcessor
- test_transfusion_network.py
- granap.cell_class.Cell
- test_vascular_regression.py
- visualization.py
- test_plot_tissues
- .for_dicot_stem
- LayerManager
- test_growth_modes.py
- .create_from_input
- DicotStemAnatomy
- Polygon
- granap.needle_class.NeedleAnatomy
- test_monocot_xylem.py
- ._create_central_layers
- granap.network_base.AbstractNetwork
- .__init__
- granap.root_class.RootAnatomy
- .for_root
- .buffer_polygon
- LineString
- .merge_intercellular_aerenchyma
- organ_class.py
- vascular_bundle.py
- test_secondary_phloem.py
- stem_class.py
- math_functions.py
- ._resolved_dimensions
- read_cellset
- secondary_growth.py
- _get_planttype
- .for_dicot_root
- GRANAP conda recipe (meta.yaml)
- test_leaf_slab.py
- LeafAnatomy
- test_param_metadata.py
- StemAnatomy
- Heavy configs (secondary growth) — `dicot_nettle` profile (2026-07)
- 6. Designing a new tissue — the useful functions
- DicotLeafAnatomy
- .add_stomata
- .for_monocot_stem
- test_stem_bundles.py
- MonocotLeafAnatomy
- main
- granap.visualization
- make_secondary_root
- Drop geopandas dissolve for Python grouping + unary_union
- __init__.py
- test_stem_features.py
- G
- granap
- P5 Optional layout/cleanup follow-ups (closed)
- openalea.granap
- _local_envelope
- DicotRootSeries
- cellset_organ.py
- RootSeries
- test_root_series.py
- test_stem_continuous.py
- .for_dicot_leaf
- ._build_layer_polygons
- Tissue refactor — the shape-first model
- Roadmap — toward simplified recipes for monocot, dicot, needle
- NeedleAnatomy
- Polygon
- Proposal
- .pole_and_corner_angles
- .__init__
- root_series.py
- CellSetOrgan
- test_param_schema_equivalence.py
- series_monocot_wheat.py
- RootSeriesResult
- ._finalize_pith_and_lacunae
- default_dicot_leaf_params
- monocot_snowdrop.py
- make_root
- build_dicot_base
- transfusion_network.py
- TissueStep
- describe_params
- Test P. pinaster anatomy
- brassicaceae_roots.py
- build
- make_dicot_root
- make_root
- make_secondary_root
- main
- _continuous
- _dicot
- Any
- dicot_nerium.py
- GRANAR package translated in Python
- ._create_base_shape
- Architecture
- ._vein_layout
- _build_topology wall-snapping
- generate_cells performance work
- Golden regression byte-identical baseline
- perf_characterize.py verification harness
- process_voronoi_groups grouping
- Vectorised point-in-polygon (contains_xy) tests
- P1 Voronoi seed density reduction
- Cell.radial seeding idiom
- Cells-first Voronoi engine
- Seed determinism invariant (self.rng)
- Shape-first tissue model
- special_tissues.py vocabulary
- Tissue (tag, shape) region
- TissueRecipe / TissueStep
- Unified vascular seed mask (region algebra)
- test_vascular_regression golden census
- P0 Needle golden safety net
- P1 Recipe ergonomics (region to fill)
- P2 Special-function vocabulary
- P3 Needle into recipe model
- P4 Unify three recipes on Organ scaffold
- Simplified declarative recipe target
- AbstractNetwork
- Cell
- CellGenerator
- CellManager
- GeometryProcessor
- Layer
- LayerManager
- NeedleAnatomy
- Organ
- RootAnatomy
- _grow_bundle_sheath
- GRANAP (GRANAR in Python)
- .resample_coords
- .fit_inner_ellipse
- .oriented_ellipse
- .add_aerenchyma
- build_gallery_needle_data
- ._build_donor
- _as_shape
- cellset_hybrid.py
- .expand_layers
- _blunt_margin_input
- solve
- .add_layer
- Cross-platform determinism figures (golden regressions)
- make_dicot_root
- main
- DicotCambiumParams
- .export_to_adjencymatrix
- .allocate_vascular_tissue
- Transfusion network
- .__init__

## God Nodes (most connected - your core abstractions)
1. `OrganInputData` - 145 edges
2. `RootAnatomy` - 100 edges
3. `Organ` - 92 edges
4. `NeedleAnatomy` - 90 edges
5. `CellManager` - 86 edges
6. `GeometryProcessor` - 84 edges
7. `Cell` - 80 edges
8. `LeafAnatomy` - 68 edges
9. `StemAnatomy` - 62 edges
10. `TissueRecipe` - 53 edges

## Surprising Connections (you probably didn't know these)
- `main()` --uses--> `LeafAnatomy`  [INFERRED]
  example/leaf/gallery_leaf.py → src/openalea/granap/leaf_class.py
- `main()` --uses--> `StemAnatomy`  [INFERRED]
  example/stem/gallery_continuous_cylinder.py → src/openalea/granap/stem_class.py
- `main()` --uses--> `StemAnatomy`  [INFERRED]
  example/stem/gallery_dicot_growth.py → src/openalea/granap/stem_class.py
- `main()` --uses--> `StemAnatomy`  [INFERRED]
  example/stem/gallery_stem_shapes.py → src/openalea/granap/stem_class.py
- `main()` --uses--> `StemAnatomy`  [INFERRED]
  example/stem/gallery_vascular_band_packing.py → src/openalea/granap/stem_class.py

## Import Cycles
- None detected.

## Communities (179 total, 44 thin omitted)

### Community 0 - "Tissue"
Cohesion: 0.11
Nodes (19): Record only the wide ("xylem") vessels in ``vascular_polygons``. The packing…, A tagged anatomical region: a shapely shape plus the tag its cells take. Shape-…, Smooth the region boundary (pre-fill, pure geometry)., Tissue, cell_type_counts(), make_arch_root(), Shape-first tissue vocabulary + recipe tests. Model under test: * A **tissue**…, test_monocot_arch_produces_vessels() (+11 more)

### Community 1 - "DicotRootAnatomy"
Cohesion: 0.06
Nodes (31): Pole of inaccessibility (largest inscribed circle): ``(cx, cy, radius)``.…, DicotRootAnatomy, Polygon, The angular wedges (one per vascular peak) the secondary-xylem vessels pack…, Build the medullar-ray corridor polygons (before vessel packing so they can cut…, Radial growth-ring bands of the secondary-xylem annulus, following the…, Build secondary xylem between the primary and secondary cambium., Build secondary phloem outside the cambium. The phloem occupies a band that… (+23 more)

### Community 2 - "OrganInputData"
Cohesion: 0.06
Nodes (24): BaseModel, _dicot(), main(), _monocot(), Demo: stem vascular bundle types and the hollow (fistular) pith. Renders the…, OrganInputData, fld(), A unified data structure for handling Organ initialization parameters from… (+16 more)

### Community 3 - "Cell"
Cohesion: 0.10
Nodes (18): Cell, Polygon, Build a seed cell whose ``angle``/``radius`` are measured from ``center``. This…, Jitter the cell position. Uses ``rng`` (the organ's seeded generator) when…, Drop cells that sit (almost) entirely inside another cell, keeping the bigger…, Layer module for plant anatomy representation. Provides the Layer class…, carve_and_insert(), consider_as_cell() (+10 more)

### Community 5 - "TissueRecipe"
Cohesion: 0.11
Nodes (8): Needle organ-specific tissues as a recipe of P2 special-tissue steps. Resin…, An ordered, inspectable sequence of :class:`TissueStep` objects. Beyond the…, Set the target CellManager + rng used by :meth:`fill` / :meth:`fill_each`.…, Run every step in order., Return ``(name, produces)`` for each step, for inspection / preview., Return ``(name, kind, produces)`` for each step — the full plan. Richer than…, Render the plan as an indented, human-readable block., TissueRecipe

### Community 6 - "generate_callgraph.py"
Cohesion: 0.10
Nodes (25): AsyncFunctionDef, ClassDef, build_dot_source(), build_html(), clean_docstring(), collect_docstrings(), DocCollector, ensure_graphviz_available() (+17 more)

### Community 7 - "RootAnatomy"
Cohesion: 0.07
Nodes (17): anatomy_metrics(), main(), Build a tomato-like dicot root from the ``for_root`` preset and plot it.…, Geometry summary of a generated root: root/stele diameters and total xylem…, Any, Polygon, Plant type: 1 = monocot, 2 = dicot., Create stele parenchyma rings from the stele edge toward the centre. (+9 more)

### Community 8 - "granap.layer_class.Layer"
Cohesion: 0.14
Nodes (29): granap.layer_class, granap.layer_class.Layer, granap.layer_class.Layer..post_init., granap.layer_class.Layer..repr., granap.layer_class.Layer.from_dict, granap.layer_class.Layer.get_total_thickness, granap.layer_class.Layer.to_dict, granap.layer_manager (+21 more)

### Community 9 - "CellManager"
Cohesion: 0.07
Nodes (12): setter, CellManager, Polygon, Recalculate the properties of all cells in the list., Drop every cell whose seed point intersects ``polygon`` (bulk predicate)., Remove exactly these cell objects, matched by identity. Unlike…, Next free ``id_group`` for appending a new cell group. ``get_last_id_group() +…, Drop every cell that intersects ``polygon``. A cell's footprint is its Voronoi… (+4 more)

### Community 10 - "AnatomyWriter"
Cohesion: 0.06
Nodes (17): main(), Manual testing script for RoiOrgan behavior. Update the folder_path to point to…, AnatomyWriter, get_v_idx(), process_polygon(), Class to export Organ anatomy to various formats (XML, OBJ, GEO)., Write the root anatomy as an XML file matching GRANAR format. Cell ordering and…, write the MECHA geometry files (retro-compatible) (+9 more)

### Community 11 - "input_data.py"
Cohesion: 0.12
Nodes (30): BaseParams, BaseShapeParams, _build_bundle_group_models(), CambiumParams, CentralCylinderParams, _ground_region_params(), InterBundleAerenchymaParams, _layer_params() (+22 more)

### Community 12 - "PolygonInterpolator"
Cohesion: 0.15
Nodes (11): as_tuple(), interpolate_point_points(), interpolate_poly(), midpoints(), normals_offset(), plot_poly_normals(), point_derivative(), PolygonInterpolator (+3 more)

### Community 13 - "LayerPolygon"
Cohesion: 0.09
Nodes (14): LayerPolygon, Typed representation of a layer polygon produced by _build_layer_polygons.…, Find the layer where vascular tissue will be allocated. Args: layers_polygons:…, Return extra tissue polygons for visualization without placing cells.…, Any, GeoDataFrame, Polygon, Define the basic shape of the root/needle from loaded polygons. (+6 more)

### Community 14 - "AbstractNetwork"
Cohesion: 0.18
Nodes (8): AbstractNetwork, ABC, Abstract network base module for hydraulic network construction. Provides the…, Fill entries of the adjacency matrix with a hydraulic conductivity. Parameters…, Abstract base class for hydraulic network graph construction. Subclasses must…, Fill entries of the adjacency matrix from a dictionary of hydraulic…, Return the set of wall-node indices whose adjacent cells match the requested…, Check whether (actual_a, actual_b) matches the requested pair in *either* order.

### Community 15 - "MonocotRootAnatomy"
Cohesion: 0.12
Nodes (14): Split a polygon into n slices using radial lines from the center., MonocotRootAnatomy, Polygon, Declarative description of how a monocot stele is assembled. Built and run by…, Phloem regions in the valleys between the xylem star arms. Monocot roots have…, Declarative phloem step: valley *regions* filled by circle-packing. Mirrors the…, The class's canonical metaxylem *centres* for ``n`` vessels — the same geometry…, Add a ring of xylem parenchyma cells around each metaxylem vessel. (+6 more)

### Community 16 - "Layer"
Cohesion: 0.10
Nodes (23): Layer, Represents a single tissue layer in plant anatomy. Attributes: name: Identifier…, Validate layer parameters., Calculate total thickness of this layer., Leaf cross-section anatomy. A leaf is a flat *slab*, not a radial organ.…, Wrap a normalized shape function so its output spans [lo, hi]. Parameters…, rescale(), __getattr__() (+15 more)

### Community 17 - "Organ"
Cohesion: 0.10
Nodes (13): Organ, ABC, Populate ``self.graph`` from the cell GeoDataFrame. Delegated to…, Add a tissue layer to the anatomy. Args: layer: Layer object to add position:…, Remove a tissue layer by name. Args: name: Name identifier of the layer…, List all layer names., Invalidate cached geometry after layer changes., Abstract base class for plant anatomical structures. Defines the interface and… (+5 more)

### Community 18 - "granap.geometry_collection.GeometryProcessor"
Cohesion: 0.16
Nodes (16): granap.cell_class, granap.generate_cell.CellGenerator.cell_border, granap.generate_cell.CellGenerator.cells_on_layer, granap.geometry_collection, granap.geometry_collection.GeometryProcessor, granap.geometry_collection.GeometryProcessor.difference_polygons, granap.geometry_collection.GeometryProcessor.draw_ellipse, granap.geometry_collection.GeometryProcessor.ellipse_to_polygon (+8 more)

### Community 19 - "granap.organ_class.Organ"
Cohesion: 0.19
Nodes (21): granap.layer_manager.LayerManager.get_layers, granap.organ_class.Organ, granap.organ_class.Organ._build_anatnetwork, granap.organ_class.Organ._build_layer_polygons, granap.organ_class.Organ._create_base_shape, granap.organ_class.Organ._create_central_layers, granap.organ_class.Organ._create_vascular_tissue, granap.organ_class.Organ._which_layer_for_vascular (+13 more)

### Community 20 - ".pack_circles"
Cohesion: 0.24
Nodes (5): Create a polygon for an ellipse, Largest ellipse (up to the target circle's area, aspect-capped) that fits…, Ellipse of the target circle's area, oriented **radially** (major axis along…, Size-first, gradient-driven radial packing. Marches outward in radius over…, Unified Apollonian circle packing with proportion stop, directional gradient,…

### Community 21 - "test_cellset_organ.py"
Cohesion: 0.07
Nodes (26): organ(), fixture, CellSet ingestion + the GRANAP-stele graft (``CellSetOrgan``). The fixture is a…, Outlines must stay exactly the wall points. ``RoiOrgan`` smooths every outline…, The digitized cells tile the section exactly -- no gaps, no overlaps. This is…, Everything outside the regenerated region keeps its measured area., The graft interface must be *shared*, not merely adjacent. A wall with one…, Plasmodesmata must cross both halves of the graft boundary. (+18 more)

### Community 22 - "test_resin_duct_placement.py"
Cohesion: 0.11
Nodes (28): _base_params(), _bearing(), _duct_data(), parametrize, Resin-duct placement in ``NeedleAnatomy._duct_zone_data``. Two placement modes…, A narrower wedge still yields a duct, and keeps it nearer the bearing., Two resin_duct blocks -> per-duct sizing, not the first block's for all., Re-derive slice placement independently and require an exact match. Recomputes… (+20 more)

### Community 23 - ".generate_cells"
Cohesion: 0.15
Nodes (8): GeoDataFrame, lil_matrix, Export cell geometries as GeoDataFrame. Returns: GeoDataFrame with cell data, Export cell data to CSV file. Args: filepath: Output file path, Build the hydraulic network from cell geometry and return the sparse adjacency…, Generate cell geometries using Voronoi tessellation. Returns: GeoDataFrame with…, Add organ-specific tissues by building this organ's organ recipe., Return the recipe of organ-specific (post-fill) tissues. Default: empty (root…

### Community 24 - "granap.cell_manager.CellManager"
Cohesion: 0.16
Nodes (23): granap.cell_manager, granap.cell_manager.CellManager, granap.cell_manager.CellManager.extend_cells, granap.cell_manager.CellManager.get_all_types, granap.cell_manager.CellManager.get_cell_by_id, granap.cell_manager.CellManager.get_cells, granap.cell_manager.CellManager.get_cells_by_group, granap.cell_manager.CellManager.get_cells_by_groups (+15 more)

### Community 25 - "CellGenerator"
Cohesion: 0.09
Nodes (24): CellGenerator, create_local_ellipse(), create_local_rectangle(), local_to_global_poly(), Any, ndarray, Point, Polygon (+16 more)

### Community 26 - "GeometryProcessor"
Cohesion: 0.12
Nodes (15): GeometryProcessor, Polygon, Geometry processor module for handling polygon operations., Generate a star-shaped polygon as a single continuous boundary. Alternates…, Star built from *peak-side* / *valley-side* radii and arcs. The star's inner…, Generate an axis-aligned rectangle centred on the origin. Args: width: Total…, Generate an upward-pointing isosceles triangle centred on the origin. Args:…, One source of truth for an organ / tissue *contour* outline, at ``(cx, cy)``.… (+7 more)

### Community 27 - "test_transfusion_network.py"
Cohesion: 0.10
Nodes (26): build_pinaster(), Needle config builders duplicated from ``example/needle/*.py``. These are…, _cell_nodes_by_type(), needle(), fixture, Tests for the needle transfusion-tissue network bridge. See…, cgroups: parenchyma nodes = 17, tracheid nodes = 18, endodermis = 3, xylem = 13., n_total == number of graph nodes; wall ids < n_walls; cell ids >= n_walls +… (+18 more)

### Community 28 - "granap.cell_class.Cell"
Cohesion: 0.22
Nodes (20): granap.cell_class.Cell, granap.cell_class.Cell..init., granap.cell_class.Cell.cell_to_dict, granap.cell_class.Cell.jitter, granap.cell_class.Cell.smooth, granap.cell_manager.CellManager..init., granap.cell_manager.CellManager.add_cell, granap.generate_cell.CellGenerator.generate_cells_info (+12 more)

### Community 29 - "test_vascular_regression.py"
Cohesion: 0.13
Nodes (21): _census(), _check(), dicot_primary(), dicot_secondary(), monocot_arch(), needle_default(), parametrize, Golden regression tests for the root anatomy pipeline. These pin the exact… (+13 more)

### Community 30 - "visualization.py"
Cohesion: 0.14
Nodes (14): build_anatomy_tissues(), _dry_run_vascular(), plot_layers_simple(), plot_section(), plot_tissues(), Any, Figure, GeoDataFrame (+6 more)

### Community 31 - "test_plot_tissues"
Cohesion: 0.67
Nodes (3): _dicot_secondary(), Verify plot_tissues and build_anatomy_tissues for monocot, dicot, and needle:…, test_plot_tissues()

### Community 32 - ".for_dicot_stem"
Cohesion: 0.16
Nodes (16): _circle_uniform(), _continuous(), _drop(), _ellipse_mixed(), main(), Demo: shaping a dicot stem — stem outline, cambium ring, and bundle pattern.…, Remove every param entry with the given name (returns the data)., _star_secondary() (+8 more)

### Community 33 - "LayerManager"
Cohesion: 0.12
Nodes (8): LayerManager, Layer manager module for handling collections of tissue layers., Manages a collection of tissue layers with add/remove operations. This class…, Return number of layers., Make LayerManager iterable., Initialize an empty layer collection., Remove a layer by name. Args: name: Name identifier of the layer to remove…, Get all layers in current order.

### Community 34 - "test_growth_modes.py"
Cohesion: 0.21
Nodes (14): Dicot root preset with secondary growth enabled (single growth ring).…, _census(), dicot_primary(), dicot_secondary(), dicot_woody(), monocot(), Smoke + structure tests for the four canonical organ growth cases. Covers the…, Woody growth thickens the stele and repeats vessel packing, so it has more… (+6 more)

### Community 35 - ".create_from_input"
Cohesion: 0.18
Nodes (9): main(), Gallery: root & needle cells and connectivity networks. Top row shows cells,…, Create OrganInputData from a plain list of dicts (no validation)., Factory method to initialize the appropriate Organ subclass (RootAnatomy or…, Tests for organ construction from XML / param-list input. Visual network…, A monocot root loaded from XML has 3 metaxylem cells and no air spaces., A needle built from a raw param list constructs and produces cells., test_needle_from_param_list_builds() (+1 more)

### Community 36 - "DicotStemAnatomy"
Cohesion: 0.06
Nodes (34): DicotStemAnatomy, pose(), _fits(), _ok(), _place(), Polygon, Radial thickness reserved outside the secondary phloem for the displaced…, Stem outer radius, grown outward to make room for secondary growth. (+26 more)

### Community 37 - "Polygon"
Cohesion: 0.15
Nodes (10): Polygon, Create the base shape for the organ. This method must be implemented by…, Generate or retrieve the base shape. Returns: Base polygon, Regions that are *meant* to hold no cell (so they are not gaps). The base organ…, The union of ``polys`` with its internal pockets filled in. Anything inside…, Detect empty space inside the organ that no cell was assigned to. A gap is the…, Absorb each detected gap into the nearest cell of its own tissue zone. For…, Create vascular tissue by building this organ's vascular recipe. Shared… (+2 more)

### Community 38 - "granap.needle_class.NeedleAnatomy"
Cohesion: 0.39
Nodes (9): granap.needle_class.NeedleAnatomy, granap.needle_class.NeedleAnatomy..init., granap.needle_class.NeedleAnatomy._initialize_default_layers, granap.needle_class.NeedleAnatomy._initialize_default_params, granap.needle_class.NeedleAnatomy._initialize_params, granap.needle_class.NeedleAnatomy._which_layer_for_vascular, granap.needle_class.NeedleAnatomy.set_central_cylinder_params, granap.needle_class.NeedleAnatomy.set_transfusion_params (+1 more)

### Community 39 - "test_monocot_xylem.py"
Cohesion: 0.16
Nodes (18): cell_type_counts(), make_arch_root(), make_star_root(), Tests for monocot arch-mode xylem (metaxylem ring + protoxylem) + pith, and…, Star mode packs xylem vessels into the star and phloem into the valleys., Star mode with a pith: no xylem vessels inside the pith circle, but stele (pith…, Phloem strands fall in the valleys (between arms), not on the arm axes., Arch mode without pith: metaxylem vessels exist. (+10 more)

### Community 40 - "._create_central_layers"
Cohesion: 0.25
Nodes (5): Any, Return tissue zone descriptors (layer rings + vascular polygons). See…, Extra plasmodesmata/membrane network bridges beyond real shared walls. Each…, Calculate anatomical statistics. Returns: Dictionary with statistics, Create central tissue layers (vascular, parenchyma, etc.). Args:…

### Community 41 - "granap.network_base.AbstractNetwork"
Cohesion: 0.29
Nodes (11): granap.network_base, granap.network_base.AbstractNetwork, granap.network_base.AbstractNetwork..init., granap.network_base.AbstractNetwork._build_anatnetwork, granap.network_base.AbstractNetwork._get_walls_for_types, granap.network_base.AbstractNetwork._types_match, granap.network_base.AbstractNetwork.export_to_adjencymatrix, granap.network_base.AbstractNetwork.fill_matrix (+3 more)

### Community 42 - ".__init__"
Cohesion: 0.33
Nodes (3): Parse the parameters that are common to all root types., Parse plant-type-specific vascular parameters. Overridden in subclasses., Initialise root layers from parsed params.

### Community 43 - "granap.root_class.RootAnatomy"
Cohesion: 0.32
Nodes (8): granap.geometry_collection.GeometryProcessor.circle_polygon, granap.root_class.RootAnatomy, granap.root_class.RootAnatomy..init., granap.root_class.RootAnatomy._create_base_shape, granap.root_class.RootAnatomy._initialize_default_layers, granap.root_class.RootAnatomy._which_layer_for_vascular, granap.root_class.RootAnatomy.add_lateral_root_primordium, granap.root_class.RootAnatomy.set_vascular_params

### Community 44 - ".for_root"
Cohesion: 0.14
Nodes (14): _census(), _dicot_primary(), _dicot_secondary(), _geom_hash(), _monocot_arch(), _monocot_default(), Byte-identical safety net for perf work (see performance_proposals.md). Pins…, run() (+6 more)

### Community 45 - ".buffer_polygon"
Cohesion: 0.18
Nodes (11): Smooth the cell polygon., Buffer a polygon with optional smoothing. Args: polygon: Input polygon…, _duct_assembly_radius(), _band(), _seat_duct(), _seat_duct_at_point(), Build one duct's concentric geometry, either at an explicit ``center`` or (when…, Add resin ducts (parenchyma ring + inner lumen) to the mesophyll. Geometry is… (+3 more)

### Community 46 - "LineString"
Cohesion: 0.16
Nodes (10): LineString, _bearing_target_point(), Needle override of the intercellular-space geometry. In the needle mesophyll…, Insert small rhombic air spaces on the walls between mesophyll cells. For every…, Carve thin, full-height air slits across every ``every``-th wall between…, Yield the 1-D wall segments from a cell/cell intersection geometry. Two…, Build a small rhombus centred on ``wall`` and aligned with it. The rhombus is a…, Build a thin rectangle centred on ``wall`` and aligned with it. Same local… (+2 more)

### Community 47 - ".merge_intercellular_aerenchyma"
Cohesion: 0.24
Nodes (6): attempt(), _find(), _union(), Fuse touching air-space cells within the same angular sector, then carve tissue…, Air-space fraction of the tissue band, ``air / (air + tissue)``, measured from…, Place aerenchyma so the realized air proportion matches the request. The fuse-…

### Community 48 - "organ_class.py"
Cohesion: 0.13
Nodes (9): NetworkExporter, # NOTE: this is not a value-preserving bugfix. For every, # IMPORTANT: apply the IDENTICAL valid-geometry filter so that both paths, Class to export Organ anatomy to an AbstractNetwork topological graph., Populate the provided network graph from the cell GeoDataFrame. Algorithm…, Plant anatomy base module providing abstract interface., Rename every cell tagged ``old_tag`` to ``new_tag``. The terminal cell-level…, retag_tissue() (+1 more)

### Community 49 - "vascular_bundle.py"
Cohesion: 0.09
Nodes (47): _anchor_shift(), build_arc_bundle(), _build(), clip(), build_bundle(), BundleResult, _fill_cambium(), _fill_parenchyma() (+39 more)

### Community 50 - "test_secondary_phloem.py"
Cohesion: 0.43
Nodes (6): cell_type_counts(), make_root(), Geometry tests for secondary phloem generation. Visual gallery lives in…, Return (phloem_at_valley, phloem_at_cambium_arm)., test_secondary_phloem_zone_placement(), _zone_geometry_ok()

### Community 51 - "stem_class.py"
Cohesion: 0.06
Nodes (31): _get_arrangement(), _get_planttype(), __getattr__(), _input_params(), Stem anatomy implementation. ``StemAnatomy`` acts as a transparent factory:…, The raw param-dict list from any accepted input form (or None)., Extract the planttype integer (1 = monocot, 2 = dicot) from raw input., Dicot stele arrangement: 'fascicular' (discrete bundle ring) or 'continuous' (a… (+23 more)

### Community 52 - "math_functions.py"
Cohesion: 0.33
Nodes (5): five_pl(), linear(), Normalized shape functions and a rescale wrapper for GRANAP gradients.…, Normalized 5-parameter logistic: f(0) = 1, f(inf) -> 0. Parameters ---------- t…, Normalized linear decrease: f(0) = 1, f(1) = 0, clamped outside [0, 1].…

### Community 53 - "._resolved_dimensions"
Cohesion: 0.25
Nodes (4): Resolve the needle's actual (width, thickness), computing from the layer stack…, Create the half-ellipse shape of a needle cross-section. Returns: Half-ellipse…, Calculate total needle width from layers., Calculate total needle thickness from layers.

### Community 54 - "read_cellset"
Cohesion: 0.11
Nodes (22): Point2, Ring, assemble_rings(), CellSetCell, CellSetSection, _key(), normalize_tag(), _polygon_from_rings() (+14 more)

### Community 55 - "secondary_growth.py"
Cohesion: 0.08
Nodes (40): angular_wedge(), _bisect_bounded(), _bisect_circular(), build_annual_bands(), build_medullar_ray_polygons(), fill_medullar_rays(), fill_phloem_zone(), fill_ray_parenchyma() (+32 more)

### Community 57 - ".for_dicot_root"
Cohesion: 0.09
Nodes (21): Profiling harness for the heavy secondary-growth path (see…, main(), main(), Build a dicot root from scratch (no ``for_*`` preset), tune to nettle cytomine…, main(), Build a dicot root from the ``for_dicot_root`` preset, update to Oak data from…, main(), Build a dicot root from the ``for_dicot_root`` preset, update to renoncule… (+13 more)

### Community 58 - "GRANAP conda recipe (meta.yaml)"
Cohesion: 0.50
Nodes (4): granap conda environment, GRANAP conda recipe (meta.yaml), OpenAlea CI Workflow, openalea/action-build-publish-anaconda openalea_ci.yml

### Community 59 - "test_leaf_slab.py"
Cohesion: 0.08
Nodes (33): _build(), _build_dicot(), _dicot_input(), Checks for the leaf cross-section (LeafAnatomy: monocot slab + dicot)., Palisade under the adaxial (upper) epidermis, spongy above the abaxial one., A dicot leaf vein is the collateral (xylem/cambium/phloem) bundle, not the…, The dicot leaf carries >1 vein size-class, with the widest (midrib) centred., Monocot leaf: epidermis + one uniform mesophyll tissue (no palisade/spongy). (+25 more)

### Community 60 - "LeafAnatomy"
Cohesion: 0.09
Nodes (14): Any, Convert layer to dictionary representation., Create a Layer from dictionary representation., LeafAnatomy, Same as the base, but split each multi-piece layer into one seedable Polygon…, Region veins occupy — the innermost mesophyll band at the mid-plane., Ground (mesophyll) cell diameter the veins sit among — turns on the outer…, The ground tissue the inter-bundle aerenchyma is carved from. (+6 more)

### Community 61 - "test_param_metadata.py"
Cohesion: 0.10
Nodes (19): Tests for the parameter-metadata layer (grouping / tiering / dead-knob lint).…, The derived group sub-models partition the bundle's flat fields exactly., Authoring a bundle via group sub-models yields the identical flat model., sheath / placement / ring_shape are both group names and flat fields; a scalar…, Every VascularBundleParams field (except name) is grouped + tiered., The point of the tiering: a handful of primary knobs, not 69., A mode-gated field set while its mode is off is reported (and the value is…, Params with no registered metadata (everything but the bundle today) are… (+11 more)

### Community 62 - "StemAnatomy"
Cohesion: 0.06
Nodes (20): Any, Polygon, Plant type: 1 = monocot, 2 = dicot., Parse the parameters common to all stem types. The central ground tissue is the…, Optional per-subclass hook for extra vascular parsing. Not needed by the stem…, Initialise stem layers from parsed params., Pith radial size gradient: ``r_norm`` (0 = centre .. 1 = pith edge) -> ground-…, Ground-tissue (pith) cell diameter at radius ``r`` mm from the organ centre,… (+12 more)

### Community 63 - "Heavy configs (secondary growth) — `dicot_nettle` profile (2026-07)"
Cohesion: 0.08
Nodes (24): ① Drop geopandas `dissolve` in `process_voronoi_groups`  *(DONE 2026-07-07 — byte-identical)*, ② Vectorise the remaining per-point containment tests  *(DONE 2026-07-07 — byte-identical)*, ③ Loosen `maximum_inscribed_circle` tolerance  *(est. −2 to −4s, changes golden)*, ④ Batch `Cell.jitter` / seed creation  *(DONE 2026-07-07 — byte-identical, NOT a repro break)*, ⑤ Drop the per-cell DataFrame in `voronoi_diagram`  *(DONE 2026-07-07 — byte-identical)*, Already done (byte-identical — golden unchanged), F1 — Drop `is_valid` / `buffer(0)` on Voronoi cells in `process_voronoi_groups`  *(DONE 2026-07-07)*, Findings — golden-changing levers, now measured on Linux (+16 more)

### Community 64 - "6. Designing a new tissue — the useful functions"
Cohesion: 0.08
Nodes (23): 1. Mental model, 2. Installation & quick start, 3.1 `OrganInputData` — configuration, 3.2 `Organ` — the anatomy object, 3.3 Plotting & inspection, 3.4 Export, 3. The public API, 4. The generation pipeline (+15 more)

### Community 65 - "DicotLeafAnatomy"
Cohesion: 0.17
Nodes (11): DicotLeafAnatomy, _flush(), Polygon, Peel ``region`` inward into ``name`` rings (appended to ``out``); returns the…, (bottom_y, top_y) of the lamina outline at abscissa ``x`` (a vertical cut)., Sclerenchyma girder rays from a bundle to the epidermis. For a bundle whose…, A single ``mesophyll`` tissue fills the whole core., Dicot leaf: ``palisade.n_layers`` columnar palisade files hugging the adaxial… (+3 more)

### Community 66 - ".add_stomata"
Cohesion: 0.14
Nodes (6): The lamina mid-line y(x): the centre of the leaf slab at each ``x``. Two…, dy/dx of the lamina mid-line at ``x`` (central finite difference). Zero on a…, World angle of a vein's abaxial (outward, lower-surface) pole at ``x``. A…, Carve stomata into both epidermes. Epidermis cells are seeded along the outline…, ``n`` evenly spaced prev/curr/next triplets along one epidermis run. ``margin``…, (carve, gc1, gc2, chamber, pore) per stoma from prev/curr/next centroids.

### Community 67 - ".for_monocot_stem"
Cohesion: 0.11
Nodes (16): main(), _monocot_bands(), Demo: radius-dependent vascular bundle bands in the monocot stem. A monocot…, Monocot stem data whose single preset bundle spec is replaced by ``bands``.…, _dicot(), main(), _monocot(), Demo: stem vascular bundle types — TISSUE view (no cells). Same six scenarios… (+8 more)

### Community 68 - "test_stem_bundles.py"
Cohesion: 0.20
Nodes (19): _build(), _census(), _mean_dist(), _mean_x(), _params(), Tests for stem vascular bundles and the hollow (fistular) pith. Topology is…, Bundle + cell-level param dicts from the dicot preset, with overrides., test_amphicribral_phloem_rings_xylem() (+11 more)

### Community 69 - "MonocotLeafAnatomy"
Cohesion: 0.15
Nodes (10): main(), Gallery: monocot vs dicot leaf cross-sections, side by side. * **monocot** —…, A grass-like monocot leaf showing the shape/vein options: a twisted lamina with…, _twisted_girdered_monocot(), main(), Monocot leaf cross-section demo (Phases 1-2). Builds a wide, thin lamina whose…, Monocot leaf preset: spongy / palisade / spongy mesophyll, an even row of…, MonocotLeafAnatomy (+2 more)

### Community 70 - "main"
Cohesion: 0.50
Nodes (4): anatomy_metrics(), main(), Build a wheat-like monocot root from the ``for_root`` preset and plot it.…, Geometry summary of a generated root: root/stele diameters and total xylem…

### Community 71 - "granap.visualization"
Cohesion: 0.67
Nodes (3): granap.visualization, granap.visualization.plot_layers_simple, granap.visualization.plot_section

### Community 72 - "make_secondary_root"
Cohesion: 0.20
Nodes (15): _center(), distinct_rays(), make_secondary_root(), Tests for dicot secondary-growth medullar rays. Covers the constant-tangential-…, Rays reach the outer edge, so the secondary phloem is still produced., Dicot secondary-growth root with a medullar-ray param entry., Number of distinct medullar rays whose cells fall in the radial band…, With rate=0, exactly n_medullar rays are present (no rate-driven extras).… (+7 more)

### Community 75 - "test_stem_features.py"
Cohesion: 0.19
Nodes (14): _bundles(), _pattern_stem(), Minimal coverage for the newer dicot/monocot stem features: * mixed-kind…, An 'even' band whose ``radius`` exceeds the pith radius carries its bundles out…, A dicot eustele with two bundle kinds arranged by a bundle_pattern., Equal (distance) spacing places the whole tiled sequence: 2 kinds x 3 = 6., grouped' spacing on a 3-arm star clusters the sequence into 3 lobes with empty…, A bicollateral eustele lays a cambium band on both radial sides of the xylem… (+6 more)

### Community 80 - "_local_envelope"
Cohesion: 0.12
Nodes (14): Teardrop / 'violin' shape: an asymmetric oval whose widest point (half width…, Map local-frame geometries (radial axis = local +y) to their place. Applies the…, The bundle footprint placed at ``(cx, cy)`` oriented radially at ``theta``.…, Dicot stem anatomy (eustele). ``DicotStemAnatomy`` builds a dicot stem: a…, bundle_cambium_anchor(), bundle_layout(), _local_envelope(), outer_sheath_mask_pad() (+6 more)

### Community 81 - "DicotRootSeries"
Cohesion: 0.24
Nodes (7): DicotRootSeries, Longitudinal series of **dicot**-root sections with identity-tracked primary…, Pith radius at ``length``, snapped to exactly 0 below a tiny epsilon — a sub-…, The sample where the pith is smallest — the star is fully packed there, so it…, Generate the reference (smallest-pith) section with the normal dicot packer and…, The packed primary-xylem vessels of a reference section as ``(x, y, radius)``.…, The vessels present at ``length`` with their pith-capped positions + sizes. A…

### Community 82 - "cellset_organ.py"
Cohesion: 0.11
Nodes (21): Polygon, Hybrid anatomy: real CellSet segmentation + a GRANAP-generated stele. Why this…, ``(annular_cell, region, inner_cells)`` for every region to rebuild. ``region``…, Clip the donor to ``region``, weld the interface, seat the real vessels., The section outline: the union of every real cell. Deliberately *not*…, absorb_residual(), carve_cells(), conform_boundaries() (+13 more)

### Community 83 - "RootSeries"
Cohesion: 0.21
Nodes (6): Longitudinal series of monocot-root sections with identity-tracked metaxylem…, The k metaxylem centres the monocot class would place in a stele of radius R…, Identity of the vessels at ``length`` (no final position yet): the fused groups…, Assign each vessel (by its current position ``reps[i]``) to a distinct slot,…, Actual vessel positions per length, walking from the highest length to the…, RootSeries

### Community 84 - "test_root_series.py"
Cohesion: 0.14
Nodes (17): _dicot_series(), Root developmental series (ROOT_SERIES_PLAN) — Phase 1: prescription plumbing.…, A dicot section with a receded pith places its prescribed vessels as tracked…, The new track_id column exists and is null for an ordinary (untracked) root., Prescribed vessels appear as metaxylem cells at their positions, each carrying…, Same prescribed set -> same tracked geometry (identity is stable)., The primordial set (positions + 5PL targets) is extracted from the smallest-…, As the pith recedes (apex -> collet) the number of differentiated vessels only… (+9 more)

### Community 85 - "test_stem_continuous.py"
Cohesion: 0.23
Nodes (12): _n_components(), Tests for the continuous (non-fascicular) dicot stem vascular cylinder. Unlike…, Inner / outer radius (from the organ centre) of the registered xylem annulus., (#vessels, #parenchyma) cells whose centre falls in the xylem annulus band., The 'files' layout keeps all three tissue *regions* continuous (the strips are…, _stem(), test_continuous_stem_generates_all_tissues(), test_files_line_up_xylem_only() (+4 more)

### Community 86 - ".for_dicot_leaf"
Cohesion: 0.16
Nodes (12): build_dicot_leaf(), main(), Dicot leaf cross-section demo (dorsiventral) — midrib bundle designs. Palisade…, A near-constant-thickness dicot lamina with one big central midrib. An ellipse…, _air_space_network_elements(), main(), plot_air_link_network(), Gallery: full feature set (layers / cells / network / air-link network) for one… (+4 more)

### Community 87 - "._build_layer_polygons"
Cohesion: 0.13
Nodes (9): Figure, Point, Plot layer boundaries. Args: show: Whether to display the plot Returns:…, Plot cell geometries. Args: show: Whether to display the plot Returns:…, Plot every tissue zone before placing any cell. See…, Generate polygons for all layers. Returns: List of layer polygon dictionaries, Build layer polygons from current layer configuration., Shrink/expand one ring layer's polygon by ``distance``. Default implementation:… (+1 more)

### Community 88 - "Tissue refactor — the shape-first model"
Cohesion: 0.18
Nodes (10): How an organ uses it, Invariants for safe refactoring, Status, Test harness, The cells-first engine (unchanged, lives in `Organ.generate_cells`), The core idea: a tissue **is a region**, The seeding idiom: `Cell.radial`, The vocabulary (`openalea/granap/tissue_builder.py`) (+2 more)

### Community 89 - "Roadmap — toward simplified recipes for monocot, dicot, needle"
Cohesion: 0.18
Nodes (10): Dependency order, Guardrails, P0 — Safety net for needle  *(DONE 2026-06-24)*, P1 — Recipe ergonomics: region→fill as a first-class step  *(DONE 2026-06-24)*, P2 — Special-function vocabulary  *(DONE 2026-06-24; companion cells deferred)*, P3 — Needle into the recipe model  *(DONE 2026-06-24)*, P4 — Unify the three recipes  *(DONE 2026-06-24)*, P5 — Optional layout/cleanup follow-ups  *(CLOSED — not pursued, 2026-06-24)* (+2 more)

### Community 90 - "NeedleAnatomy"
Cohesion: 0.08
Nodes (20): NeedleAnatomy, Any, Needle cross-sectional anatomy. Implements the specific structure of gymnosperm…, Compute stomata geometry from triplet positions without placing cells.…, Initialize needle anatomy. Args: input_data: Parameter data (OrganInputData,…, Initialize central layers., Fill the reserved transfusion-tissue zone by circle-packing. Only runs when…, Prune any layer carrying a ``zone_angles`` entry down to only the cells whose… (+12 more)

### Community 91 - "Polygon"
Cohesion: 0.10
Nodes (13): place(), Point, Polygon, The part of ``annulus`` on a given bearing -- the explicit-placement…, Circularly interpolate a list of ``(angle_degrees, multiplier)`` control points…, Oriented probe rectangle for :meth:`_clear_hypodermis_under_chambers`. Built…, Needle override: dispatch to an angle-varying offset when the layer dict…, When "central_cylinder" has shape="ellipse", morph the layer polygons between… (+5 more)

### Community 92 - "Proposal"
Cohesion: 0.20
Nodes (9): Alternatives considered, Implementation sketch, New params (`VascularBundleParams`), Proposal, Status, Suggested dicot-stem preset, Vascular-bundle sclerenchyma caps — per-pole layer control, What already exists (+1 more)

### Community 93 - ".pole_and_corner_angles"
Cohesion: 0.11
Nodes (19): _air_space_network_elements(), build_gallery_needle_data(), main(), plot_air_link_network(), Return the nodes/edges of the network graph that belong to air-space cells.…, Draw the full network faded, then highlight the air-space nodes/edges., Build the ``OrganInputData`` for the feature-showcase needle., build_nigra() (+11 more)

### Community 94 - ".__init__"
Cohesion: 0.24
Nodes (4): Schedule, A ``(start, end)`` tuple -> linear ramp from the first to the last sampled…, The N primordial ids in ring (angular) order, the non-fusers (terminators +…, Distance-based merge tree over the fuser positions ``pos`` ({id: (x, y)}): for…

### Community 95 - "root_series.py"
Cohesion: 0.22
Nodes (8): _fill_poly(), _fmt_members(), Root developmental series with identity-tracked xylem vessels…, Label of the primordial ids inside a (possibly fused) vessel — the exact ids…, Read one numeric field off a config param entry (model or raw dict), or…, Fill a (possibly Multi)Polygon with a solid colour, on top of the tissue., Render the whole series as a grid, each tracked vessel in a fixed colour by its…, _read_field()

### Community 96 - "CellSetOrgan"
Cohesion: 0.11
Nodes (11): CellSetOrgan, Any, GeoDataFrame, An :class:`Organ` whose cells come from a CellSet file, with unresolved regions…, Read the file and turn every digitized cell into a :class:`Cell`., Orientation (degrees) for the donor's xylem plate., Import the real cells, regenerate the unresolved regions, export a gdf.…, Wall-sharing summary of the exported network — the graft's acceptance test.… (+3 more)

### Community 97 - "test_param_schema_equivalence.py"
Cohesion: 0.36
Nodes (8): _emit(), _load(), _norm(), Schema-equivalence guard for the parameter-schema unification. The parameter…, Every preset -> its ``to_dict_list()`` as a list of name-keyed dicts (the exact…, JSON-normalise for a value-only compare (tuples -> lists, as JSON round-trips)., test_preset_param_dicts_unchanged(), _write_golden()

### Community 98 - "series_monocot_wheat.py"
Cohesion: 0.36
Nodes (7): build_series(), build_wheat_base(), main(), n_fused(), Wheat monocot root — a developmental *series* (apex -> collet) with tracked…, A wheat 'Salmone' monocot-root template (the tissue that refits around the…, Number of FUSED metaxylem along the root: ``APEX_META`` at the apex (the fusion…

### Community 99 - "RootSeriesResult"
Cohesion: 0.25
Nodes (3): Per-section results + the vessel track table (with fusion membership)., Every (length, track_id, members) the primordial ``p`` belongs to — i.e. which…, RootSeriesResult

### Community 100 - "._finalize_pith_and_lacunae"
Cohesion: 0.29
Nodes (4): Add cortex air spaces, then tidy the hollow cavity + the lacunae. The pith…, True only when aerenchyma is explicitly requested for the pith. (An…, Largest Polygon piece of a (possibly Multi)Polygon, or None., Post-Voronoi tidy-up of the pith centre and the protoxylem lacunae. The solid…

### Community 101 - "default_dicot_leaf_params"
Cohesion: 0.11
Nodes (17): model_validator, default_dicot_leaf_params(), default_monocot_leaf_params(), InterCellularSpacesParams, _LeafGroundParams, LeafMesophyllParams, LeafPalisadeParams, LeafSpongyParams (+9 more)

### Community 102 - "monocot_snowdrop.py"
Cohesion: 0.47
Nodes (5): build_snowdrop(), _bundle(), main(), Snowdrop (*Galanthus*) leaf cross-section — a measured monocot leaf. All…, A snowdrop face bundle: protoxylem + phloem, no metaxylem, no sclerenchyma.

### Community 103 - "make_root"
Cohesion: 0.47
Nodes (5): cell_type_counts(), main(), make_root(), Gallery: monocot xylem modes — 'default' (metaxylem ring), 'arch', and 'star'.…, Build a monocot root in the given xylem ``shape`` with per-layer overrides.

### Community 104 - "build_dicot_base"
Cohesion: 0.47
Nodes (5): build_dicot_base(), build_series(), main(), Dicot root — a primary-growth *series* (apex -> collet) with tracked xylem.…, A dicot-root template (the tissue that refits around the tracked vessels).…

### Community 105 - "transfusion_network.py"
Cohesion: 0.18
Nodes (18): _cell_nodes_by_type(), _draw_symplast_panel(), main(), plot_apoplastic_isolation(), plot_symplast_subgraph(), plot_symplast_zoom(), plot_transfusion_cells(), print_report() (+10 more)

### Community 106 - "TissueStep"
Cohesion: 0.12
Nodes (8): One named step of a tissue-build recipe. A step is just a label, the tags it is…, Add a step that fills one region ``tissue`` by ``strategy``. ``record(tissue,…, Add a step that fills several regions ``tissues`` by ``strategy``. ``tissues``…, Add a cell/group-level cleanup step (produces nothing new)., Add a bespoke placement step (sheath, bundles, ...) that isn't a plain fill., _run(), _run(), TissueStep

### Community 107 - "describe_params"
Cohesion: 0.12
Nodes (17): describe_params(), _field_is_active(), _field_kind(), _field_tier(), _preset_moved_fields(), _preset_names(), Is a field meaningful given the current param values? (its mode gate holds)., classname -> {fields any preset sets to a non-default} (the primary set). (+9 more)

### Community 108 - "Test P. pinaster anatomy"
Cohesion: 0.40
Nodes (4): 28.08.2026, 31.08.2026, Open / for review, Test P. pinaster anatomy

### Community 109 - "brassicaceae_roots.py"
Cohesion: 0.20
Nodes (14): build(), census(), check_topology(), contact_sheet(), draw(), main(), Any, Template: a batch of CellSet-digitized Brassicaceae roots -> MECHA-ready XML.… (+6 more)

### Community 110 - "build"
Cohesion: 0.50
Nodes (4): build(), main(), Build a default monocot root in each base shape; plot tissues and cells., Default monocot root with the given base_shape param.

### Community 111 - "make_dicot_root"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_dicot_root(), Gallery: dicot root — star-shaped xylem with Apollonian packing.

### Community 112 - "make_root"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_root(), Gallery: dicot root — secondary phloem generation.

### Community 113 - "make_secondary_root"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_secondary_root(), Smoke test and visualisation for secondary xylem / secondary growth.

### Community 114 - "main"
Cohesion: 0.50
Nodes (4): anatomy_metrics(), main(), Build a iris-like monocot root from the ``for_root`` preset and plot it.…, Geometry summary of a generated root: root/stele diameters and total xylem…

### Community 115 - "_continuous"
Cohesion: 0.40
Nodes (4): _continuous(), main(), Demo: fascicular eustele vs the continuous (non-fascicular) vascular cylinder.…, Continuous-cylinder preset with the xylem layout set. ``n_xylem_files=0``…

### Community 116 - "_dicot"
Cohesion: 0.40
Nodes (4): _dicot(), main(), Demo: dicot stem growth — primary bundles vs the closed cambium ring. The…, Dicot stem preset with the secondary-growth flag set.

### Community 117 - "Any"
Cohesion: 0.13
Nodes (16): _build_bundle_field_meta(), _dicot_leaf_vein(), LeafVascularBundleParams, _monocot_leaf_vein(), Any, The monocot 'face' vein (metaxylem eyes + protoxylem + a lacuna), leaf-scaled —…, A dicot collateral vein (xylem / cambium / phloem in radial files, no metaxylem…, Vessel diameter + its min floor + per-vessel jitter (every xylem uses these). (+8 more)

### Community 118 - "dicot_nerium.py"
Cohesion: 0.67
Nodes (3): build_nerium(), main(), Oleander (*Nerium oleander*) leaf cross-section — a measured dicot leaf. All…

### Community 119 - "GRANAR package translated in Python"
Cohesion: 0.50
Nodes (3): Code structure, GRANAR package translated in Python, Installation

### Community 120 - "._create_base_shape"
Cohesion: 0.14
Nodes (7): Half the lamina thickness at each ``xg``. Default is the ellipse profile.…, Half the lamina thickness at ``x``., The lamina outline: a wide, thin lens, ribbed where each vein thickens it. The…, Along-surface smoothing length (mm) for the outline: the ``outline_smooth``…, Gaussian-smooth the adaxial/abaxial surface profiles along x, rounding the…, Round sharp convex features (the pointed margins/tips and a narrow rib crest)…, The vein's y, shifted by its ``relative_distance`` toward a face: 0.5 = mid-…

### Community 121 - "Architecture"
Cohesion: 0.13
Nodes (13): Architecture, CellSet ingestion (`cellset_reader.py` + `cellset_organ.py`), Commands, graphify, Known stale/fragile things worth knowing before touching them, Organ family tree, Project, Recipe-dict input model (+5 more)

### Community 122 - "._vein_layout"
Cohesion: 0.14
Nodes (8): _leaf_planttype(), Any, All ``vascular_bundle`` specs (vein size-classes), widest first so a big vein…, ``(x, spec)`` for every vein along the mid-plane — the single source for the…, x of each vein (any size-class)., Lay the transverse vein row, then (dicot) the columnar palisade files., Seed the columnar palisade as ``n_layers`` horizontal cell files hugging the…, planttype (1 = monocot, 2 = dicot) from any accepted input form.

### Community 156 - "_grow_bundle_sheath"
Cohesion: 0.17
Nodes (11): _grow_bundle_sheath(), _majority_tissue_name(), _place_ring(), Normalise ``ground_tissue_name`` (callable ``f(x, y) -> name`` or None) into a…, Tissue name occupying most of a sheath cell centred at ``(px, py)`` of radius…, Seed one round cell as a ring of ``n_border`` (>= 8) border points — one…, Place a single outer bundle-sheath ring and return the removal mask (footprint…, Normalise ``ground_cell_size`` (scalar, callable ``f(x,y)``, or None) into a… (+3 more)

### Community 158 - ".resample_coords"
Cohesion: 0.20
Nodes (7): ndarray, Point, Resample coordinates to have uniform spacing. Args: coords: Array of (x, y)…, Smooth coordinates using Laplacian smoothing. Args: coords: Array of (x, y)…, Buffer a polygon boundary by a per-angle distance around ``center``, instead of…, Generate points along an ellipse boundary. Args: center: (x, y) center…, Build the star-shaped xylem *region* (pure geometry). Shape-first: the star is…

### Community 159 - ".fit_inner_ellipse"
Cohesion: 0.20
Nodes (4): Fit an inner ellipse to a polygon. ``angle`` (degrees), when given, overrides…, Translate ``ellipse`` (a :func:`fit_inner_ellipse`-style dict) along…, Finds the approximate center of the Maximum Inscribed Circle (Pole of…, Return (cx, cy, radius) of the largest circle that fits inside *polygon*. Uses…

### Community 160 - ".oriented_ellipse"
Cohesion: 0.20
Nodes (7): Axis-aligned unit disc scaled to ``width`` x ``height``, rotated so its major…, Place every vein transversely along the mid-plane, each built from its own…, Compute the gap region **between** each pair of adjacent veins (the monocot…, partition_concentric(), _phloem_ellipse(), Split ``env`` into a central core ellipse and the surrounding ring., Phloem sieve-cluster ellipse, placed along ``axis`` between radial offsets…

### Community 161 - ".add_aerenchyma"
Cohesion: 0.20
Nodes (3): Denominator for the per-quadrant aerenchyma target area. Override in subclasses., Order ``cells`` so that every prefix is a connected region, growing outward…, Generate aerenchyma in the tissue defined in aerenchyma_params.…

### Community 162 - "build_gallery_needle_data"
Cohesion: 0.22
Nodes (9): _is_duct_home_layer(), build_gallery_needle_data(), Build the ``OrganInputData`` for the feature-showcase needle., _assert_real_config_ducts(), Shared checks for a real config's ``NeedleAnatomy`` instance. Always hard-…, test_gallery_ducts_clear_boundaries_and_keep_full_size(), test_pinaster_ducts_clear_boundaries_and_keep_full_size(), needle_features() (+1 more)

### Community 163 - "._build_donor"
Cohesion: 0.25
Nodes (6): _make(), spec(), default_stele_input(), A dicot-root donor configuration scaled to a stele of radius ``radius`` (mm).…, Inscribed radius of the polygon the donor's vascular recipe will get.…, Generate a stele on ``region``, rotated so its xylem plate follows ``angle``.

### Community 164 - "_as_shape"
Cohesion: 0.25
Nodes (5): _as_shape(), Accept a :class:`Tissue` or a raw shapely geometry; return the geometry., Subtract another region (Tissue or shapely geometry) from this one., Clip this region to another (Tissue or shapely geometry)., Merge another region (Tissue or shapely geometry) into this one.

### Community 165 - "cellset_hybrid.py"
Cohesion: 0.43
Nodes (6): build(), main(), Mix real CellSet data with a GRANAP stele, and export it for MECHA. A CellSet-…, Per-tissue count *and* area -- cell counts hide size/proportion errors., render(), report()

### Community 166 - ".expand_layers"
Cohesion: 0.29
Nodes (4): Any, Expand layers with n_layers > 1 into individual layer entries. Used for…, Get parameters of all layers. Returns: List of dictionaries with layer…, Get layers sorted by their order attribute. Args: reverse: If True, sort…

### Community 167 - "_blunt_margin_input"
Cohesion: 0.29
Nodes (7): _blunt_margin_input(), _longest_extent(), Dicot leaf ending in a blunt rounded margin instead of a thin taper. Full…, ``fuse_gaps`` absorbs the holes the tissue closes around and leaves the rest.…, Length of the long side of a cell's minimum rotated rectangle., test_fusion_fills_enclosed_holes_but_not_the_surface_ribbon(), longest_epidermis()

### Community 168 - "solve"
Cohesion: 0.40
Nodes (5): main(), Any, Estimate radial hydraulic conductivity (kr) for every hybrid section, in MECHA.…, Run MECHA on one section and pull out kr per maturity stage., solve()

### Community 169 - ".add_layer"
Cohesion: 0.33
Nodes (3): Add a layer to the collection. Args: layer: Layer object to add position:…, Retrieve a layer by name. Args: name: Name identifier of the layer Returns:…, Check if a layer exists.

### Community 170 - "Cross-platform determinism figures (golden regressions)"
Cohesion: 0.33
Nodes (5): A note on reading the seed figures, Cross-platform determinism figures (golden regressions), Figures, Result, The fixes

### Community 171 - "make_dicot_root"
Cohesion: 0.40
Nodes (5): make_dicot_root(), Tests for dicot star-shaped xylem with Apollonian packing. Visual scenario…, Construct a dicot RootAnatomy with custom xylem and optional cambium parameters., Every scenario yields xylem cells, and no xylem cell is smaller than…, test_dicot_star_xylem_size_classification()

### Community 172 - "main"
Cohesion: 0.50
Nodes (4): anatomy_metrics(), main(), Build a maize-like monocot root from the ``for_root`` preset and plot it.…, Geometry summary of a generated root: root/stele diameters and total xylem…

### Community 173 - "DicotCambiumParams"
Cohesion: 0.40
Nodes (5): BundleCambiumParams, DicotCambiumParams, LeafBundleCambiumParams, Cambium cell sizing for a vascular bundle — the fascicular cambium of a dicot…, Cambium cell sizing for dicot leaf veins (ignored by closed monocot veins).

### Community 174 - ".export_to_adjencymatrix"
Cohesion: 0.40
Nodes (3): lil_matrix, Populate ``self.graph`` with wall, junction and cell nodes, and connect them…, Build the network (if needed) and return the sparse adjacency matrix.…

## Knowledge Gaps
- **134 isolated node(s):** `openalea.granap`, `Project`, `Setup & environment`, `Commands`, `graphify` (+129 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1046 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **44 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OrganInputData` connect `OrganInputData` to `Tissue`, `RootAnatomy`, `input_data.py`, `Layer`, `Organ`, `CellGenerator`, `test_vascular_regression.py`, `test_plot_tissues`, `.for_dicot_stem`, `test_growth_modes.py`, `.create_from_input`, `._build_donor`, `build_gallery_needle_data`, `_blunt_margin_input`, `test_monocot_xylem.py`, `make_dicot_root`, `.for_root`, `main`, `organ_class.py`, `test_secondary_phloem.py`, `stem_class.py`, `_get_planttype`, `.for_dicot_root`, `test_leaf_slab.py`, `LeafAnatomy`, `test_param_metadata.py`, `StemAnatomy`, `DicotLeafAnatomy`, `.for_monocot_stem`, `test_stem_bundles.py`, `MonocotLeafAnatomy`, `main`, `make_secondary_root`, `test_stem_features.py`, `DicotRootSeries`, `cellset_organ.py`, `RootSeries`, `test_root_series.py`, `test_stem_continuous.py`, `.for_dicot_leaf`, `NeedleAnatomy`, `.pole_and_corner_angles`, `.__init__`, `root_series.py`, `CellSetOrgan`, `test_param_schema_equivalence.py`, `series_monocot_wheat.py`, `make_root`, `build_dicot_base`, `build`, `make_dicot_root`, `make_root`, `make_secondary_root`, `main`, `_continuous`, `_dicot`, `._vein_layout`?**
  _High betweenness centrality (0.198) - this node is a cross-community bridge._
- **Why does `NeedleAnatomy` connect `NeedleAnatomy` to `Tissue`, `OrganInputData`, `Cell`, `TissueRecipe`, `CellManager`, `PolygonInterpolator`, `LayerPolygon`, `Layer`, `Organ`, `test_resin_duct_placement.py`, `CellGenerator`, `GeometryProcessor`, `test_transfusion_network.py`, `test_vascular_regression.py`, `test_plot_tissues`, `build_gallery_needle_data`, `.create_from_input`, `.buffer_polygon`, `LineString`, `organ_class.py`, `._resolved_dimensions`, `Polygon`, `.pole_and_corner_angles`, `transfusion_network.py`?**
  _High betweenness centrality (0.180) - this node is a cross-community bridge._
- **Why does `Organ` connect `Organ` to `OrganInputData`, `Cell`, `TissueRecipe`, `RootAnatomy`, `CellManager`, `AnatomyWriter`, `LayerPolygon`, `AbstractNetwork`, `Layer`, `.generate_cells`, `CellGenerator`, `GeometryProcessor`, `visualization.py`, `LayerManager`, `.add_aerenchyma`, `.create_from_input`, `._build_donor`, `Polygon`, `._create_central_layers`, `.merge_intercellular_aerenchyma`, `organ_class.py`, `.allocate_vascular_tissue`, `.__init__`, `stem_class.py`, `LeafAnatomy`, `StemAnatomy`, `MonocotLeafAnatomy`, `cellset_organ.py`, `._build_layer_polygons`, `NeedleAnatomy`, `CellSetOrgan`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Are the 102 inferred relationships involving `OrganInputData` (e.g. with `_dicot_primary()` and `_dicot_secondary()`) actually correct?**
  _`OrganInputData` has 102 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `RootAnatomy` (e.g. with `_dicot_primary()` and `_dicot_secondary()`) actually correct?**
  _`RootAnatomy` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Organ` (e.g. with `main()` and `AnatomyWriter`) actually correct?**
  _`Organ` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `NeedleAnatomy` (e.g. with `main()` and `build_gallery_needle_data()`) actually correct?**
  _`NeedleAnatomy` has 37 INFERRED edges - model-reasoned connections that need verification._