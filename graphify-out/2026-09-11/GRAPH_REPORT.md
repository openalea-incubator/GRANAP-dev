# Graph Report - GRANAP  (2026-09-01)

## Corpus Check
- 96 files · ~294,020 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2040 nodes · 3757 edges · 158 communities (115 shown, 43 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 409 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f289bb19`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Tissue Region Geometry
- Dicot Root Anatomy
- OrganInputData Model
- Cell Class
- Perf & Tissue Refactor Docs
- TissueRecipe Steps
- Callgraph Generator
- Root Feature Demos
- LayerManager (callgraph)
- CellManager
- Anatomy Writer / Export
- Input Params Classes
- Shape Interpolation
- RoiOrgan / LayerPolygon
- AbstractNetwork
- Monocot Root Anatomy
- Cell Border Generation
- Organ Base Class
- Cell/Geometry (callgraph)
- Organ (callgraph)
- GeometryProcessor
- Needle Anatomy
- Central Layers & Xylem Star
- Cell/Layer Generation & Plot
- CellManager (callgraph)
- CellGenerator (Voronoi)
- Base Shape Polygons
- Medullar Rays & Secondary Xylem
- Cell Build Flow (callgraph)
- Vascular Regression Tests
- Visualization / plot_tissues
- Dicot Secondary Params
- Root Params & XML Parse
- LayerManager Class
- Growth Modes Tests
- Organ Factory & Network Tests
- Layer Class
- Vascular Recipe (Organ)
- NeedleAnatomy (callgraph)
- Monocot Xylem Tests
- Layer Polygons & Statistics
- AbstractNetwork (callgraph)
- Layer Dict & Needle Init
- RootAnatomy Factory (callgraph)
- Perf Characterize
- Stomata
- Buffer/Union & Resin Duct
- Aerenchyma & Intercellular Spaces
- NetworkExporter
- Layer Expansion
- Secondary Phloem Tests
- Layer Add/Get
- Math Shape Functions
- Needle Base Shape
- Transfusion Layers
- Layer Add/Remove
- RootAnatomy Factory
- Dicot Star Xylem Tests
- Conda/CI Packaging
- Tomato Dicot Example
- Monocot Base Shapes Demo
- Iris Monocot Example
- Maize Monocot Example
- Wheat Monocot Example
- Monocot Xylem Gallery
- Dicot Root Gallery
- Secondary Phloem Gallery
- Secondary Xylem Demo
- Needle Vascular Recipe
- Perf Profiling (Nettle)
- Medullar Rays Demo
- Visualization (callgraph)
- Oak Dicot Example
- Perf Dissolve Removal
- Package Init
- Resin Canal
- Network Graph Object
- granap Package
- Roadmap P5 Followups
- openalea.granap Namespace
- Pinus Params Demo
- DicotRootSeries
- ._create_central_layers
- RootSeries
- test_root_series.py
- test_stem_continuous.py
- .for_dicot_leaf
- .buffer_polygon
- Tissue refactor — the shape-first model
- Roadmap — toward simplified recipes for monocot, dicot, needle
- ._directional_stomata_triplets
- ._offset_layer_polygon
- Proposal
- build_gallery_needle_data
- .__init__
- root_series.py
- .pole_and_corner_angles
- test_param_schema_equivalence.py
- series_monocot_wheat.py
- RootSeriesResult
- ._finalize_pith_and_lacunae
- bundle_cambium_anchor
- monocot_snowdrop.py
- make_root
- build_dicot_base
- ._convert_inter_bundle_aerenchyma
- .__init__
- main
- Test P. pinaster anatomy
- anatomy_metrics
- build
- gallery_dicot_features.py
- gallery_secondary_phloem.py
- make_secondary_root
- anatomy_metrics
- gallery_continuous_cylinder.py
- gallery_dicot_growth.py
- retag_tissue
- dicot_nerium.py
- GRANAR package translated in Python
- ._half_profile
- main
- gallery_vascular_band_packing.py
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
- GRANAP (GRANAR in Python)

## God Nodes (most connected - your core abstractions)
1. `RootAnatomy` - 95 edges
2. `CellManager` - 79 edges
3. `Organ` - 78 edges
4. `LeafAnatomy` - 67 edges
5. `OrganInputData` - 64 edges
6. `NeedleAnatomy` - 63 edges
7. `StemAnatomy` - 59 edges
8. `GeometryProcessor` - 51 edges
9. `Cell` - 50 edges
10. `TissueRecipe` - 50 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `LeafAnatomy`  [INFERRED]
  example/leaf/gallery_leaf.py → src/openalea/granap/leaf_class.py
- `main()` --calls--> `StemAnatomy`  [INFERRED]
  example/stem/gallery_continuous_cylinder.py → src/openalea/granap/stem_class.py
- `main()` --calls--> `StemAnatomy`  [INFERRED]
  example/stem/gallery_dicot_growth.py → src/openalea/granap/stem_class.py
- `main()` --calls--> `StemAnatomy`  [INFERRED]
  example/stem/gallery_stem_shapes.py → src/openalea/granap/stem_class.py
- `main()` --calls--> `StemAnatomy`  [INFERRED]
  example/stem/gallery_vascular_band_packing.py → src/openalea/granap/stem_class.py

## Import Cycles
- None detected.

## Communities (158 total, 43 thin omitted)

### Community 0 - "Tissue Region Geometry"
Cohesion: 0.08
Nodes (23): Record only the wide ("xylem") vessels in ``vascular_polygons``.          The, _as_shape(), Accept a :class:`Tissue` or a raw shapely geometry; return the geometry., A tagged anatomical region: a shapely shape plus the tag its cells take., Smooth the region boundary (pre-fill, pure geometry)., Subtract another region (Tissue or shapely geometry) from this one., Clip this region to another (Tissue or shapely geometry)., Merge another region (Tissue or shapely geometry) into this one. (+15 more)

### Community 1 - "Dicot Root Anatomy"
Cohesion: 0.08
Nodes (23): DicotRootAnatomy, Polygon, Dicot root anatomy.  ``DicotRootAnatomy`` builds a dicot stele: star-shaped pr, The angular wedges (one per vascular peak) the secondary-xylem vessels, Build secondary xylem between the primary and secondary cambium., Build secondary phloem outside the cambium.          The phloem occupies a ban, Place sieve tubes (+ companion cells when alive=True), then fill the         pa, Declarative description of how a dicot stele is assembled.          Built and (+15 more)

### Community 2 - "OrganInputData Model"
Cohesion: 0.06
Nodes (32): BaseModel, _build_bundle_field_meta(), _field_is_active(), OrganInputData, _preset_moved_fields(), _preset_names(), Any, Is a field meaningful given the current param values? (its mode gate holds). (+24 more)

### Community 3 - "Cell Class"
Cohesion: 0.09
Nodes (22): Cell, Polygon, Build a seed cell whose ``angle``/``radius`` are measured from ``center``., Jitter the cell position.          Uses ``rng`` (the organ's seeded generator), Smooth the cell polygon., carve_and_insert(), consider_as_cell(), place_resin_duct() (+14 more)

### Community 5 - "TissueRecipe Steps"
Cohesion: 0.07
Nodes (15): Needle organ-specific tissues as a recipe of P2 special-tissue steps., One named step of a tissue-build recipe.      A step is just a label, the tags, An ordered, inspectable sequence of :class:`TissueStep` objects.      Beyond t, Set the target CellManager + rng used by :meth:`fill` / :meth:`fill_each`., Add a step that fills one region ``tissue`` by ``strategy``.          ``record, Add a step that fills several regions ``tissues`` by ``strategy``.          ``, Add a cell/group-level cleanup step (produces nothing new)., Add a bespoke placement step (sheath, bundles, ...) that isn't a plain fill. (+7 more)

### Community 6 - "Callgraph Generator"
Cohesion: 0.10
Nodes (25): AsyncFunctionDef, ClassDef, build_dot_source(), build_html(), clean_docstring(), collect_docstrings(), DocCollector, ensure_graphviz_available() (+17 more)

### Community 7 - "Root Feature Demos"
Cohesion: 0.08
Nodes (17): main(), main(), Build a dicot root from the ``for_dicot_root`` preset, update to Oak data from B, anatomy_metrics(), main(), Build a maize-like monocot root from the ``for_root`` preset and plot it.  Con, Geometry summary of a generated root: root/stele diameters and total     xylem, Polygon (+9 more)

### Community 8 - "LayerManager (callgraph)"
Cohesion: 0.14
Nodes (29): granap.layer_class, granap.layer_class.Layer, granap.layer_class.Layer..post_init., granap.layer_class.Layer..repr., granap.layer_class.Layer.from_dict, granap.layer_class.Layer.get_total_thickness, granap.layer_class.Layer.to_dict, granap.layer_manager (+21 more)

### Community 9 - "CellManager"
Cohesion: 0.09
Nodes (7): CellManager, Polygon, Recalculate the properties of all cells in the list., Drop every cell whose seed point intersects ``polygon`` (bulk predicate)., Remove exactly these cell objects, matched by identity.          Unlike :meth:, Next free ``id_group`` for appending a new cell group.          ``get_last_id_, Drop every cell that intersects ``polygon``.          A cell's footprint is it

### Community 10 - "Anatomy Writer / Export"
Cohesion: 0.08
Nodes (14): main(), Manual testing script for RoiOrgan behavior.     Update the folder_path to poin, AnatomyWriter, write the MECHA geometry files (retro-compatible), Write a .obj from the generated cross section geometry.         If membrane is, Write a .svg from the generated cross section geometry.         Uses prep_geo l, Pre-proc for .geo file generation.         Returns list of shrunken inner lumin, Class to export Organ anatomy to various formats (XML, OBJ, GEO). (+6 more)

### Community 11 - "Input Params Classes"
Cohesion: 0.07
Nodes (52): BaseParams, BaseShapeParams, _build_bundle_group_models(), CambiumParams, CentralCylinderParams, default_dicot_leaf_params(), default_monocot_leaf_params(), _dicot_leaf_vein() (+44 more)

### Community 12 - "Shape Interpolation"
Cohesion: 0.16
Nodes (11): as_tuple(), interpolate_point_points(), interpolate_poly(), midpoints(), normals_offset(), plot_poly_normals(), point_derivative(), PolygonInterpolator (+3 more)

### Community 13 - "RoiOrgan / LayerPolygon"
Cohesion: 0.08
Nodes (14): LayerPolygon, Typed representation of a layer polygon produced by _build_layer_polygons., Allocate vascular tissue.         Define the region where vascular tissue will, Find the layer where vascular tissue will be allocated.                  Args:, Return extra tissue polygons for visualization without placing cells.         S, Any, GeoDataFrame, Polygon (+6 more)

### Community 14 - "AbstractNetwork"
Cohesion: 0.11
Nodes (12): AbstractNetwork, ABC, lil_matrix, Abstract network base module for hydraulic network construction.  Provides the, Fill entries of the adjacency matrix with a hydraulic conductivity.          P, Fill entries of the adjacency matrix from a dictionary of hydraulic conductiviti, Return the set of wall-node indices whose adjacent cells         match the requ, Abstract base class for hydraulic network graph construction.      Subclasses (+4 more)

### Community 15 - "Monocot Root Anatomy"
Cohesion: 0.12
Nodes (15): Split a polygon into n slices using radial lines from the center., MonocotRootAnatomy, Polygon, Monocot root anatomy.  ``MonocotRootAnatomy`` builds a monocot stele: either t, Declarative description of how a monocot stele is assembled.          Built an, Phloem regions in the valleys between the xylem star arms.          Monocot ro, Declarative phloem step: valley *regions* filled by circle-packing.          M, The class's canonical metaxylem *centres* for ``n`` vessels — the same (+7 more)

### Community 16 - "Cell Border Generation"
Cohesion: 0.16
Nodes (14): Fill the reserved transfusion-tissue zone by circle-packing.          Only run, Place cell seeds along a geometry (Polygon exterior or LineString/MultiLineStrin, _dispatch_fill(), fill_along(), fill_by_packing(), fill_by_rings(), place_packed_group(), Polygon (+6 more)

### Community 17 - "Organ Base Class"
Cohesion: 0.09
Nodes (15): Organ, ABC, Plant anatomy base module providing abstract interface., Add a tissue layer to the anatomy.                  Args:             layer:, Remove a tissue layer by name.                  Args:             name: Name, List all layer names., Invalidate cached geometry after layer changes., Abstract base class for plant anatomical structures.          Defines the inte (+7 more)

### Community 18 - "Cell/Geometry (callgraph)"
Cohesion: 0.16
Nodes (16): granap.cell_class, granap.generate_cell.CellGenerator.cell_border, granap.generate_cell.CellGenerator.cells_on_layer, granap.geometry_collection, granap.geometry_collection.GeometryProcessor, granap.geometry_collection.GeometryProcessor.difference_polygons, granap.geometry_collection.GeometryProcessor.draw_ellipse, granap.geometry_collection.GeometryProcessor.ellipse_to_polygon (+8 more)

### Community 19 - "Organ (callgraph)"
Cohesion: 0.19
Nodes (21): granap.layer_manager.LayerManager.get_layers, granap.organ_class.Organ, granap.organ_class.Organ._build_anatnetwork, granap.organ_class.Organ._build_layer_polygons, granap.organ_class.Organ._create_base_shape, granap.organ_class.Organ._create_central_layers, granap.organ_class.Organ._create_vascular_tissue, granap.organ_class.Organ._which_layer_for_vascular (+13 more)

### Community 20 - "GeometryProcessor"
Cohesion: 0.23
Nodes (8): Point, Create a polygon for an ellipse, Largest ellipse (up to the target circle's area, aspect-capped) that         fi, Ellipse of the target circle's area, oriented **radially** (major axis, Size-first, gradient-driven radial packing.          Marches outward in radius, Unified Apollonian circle packing with proportion stop, directional gradient,, fill_phloem_zone(), Place sieve tubes (+ companion cells when ``alive``) then parenchyma around

### Community 21 - "Needle Anatomy"
Cohesion: 0.13
Nodes (9): NeedleAnatomy, Needle anatomy implementation., Update central cylinder parameters.                  Args:             **kwar, Update transfusion tissue parameters.                  Args:             **kw, Find the layer where vascular tissue will be allocated.                  Args:, Needle vascular tissue as an inspectable recipe.          Built and run by the, The sliver of ``polygon`` left over beyond each ellipse's own         outer (la, Seed the rectangular xylem/phloem grid (with a central cambium row         and (+1 more)

### Community 22 - "Central Layers & Xylem Star"
Cohesion: 0.25
Nodes (3): Pole of inaccessibility (largest inscribed circle): ``(cx, cy, radius)``., Radial growth-ring bands of the secondary-xylem annulus, following the, Translate the region by ``(dx, dy)``.

### Community 23 - "Cell/Layer Generation & Plot"
Cohesion: 0.11
Nodes (11): Figure, GeoDataFrame, lil_matrix, Plot layer boundaries.                  Args:             show: Whether to di, Plot cell geometries.                  Args:             show: Whether to dis, Export cell geometries as GeoDataFrame.                  Returns:, Export cell data to CSV file.                  Args:             filepath: Ou, Plot every tissue zone before placing any cell.         See :func:`openalea.gra (+3 more)

### Community 24 - "CellManager (callgraph)"
Cohesion: 0.16
Nodes (23): granap.cell_manager, granap.cell_manager.CellManager, granap.cell_manager.CellManager.extend_cells, granap.cell_manager.CellManager.get_all_types, granap.cell_manager.CellManager.get_cell_by_id, granap.cell_manager.CellManager.get_cells, granap.cell_manager.CellManager.get_cells_by_group, granap.cell_manager.CellManager.get_cells_by_groups (+15 more)

### Community 25 - "CellGenerator (Voronoi)"
Cohesion: 0.10
Nodes (16): CellGenerator, ndarray, Point, Polygon, Cell generator module for creating cells using Voronoi tessellation., Generate cell information from layer polygons.                  Args:, Generates plant cells using Voronoi tessellation.          Handles cell placem, Remove cell_border points from lower-priority id_groups that overlap         wi (+8 more)

### Community 26 - "Base Shape Polygons"
Cohesion: 0.08
Nodes (22): GeometryProcessor, Polygon, Geometry processor module for handling polygon operations., Generate a star-shaped polygon as a single continuous boundary.          Alter, Fit an inner ellipse to a polygon.          ``angle`` (degrees), when given, o, Translate ``ellipse`` (a :func:`fit_inner_ellipse`-style dict) along         ``, Star built from *peak-side* / *valley-side* radii and arcs.          The star', Generate an axis-aligned rectangle centred on the origin.          Args: (+14 more)

### Community 27 - "Medullar Rays & Secondary Xylem"
Cohesion: 0.20
Nodes (5): Build the medullar-ray corridor polygons (before vessel packing so they, Midpoint of the widest gap around a circle of thetas in ``[0, 2pi)``., Midpoint of the widest interior gap in a sorted, bounded list.          ``boun, Start radii for the ``n_new`` rate-driven rays.          Spread evenly across, Build constant-width radial-strip polygons for each medullar ray.          ``n

### Community 28 - "Cell Build Flow (callgraph)"
Cohesion: 0.22
Nodes (20): granap.cell_class.Cell, granap.cell_class.Cell..init., granap.cell_class.Cell.cell_to_dict, granap.cell_class.Cell.jitter, granap.cell_class.Cell.smooth, granap.cell_manager.CellManager..init., granap.cell_manager.CellManager.add_cell, granap.generate_cell.CellGenerator.generate_cells_info (+12 more)

### Community 29 - "Vascular Regression Tests"
Cohesion: 0.13
Nodes (22): _census(), _check(), dicot_primary(), dicot_secondary(), dicot_stem(), monocot_arch(), monocot_stem(), needle_default() (+14 more)

### Community 30 - "Visualization / plot_tissues"
Cohesion: 0.14
Nodes (14): build_anatomy_tissues(), _dry_run_vascular(), plot_layers_simple(), plot_section(), plot_tissues(), Any, Figure, GeoDataFrame (+6 more)

### Community 31 - "Dicot Secondary Params"
Cohesion: 0.67
Nodes (3): _dicot_secondary(), Verify plot_tissues and build_anatomy_tissues for monocot, dicot, and needle:, test_plot_tissues()

### Community 32 - "Root Params & XML Parse"
Cohesion: 0.13
Nodes (18): _circle_uniform(), _continuous(), _drop(), _ellipse_mixed(), main(), Demo: shaping a dicot stem — stem outline, cambium ring, and bundle pattern., Remove every param entry with the given name (returns the data)., _star_secondary() (+10 more)

### Community 33 - "LayerManager Class"
Cohesion: 0.05
Nodes (24): Layer, Any, Layer module for plant anatomy representation. Provides the Layer class represe, Create a Layer from dictionary representation., Represents a single tissue layer in plant anatomy.          Attributes:, Validate layer parameters., Calculate total thickness of this layer., Convert layer to dictionary representation. (+16 more)

### Community 34 - "Growth Modes Tests"
Cohesion: 0.25
Nodes (13): _census(), dicot_primary(), dicot_secondary(), dicot_woody(), monocot(), Smoke + structure tests for the four canonical organ growth cases.  Covers the, Woody growth thickens the stele and repeats vessel packing, so it has     more, Same preset + seed must yield an identical cell census. (+5 more)

### Community 35 - "Organ Factory & Network Tests"
Cohesion: 0.14
Nodes (11): MonocotLeafAnatomy, Peel ``region`` inward into ``name`` rings (appended to ``out``); returns, Monocot leaf: one uniform mesophyll tissue fills the core (no palisade/spongy, A single ``mesophyll`` tissue fills the whole core., Factory method to initialize the appropriate Organ subclass          (RootAnato, test_leaf_presets_and_routing(), Tests for organ construction from XML / param-list input.  Visual network gall, A monocot root loaded from XML has 3 metaxylem cells and no air spaces. (+3 more)

### Community 36 - "Layer Class"
Cohesion: 0.07
Nodes (30): DicotStemAnatomy, Polygon, Dicot stem anatomy (eustele).  ``DicotStemAnatomy`` builds a dicot stem: a cen, Radial thickness reserved outside the secondary phloem for the displaced, Stem outer radius, grown outward to make room for secondary growth., Radius of the primary bundle ring (where the bundles / fascicular cambium, A ring contour in the ``ring_shape`` family from a bundle / cylinder spec., The drawn contour the eustele bundles are placed on — the (primary)         cam (+22 more)

### Community 37 - "Vascular Recipe (Organ)"
Cohesion: 0.15
Nodes (10): Polygon, Create the base shape for the organ.                  This method must be impl, Generate or retrieve the base shape.                  Returns:             Ba, Regions that are *meant* to hold no cell (so they are not gaps).          The, The union of ``polys`` with its internal pockets filled in.          Anything, Detect empty space inside the organ that no cell was assigned to.          A g, Absorb each detected gap into the nearest cell of its own tissue zone., Create vascular tissue by building this organ's vascular recipe.          Shar (+2 more)

### Community 38 - "NeedleAnatomy (callgraph)"
Cohesion: 0.39
Nodes (9): granap.needle_class.NeedleAnatomy, granap.needle_class.NeedleAnatomy..init., granap.needle_class.NeedleAnatomy._initialize_default_layers, granap.needle_class.NeedleAnatomy._initialize_default_params, granap.needle_class.NeedleAnatomy._initialize_params, granap.needle_class.NeedleAnatomy._which_layer_for_vascular, granap.needle_class.NeedleAnatomy.set_central_cylinder_params, granap.needle_class.NeedleAnatomy.set_transfusion_params (+1 more)

### Community 39 - "Monocot Xylem Tests"
Cohesion: 0.16
Nodes (18): cell_type_counts(), make_arch_root(), make_star_root(), Tests for monocot arch-mode xylem (metaxylem ring + protoxylem) + pith, and sta, Star mode packs xylem vessels into the star and phloem into the valleys., Star mode with a pith: no xylem vessels inside the pith circle, but stele     (, Phloem strands fall in the valleys (between arms), not on the arm axes., Arch mode without pith: metaxylem vessels exist. (+10 more)

### Community 40 - "Layer Polygons & Statistics"
Cohesion: 0.29
Nodes (4): Any, Return tissue zone descriptors (layer rings + vascular polygons).         See :, Calculate anatomical statistics.                  Returns:             Dictio, Create central tissue layers (vascular, parenchyma, etc.).          Args:

### Community 41 - "AbstractNetwork (callgraph)"
Cohesion: 0.29
Nodes (11): granap.network_base, granap.network_base.AbstractNetwork, granap.network_base.AbstractNetwork..init., granap.network_base.AbstractNetwork._build_anatnetwork, granap.network_base.AbstractNetwork._get_walls_for_types, granap.network_base.AbstractNetwork._types_match, granap.network_base.AbstractNetwork.export_to_adjencymatrix, granap.network_base.AbstractNetwork.fill_matrix (+3 more)

### Community 42 - "Layer Dict & Needle Init"
Cohesion: 0.29
Nodes (3): Parse the parameters that are common to all root types., Parse plant-type-specific vascular parameters. Overridden in subclasses., Initialise root layers from parsed params.

### Community 43 - "RootAnatomy Factory (callgraph)"
Cohesion: 0.32
Nodes (8): granap.geometry_collection.GeometryProcessor.circle_polygon, granap.root_class.RootAnatomy, granap.root_class.RootAnatomy..init., granap.root_class.RootAnatomy._create_base_shape, granap.root_class.RootAnatomy._initialize_default_layers, granap.root_class.RootAnatomy._which_layer_for_vascular, granap.root_class.RootAnatomy.add_lateral_root_primordium, granap.root_class.RootAnatomy.set_vascular_params

### Community 44 - "Perf Characterize"
Cohesion: 0.28
Nodes (8): _census(), _dicot_primary(), _dicot_secondary(), _geom_hash(), _monocot_arch(), _monocot_default(), Byte-identical safety net for perf work (see performance_proposals.md).  Pins, run()

### Community 45 - "Stomata"
Cohesion: 0.20
Nodes (5): Add resin ducts (parenchyma ring + inner lumen) to the mesophyll.          Geo, Add stomata to the needle epidermis.          With ``n_files`` only, stomata a, Return resin-duct and stomata polygons for plot_tissues visualization,, Compute resin duct geometry from layer polygons without placing cells., Compute stomata geometry from triplet positions without placing cells.

### Community 46 - "Buffer/Union & Resin Duct"
Cohesion: 0.15
Nodes (10): LineString, Union a list of polygons.                  Args:             polygons: List o, Insert small rhombic air spaces on the walls between mesophyll cells., Carve thin, full-height air slits across every ``every``-th wall         betwee, Yield the 1-D wall segments from a cell/cell intersection geometry.          T, Build a small rhombus centred on ``wall`` and aligned with it.          The rh, Build a thin rectangle centred on ``wall`` and aligned with it.          Same, Needle override of the intercellular-space geometry.          In the needle me (+2 more)

### Community 47 - "Aerenchyma & Intercellular Spaces"
Cohesion: 0.17
Nodes (6): Simplify cell boundaries by retaining only junction vertices.          Delegat, Fuse touching air-space cells within the same angular sector, then carve tissue, Orchestrate intercellular space and aerenchyma generation., Place aerenchyma so the realized air proportion matches the request., Compute air spaces for each inter_cellular_spaces entry., Apply one inter_cellular_spaces entry to the relevant tissue cells.

### Community 48 - "NetworkExporter"
Cohesion: 0.14
Nodes (8): NetworkExporter, Write the root anatomy as an XML file matching GRANAR format.          Cell or, Class to export Organ anatomy to an AbstractNetwork topological graph., Populate the provided network graph from the cell GeoDataFrame.          Algor, # IMPORTANT: apply the IDENTICAL valid-geometry filter so that both paths, Any, Build the shared vertex/edge topology for a collection of polygons.          R, Populate ``self.graph`` from the cell GeoDataFrame.         Delegated to Anatom

### Community 49 - "Layer Expansion"
Cohesion: 0.08
Nodes (54): _anchor_shift(), build_arc_bundle(), build_bundle(), BundleResult, _fill_cambium(), _fill_parenchyma(), _fill_phloem(), _fill_xylem_face() (+46 more)

### Community 50 - "Secondary Phloem Tests"
Cohesion: 0.43
Nodes (6): cell_type_counts(), make_root(), Geometry tests for secondary phloem generation.  Visual gallery lives in ``exa, Return (phloem_at_valley, phloem_at_cambium_arm)., test_secondary_phloem_zone_placement(), _zone_geometry_ok()

### Community 51 - "Layer Add/Get"
Cohesion: 0.06
Nodes (31): _get_arrangement(), _get_planttype(), __getattr__(), _input_params(), Stem anatomy implementation.  ``StemAnatomy`` acts as a transparent factory: c, The raw param-dict list from any accepted input form (or None)., Extract the planttype integer (1 = monocot, 2 = dicot) from raw input., Dicot stele arrangement: 'fascicular' (discrete bundle ring) or     'continuous (+23 more)

### Community 52 - "Math Shape Functions"
Cohesion: 0.25
Nodes (7): five_pl(), linear(), Normalized shape functions and a rescale wrapper for GRANAP gradients.  Conven, Normalized 5-parameter logistic: f(0) = 1, f(inf) -> 0.      Parameters     -, Normalized linear decrease: f(0) = 1, f(1) = 0, clamped outside [0, 1].      P, Wrap a normalized shape function so its output spans [lo, hi].      Parameters, rescale()

### Community 53 - "Needle Base Shape"
Cohesion: 0.33
Nodes (3): Resolve the needle's actual (width, thickness), computing from the         laye, Calculate total needle width from layers., Calculate total needle thickness from layers.

### Community 54 - "Transfusion Layers"
Cohesion: 0.33
Nodes (4): Polygon, Create the half-ellipse shape of a needle cross-section.          Returns:, When "central_cylinder" has shape="ellipse", interpolate each layer         pol, Create transfusion tissue and parenchyma layers.                  Args:

### Community 55 - "Layer Add/Remove"
Cohesion: 0.07
Nodes (41): angular_wedge(), _bisect_bounded(), _bisect_circular(), build_annual_bands(), build_medullar_ray_polygons(), cambium_local_frame(), fill_medullar_rays(), fill_ray_parenchyma() (+33 more)

### Community 56 - "RootAnatomy Factory"
Cohesion: 0.33
Nodes (4): _get_planttype(), __getattr__(), Root anatomy implementation.  `RootAnatomy` acts as a transparent factory: cal, Extract the planttype integer (1 = monocot, 2 = dicot) from raw input.

### Community 57 - "Dicot Star Xylem Tests"
Cohesion: 0.09
Nodes (24): Profiling harness for the heavy secondary-growth path (see performance_proposals, main(), Build a dicot root from scratch (no ``for_*`` preset), tune to nettle cytomine d, main(), make_root(), Demo: radius-dependent medullar-ray initiation (``n_medullar_rate``).  ``n_med, BundleCambiumParams, DicotCambiumParams (+16 more)

### Community 58 - "Conda/CI Packaging"
Cohesion: 0.50
Nodes (4): granap conda environment, GRANAP conda recipe (meta.yaml), OpenAlea CI Workflow, openalea/action-build-publish-anaconda openalea_ci.yml

### Community 59 - "Tomato Dicot Example"
Cohesion: 0.08
Nodes (34): _build(), _build_dicot(), _dicot_input(), _longest_extent(), Checks for the leaf cross-section (LeafAnatomy: monocot slab + dicot)., Palisade under the adaxial (upper) epidermis, spongy above the abaxial one., Length of the long side of a cell's minimum rotated rectangle., A dicot leaf vein is the collateral (xylem/cambium/phloem) bundle, not the (+26 more)

### Community 60 - "Monocot Base Shapes Demo"
Cohesion: 0.08
Nodes (16): _leaf_planttype(), LeafAnatomy, Any, Leaf cross-section anatomy.  A leaf is a flat *slab*, not a radial organ.  ``L, All ``vascular_bundle`` specs (vein size-classes), widest first so a big, Same as the base, but split each multi-piece layer into one seedable         Po, Region veins occupy — the innermost mesophyll band at the mid-plane., Lay the transverse vein row, then (dicot) the columnar palisade files. (+8 more)

### Community 61 - "Iris Monocot Example"
Cohesion: 0.07
Nodes (30): describe_params(), _field_kind(), _field_tier(), Tier of a field.  Rendering (algorithm) knobs are always advanced; the bundle's, rendering' for algorithm-tuning knobs, else 'anatomy'., A grouped, human-readable view of a param model's fields — the antidote to a, A curated, per-block view of this organ's parameters.  Defaults show only the, Tests for the parameter-metadata layer (grouping / tiering / dead-knob lint). (+22 more)

### Community 62 - "Maize Monocot Example"
Cohesion: 0.08
Nodes (13): Plant type: 1 = monocot, 2 = dicot., Parse the parameters common to all stem types.          The central ground tis, Optional per-subclass hook for extra vascular parsing.          Not needed by, Build the vascular tissue, then register the medullary cavity (if any)., Record one built bundle for the removal mask and the tissue view.          The, Minimum ground-tissue gap required between two bundle envelopes.          Bund, All ``vascular_bundle`` param specs, sorted by inner band radius.          A s, Radial band ``(r_lo, r_hi)`` of a bundle spec, in mm.          ``radius_max <= (+5 more)

### Community 63 - "Wheat Monocot Example"
Cohesion: 0.08
Nodes (24): ① Drop geopandas `dissolve` in `process_voronoi_groups`  *(DONE 2026-07-07 — byte-identical)*, ② Vectorise the remaining per-point containment tests  *(DONE 2026-07-07 — byte-identical)*, ③ Loosen `maximum_inscribed_circle` tolerance  *(est. −2 to −4s, changes golden)*, ④ Batch `Cell.jitter` / seed creation  *(DONE 2026-07-07 — byte-identical, NOT a repro break)*, ⑤ Drop the per-cell DataFrame in `voronoi_diagram`  *(DONE 2026-07-07 — byte-identical)*, Already done (byte-identical — golden unchanged), F1 — Drop `is_valid` / `buffer(0)` on Voronoi cells in `process_voronoi_groups`  *(DONE 2026-07-07)*, Findings — golden-changing levers, now measured on Linux (+16 more)

### Community 64 - "Monocot Xylem Gallery"
Cohesion: 0.08
Nodes (23): 1. Mental model, 2. Installation & quick start, 3.1 `OrganInputData` — configuration, 3.2 `Organ` — the anatomy object, 3.3 Plotting & inspection, 3.4 Export, 3. The public API, 4. The generation pipeline (+15 more)

### Community 65 - "Dicot Root Gallery"
Cohesion: 0.13
Nodes (12): DicotLeafAnatomy, Polygon, ``(x, spec)`` for every vein along the mid-plane — the single source for, x of each vein (any size-class)., The lamina outline: a wide, thin lens, ribbed where each vein thickens it., Along-surface smoothing length (mm) for the outline: the ``outline_smooth``, Gaussian-smooth the adaxial/abaxial surface profiles along x, rounding the, Round sharp convex features (the pointed margins/tips and a narrow rib (+4 more)

### Community 66 - "Secondary Phloem Gallery"
Cohesion: 0.11
Nodes (10): The lamina mid-line y(x): the centre of the leaf slab at each ``x``., dy/dx of the lamina mid-line at ``x`` (central finite difference).          Ze, World angle of a vein's abaxial (outward, lower-surface) pole at ``x``., Place every vein transversely along the mid-plane, each built from its own, Ground (mesophyll) cell diameter the veins sit among — turns on the outer, The vein's y, shifted by its ``relative_distance`` toward a face: 0.5 =, Compute the gap region **between** each pair of adjacent veins (the monocot, (bottom_y, top_y) of the lamina outline at abscissa ``x`` (a vertical cut). (+2 more)

### Community 67 - "Secondary Xylem Demo"
Cohesion: 0.11
Nodes (15): _monocot_bands(), Monocot stem data whose single preset bundle spec is replaced by ``bands``., _dicot(), main(), _monocot(), Demo: stem vascular bundle types and the hollow (fistular) pith.  Renders the, _dicot(), main() (+7 more)

### Community 68 - "Needle Vascular Recipe"
Cohesion: 0.20
Nodes (19): _build(), _census(), _mean_dist(), _mean_x(), _params(), Tests for stem vascular bundles and the hollow (fistular) pith.  Topology is a, Bundle + cell-level param dicts from the dicot preset, with overrides., test_amphicribral_phloem_rings_xylem() (+11 more)

### Community 69 - "Perf Profiling (Nettle)"
Cohesion: 0.12
Nodes (14): _air_space_network_elements(), main(), plot_air_link_network(), Gallery: full feature set (layers / cells / network / air-link network) for one, Return the nodes/edges of the network graph that belong to air-space cells., Draw the full network faded, then highlight the air-space nodes/edges., run_leaf_features(), main() (+6 more)

### Community 70 - "Medullar Rays Demo"
Cohesion: 0.11
Nodes (16): main(), anatomy_metrics(), main(), Build a wheat-like monocot root from the ``for_root`` preset and plot it.  Con, Geometry summary of a generated root: root/stele diameters and total     xylem, Return OrganInputData pre-loaded with default root anatomy parameters., Params with no registered metadata (everything but the bundle today) are     sk, test_lint_ignores_unregistered_params() (+8 more)

### Community 71 - "Visualization (callgraph)"
Cohesion: 0.67
Nodes (3): granap.visualization, granap.visualization.plot_layers_simple, granap.visualization.plot_section

### Community 72 - "Oak Dicot Example"
Cohesion: 0.20
Nodes (15): _center(), distinct_rays(), make_secondary_root(), Tests for dicot secondary-growth medullar rays.  Covers the constant-tangentia, Rays reach the outer edge, so the secondary phloem is still produced., Dicot secondary-growth root with a medullar-ray param entry., Number of distinct medullar rays whose cells fall in the radial band     ``[r_l, With rate=0, exactly n_medullar rays are present (no rate-driven extras). (+7 more)

### Community 75 - "Resin Canal"
Cohesion: 0.17
Nodes (15): _bundles(), _pattern_stem(), Minimal coverage for the newer dicot/monocot stem features:  * mixed-kind ``bu, An 'even' band whose ``radius`` exceeds the pith radius carries its bundles out, A dicot eustele with two bundle kinds arranged by a bundle_pattern., Equal (distance) spacing places the whole tiled sequence: 2 kinds x 3 = 6., grouped' spacing on a 3-arm star clusters the sequence into 3 lobes with     em, A bicollateral eustele lays a cambium band on both radial sides of the xylem (+7 more)

### Community 80 - "Pinus Params Demo"
Cohesion: 0.14
Nodes (10): Axis-aligned unit disc scaled to ``width`` x ``height``, rotated so its, Map local-frame geometries (radial axis = local +y) to their place.          A, The bundle footprint placed at ``(cx, cy)`` oriented radially at ``theta``., Tangential footprint of a bundle at the ring (envelope width + outer         sh, _local_envelope(), outer_sheath_mask_pad(), partition_concentric(), Conservative outward reservation for a *candidate* bundle placement.      The (+2 more)

### Community 81 - "DicotRootSeries"
Cohesion: 0.24
Nodes (7): DicotRootSeries, Longitudinal series of **dicot**-root sections with identity-tracked primary xyl, Pith radius at ``length``, snapped to exactly 0 below a tiny epsilon — a, The sample where the pith is smallest — the star is fully packed there, so it is, Generate the reference (smallest-pith) section with the normal dicot packer and, The packed primary-xylem vessels of a reference section as ``(x, y, radius)``., The vessels present at ``length`` with their pith-capped positions + sizes.

### Community 82 - "._create_central_layers"
Cohesion: 0.14
Nodes (8): Any, Polygon, Pith radial size gradient: ``r_norm`` (0 = centre .. 1 = pith edge) ->, Ground-tissue (pith) cell diameter at radius ``r`` mm from the organ         ce, Create pith parenchyma rings from the pith edge toward the centre.          Id, True if ``env`` comes within ``gap`` of any already-placed envelope., Cell diameter of the tissue at ``(x, y)`` — the ground a bundle sits in., Representative ground-cell size of the bundle's *neighbours*.          Samples

### Community 83 - "RootSeries"
Cohesion: 0.21
Nodes (6): Longitudinal series of monocot-root sections with identity-tracked metaxylem tha, The k metaxylem centres the monocot class would place in a stele of radius R, Identity of the vessels at ``length`` (no final position yet): the fused groups, Assign each vessel (by its current position ``reps[i]``) to a distinct slot,, Actual vessel positions per length, walking from the highest length to the, RootSeries

### Community 84 - "test_root_series.py"
Cohesion: 0.21
Nodes (12): _dicot_series(), Root developmental series (ROOT_SERIES_PLAN) — Phase 1: prescription plumbing., A dicot section with a receded pith places its prescribed vessels as tracked, Same prescribed set -> same tracked geometry (identity is stable)., The primordial set (positions + 5PL targets) is extracted from the smallest-pith, As the pith recedes (apex -> collet) the number of differentiated vessels only g, Centripetal maturation: at a mid pith the present vessels are the outer ones (la, test_dicot_extraction_is_stable_regardless_of_span() (+4 more)

### Community 85 - "test_stem_continuous.py"
Cohesion: 0.26
Nodes (12): _n_components(), Tests for the continuous (non-fascicular) dicot stem vascular cylinder.  Unlik, Inner / outer radius (from the organ centre) of the registered xylem annulus., (#vessels, #parenchyma) cells whose centre falls in the xylem annulus band., The 'files' layout keeps all three tissue *regions* continuous (the strips, _stem(), test_continuous_stem_generates_all_tissues(), test_files_line_up_xylem_only() (+4 more)

### Community 86 - ".for_dicot_leaf"
Cohesion: 0.18
Nodes (9): build_dicot_leaf(), main(), Dicot leaf cross-section demo (dorsiventral) — midrib bundle designs.  Palisad, A near-constant-thickness dicot lamina with one big central midrib.      An el, Dicot leaf preset: dorsiventral palisade (adaxial) / spongy (abaxial), a, _blunt_margin_input(), Dicot leaf ending in a blunt rounded margin instead of a thin taper.      Full, ``fuse_gaps`` absorbs the holes the tissue closes around and leaves the rest. (+1 more)

### Community 87 - ".buffer_polygon"
Cohesion: 0.18
Nodes (7): Buffer a polygon with optional smoothing.                  Args:, Point, Build layer polygons from current layer configuration., Shrink/expand one ring layer's polygon by ``distance``.          Default imple, Optionally reshape layer polygons after they have been built.          The def, Any, Create stele parenchyma rings from the stele edge toward the centre.

### Community 88 - "Tissue refactor — the shape-first model"
Cohesion: 0.18
Nodes (10): How an organ uses it, Invariants for safe refactoring, Status, Test harness, The cells-first engine (unchanged, lives in `Organ.generate_cells`), The core idea: a tissue **is a region**, The seeding idiom: `Cell.radial`, The vocabulary (`openalea/granap/tissue_builder.py`) (+2 more)

### Community 89 - "Roadmap — toward simplified recipes for monocot, dicot, needle"
Cohesion: 0.18
Nodes (10): Dependency order, Guardrails, P0 — Safety net for needle  *(DONE 2026-06-24)*, P1 — Recipe ergonomics: region→fill as a first-class step  *(DONE 2026-06-24)*, P2 — Special-function vocabulary  *(DONE 2026-06-24; companion cells deferred)*, P3 — Needle into the recipe model  *(DONE 2026-06-24)*, P4 — Unify the three recipes  *(DONE 2026-06-24)*, P5 — Optional layout/cleanup follow-ups  *(CLOSED — not pursued, 2026-06-24)* (+2 more)

### Community 90 - "._directional_stomata_triplets"
Cohesion: 0.22
Nodes (5): Any, Prune any layer carrying a ``zone_angles`` entry down to only the         cells, Split epidermis groups into an adaxial (flat) and abaxial (domed)         angul, ``n`` evenly spaced prev/curr/next triplets along one epidermis         angular, Greedily keep triplets in order, dropping any whose stoma centre         (the t

### Community 91 - "._offset_layer_polygon"
Cohesion: 0.18
Nodes (6): Point, Circularly interpolate a list of ``(angle_degrees, multiplier)``         contro, Needle override: dispatch to an angle-varying offset when the layer         dic, Delete the innermost hypodermis *seed cell* under each sub-stomatal chamber., Oriented probe rectangle for :meth:`_clear_hypodermis_under_chambers`., Retag parenchyma cells in the corner pockets (see         ``_corner_parenchyma_

### Community 92 - "Proposal"
Cohesion: 0.20
Nodes (9): Alternatives considered, Implementation sketch, New params (`VascularBundleParams`), Proposal, Status, Suggested dicot-stem preset, Vascular-bundle sclerenchyma caps — per-pole layer control, What already exists (+1 more)

### Community 93 - "build_gallery_needle_data"
Cohesion: 0.27
Nodes (9): _air_space_network_elements(), build_gallery_needle_data(), main(), plot_air_link_network(), Return the nodes/edges of the network graph that belong to air-space cells., Draw the full network faded, then highlight the air-space nodes/edges., Build the ``OrganInputData`` for the feature-showcase needle., needle_features() (+1 more)

### Community 94 - ".__init__"
Cohesion: 0.24
Nodes (4): Schedule, A ``(start, end)`` tuple -> linear ramp from the first to the last sampled, The N primordial ids in ring (angular) order, the non-fusers (terminators +, Distance-based merge tree over the fuser positions ``pos`` ({id: (x, y)}): for

### Community 95 - "root_series.py"
Cohesion: 0.22
Nodes (8): _fill_poly(), _fmt_members(), Root developmental series with identity-tracked xylem vessels (ROOT_SERIES_PLAN), Label of the primordial ids inside a (possibly fused) vessel — the exact ids joi, Read one numeric field off a config param entry (model or raw dict), or ``defaul, Fill a (possibly Multi)Polygon with a solid colour, on top of the tissue., Render the whole series as a grid, each tracked vessel in a fixed colour by its, _read_field()

### Community 96 - ".pole_and_corner_angles"
Cohesion: 0.25
Nodes (5): build_pinaster(), main(), Pinus (*Pinus pinaster*) needle cross-section — a measured pine needle.  All l, Locate the needle cross-section's adaxial pole, abaxial pole, and         two c, Build a circular (angle_deg, multiplier) thickness_profile from a list

### Community 97 - "test_param_schema_equivalence.py"
Cohesion: 0.33
Nodes (8): _emit(), _load(), _norm(), Schema-equivalence guard for the parameter-schema unification.  The parameter, Every preset -> its ``to_dict_list()`` as a list of name-keyed dicts (the exact, JSON-normalise for a value-only compare (tuples -> lists, as JSON round-trips)., test_preset_param_dicts_unchanged(), _write_golden()

### Community 98 - "series_monocot_wheat.py"
Cohesion: 0.36
Nodes (7): build_series(), build_wheat_base(), main(), n_fused(), Wheat monocot root — a developmental *series* (apex -> collet) with tracked xyle, A wheat 'Salmone' monocot-root template (the tissue that refits around the track, Number of FUSED metaxylem along the root: ``APEX_META`` at the apex (the fusion

### Community 99 - "RootSeriesResult"
Cohesion: 0.25
Nodes (3): Per-section results + the vessel track table (with fusion membership)., Every (length, track_id, members) the primordial ``p`` belongs to — i.e. which, RootSeriesResult

### Community 100 - "._finalize_pith_and_lacunae"
Cohesion: 0.25
Nodes (4): Add cortex air spaces, then tidy the hollow cavity + the lacunae.          The, True only when aerenchyma is explicitly requested for the pith.          (An `, Largest Polygon piece of a (possibly Multi)Polygon, or None., Post-Voronoi tidy-up of the pith centre and the protoxylem lacunae.          T

### Community 101 - "bundle_cambium_anchor"
Cohesion: 0.25
Nodes (8): bundle_cambium_anchor(), bundle_layout(), partition_banded(), Resolve bundle params into a partition spec.      Returns ``("banded", [(role,, Cut ``env`` into horizontal strips (inner->outer along local +y)., Peel the sheath off the working envelope (local frame).      Every bundle gets, Local +y (radial) offset of the (outer) cambium band's centre.      Placement, _sheath_zones()

### Community 102 - "monocot_snowdrop.py"
Cohesion: 0.47
Nodes (5): build_snowdrop(), _bundle(), main(), Snowdrop (*Galanthus*) leaf cross-section — a measured monocot leaf.  All leng, A snowdrop face bundle: protoxylem + phloem, no metaxylem, no sclerenchyma.

### Community 103 - "make_root"
Cohesion: 0.47
Nodes (5): cell_type_counts(), main(), make_root(), Gallery: monocot xylem modes — 'default' (metaxylem ring), 'arch', and 'star'., Build a monocot root in the given xylem ``shape`` with per-layer overrides.

### Community 104 - "build_dicot_base"
Cohesion: 0.47
Nodes (5): build_dicot_base(), build_series(), main(), Dicot root — a primary-growth *series* (apex -> collet) with tracked xylem.  D, A dicot-root template (the tissue that refits around the tracked vessels).  Prim

### Community 105 - "._convert_inter_bundle_aerenchyma"
Cohesion: 0.33
Nodes (3): The ground tissue the inter-bundle aerenchyma is carved from., Run the shared intercellular / aerenchyma pass, then convert the         inter-, Turn the mesophyll cells inside each stored inter-bundle region into a

### Community 106 - ".__init__"
Cohesion: 0.33
Nodes (3): Initialize default needle layers., Initialize needle anatomy.          Args:             input_data: Parameter d, Initialize central layers.

### Community 107 - "main"
Cohesion: 0.40
Nodes (3): main(), Gallery: root & needle cells and connectivity networks.  Top row shows cells,, Create OrganInputData from a plain list of dicts (no validation).

### Community 108 - "Test P. pinaster anatomy"
Cohesion: 0.40
Nodes (4): 28.08.2026, 31.08.2026, Open / for review, Test P. pinaster anatomy

### Community 109 - "anatomy_metrics"
Cohesion: 0.50
Nodes (4): anatomy_metrics(), main(), Build a tomato-like dicot root from the ``for_root`` preset and plot it.  Conf, Geometry summary of a generated root: root/stele diameters and total     xylem

### Community 110 - "build"
Cohesion: 0.50
Nodes (4): build(), main(), Build a default monocot root in each base shape; plot tissues and cells., Default monocot root with the given base_shape param.

### Community 111 - "gallery_dicot_features.py"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_dicot_root(), Gallery: dicot root — star-shaped xylem with Apollonian packing.

### Community 112 - "gallery_secondary_phloem.py"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_root(), Gallery: dicot root — secondary phloem generation.

### Community 113 - "make_secondary_root"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_secondary_root(), Smoke test and visualisation for secondary xylem / secondary growth.

### Community 114 - "anatomy_metrics"
Cohesion: 0.50
Nodes (4): anatomy_metrics(), main(), Build a iris-like monocot root from the ``for_root`` preset and plot it.  Conf, Geometry summary of a generated root: root/stele diameters and total     xylem

### Community 115 - "gallery_continuous_cylinder.py"
Cohesion: 0.40
Nodes (4): _continuous(), main(), Demo: fascicular eustele vs the continuous (non-fascicular) vascular cylinder., Continuous-cylinder preset with the xylem layout set.      ``n_xylem_files=0``

### Community 116 - "gallery_dicot_growth.py"
Cohesion: 0.40
Nodes (4): _dicot(), main(), Demo: dicot stem growth — primary bundles vs the closed cambium ring.  The *sa, Dicot stem preset with the secondary-growth flag set.

### Community 117 - "retag_tissue"
Cohesion: 0.40
Nodes (4): Rename every cell tagged ``old_tag`` to ``new_tag``.          Retags the live, Rename every cell tagged ``old_tag`` to ``new_tag``.      The terminal cell-le, retag_tissue(), test_retag_is_terminal_cell_verb()

### Community 118 - "dicot_nerium.py"
Cohesion: 0.67
Nodes (3): build_nerium(), main(), Oleander (*Nerium oleander*) leaf cross-section — a measured dicot leaf.  All

### Community 119 - "GRANAR package translated in Python"
Cohesion: 0.50
Nodes (3): Code structure, GRANAR package translated in Python, Installation

## Knowledge Gaps
- **117 isolated node(s):** `openalea.granap`, `Installation`, `Code structure`, `1. Mental model`, `2. Installation & quick start` (+112 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **43 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OrganInputData` connect `OrganInputData Model` to `Dicot Root Anatomy`, `Root Feature Demos`, `Input Params Classes`, `Monocot Root Anatomy`, `Organ Base Class`, `Needle Anatomy`, `Root Params & XML Parse`, `Organ Factory & Network Tests`, `Layer Dict & Needle Init`, `Layer Add/Get`, `RootAnatomy Factory`, `Dicot Star Xylem Tests`, `Monocot Base Shapes Demo`, `Iris Monocot Example`, `Maize Monocot Example`, `Dicot Root Gallery`, `Secondary Xylem Demo`, `Perf Profiling (Nettle)`, `Medullar Rays Demo`, `DicotRootSeries`, `RootSeries`, `.for_dicot_leaf`, `build_gallery_needle_data`, `.__init__`, `root_series.py`, `test_param_schema_equivalence.py`, `series_monocot_wheat.py`, `RootSeriesResult`, `build_dicot_base`, `.__init__`, `main`?**
  _High betweenness centrality (0.191) - this node is a cross-community bridge._
- **Why does `RootAnatomy` connect `Root Feature Demos` to `Tissue Region Geometry`, `Dicot Root Anatomy`, `OrganInputData Model`, `Cell Class`, `TissueRecipe Steps`, `CellManager`, `RoiOrgan / LayerPolygon`, `Monocot Root Anatomy`, `Cell Border Generation`, `Organ Base Class`, `CellGenerator (Voronoi)`, `Base Shape Polygons`, `Vascular Regression Tests`, `Dicot Secondary Params`, `LayerManager Class`, `Growth Modes Tests`, `Organ Factory & Network Tests`, `Monocot Xylem Tests`, `Layer Dict & Needle Init`, `Perf Characterize`, `Secondary Phloem Tests`, `RootAnatomy Factory`, `Dicot Star Xylem Tests`, `Medullar Rays Demo`, `Oak Dicot Example`, `DicotRootSeries`, `RootSeries`, `.buffer_polygon`, `RootSeriesResult`, `make_root`, `main`, `anatomy_metrics`, `build`, `gallery_dicot_features.py`, `gallery_secondary_phloem.py`, `make_secondary_root`, `anatomy_metrics`, `main`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Why does `NeedleAnatomy` connect `Needle Anatomy` to `Tissue Region Geometry`, `OrganInputData Model`, `Cell Class`, `TissueRecipe Steps`, `CellManager`, `Shape Interpolation`, `RoiOrgan / LayerPolygon`, `Cell Border Generation`, `Organ Base Class`, `CellGenerator (Voronoi)`, `Base Shape Polygons`, `Vascular Regression Tests`, `Dicot Secondary Params`, `LayerManager Class`, `Organ Factory & Network Tests`, `Stomata`, `Buffer/Union & Resin Duct`, `Needle Base Shape`, `Transfusion Layers`, `._directional_stomata_triplets`, `._offset_layer_polygon`, `build_gallery_needle_data`, `.pole_and_corner_angles`, `.__init__`, `main`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Are the 34 inferred relationships involving `RootAnatomy` (e.g. with `_dicot_primary()` and `_dicot_secondary()`) actually correct?**
  _`RootAnatomy` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `CellManager` (e.g. with `Cell` and `CellGenerator`) actually correct?**
  _`CellManager` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `Organ` (e.g. with `AnatomyWriter` and `NetworkExporter`) actually correct?**
  _`Organ` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `LeafAnatomy` (e.g. with `main()` and `main()`) actually correct?**
  _`LeafAnatomy` has 22 INFERRED edges - model-reasoned connections that need verification._