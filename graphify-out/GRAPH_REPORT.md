# Graph Report - .  (2026-07-09)

## Corpus Check
- 65 files · ~90,918 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1105 nodes · 2136 edges · 81 communities (71 shown, 10 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 255 edges (avg confidence: 0.67)
- Token cost: 61,438 input · 0 output

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

## God Nodes (most connected - your core abstractions)
1. `RootAnatomy` - 84 edges
2. `Organ` - 68 edges
3. `CellManager` - 65 edges
4. `BaseParams` - 43 edges
5. `NeedleAnatomy` - 40 edges
6. `DicotRootAnatomy` - 40 edges
7. `Cell` - 39 edges
8. `GeometryProcessor` - 39 edges
9. `TissueRecipe` - 38 edges
10. `Tissue` - 34 edges

## Surprising Connections (you probably didn't know these)
- `test_needle_recipes_are_inspectable()` --calls--> `NeedleAnatomy`  [INFERRED]
  test/test_recipe_vocabulary.py → src/openalea/granap/needle_class.py
- `_monocot_default()` --calls--> `RootAnatomy`  [INFERRED]
  doc/perf_characterize.py → src/openalea/granap/root_class.py
- `_monocot_arch()` --calls--> `RootAnatomy`  [INFERRED]
  doc/perf_characterize.py → src/openalea/granap/root_class.py
- `_dicot_primary()` --calls--> `RootAnatomy`  [INFERRED]
  doc/perf_characterize.py → src/openalea/granap/root_class.py
- `_dicot_secondary()` --calls--> `RootAnatomy`  [INFERRED]
  doc/perf_characterize.py → src/openalea/granap/root_class.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Organ anatomy class hierarchy** — granap_diagram_abstractnetwork, granap_diagram_organ, granap_diagram_rootanatomy, granap_diagram_needleanatomy [EXTRACTED 1.00]
- **Cells-first generate_cells pipeline** — doc_tissue_refactor_cells_first_engine, doc_performance_proposals_process_voronoi_groups, doc_performance_proposals_generate_cells_perf, doc_tissue_refactor_vascular_mask [INFERRED 0.75]
- **Tissue refactor roadmap phases P0-P4** — doc_tissue_roadmap_p0_needle_golden, doc_tissue_roadmap_p1_recipe_ergonomics, doc_tissue_roadmap_p2_special_vocab, doc_tissue_roadmap_p3_needle_recipe, doc_tissue_roadmap_p4_unify_recipes [EXTRACTED 1.00]

## Communities (81 total, 10 thin omitted)

### Community 0 - "Tissue Region Geometry"
Cohesion: 0.08
Nodes (27): Point, Record only the wide ("xylem") vessels in ``vascular_polygons``.          The pa, _as_shape(), Accept a :class:`Tissue` or a raw shapely geometry; return the geometry., A tagged anatomical region: a shapely shape plus the tag its cells take.      Sh, Smooth the region boundary (pre-fill, pure geometry)., Subtract another region (Tissue or shapely geometry) from this one., Clip this region to another (Tissue or shapely geometry). (+19 more)

### Community 1 - "Dicot Root Anatomy"
Cohesion: 0.10
Nodes (17): DicotRootAnatomy, Polygon, Dicot root anatomy.  ``DicotRootAnatomy`` builds a dicot stele: star-shaped prim, Build secondary phloem as tapering trapezes outside the cambium.          The ph, Pack sieve tubes (+ companion cells when alive=True) then fill parenchyma., Declarative description of how a dicot stele is assembled.          Built and ru, Build the phloem regions: one :class:`Tissue` per valley between peaks., Clearance buffer used when recording phloem regions for the stele mask. (+9 more)

### Community 2 - "OrganInputData Model"
Cohesion: 0.09
Nodes (18): BaseModel, OrganInputData, Any, A unified data structure for handling Organ initialization parameters     from d, Return params as a plain list of dicts (for backward compatibility)., Retrieve a param entry by its `name` field, or None if absent.          Prefer `, The `name` of every param entry (the keys accepted by ``get`` / ``[]``)., Return the param entry named ``name`` or raise a KeyError listing the         av (+10 more)

### Community 3 - "Cell Class"
Cohesion: 0.09
Nodes (22): Cell, Polygon, Build a seed cell whose ``angle``/``radius`` are measured from ``center``., Jitter the cell position.          Uses ``rng`` (the organ's seeded generator) w, Smooth the cell polygon., carve_and_insert(), consider_as_cell(), place_resin_duct() (+14 more)

### Community 4 - "Perf & Tissue Refactor Docs"
Cohesion: 0.09
Nodes (32): Batch Cell.jitter / seed creation, _build_topology wall-snapping, generate_cells performance work, Golden regression byte-identical baseline, perf_characterize.py verification harness, Vectorised point-in-polygon (contains_xy) tests, P1 Voronoi seed density reduction, Cell.radial seeding idiom (+24 more)

### Community 5 - "TissueRecipe Steps"
Cohesion: 0.07
Nodes (14): One named step of a tissue-build recipe.      A step is just a label, the tags i, An ordered, inspectable sequence of :class:`TissueStep` objects.      Beyond the, Set the target CellManager + rng used by :meth:`fill` / :meth:`fill_each`., Add a step that fills one region ``tissue`` by ``strategy``.          ``record(t, Add a step that fills several regions ``tissues`` by ``strategy``.          ``ti, Add a cell/group-level cleanup step (produces nothing new)., Add a bespoke placement step (sheath, bundles, ...) that isn't a plain fill., Run every step in order. (+6 more)

### Community 6 - "Callgraph Generator"
Cohesion: 0.10
Nodes (25): AsyncFunctionDef, ClassDef, build_dot_source(), build_html(), clean_docstring(), collect_docstrings(), DocCollector, ensure_graphviz_available() (+17 more)

### Community 7 - "Root Feature Demos"
Cohesion: 0.07
Nodes (14): main(), Build a dicot root from the ``for_dicot_root`` preset, update to renoncule cytom, main(), main(), main(), Plant type: 1 = monocot, 2 = dicot., Parse the parameters that are common to all root types., Parse plant-type-specific vascular parameters. Overridden in subclasses. (+6 more)

### Community 8 - "LayerManager (callgraph)"
Cohesion: 0.14
Nodes (29): granap.layer_class, granap.layer_class.Layer, granap.layer_class.Layer..post_init., granap.layer_class.Layer..repr., granap.layer_class.Layer.from_dict, granap.layer_class.Layer.get_total_thickness, granap.layer_class.Layer.to_dict, granap.layer_manager (+21 more)

### Community 9 - "CellManager"
Cohesion: 0.08
Nodes (9): CellManager, Polygon, Recalculate the properties of all cells in the list., Drop every cell whose seed point intersects ``polygon`` (bulk predicate)., Next free ``id_group`` for appending a new cell group.          ``get_last_id_gr, Drop every cell that intersects ``polygon``.          A cell's footprint is its, Rename every cell tagged ``old_tag`` to ``new_tag``.      The terminal cell-leve, retag_tissue() (+1 more)

### Community 10 - "Anatomy Writer / Export"
Cohesion: 0.08
Nodes (15): main(), Manual testing script for RoiOrgan behavior.     Update the folder_path to point, AnatomyWriter, write the MECHA geometry files (retro-compatible), Write a .obj from the generated cross section geometry.         If membrane is T, Write a .svg from the generated cross section geometry.         Uses prep_geo lo, Class to export Organ anatomy to various formats (XML, OBJ, GEO)., Pre-proc for .geo file generation.         Returns list of shrunken inner lumina (+7 more)

### Community 11 - "Input Params Classes"
Cohesion: 0.17
Nodes (25): BaseParams, BaseShapeParams, CambiumParams, CentralCylinderParams, DicotMedularRaysParams, DicotPhloemParams, DicotSecondaryPhellodermParams, DicotSecondaryPhellogenParams (+17 more)

### Community 12 - "Shape Interpolation"
Cohesion: 0.16
Nodes (11): as_tuple(), interpolate_point_points(), interpolate_poly(), midpoints(), normals_offset(), plot_poly_normals(), point_derivative(), PolygonInterpolator (+3 more)

### Community 13 - "RoiOrgan / LayerPolygon"
Cohesion: 0.10
Nodes (12): LayerPolygon, Typed representation of a layer polygon produced by _build_layer_polygons., Return extra tissue polygons for visualization without placing cells.         Su, Any, GeoDataFrame, Polygon, Define the basic shape of the root/needle from loaded polygons., An Organ class generated from a folder of ImageJ ROI files.     Bypasses procedu (+4 more)

### Community 14 - "AbstractNetwork"
Cohesion: 0.10
Nodes (13): Populate the provided network graph from the cell GeoDataFrame.          Algorit, AbstractNetwork, ABC, lil_matrix, Abstract network base module for hydraulic network construction.  Provides the A, Fill entries of the adjacency matrix with a hydraulic conductivity.          Par, Fill entries of the adjacency matrix from a dictionary of hydraulic conductiviti, Return the set of wall-node indices whose adjacent cells         match the reque (+5 more)

### Community 15 - "Monocot Root Anatomy"
Cohesion: 0.14
Nodes (12): Split a polygon into n slices using radial lines from the center., MonocotRootAnatomy, Polygon, Monocot root anatomy.  ``MonocotRootAnatomy`` builds a monocot stele: either the, Add a ring of xylem parenchyma cells around each metaxylem vessel., Pack one ``kind`` ('protoxylem' or 'phloem') bundle inscribed in a         pizza, Radius separating the inner metaxylem zone from the outer protoxylem         ban, Place ``n_metaxylem`` metaxylem **evenly spaced** in a central ring,         ind (+4 more)

### Community 16 - "Cell Border Generation"
Cohesion: 0.13
Nodes (18): ndarray, Generate cell center positions along a layer polygon.                  Args:, Generate border points for elliptical cells.                  Args:, ndarray, Resample coordinates to have uniform spacing.                  Args:, Generate points along an ellipse boundary.                  Args:             ce, _dispatch_fill(), fill_along() (+10 more)

### Community 17 - "Organ Base Class"
Cohesion: 0.10
Nodes (13): Organ, ABC, Plant anatomy base module providing abstract interface., List all layer names., Rename every cell tagged ``old_tag`` to ``new_tag``.          Retags the live ce, Abstract base class for plant anatomical structures.          Defines the interf, Add organ-specific tissues by building this organ's organ recipe., Return the recipe of organ-specific (post-fill) tissues.          Default: empty (+5 more)

### Community 18 - "Cell/Geometry (callgraph)"
Cohesion: 0.13
Nodes (21): granap.cell_class, granap.generate_cell, granap.generate_cell.CellGenerator, granap.generate_cell.CellGenerator._build_topology, granap.generate_cell.CellGenerator.cell_border, granap.generate_cell.CellGenerator.cells_on_layer, granap.generate_cell.CellGenerator.voronoi_diagram, granap.geometry_collection (+13 more)

### Community 19 - "Organ (callgraph)"
Cohesion: 0.19
Nodes (21): granap.layer_manager.LayerManager.get_layers, granap.organ_class.Organ, granap.organ_class.Organ._build_anatnetwork, granap.organ_class.Organ._build_layer_polygons, granap.organ_class.Organ._create_base_shape, granap.organ_class.Organ._create_central_layers, granap.organ_class.Organ._create_vascular_tissue, granap.organ_class.Organ._which_layer_for_vascular (+13 more)

### Community 20 - "GeometryProcessor"
Cohesion: 0.15
Nodes (11): GeometryProcessor, Geometry processor module for handling polygon operations., Handles all geometric operations for anatomy generation.          Provides metho, Create a polygon for an ellipse, Largest ellipse (up to the target circle's area, aspect-capped) that         fit, Ellipse of the target circle's area, oriented **radially** (major axis         a, Size-first, gradient-driven radial packing.          Marches outward in radius o, Finds the approximate center of the Maximum Inscribed Circle (Pole of Inaccessib (+3 more)

### Community 21 - "Needle Anatomy"
Cohesion: 0.10
Nodes (12): main(), NeedleAnatomy, Needle anatomy implementation., Update central cylinder parameters.                  Args:             **kwargs:, Update transfusion tissue parameters.                  Args:             **kwarg, Find the layer where vascular tissue will be allocated.                  Args:, Needle organ-specific tissues as a recipe of P2 special-tissue steps.          B, Needle cross-sectional anatomy.      Implements the specific structure of gymnos (+4 more)

### Community 22 - "Central Layers & Xylem Star"
Cohesion: 0.12
Nodes (11): Smooth coordinates using Laplacian smoothing.                  Args:, Pole of inaccessibility (largest inscribed circle): ``(cx, cy, radius)``., Wrap a normalized shape function so its output spans [lo, hi].      Parameters, rescale(), Any, Polygon, Create stele parenchyma rings from the stele edge toward the centre., Build the star-shaped xylem *region* (pure geometry).          Shape-first: the (+3 more)

### Community 23 - "Cell/Layer Generation & Plot"
Cohesion: 0.12
Nodes (11): Figure, GeoDataFrame, lil_matrix, Generate polygons for all layers.                  Returns:             List of, Generate cell geometries using Voronoi tessellation.                  Returns:, Plot layer boundaries.                  Args:             show: Whether to displ, Plot cell geometries.                  Args:             show: Whether to displa, Export cell geometries as GeoDataFrame.                  Returns:             Ge (+3 more)

### Community 24 - "CellManager (callgraph)"
Cohesion: 0.20
Nodes (18): granap.cell_manager, granap.cell_manager.CellManager, granap.cell_manager.CellManager.extend_cells, granap.cell_manager.CellManager.get_all_types, granap.cell_manager.CellManager.get_cell_by_id, granap.cell_manager.CellManager.get_cells, granap.cell_manager.CellManager.get_cells_by_group, granap.cell_manager.CellManager.get_cells_by_groups (+10 more)

### Community 25 - "CellGenerator (Voronoi)"
Cohesion: 0.14
Nodes (11): CellGenerator, Any, Polygon, Cell generator module for creating cells using Voronoi tessellation., Generate cell information from layer polygons.                  Args:, Generates plant cells using Voronoi tessellation.          Handles cell placemen, Remove cell_border points from lower-priority id_groups that overlap         wit, Process Voronoi diagram into grouped cell geometries.                  Args: (+3 more)

### Community 26 - "Base Shape Polygons"
Cohesion: 0.14
Nodes (9): Polygon, Generate an axis-aligned rectangle centred on the origin.          Args:, Generate an upward-pointing isosceles triangle centred on the origin.          A, Generate a polygon representing the upper half of an ellipse.                  A, A superellipse / Lamé curve ``|x/rx|**e + |y/ry|**e = 1``.          Named ``focu, Best-fit :meth:`focus_ellipse_polygon` to a measured contour profile.          `, Teardrop / 'violin' shape: an asymmetric oval whose widest point (half         w, Generate a circular polygon.                  Args:             radius: Radius o (+1 more)

### Community 27 - "Medullar Rays & Secondary Xylem"
Cohesion: 0.12
Nodes (9): Pie-wedge polygon: apex at ``(cx, cy)``, spanning ``theta_center ± half_angle``, Build wedge-shaped polygons for each medullar ray.          When allow_non_vascu, Seed one radially-oriented elliptical cell as ``len(border_cos)`` border, Fill a medullar ray polygon with medullar_ray cells.          The tangential wid, Fill angular gaps between pizza slices with radially-oriented ray parenchyma., Render the secondary cambium as ``n_layers`` concentric cell files and         r, The angular wedges (one per vascular peak) the secondary-xylem vessels         p, Build the medullar-ray corridor polygons (before vessel packing so they (+1 more)

### Community 28 - "Cell Build Flow (callgraph)"
Cohesion: 0.26
Nodes (17): granap.cell_class.Cell, granap.cell_class.Cell..init., granap.cell_class.Cell.cell_to_dict, granap.cell_class.Cell.jitter, granap.cell_class.Cell.smooth, granap.cell_manager.CellManager..init., granap.cell_manager.CellManager.add_cell, granap.generate_cell.CellGenerator.generate_cells_info (+9 more)

### Community 29 - "Vascular Regression Tests"
Cohesion: 0.18
Nodes (16): _census(), _check(), dicot_primary(), dicot_secondary(), monocot_arch(), monocot_default(), Golden regression tests for the root anatomy pipeline.  These pin the exact ``se, Two builds of the same config must be identical (no global-RNG leakage). (+8 more)

### Community 30 - "Visualization / plot_tissues"
Cohesion: 0.14
Nodes (14): build_anatomy_tissues(), _dry_run_vascular(), plot_layers_simple(), plot_section(), plot_tissues(), Any, Figure, GeoDataFrame (+6 more)

### Community 31 - "Dicot Secondary Params"
Cohesion: 0.22
Nodes (13): main(), DicotCambiumParams, DicotSecondaryCambiumParams, DicotSecondaryGrowthParams, DicotSecondaryPhloemParams, DicotSecondaryXylemParams, DicotXylemParams, PlantTypeParams (+5 more)

### Community 32 - "Root Params & XML Parse"
Cohesion: 0.22
Nodes (12): AerenchymaParams, CortexParams, EndodermisParams, EpidermisParams, ExodermisParams, InterCellularSpacesParams, PericycleParams, Parse a GRANAR-style XML file into OrganInputData.          Each parsed dict is (+4 more)

### Community 33 - "LayerManager Class"
Cohesion: 0.14
Nodes (7): LayerManager, Layer manager module for handling collections of tissue layers., Manages a collection of tissue layers with add/remove operations.          This, Return number of layers., Make LayerManager iterable., Initialize an empty layer collection., Get all layers in current order.

### Community 34 - "Growth Modes Tests"
Cohesion: 0.25
Nodes (13): _census(), dicot_annual(), dicot_primary(), dicot_secondary(), monocot(), Smoke + structure tests for the four canonical organ growth cases.  Covers the p, N annual rings repeat the vessel packing, so annual has more xylem cells., Same preset + seed must yield an identical cell census. (+5 more)

### Community 35 - "Organ Factory & Network Tests"
Cohesion: 0.18
Nodes (9): main(), Gallery: root & needle cells and connectivity networks.  Top row shows cells, bo, Create OrganInputData from a plain list of dicts (no validation)., Factory method to initialize the appropriate Organ subclass          (RootAnatom, Tests for organ construction from XML / param-list input.  Visual network galler, A monocot root loaded from XML has 3 metaxylem cells and no air spaces., A needle built from a raw param list constructs and produces cells., test_needle_from_param_list_builds() (+1 more)

### Community 36 - "Layer Class"
Cohesion: 0.15
Nodes (6): Layer, Layer module for plant anatomy representation. Provides the Layer class represen, Represents a single tissue layer in plant anatomy.          Attributes:, Validate layer parameters., Calculate total thickness of this layer., Remove a layer by name.                  Args:             name: Name identifier

### Community 37 - "Vascular Recipe (Organ)"
Cohesion: 0.18
Nodes (7): Polygon, Create the base shape for the organ.                  This method must be implem, Generate or retrieve the base shape.                  Returns:             Base, Allocate vascular tissue.         Define the region where vascular tissue will b, Find the layer where vascular tissue will be allocated.                  Args:, Create vascular tissue by building this organ's vascular recipe.          Shared, Return the ordered recipe that builds this organ's vascular tissue.          Def

### Community 38 - "NeedleAnatomy (callgraph)"
Cohesion: 0.30
Nodes (12): granap.needle_class.NeedleAnatomy, granap.needle_class.NeedleAnatomy..init., granap.needle_class.NeedleAnatomy._create_vascular_tissue, granap.needle_class.NeedleAnatomy._initialize_default_layers, granap.needle_class.NeedleAnatomy._initialize_default_params, granap.needle_class.NeedleAnatomy._initialize_params, granap.needle_class.NeedleAnatomy._which_layer_for_vascular, granap.needle_class.NeedleAnatomy.fit_vascular_elements (+4 more)

### Community 39 - "Monocot Xylem Tests"
Cohesion: 0.26
Nodes (11): cell_type_counts(), make_arch_root(), Tests for monocot arch-mode xylem (metaxylem ring + protoxylem) + pith.  Visual, Arch mode without pith: metaxylem vessels exist., Arch mode with a pith: no vessels inside the pith circle, but stele     (pith pa, n_metaxylem places exactly that many metaxylem vessels (Voronoi groups)., Both modes produce a reasonable number of cells., test_arch_exact_metaxylem_count() (+3 more)

### Community 40 - "Layer Polygons & Statistics"
Cohesion: 0.18
Nodes (6): Any, Calculate anatomical statistics.                  Returns:             Dictionar, Create central tissue layers (vascular, parenchyma, etc.).          Args:, Build layer polygons from current layer configuration., Optionally reshape layer polygons after they have been built.          The defau, Return tissue zone descriptors (layer rings + vascular polygons).         See :f

### Community 41 - "AbstractNetwork (callgraph)"
Cohesion: 0.33
Nodes (10): granap.network_base, granap.network_base.AbstractNetwork, granap.network_base.AbstractNetwork..init., granap.network_base.AbstractNetwork._build_anatnetwork, granap.network_base.AbstractNetwork._get_walls_for_types, granap.network_base.AbstractNetwork._types_match, granap.network_base.AbstractNetwork.export_to_adjencymatrix, granap.network_base.AbstractNetwork.fill_matrix (+2 more)

### Community 42 - "Layer Dict & Needle Init"
Cohesion: 0.20
Nodes (5): Any, Create a Layer from dictionary representation., Convert layer to dictionary representation., Initialize default needle layers., Initialise root layers from parsed params.

### Community 43 - "RootAnatomy Factory (callgraph)"
Cohesion: 0.28
Nodes (9): granap.geometry_collection.GeometryProcessor.circle_polygon, granap.root_class, granap.root_class.RootAnatomy, granap.root_class.RootAnatomy..init., granap.root_class.RootAnatomy._create_base_shape, granap.root_class.RootAnatomy._initialize_default_layers, granap.root_class.RootAnatomy._which_layer_for_vascular, granap.root_class.RootAnatomy.add_lateral_root_primordium (+1 more)

### Community 44 - "Perf Characterize"
Cohesion: 0.28
Nodes (8): _census(), _dicot_primary(), _dicot_secondary(), _geom_hash(), _monocot_arch(), _monocot_default(), Byte-identical safety net for perf work (see performance_proposals.md).  Pins th, run()

### Community 45 - "Stomata"
Cohesion: 0.25
Nodes (4): Create stomata on a cell.          Args:             cells: triplet of Cell obje, Compute stomata geometry from triplet positions without placing cells., Add stomata to the needle epidermis., Return resin-duct and stomata polygons for plot_tissues visualization,         w

### Community 46 - "Buffer/Union & Resin Duct"
Cohesion: 0.25
Nodes (4): Buffer a polygon with optional smoothing.                  Args:             pol, Union a list of polygons.                  Args:             polygons: List of p, Compute resin duct geometry from layer polygons without placing cells., Apply one inter_cellular_spaces entry to the relevant tissue cells.

### Community 47 - "Aerenchyma & Intercellular Spaces"
Cohesion: 0.25
Nodes (4): Orchestrate intercellular space and aerenchyma generation., Place aerenchyma so the realized air proportion matches the request.          Th, Compute air spaces for each inter_cellular_spaces entry., Fuse touching air-space cells within the same angular sector, then carve tissue

### Community 48 - "NetworkExporter"
Cohesion: 0.29
Nodes (4): NetworkExporter, Class to export Organ anatomy to an AbstractNetwork topological graph., # IMPORTANT: apply the IDENTICAL valid-geometry filter so that both paths, Populate ``self.graph`` from the cell GeoDataFrame.         Delegated to Anatomy

### Community 49 - "Layer Expansion"
Cohesion: 0.29
Nodes (4): Any, Expand layers with n_layers > 1 into individual layer entries.         Used for, Get parameters of all layers.                  Returns:             List of dict, Get layers sorted by their order attribute.                  Args:             r

### Community 50 - "Secondary Phloem Tests"
Cohesion: 0.43
Nodes (6): cell_type_counts(), make_root(), Geometry tests for secondary phloem generation.  Visual gallery lives in ``examp, Return (phloem_at_valley, phloem_at_cambium_arm)., test_secondary_phloem_zone_placement(), _zone_geometry_ok()

### Community 51 - "Layer Add/Get"
Cohesion: 0.33
Nodes (3): Add a layer to the collection.                  Args:             layer: Layer o, Retrieve a layer by name.                  Args:             name: Name identifi, Check if a layer exists.

### Community 52 - "Math Shape Functions"
Cohesion: 0.33
Nodes (5): five_pl(), linear(), Normalized shape functions and a rescale wrapper for GRANAP gradients.  Conventi, Normalized 5-parameter logistic: f(0) = 1, f(∞) → 0.      Parameters     -------, Normalized linear decrease: f(0) = 1, f(1) = 0, clamped outside [0, 1].      Par

### Community 53 - "Needle Base Shape"
Cohesion: 0.33
Nodes (3): Calculate total needle width from layers., Calculate total needle thickness from layers., Create the half-ellipse shape of a needle cross-section.                  Return

### Community 54 - "Transfusion Layers"
Cohesion: 0.40
Nodes (4): Any, Polygon, When "central_cylinder" has shape="ellipse", interpolate each layer         poly, Create transfusion tissue and parenchyma layers.                  Args:

### Community 55 - "Layer Add/Remove"
Cohesion: 0.33
Nodes (3): Remove a tissue layer by name.                  Args:             name: Name ide, Invalidate cached geometry after layer changes., Add a tissue layer to the anatomy.                  Args:             layer: Lay

### Community 56 - "RootAnatomy Factory"
Cohesion: 0.33
Nodes (4): _get_planttype(), __getattr__(), Root anatomy implementation.  `RootAnatomy` acts as a transparent factory: calli, Extract the planttype integer (1 = monocot, 2 = dicot) from raw input.

### Community 57 - "Dicot Star Xylem Tests"
Cohesion: 0.40
Nodes (5): make_dicot_root(), Tests for dicot star-shaped xylem with Apollonian packing.  Visual scenario gall, Construct a dicot RootAnatomy with custom xylem and optional cambium parameters., Every scenario yields xylem cells, and no xylem cell is smaller than     vessel_, test_dicot_star_xylem_size_classification()

### Community 58 - "Conda/CI Packaging"
Cohesion: 0.40
Nodes (5): granap conda environment, GRANAP conda recipe (meta.yaml), OpenAlea CI Workflow, openalea/action-build-publish-anaconda openalea_ci.yml, GRANAP (GRANAR in Python)

### Community 59 - "Tomato Dicot Example"
Cohesion: 0.50
Nodes (4): anatomy_metrics(), main(), Build a tomato-like dicot root from the ``for_root`` preset and plot it.  Config, Geometry summary of a generated root: root/stele diameters and total     xylem (

### Community 60 - "Monocot Base Shapes Demo"
Cohesion: 0.50
Nodes (4): build(), main(), Build a default monocot root in each base shape; plot tissues and cells., Default monocot root with the given base_shape param.

### Community 61 - "Iris Monocot Example"
Cohesion: 0.50
Nodes (4): anatomy_metrics(), main(), Build a iris-like monocot root from the ``for_root`` preset and plot it.  Config, Geometry summary of a generated root: root/stele diameters and total     xylem (

### Community 62 - "Maize Monocot Example"
Cohesion: 0.50
Nodes (4): anatomy_metrics(), main(), Build a maize-like monocot root from the ``for_root`` preset and plot it.  Confi, Geometry summary of a generated root: root/stele diameters and total     xylem (

### Community 63 - "Wheat Monocot Example"
Cohesion: 0.50
Nodes (4): anatomy_metrics(), main(), Build a wheat-like monocot root from the ``for_root`` preset and plot it.  Confi, Geometry summary of a generated root: root/stele diameters and total     xylem (

### Community 64 - "Monocot Xylem Gallery"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_star_root(), Gallery: monocot star-shaped xylem mode across several parameterisations.

### Community 65 - "Dicot Root Gallery"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_dicot_root(), Gallery: dicot root — star-shaped xylem with Apollonian packing.

### Community 66 - "Secondary Phloem Gallery"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_root(), Gallery: dicot root — secondary phloem generation.

### Community 67 - "Secondary Xylem Demo"
Cohesion: 0.60
Nodes (4): cell_type_counts(), main(), make_secondary_root(), Smoke test and visualisation for secondary xylem / secondary growth.

### Community 70 - "Medullar Rays Demo"
Cohesion: 0.67
Nodes (3): main(), make_root(), Visualisation demo for medullar ray placement (n_medullar × allow_non_vascular).

### Community 71 - "Visualization (callgraph)"
Cohesion: 0.67
Nodes (3): granap.visualization, granap.visualization.plot_layers_simple, granap.visualization.plot_section

## Knowledge Gaps
- **30 isolated node(s):** `openalea.granap`, `openalea/action-build-publish-anaconda openalea_ci.yml`, `AbstractNetwork`, `GRANAP (GRANAR in Python)`, `process_voronoi_groups grouping` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Organ` connect `Organ Base Class` to `Dicot Root Anatomy`, `OrganInputData Model`, `Cell Class`, `TissueRecipe Steps`, `Root Feature Demos`, `CellManager`, `Anatomy Writer / Export`, `RoiOrgan / LayerPolygon`, `AbstractNetwork`, `Monocot Root Anatomy`, `GeometryProcessor`, `Needle Anatomy`, `Cell/Layer Generation & Plot`, `CellGenerator (Voronoi)`, `LayerManager Class`, `Organ Factory & Network Tests`, `Layer Class`, `Vascular Recipe (Organ)`, `Layer Polygons & Statistics`, `Buffer/Union & Resin Duct`, `Aerenchyma & Intercellular Spaces`, `NetworkExporter`, `Layer Add/Remove`?**
  _High betweenness centrality (0.209) - this node is a cross-community bridge._
- **Why does `RootAnatomy` connect `Root Feature Demos` to `Tissue Region Geometry`, `Dicot Root Anatomy`, `OrganInputData Model`, `Cell Class`, `TissueRecipe Steps`, `CellManager`, `RoiOrgan / LayerPolygon`, `Monocot Root Anatomy`, `Organ Base Class`, `GeometryProcessor`, `Central Layers & Xylem Star`, `CellGenerator (Voronoi)`, `Base Shape Polygons`, `Vascular Regression Tests`, `Dicot Secondary Params`, `Growth Modes Tests`, `Organ Factory & Network Tests`, `Layer Class`, `Monocot Xylem Tests`, `Layer Dict & Needle Init`, `Perf Characterize`, `Secondary Phloem Tests`, `RootAnatomy Factory`, `Dicot Star Xylem Tests`, `Tomato Dicot Example`, `Monocot Base Shapes Demo`, `Iris Monocot Example`, `Maize Monocot Example`, `Wheat Monocot Example`, `Monocot Xylem Gallery`, `Dicot Root Gallery`, `Secondary Phloem Gallery`, `Secondary Xylem Demo`, `Medullar Rays Demo`, `Oak Dicot Example`?**
  _High betweenness centrality (0.189) - this node is a cross-community bridge._
- **Why does `CellManager` connect `CellManager` to `Tissue Region Geometry`, `Dicot Root Anatomy`, `OrganInputData Model`, `Cell Class`, `Needle Vascular Recipe`, `TissueRecipe Steps`, `Root Feature Demos`, `Stomata`, `AbstractNetwork`, `Monocot Root Anatomy`, `Cell Border Generation`, `Organ Base Class`, `Needle Anatomy`, `Cell/Layer Generation & Plot`, `CellGenerator (Voronoi)`, `Visualization / plot_tissues`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `RootAnatomy` (e.g. with `_dicot_primary()` and `_dicot_secondary()`) actually correct?**
  _`RootAnatomy` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `Organ` (e.g. with `AnatomyWriter` and `NetworkExporter`) actually correct?**
  _`Organ` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `CellManager` (e.g. with `Cell` and `CellGenerator`) actually correct?**
  _`CellManager` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `NeedleAnatomy` (e.g. with `main()` and `main()`) actually correct?**
  _`NeedleAnatomy` has 15 INFERRED edges - model-reasoned connections that need verification._