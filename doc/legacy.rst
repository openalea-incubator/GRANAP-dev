=================================
The legacy: algorithms and quirks
=================================

The legacy is GRANAP as it was before the port, kept frozen as ``openalea.granap.legacy``
(``src/openalea/granap/legacy/``): the online source (``origin/to_metafspm`` at ``248adca``) merged with the
``origin/tissue`` branch (``03d5a5c``), verbatim, with only its internal imports rewritten so that it runs on its
own. It is not developed further. The port is the main package, ``openalea.granap``, under GRANAP's module and class
names: ``from openalea.granap.root_class import RootAnatomy`` gives the port.

The port reproduces the legacy exactly at a given seed on the pinned geometry stack (shapely 2.1.2, GEOS 3.14.1):
the same cells (every attribute, the polygons equal), the same cell GeoDataFrame, the same network (nodes, edges,
every attribute, the adjacency matrix), the same XML file byte for byte. The only differences are the reviewed ones
of the registry below (the air rules, the added cell groups, the GRANAR reader). The comparisons are
``test/comparison`` (run locally only; the CI runs ``pytest --ignore=comparison``) and
``example/comparison/compare_sections.py`` (14 organ kinds, with a panel of the differences). The equations are in
:doc:`equations`, the times in :doc:`performance`.


The legacy mapped to the port
=============================

24 of the legacy's modules are in the port unchanged, apart from their imports: the organs' classes and their
building blocks (``root_class``, ``root_monocot_class``, ``root_dicot_class``, ``root_series``, ``stem_class``,
``stem_monocot_class``, ``stem_dicot_class``, ``stem_dicot_continuous_class``, ``leaf_class``, ``secondary_growth``,
``vascular_bundle``, ``tissue_class``, ``geometry_collection``, ``shapes``, ``distribution``, ``math_functions``,
``input_data``, ``layer_class``, ``layer_manager``, ``cell_class``, ``network_base``, ``generate_cell_3d``,
``mesh_utils_3d``, ``visualization``). The others carry the port:

======================================= ========================================================================
legacy                                  port
======================================= ========================================================================
``organ_class.Organ.generate_cells``    ``generate_cells`` → ``generate_section`` → ``_generate_pipeline``: the
                                        same calls in the same order, each stage's cells written into the
                                        section's MPG, the network stored at generation; the cells become views
                                        of the MPG after the grouping; the GeoDataFrame and the networkx graph
                                        are views built at their first read
``Organ._apply_intercellular``          the cells' smoothing in bulk (``_smoothed_polygons``)
``Organ._add_aerenchyma_to_proportion`` shallow copies instead of deep copies; the fill order planned once
                                        (``_fill_plan``); the best attempt's cells kept instead of run again
``Organ.merge_intercellular_aeren...``  the carving in bulk; the simplification through the organ's topology
``Organ.find_gaps``, ``fuse_gaps``      the cells' union and its filled mass computed once
``Organ.section_topology`` (new)        the section's topology: the rings stored in the MPG, else the snap
                                        through a cache keyed by the polygons, the ids and the protected ids
``generate_cell.CellGenerator``         ``_build_topology`` on arrays (the same clusters and maps, the same dict
                                        and set orders); ``simplify_cells`` rebuilt in bulk;
                                        ``resolve_cell_border_overlaps`` footprints in bulk
``cell_manager``                        ``recalculate_cell_properties`` in bulk; ``set_polygons`` writes into
                                        the MPG's column
``special_tissues.seat_air_spaces``     the hosts carved in bulk, the organ's topology
``needle_class``                        its lacunae seated with the organ's topology
``anatomy_writer``                      the exporter (``NetworkExporter``) and the XML writer ask the organ for
                                        the topology; the exporter reads the rows once (``_Rows``); the air
                                        rules; the added cell groups (``CGROUPS_ADDED``)
``cellset_organ``, ``roi_organ``        ``_generate_pipeline`` (written into the MPG by ``generate_cells``);
                                        the CellSet donor section stores no network
``cellset_reader``                      reads GRANAR's ``<granardata>`` files
— (new)                                 ``openalea.granap.metafspm``: ``build`` (the MPG's layout), ``cells``
                                        (the views), ``topology``, ``network`` (the stored network, built
                                        from the MPG), ``columns`` (the network as numeric variables),
                                        ``views``, ``vessels``, ``library`` (classes, templates), ``component``
                                        (``GranapAnatomy``), ``architectures``, ``plots``
======================================= ========================================================================


The pipeline as the port runs it
================================

``Organ._generate_pipeline``, for every organ but the CellSet and ROI organs (which load their cells), in the
legacy's order and with its random stream (one ``np.random.default_rng(seed)`` per organ):

1. the base shape and the layer rings, peeled from it (``generate_layer_polygons``);
2. the seeds on each ring, an ellipse of border points per cell, lower-priority overlaps removed
   (``CellGenerator.generate_cells_info``);
3. the vascular recipe (packing, bundles); the ring seeds inside the vascular tissue removed; the organ's own
   tissues (resin ducts, transfusion tissue, stomata);
4. one scipy Voronoi diagram of every seed, jittered;
5. the regions unioned per cell (``id_group``), simplified to their junctions, nested cells merged: stage
   ``grouping``, the first written into the MPG;
6. the intercellular spaces (stage ``intercellular``), then the aerenchyma, calibrated (stage ``aerenchyma``);
7. the gaps fused into their tissue (stage ``fused_gaps``);
8. the layers populated, the cells' properties recomputed: stage ``final``;
9. the topology snapped once and stored on the protoplasts (each cell's snapped ring, ``boundary``); the network
   built from it in the MPG (``metafspm.network.build_network``).

**The MPG layout** (``metafspm.build``). A section under one SubOrgan vertex: a Layer per connected region of one
tissue type, a Cell per GRANAP cell (every attribute, the polygon included), one protoplast Compartment (label
Symplastic) per Cell; the walls and wall junctions as Compartments (label Apoplastic, ``kind`` 1 and 2) anchored to
the SubOrgan, since each belongs to several cells; every network edge a Connection (membrane and wall_air:
Transmembrane; plasmodesmata: Symplastic; wall and air_link: Apoplastic). Each Compartment keeps its MECHA index
(``mecha_index``), so that the networkx graph rebuilt from the MPG is the exporter's, node for node. A cell keeps a
stable ``cell_uid`` through the stages, which GRANAP's ``id_cell`` is not.

**The stages.** A stage write keeps each surviving cell's vertex, removes the vertices of the cells gone (fusions,
nested cells, fused gaps) with their protoplasts, adds the new ones (air spaces), and rebuilds the Layers. After the
grouping the cells are views (``metafspm.cells.CellView``): a view keeps nothing itself, its attributes are read
from and written to its Cell vertex, so that GRANAP's algorithms run on them unchanged. The plain cells GRANAP's
code creates (an air space, a fusion, the calibration's copies) are adopted by the section; at the end the organ
gets plain cells back, rebuilt from the MPG.

**The network** (``metafspm.network``). The exporter's rules as array operations, from the stored rings: every ring
cut at its junctions in one pass, the walls numbered in first-encounter order, the cells of each wall in encounter
order; the lengths, midpoints, thicknesses and distances with the exporter's sums in its order; the junctions sorted
as tuples; the air links searched in the iteration order of GRANAP's sets and of networkx's neighbours. The
Connections come in the exporter's order of creation. The exporter itself remains for the exports with other
settings (the needle's bridges, another air-link radius or wall thickness), which replace the stored network.

**Exactness.** What it took to reproduce the legacy bit for bit:

- numpy's ``mean`` (a pairwise sum: sequential below 8 values, eight accumulators up to 128) emulated element by
  element over the arrays, checked against ``np.mean`` for every length;
- the snap's cluster means summed in a plain loop: since Python 3.12, ``sum()`` of floats is compensated (Neumaier),
  not a sum in order;
- shapely's ``buffer`` function defaults to 8 segments per quarter circle, the geometry method GRANAP calls to 16:
  passed explicitly in the bulk calls;
- the same GEOS operation per geometry in every bulk call (``difference``, ``intersects``, ``centroid``, ``area``,
  ``convex_hull``), so the same results;
- the dict and set orders the legacy's code iterates in, reproduced where they decide an order (the snap's maps, the
  walls, the air links).


The component
=============

``openalea.granap.metafspm.component.GranapAnatomy`` has no counterpart in the legacy. A metafspm
``StructuralComponent``: each segment of a population gets the anatomy of its class, one template generated per
class and plant parametrization (``library.AnatomyLibrary``), a copy per segment (its own vertices and state, the
template's polygons shared); the vessels of successive sections are joined by axial junctions
(``junction_rules``); a growing segment gets its anatomy when it emerges, a new one when its class changes, and
loses it when pruned; its new Compartments inherit their values from the same template node upstream
(``granap_inheritance``). Its declared inputs and outputs, and the rules it computes, are in :doc:`equations`.


The registry
============

Each behaviour of the legacy that is odd or was found during the port, with what the port does: **reproduced** (the
port gives the legacy's result), **corrected** (a reviewed difference from the legacy, replayed on it in the
comparisons), **fixed** (a bug outside the anatomy itself, fixed in the port), **kept** (a modelling choice kept as
it is), **dropped** (nothing to port).

Cells and identifiers
---------------------

====== =================================================================================== ============================
ID     legacy                                                                              port
====== =================================================================================== ============================
GR-C01 ``id_cell`` is reset to the list index at the end; before that it is not unique     reproduced; a stable
       (air spaces numbered ``len(cells) + k`` collide with earlier ids)                   ``cell_uid`` added in the
                                                                                           MPG
GR-C02 ``id_layer`` is the peeled ring's index, −1 for vascular and fill cells, 0 for air  reproduced (a Cell
       spaces, the ceiling of the mean after a fusion: not a layer pointer                 property; the MPG's Layers
                                                                                           are tissue regions)
GR-C03 a cell's ``x``, ``y`` is its first seed's (jittered), not its centroid; its         reproduced; the component's
       ``angle`` is taken from that seed and its ``radius`` from its centroid, both about  vessel positions from the
       the origin                                                                          centroids about the section
                                                                                           centre
GR-C04 the cells' ids and the Voronoi jitter follow the seed list's order: one seed added  reproduced
       or removed changes every later cell
GR-C05 ``process_voronoi_groups`` does not carry the seeds' ``cgroup`` to the grouped      reproduced; the network
       cell                                                                                falls back on the type map
GR-C06 ``create_from_input`` passes no seed: the organ is not reproducible                 fixed:
                                                                                           ``create_from_input(data,
                                                                                           seed=None)``
GR-C07 no vessel identity, except ``track_id`` on the root series' prescribed metaxylem    kept; ``vessel_id``,
                                                                                           ``pole_id``, ``family``
                                                                                           recovered after generation
                                                                                           (``metafspm.vessels``)
GR-C08 the aerenchyma calibration deep-copies every cell before each attempt,              dropped: shallow copies (a
       and runs the best attempt again at the end                                          cell's values are
                                                                                           immutable), the best
                                                                                           attempt's cells kept
====== =================================================================================== ============================

Geometry
--------

====== ==================================================================================== ==========================
ID     legacy                                                                               port
====== ==================================================================================== ==========================
GR-G01 a ring's cells: ``N = ceil(perimeter / width)`` positions resampled on the closed    reproduced
       ring, its first and last on the same point; the first dropped, so ``N − 1`` cells,
       spaced ``perimeter / (N − 1)``
GR-G02 a cell's seeds: ``n_points`` on its ellipse, the closing point (equal to the first)  reproduced
       dropped: 14 or 9 seeds; the default is 15 when the ring's width differs from its
       diameter and 10 when they are equal, so a ring given no width (0) gets 15
GR-G03 a multipart polygon is reduced to its largest part by the snap; holes are ignored    reproduced
GR-G04 the snap clusters greedily in vertex order; a vertex inside a later cluster's ball   reproduced
       takes that cluster's point; tolerance 1 % of the 5th-percentile edge, 6 decimals
GR-G05 the goldens are exact only on the pinned shapely and GEOS (their tie-breaks)         kept (pinned)
GR-G06 the layer smoothing cuts corners past the buffer, so smoothing once per ring         kept; roots, stems and
       shrinks the innermost region                                                         leaves smooth their rings
                                                                                            with factor 0
GR-G07 ``simplify_cells`` rebuilds each polygon on its own and can bury a small cell in a   reproduced
       neighbour: ``remove_nested_cells`` merges it into the smallest enclosing cell
GR-G08 the realised aerenchyma proportion depends on the geometry after the fusion, the     reproduced: the bisection
       re-simplification and the carving: calibrated by bisection over attempts             kept; calibrating
                                                                                            from the areas cannot be
                                                                                            exact
GR-G09 the Voronoi jitter is one block drawn in list order; drawing it from each seed's     reproduced
       position was tried upstream and reverted
GR-G10 the topology snapped again at every simplification (4 to 5 times per section), and   dropped: computed once per
       again at each export and XML write                                                   distinct input (a cache),
                                                                                            stored at generation
====== ==================================================================================== ==========================

Roots
-----

====== ===================================================================================== ===========================
ID     legacy                                                                                port
====== ===================================================================================== ===========================
GR-R01 ``id_layer=i_slice`` stamped on the monocot root's metaxylem                          reproduced
GR-R02 ``DicotRootAnatomy._phloem_adjustment``: the primary cambium's cell diameter with     reproduced
       secondary growth, the secondary cambium's (0 when absent) without: the branches look
       swapped
GR-R03 the dicot recipe's comment says stele cells ≥ 75 % engulfed by vessels are dropped;   reproduced (0.6)
       the code's default is 0.6
GR-R04 the valley phloem of the star monocot and dicot roots draws its sieve diameters       reproduced
       with ``diameter_min = diameter_max``: the standard deviation has no effect (the
       draws are clipped to the mean, the random stream still consumed)
GR-R05 the default monocot bundles pass no ``diameter_min``, so the draws are centred on     reproduced
       ``(d_max + d_min)/2`` with ``d_min = max(d − 3 sd, 0.01 d)``, below the requested
       diameter
GR-R06 the arch xylem parses ``relative_phloem`` and does not use it (the phloem sits        reproduced
       mid-band)
GR-R07 the bundle orientation, the aerenchyma files' angles and the dicot series'            reproduced
       distances are taken about the origin, not the stele's centre
GR-R08 ``n_vascular_bundles`` 0 gives no vascular bundles at all outside a root series       kept
       (in a series, the protoxylem poles stay)
GR-R09 GRANAP has no rule from a root's diameter to its parameters (only the root series'    kept: a class gets its
       schedules along a root)                                                               preset unchanged unless its
                                                                                             own parametrization says
                                                                                             otherwise
====== ===================================================================================== ===========================

Shoots and needles
------------------

====== ====================================================================================== =========================
ID     legacy                                                                                 port
====== ====================================================================================== =========================
GR-S01 the monocot stem's random bundle placement over the whole section draws                reproduced (kept upstream
       :math:`r = R\,U^{0.35}`, biased to the periphery                                       as the historical rule)
GR-S02 the leaf's vein placement ``center`` is the same as ``even`` for more than one vein    reproduced
       (the docstring says a midrib at 0)
GR-S03 the needle's elliptical central cylinder is shifted by the parameter                   reproduced
       ``thickness / 2.2``, not by the resolved thickness: 0 when the thickness is automatic
GR-S04 the transfusion tracheids' ratio defaults to 0.5 in the ring mode and 1.0 in the       reproduced
       packing mode
GR-S05 a stoma's sub-stomatal chamber of several parts keeps its first part, not its          reproduced
       largest
GR-S06 ``fit_inner_ellipse`` returns its last ellipse when none fits                          reproduced
GR-S07 the needle's mesophyll rhombi take a smoothness they do not use                        reproduced
GR-S08 GEOS ties resolved per platform (inscribed circles, boundary tests, the xylem files'   reproduced (the upstream
       strips): snapped to a grid or tested with a tolerance upstream                         fixes are in the legacy)
====== ====================================================================================== =========================


The network
-----------

====== ===================================================================================== ============================
ID     legacy                                                                                port
====== ===================================================================================== ============================
GR-N01 walls and junctions exist only inside the exporter, with throwaway ids                corrected: stored at
                                                                                             generation as Compartments
                                                                                             under the SubOrgan, their
                                                                                             MECHA index kept
GR-N02 the XML writer keys a wall by its junction pair, the exporter by its junction pair    reproduced (both; the XML
       and its protected interior vertices: two numberings when cells are protected          writer reads the stored
                                                                                             topology)
GR-N03 a cell with fewer than two junctions has one closed wall; a wall of one cell is a     reproduced
       boundary wall (the outer wall's thickness added)
GR-N04 the air-link radius defaults to 1× the median wall length; the comment said 3×        kept at 1×, the comment
                                                                                             corrected
GR-N05 an air space's membrane is ``wall_air`` only on a wall with a protected interior      corrected: every membrane of
       vertex; a sub-stomatal chamber (``protect_topology`` only) keeps membranes, a liquid  an air node is ``wall_air``
       cell
GR-N06 a stomatal pore is a protoplast with membranes and plasmodesmata to its guard cells,  corrected: pores are air
       never linked to its chamber (the guard cells meet between them)                       nodes by type (wall_air, no
                                                                                             plasmodesmata), each air-
                                                                                             linked to its nearest
                                                                                             protected air space
GR-N07 nine cell types have no MECHA cell group and fall back on 4 (cortex): ``sieve         corrected: sieve
       element`` and ``companion cell`` (no phloem on shoots for MECHA), ``spongy``,         element 11, companion cell
       ``transfusion``, ``sclerenchyma``, ``phellem``, ``phelloderm``, ``phellogen``,        12, transfusion 17, the
       ``medullar_ray``                                                                      others explicitly 4
GR-N08 ``companion cell`` and ``companion_cell`` are both used as tags                       kept (both 12)
GR-N09 the monocot leaf preset has no intercellular spaces, and its lacunae are too far      kept (the preset unchanged)
       apart for an air link: no air network but its stomatal chambers
GR-N10 the exporter reads the cells' frame row by row (pandas)                               dropped: the network built
                                                                                             from the MPG's arrays; the
                                                                                             exporter reads the rows
                                                                                             once
— (*)  the network built from the MPG failed on sections without shared junctions (ROI)      fixed
— (*)  the air links' search on the holly leaf (``wall_neighbours`` 26 million calls)        fixed: the same links, each
                                                                                             wall's neighbours once, the
                                                                                             pairs without a motif
                                                                                             skipped
====== ===================================================================================== ============================

Imported anatomies
------------------

====== ================================================================================== ===========================
ID     legacy                                                                             port
====== ================================================================================== ===========================
GR-I01 ``CellSetOrgan`` reads a GRANAR ``<granardata>`` file as a CellSet file: the last  fixed: the first name kept,
       name of a group id kept (GRANAR lists aliases after the tissue), its millimetres   the scale from the root
       scaled as micrometres, its resolved stele regenerated in every stele cell          element, no regeneration
                                                                                          for ``<granardata>``
GR-I02 the CellSet organ's donor section runs the whole pipeline with its network         corrected: the donor stores
                                                                                          no network (its cells only)
====== ================================================================================== ===========================

The legacy snapshot
-------------------

====== ================================================================================== ===========================
ID     legacy                                                                             port
====== ================================================================================== ===========================
GR-L01 the ``tissue`` branch was merged into the legacy here, before upstream merges it:  kept; snapshot again when
       its conflicts (both sides adding layer parameters) resolved by keeping both        upstream merges
GR-L02 the legacy's ``stem_dicot_class`` imported the port's ``secondary_growth``         fixed in the legacy copy
                                                                                          (imports rewritten)
GR-L03 the 3D extrusion regenerates the 2D section, then restores it                      kept; nothing 3D in the MPG
====== ================================================================================== ===========================

(*) Found in the port's own code while porting, not in the legacy.
