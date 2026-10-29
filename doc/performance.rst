===========
Performance
===========

The time a section takes to generate, the time a population of sections takes in the MPG, where it goes, and what
changes it. Each number was measured during the port and is given with the GRANAP commit that recorded it (``git
show <commit>`` in GRANAP's repository; a metafspm commit where stated), or, for the times before the port, with
GRANAP's ``doc/performance_proposals.md``; the holly leaf's times, recorded in no commit, are given with their date. Times are wall-clock times of ``generate_cells`` (or of the stated step) at
``seed=0``, on the pinned geometry stack (shapely 2.1.2, GEOS 3.14.1), without the profiler unless stated. Absolute
times vary from run to run on the development machine; the speedups were measured as medians
of interleaved runs (the number of runs is given where it was recorded).

Every speedup listed for the port is exact: the sections, the networks and the XML files are those of the legacy
(:doc:`legacy`), checked by the comparison folder (``test/comparison``) and ``example/comparison/compare_sections.py``.
The changes that would change the anatomies were measured or proposed, not applied (see the last section).


Before the port
===============

GRANAP's own optimisations, before the port (``doc/performance_proposals.md``, July 2026, Windows): vectorised
point-in-polygon tests, a batched KD-tree query in the snap, an STRtree in the aerenchyma merge, the geopandas
``dissolve`` of the Voronoi grouping replaced by a union per group, and the validity check on Voronoi regions
dropped. All byte-identical. They took ``dicot_primary`` from 8.74 to about 6.5 s, the needle from 5.06 to about
3.9 s and the monocot root from 4.02 to about 3.0 s.

The same document profiled a secondary-growth dicot root (``dicot_nettle``: 504,259 Voronoi seeds for 36,204
cells, about 14 seeds per cell; about 145 s on Windows):

==================================== ========== =====
phase                                time       share
==================================== ========== =====
Voronoi grouping                     63.0 s     43 %
seeds (``generate_cells_info``)      33.7 s     23 %
vascular and organ tissues           34.0 s     23 %
Voronoi diagram                      8.0 s      6 %
intercellular spaces                 5.3 s      4 %
cell properties, export, layers      about 1 s
==================================== ========== =====

At the start of the port (Linux), the same root took about 53 s, and GRANAP's test suite (166 tests) 12 min 28 s
(``425dc4f``).

The cost follows the number of Voronoi seeds: each cell is tessellated from a ring of 9 to 14 seeds (its border
points), so the seeding, the Voronoi diagram, the grouping unions and the intercellular spaces all scale with about
ten to twenty-five times the number of cells.


One section
===========

At the first measurement of the port (cells and network; the MPG-backed column also writes the stages, the stored
topology and the network into the MPG; ``f2908a7``):

================ ====== ========= ==========
organ            cells  legacy    MPG-backed
================ ====== ========= ==========
monocot root     1384   7.3 s     7.7 s
dicot root       2834   16.0 s    16.7 s
dicot stem       4800   5.5 s     5.9 s
================ ====== ========= ==========

Writing and reading the MPG added about 0.8 s to a 10 s generation of the monocot root (a section of 30.5 k
vertices; ``80e8643``).

The exact speedups, in the order they were made (generation plus the network's matrix; medians of 3 to 5
interleaved runs), each with the commit that recorded it:

================================================== ================== ================== ==================
change                                             dicot root         monocot root       dicot leaf
================================================== ================== ================== ==================
a topology cache (``3cc3781``)                     6.49 → 5.71 s      3.02 → 2.59 s
shallow copies in the calibration (``72d2f3e``)    5.81 → 5.60 s      2.77 → 2.53 s
exporter rows read once (``722d417``)              8.02 → 7.02 s      3.92 → 3.26 s
                                                   (machine loaded)
network in the MPG, no GeoDataFrame (``bc54752``)  5.71 → 5.22 s      2.71 → 2.42 s      3.98 → 3.61 s
merge, simplification in bulk (``dcb8184``)        −4 %               2.45 → 2.21 s      unchanged
the snap on arrays (``d8019dc``)                   4.90 → 4.58 s      2.24 → 2.01 s      3.61 → 3.13 s
the fill planned once (``6214e84``)                4.47 → 4.24 s      2.00 → 1.89 s
intercellular spaces in bulk (``d7ea0dd``)         4.09 → 3.95 s      1.85 → 1.76 s      unchanged
gap fusion's unions once (``73a7e1b``)             5.12 → 4.86 s
                                                   (machine loaded)
seed footprints in bulk (``cd96edc``)              4.05 → 3.84 s      1.84 → 1.71 s      −2 %
================================================== ================== ================== ==================

From ``75bd5aa`` to the gap fusion, 13 October 2026 (medians of 3; ``73a7e1b``): dicot root 5.66 → 3.80 s
(−33 %), monocot root 2.80 → 1.75 s (−37 %), dicot leaf 3.84 → 2.89 s (−25 %). The gap fusion also took the
monocot stem from 8.94 to 8.39 s (machine loaded; ``73a7e1b``). Keeping the best aerenchyma attempt instead of
running it again saved 3 % on the monocot root and 1 % on the dicot root (``5252525``, recorded in ``8994627``).
Within these, the network's storage went from 0.65 to 0.24 s and the adjacency matrix from 0.21 to 0.02 s
(``bc54752``); the merge, over 8 calibration attempts, from 0.45 to 0.25 s (``dcb8184``).

The holly leaf took 52.8 s on the port against 28.7 s on the legacy, the difference being the network the port
builds at generation (the air links' search: ``wall_neighbours`` called 26 million times, 91 s of 136 s under the
profiler). With each wall's neighbours computed once, the pairs without a possible motif skipped and the distance
candidates taken from an STRtree, it takes 31.8 s on the port with its network, against 33.2 s for the legacy's
cells plus 31.4 s for its network export (idle machine). These holly leaf times were measured on 20 October 2026,
before and after the change (``944febe``), and recorded in no commit.

Of GRANAP's 36 examples run on the port, the longest is ``example/root/dicot_oak.py`` (227 s;
``d4fe6ee``).


Where the time goes
===================

A dicot root, one template, after the network's conversion (GRANAP's stage log; ``bc54752``):

======================================= ========
stage                                   time
======================================= ========
seeds                                   0.64 s
vascular and organ tissues              0.34 s
Voronoi diagram                         0.17 s
grouping (unions, simplification)       1.21 s
intercellular spaces and aerenchyma     1.87 s
fused gaps                              0.15 s
network                                 0.24 s
stage writes into the MPG               0.13 s
======================================= ========

Under the profiler, the aerenchyma calibration (9 attempts, each re-simplifying the whole section) took about 28 %,
the seeds (68 k polygons and 46 k points created one by one) 16 %, the cell views' attribute reads and adoptions
about 11 %, the grouping's GEOS unions 9 % (``bc54752``).

The snap (``CellGenerator._build_topology``) runs 4 to 5 times per section (the grouping, the air spaces, the
calibration, the network). Before its conversion it took 0.90 s of a dicot root's 4.34 s (21 %) and 0.44 s of a
monocot root's 2.29 s (``dcb8184``). After it, 0.06 to 0.13 s per call (2,834 cells for the largest), about 0.25 s per dicot root
over its 4 calls, 6 % of 3.8 s; by phase for the largest call: rings and tolerance 0.015 s, KD-tree query 0.007 s,
clustering 0.011 s, vertex map 0.011 s, edge map 0.008 s, junctions 0.005 s; without the profiler (``fc3d54f``). cProfile had
given 0.77 s (``d753434``), as it inflates the pure-Python loops several times over.

What remains is GEOS (the grouping's unions, the intercellular unions) and the views of the cells in the stages
after the grouping (about 3 % of a dicot root: adoption and attribute reads; ``55c618f``).

The continuous dicot stem takes 18.7 s, mostly in its vascular packing (``pack_circles``): a ``set_precision`` and a
``maximum_inscribed_circle`` per packed circle, 1,315 of each, 14 s under the profiler. The stems' and leaves'
finishers (``_finalize_pith_and_lacunae``, ``_convert_inter_bundle_aerenchyma``) take 6 to 18 ms on sections of
2.4 to 18.7 s (``53e939e``).


A population of sections
========================

The component generates one template per class and copies it onto every segment of the class: the generation is
paid once per class, the copies per segment.

**A seedling of roots** (``test/component/bench_seedling.py``: 48 segments in 3 diameter classes, 886 k vertices,
241 k nodes, 594 k edges, 2,231 junctions): the 3 templates 6.6 s; the architecture and its 48 copies 2.6 s; the
properties converted to arrays 14.6 s, then 3.6 s once metafspm converts each property in one pass
(``ArrayDict.from_items``); the DataStructure with its junctions 0.7 s; the graph view 0.09 s (``4680eda``; the conversion in one pass
recorded in ``6955b45``).

**A seedling with stems and leaves** (``example/mpg/seedling.py`` at the time: 12 sections from 4 templates, 212 k
Compartments, 526 k Connections, 1,555 junctions): built in 24 s (``35f8cf3``).

**Growth steps** on the seedling of 48 segments (1.46 M vertices), each change exact (the same junctions, the same
inherited values, checked against a reference run). The columns' changes were recorded in ``3faa1d9``,
``ae49977`` and ``fc3d54f``:

.. list-table::
   :header-rows: 1

   * - step
     - first
     - pairs per template, bulk removals
     - copies as appends
     - array mirrors
   * - 9 new segments
     - 5.3 s
     - 3.3 s
     - 2.2 s
     - 1.6 s
   * - a class change (with the new template's generation)
     - 669 s
     - 7.9 s
     - 7.8 s
     - 6.4 s
   * - pruning
     - 0.05 s
     - 0.04 s
     -
     -
   * - seedling build
     -
     - 27 s
     - 21 s
     - 13.6 s

The columns: the first measurement; the vessel pairs and the inheritance computed once per pair of templates, with
metafspm's owner maps cached and junctions wired only for the changed sections, and its removals in one pass (the
669 s were vertex-by-vertex removals: a bulk property deletion per protoplast and a list search per wall); the
copies' values appended to the array properties in one copy; metafspm's array mirrors of parents and complexes.
Under the profiler the emergence step went from 5.1 s to 4.3 s and 2.9 s; what remains of it is the
DataStructure's passes over the whole MPG (owner scales and variable stores 0.58 s, topology arrays 0.55 s,
rewiring 0.45 s, mostly the linked parent of every segment; ``ae49977``).

In metafspm, reading the anatomy-mode variables in bulk (metafspm ``4051fba``) took the component's construction on
its seedling from 477 s to 0.6 s and its call from 10 s to 0.8 s (metafspm ``c06b9c6``); deleting array-backed
properties in bulk took the replacement of a section's stored network from 301 s to 3.6 s (``b15d496``); a
section's MPG writes went from 0.25 to 0.07 s with ``ArrayDict.update`` in bulk (metafspm ``add82ba``, recorded in
``bc54752``); bulk creation takes 0.17 s for a section of 45.7 k vertices (``80e8643``).


Options that change the time
============================

**The classes.** The bin widths of the class inputs (by default the diameter in bins of 0.05 mm) set how many
templates a population needs and how often a growing segment is replaced. A class change costs a template's
generation the first time the class appears, then a copy.

**The aerenchyma.** A section with ``aerenchyma_proportion > 0`` is calibrated: up to 21 attempts (the request, then
up to 20 bisection steps), each fusing and re-simplifying the section, until the realised proportion is within
0.005 of the request or the interval is under 1e-4.

**The seeds per cell.** A layer's ``n_points`` (by default 15, or 10 for a ring whose cell width equals its
diameter; one seed fewer, the ellipse's closing point dropped) sets its seeds per cell, which drive most of the
cost. Lowering it changes every cell's shape.

**The network's settings.** The stored network is built at generation with the default wall thicknesses and air-link
radius. An export with other settings, or with the needle's bridges, runs the exporter again and replaces the
stored network. ``stores_network = False`` (the CellSet organ's donor section) skips the network.

**Measured, not applied.** These change the anatomies and would need reviewed golden changes; the port does exact
changes only.

- Fewer seeds per cell (8 and 6 instead of 15 and 10): ``dicot_primary`` 7.5 → 5.7 s, 2617 → 2641 cells, coarser
  walls (``doc/performance_proposals.md``).
- Proposed in ``doc/performance_proposals.md``, not measured in the port: fewer resampling points (200) in the
  intercellular spaces' smoothing, no intermediate simplification of the intercellular spaces, a looser
  inscribed-circle tolerance in the packing, coverage unions for the grouping.
- Proposed during the port, not measured: a coarser aerenchyma calibration (``f2908a7``), a secant search instead of
  the bisection (``8994627``).
- The aerenchyma calibrated from the areas alone: it cannot reproduce the legacy, since the realised proportion is
  measured after the air spaces' fusion, the re-simplification and the carving (``8994627``).
- The topology kept through the stages instead of snapped again: at most about 0.6 s (8 %) of a dicot root, and not
  exact, since the snap is global (its tolerance from the section's 5th-percentile edge, its clustering greedy in
  vertex order). Dropped (``a1cd820``).
- Compiled kernels (numba) for the snap: at best about 3 %, and a compilation of about a second per kernel at the
  first call of each process. Dropped (``fc3d54f``).
- Several sections in one pass: only the per-call overhead of the bulk calls would go. Dropped (``a40c3f2``).
- The solve per template: belongs to the MECHA port (``a40c3f2``).
