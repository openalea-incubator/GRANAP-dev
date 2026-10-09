# Cross-platform determinism figures (golden regressions)

Evidence for the determinism fixes that let `test_vascular_regression.py`'s exact census
pass on Linux, Windows **and** macOS at once. The stem figures were produced at `seed=0`
on numpy 2.4.2, shapely 2.1.2, GEOS 3.14.1, scipy 1.17.0 — Ubuntu under WSL and native
Windows. macOS CI runs **osx-arm64**, which contracts `a*b+c` into a single FMA with one
rounding and so diverges from x86 on tie predicates even at identical library versions.

## The fixes

All three were **exact ties**, decided by the last bits of coordinates that glibc, UCRT
and Apple/arm64 `sin`/`cos`/`arctan2` compute differently:

1. `GeometryProcessor._chebyshev_center` (`geometry_collection.py`) fed an unsnapped
   polygon to `shapely.maximum_inscribed_circle`. A zone symmetric about an axis has
   two tied optimal centres — `dicot_stem`'s 3-o'clock bundle sits exactly on the
   x-axis — so the noise picked the branch and the whole Apollonian packing followed.
   The polygon is now snapped to a 1e-9 grid first.
2. `_grow_bundle_sheath` (`vascular_bundle.py`) asked `env.contains(b)` for points `b`
   on the bundle *footprint*'s boundary, to distinguish "sits on a protruding fibre
   cap" from "sits on the envelope". But the footprint is the envelope plus the caps,
   built from the same coordinates through the same transform, so a non-cap point lies
   exactly *on* `env`'s boundary — and in the arc-bundle path `foot` **is** `env`. The
   test is now tolerant, so only a genuinely protruding point reads as a cap.
3. `NeedleAnatomy.retag_corner_parenchyma` (`needle_class.py`) asked
   `zone.contains(Point(c.x, c.y))` where the corner pocket is cut out of the very layer
   polygon those cells were seeded in, so it inherits that outline — and the layer's
   *border* seeds sit exactly on it. 44 of 250 candidates were boundary cases, so **half
   the Strasburger cells were tagged by arithmetic noise**; arm64 simply resolved 4 of
   them the other way. Boundary seeds now count as inside. This one changed anatomy, so
   the needle goldens were deliberately refrozen — see the retag figures below.

A fourth change — making `CellGenerator.voronoi_diagram`'s jitter a hash of each seed's
own position instead of indexing one `rng.uniform(size=(n, 2))` block by list order —
was implemented, measured and **reverted**. It changed no census at all, so it fixed
nothing measurable, while re-rolling the jitter realisation, which
`test_cellset_organ.py::test_no_interior_border_walls` could not survive at the time: it
held only at `seed=0` (seeds 1-5 gave 22-49 interior border walls on completely
unmodified code). That graft fragility has **since been fixed** — interior border walls
are now 0 at every seed 0-15, asserted by `test_no_interior_border_walls_at_any_seed` —
so the blocker is gone and revisiting the jitter now costs a deliberate golden refreeze
rather than a correctness regression.

## Figures

| file | what it shows |
| --- | --- |
| `crossplatform_dicot_stem_seeds.png` | every seed point before the Voronoi step: tagged by tissue, both platforms overlaid, the displacement field, and displacement vs. seed index |
| `crossplatform_dicot_stem_tissues.png` | the finished cells filled by tissue type, per platform, plus a map of cells lacking a bit-identical counterpart |
| `crossplatform_dicot_stem_cells.png` | the cell polygons themselves, per platform, overlaid, and zoomed |
| `crossplatform_monocot_stem_*.png` | the same three views for `monocot_stem` |
| `sheath_fix_monocot_stem_before_after.png` | the deliberate anatomy change behind the `monocot_stem` golden update: whole section before/after, the census delta, and a bundle zoom on the sheath ring |
| `retag_tie_zoom_needle_default.png` | the needle retag decision: central cylinder under both candidate tie rules, with the 44 contested cells outlined in red. They form the single-cell rim where the cylinder meets the transfusion tissue — which is what turned a numeric tie into an answerable anatomical question |
| `retag_tie_zoom_needle_features.png` | the same for `needle_features` (30 contested cells; excluding them would have cut the cluster to 10) |

## Result

Both stem organs are **bit-identical across Linux and Windows**: equal seed counts in
every one of the 37 (`dicot_stem`) / 43 (`monocot_stem`) `(tissue, id_layer)` buckets,
every matched seed agreeing to <1e-14, and every cell centroid matching at a distance
of exactly `0.0`. The diverged-cell panel is empty in both tissue figures. All nine
`GOLDEN` entries produce the same census on Linux and Windows, and the needle fix
removes the last macOS/arm64 divergence.

Why these are fixes rather than tuned constants: in each case the tie cells are separated
from the genuine decisions by many orders of magnitude — for the needle, ties within
5.3e-17 against a nearest true interior cell beyond 1e-6 — so any tolerance in that gap
selects the same set. `test_needle_retag_has_no_knife_edge_decisions` pins the gap itself,
so if seeding ever parks a cell mid-band it fails there rather than as a census drift on
one CI runner. **Measure that distribution before adding a tolerance elsewhere**: without
a gap, a tolerance only relocates the arbitrariness.

## A note on reading the seed figures

The bottom panel of each `_seeds.png` plots *index-aligned* displacement. That measure
is only meaningful while both platforms emit the same seeds in the same order: once one
count changes, index `i` refers to a different seed on each side and the curve jumps to
~1e-2 mm regardless of whether anything moved. The pre-fix version of these figures
showed exactly that artifact. Judge divergence from the per-bucket nearest-neighbour
match instead — which is what the tissue figures' diverged-cell panel uses.
