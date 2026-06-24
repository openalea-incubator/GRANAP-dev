# Performance — done & proposed speed-ups

Companion note to the tissue refactor docs. It records (a) the optimisations
already applied to `Organ.generate_cells` and its helpers, and (b) the proposed
further speed-ups that were **not** applied because they change the golden
baseline (cell geometry) and therefore need a deliberate visual review first.

All timings are `generate_cells()` wall-clock, `seed=0`, min of several runs,
conda env `granap`.

## Where the time goes (per-phase, dicot_primary)

| phase | cost |
|---|---|
| Cell seeds (`generate_cells_info`) | ~1.2s |
| Voronoi diagram | ~0.4s |
| Voronoi grouping (`process_voronoi_groups` + `simplify_cells`) | ~2.5s |
| Intercellular spaces (`add_intercellular_spaces`) | ~2.0s |
| everything else | <0.1s |

The two giants — *Voronoi grouping* and *Intercellular spaces* — both scale with
the **number of Voronoi seeds**: each biological cell is tessellated from a ring
of ~10–15 border-point seeds (so neighbouring cells interlock), so dicot builds
~42k seeds and dissolves them down to ~2.6k cells (~16 seeds/cell).

## Already done (byte-identical — golden unchanged)

These are pure mechanical/algorithmic wins; the produced anatomy is identical.

- **Vectorised point-in-polygon tests** in `generate_cells_info` and
  `resolve_cell_border_overlaps`: replaced ~150k per-point `shapely.Point` +
  `.contains()`/`.covers()` with batched `contains_xy`/`covers` over arrays,
  prepared geometries, and hoisted the loop-invariant `next_inner` polygon.
- **`_build_topology`**: vectorised the per-edge length computation; replaced the
  ~42k per-vertex `cKDTree.query_ball_point` calls with one parallel batch query
  (`workers=-1`) feeding the identical greedy snap loop.
- **`merge_intercellular_aerenchyma`**: replaced the O(n²) all-pairs
  `touches`/`intersects` air-cell scan (~180k predicate calls) with an STRtree
  bbox query; collapsed `touches OR intersects` to just `intersects`.
- **`smoothing_polygon`**: hoisted the invariant `is_closed` test out of the
  Laplacian iteration loop.

**Net effect so far:** dicot_primary 8.74s → ~6.5s, needle 5.06s → ~3.9s,
monocot 4.02s → ~3.0s (~25% across the board), with all 7 golden census
configs byte-identical.

---

## Proposed (changes the golden baseline — needs visual sign-off)

Ranked by expected payoff. Each rebaselines `test/test_vascular_regression.py`
and alters cell shapes, so each wants a rendered-section comparison before
commit.

### P1 — Reduce the per-cell Voronoi seed density  *(biggest lever, ~24%)*

Lower the border-ring resolution in `CellGenerator.cell_border`
(`n_points = 15 / 10`) to e.g. `8 / 6`.

- **Measured:** dicot_primary 7.5s → **5.7s** (~24%), biological cell *count*
  essentially unchanged (2617 → 2641).
- **Why it works:** seeding, Voronoi, and polygon construction/dissolve all scale
  with seed count; halving the ring roughly halves the seeds.
- **Cost:** cell boundaries become **coarser / more polygonal** (straighter shared
  edges, less organic wiggle). Purely visual, but it changes every cell and
  rebaselines the golden.
- **Effort:** one line. Push lower (6/5) for more.

### P2 — Reduce intercellular smoothing resolution  *(~0.3–0.5s)*

In `GeometryProcessor.smoothing_polygon`, the per-cell Laplacian smoothing
resamples each cell to **200 points** then runs 10 passes. Cells have ~16–30
vertices, so 200 is generous.

- Lower `target_n_points` (e.g. 80–120) and/or the iteration count.
- **Cost:** air-space / intercellular gap outlines get slightly less smooth.
- **Effort:** one or two constants. Changes only tissues with `smoothness > 0`.

### P3 — Skip the intermediate `simplify_cells` in `_apply_intercellular`

`add_intercellular_spaces` simplifies the full cell set in `_apply_intercellular`
**and** again in `merge_intercellular_aerenchyma`. The first pass feeds the
aerenchyma area accounting, so dropping it is not byte-identical (it nudges
cell areas, which can change which cells are selected as aerenchyma).

- **Saves:** ~0.3s (one `_build_topology` pass).
- **Cost:** small geometry drift; safe only for configs without aerenchyma, or
  with a reviewed golden update.

---

## Free win (no golden change) — not yet applied

### F1 — Drop `is_valid` / `buffer(0)` on Voronoi cells in `process_voronoi_groups`

Voronoi cells are **convex by construction**, hence always valid, so the
per-polygon `is_valid` check + `buffer(0)` repair (~42k calls, ~0.6s) is almost
always a no-op.

- **Saves:** ~0.5s, byte-identical in the normal case.
- **Caveat:** keep a guard for genuinely degenerate/empty regions (the few that
  the current code already special-cases) rather than removing the check blindly.

---

## Not worth it

- **Replacing the geopandas `dissolve`** in `process_voronoi_groups` with a manual
  group-union: profiling shows its cost is the polygon unions themselves
  (unavoidable work), not the pandas overhead — marginal gain, real risk.
- **Dropping below ~6 border points** (P1 taken too far): cells become visibly
  blocky; not recommended past 6/5.
