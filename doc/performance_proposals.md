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

### F1 — Drop `is_valid` / `buffer(0)` on Voronoi cells in `process_voronoi_groups`  *(DONE 2026-07-07)*

**Applied.** Removed the per-seed `is_valid` check + `buffer(0)` repair; the
`-1`/empty-region guard already covers degenerate regions. Byte-identical across
all four `perf_characterize.py` configs + the 7-config golden regression suite.


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
  **⚠ Revised for large configs — see "Heavy configs (secondary growth)" below.**
  On `dicot_primary` (~2.6k cells) the pandas overhead really is marginal, but on
  `dicot_nettle` (~36k cells) it becomes ~13s of pure groupby/`iterrows` overhead
  and is now the single biggest byte-identical win.
- **Dropping below ~6 border points** (P1 taken too far): cells become visibly
  blocky; not recommended past 6/5.

---

# Heavy configs (secondary growth) — `dicot_nettle` profile (2026-07)

The section above was measured on `dicot_primary` (~2.6k cells, ~42k seeds).
Secondary-growth dicots are an order of magnitude heavier and the profile shifts,
so this section adds that datapoint and the implementation plan for the wins that
matter *at scale*. All timings `seed=0`, conda env `granap`, Windows.

## How to reproduce

```
python doc/perf_profile_nettle.py
```

`doc/perf_profile_nettle.py` runs `example/dicot_nettle` with INFO logging on (so
the per-phase timers, which are `log.info(...)` calls already in
`Organ.generate_cells`, `organ_class.py:210`, print) plus a cProfile pass sorted
by `tottime` and `cumulative`.

## Measured (dicot_nettle, one `generate_cells`, ~145s)

**Scale: 504,259 Voronoi seeds → 36,204 biological cells (~14 seeds/cell).**

| phase | time | % | dominated by |
|---|---:|---:|---|
| **Voronoi grouping** | **63.0s** | 43% | `process_voronoi_groups` 41.6s + `simplify_cells` ~21s |
| Cell seeds (`generate_cells_info`) | 33.7s | 23% | 504k seed `Cell`/`Point`, `Cell.jitter` ×504k |
| Vascular + organ tissues | 34.0s | 23% | `fit_secondary_xylem`→`fill_by_rings` 13.6s; packing inscribed-circle 5.6s |
| Voronoi diagram | 8.0s | 6% | scipy Voronoi on 504k pts |
| Intercellular spaces | 5.3s | 4% | |
| cell properties / export / layers | ~1s | — | |

Top raw hotspots (`tottime`): shapely dispatch wrapper ×8.8M (12.5s),
`shapely.union_all` ×33.8k (11.0s), `_build_topology` (9.8s), `numpy.asarray`
×2.3M (6.7s), `shapely.points` ×1.66M (6.1s), `maximum_inscribed_circle` ×593
(5.6s), `Cell.jitter` ×504k (2.5s).

**Root cause:** everything scales with the 504k seeds, processed one Python/
shapely object at a time (a `shapely.Point` per candidate, a pandas row per cell).
The wins below remove *per-object overhead*; they are not algorithmic changes.

## Verification harness (use for every change here)

`doc/perf_characterize.py` pins the `seed=0` cell-type census + a geometry hash
for `{monocot_default, monocot_arch, dicot_primary, dicot_secondary}`. Run
`python doc/perf_characterize.py save` once on the current tree, then
`python doc/perf_characterize.py` (check) after each edit. Wins ①–② below must
stay **byte-identical** (hash unchanged); ③–④ change the hash and need
`test/test_vascular_regression.py` re-baselined + a rendered comparison.

---

## ① Drop geopandas `dissolve` in `process_voronoi_groups`  *(DONE 2026-07-07 — byte-identical)*

**Applied** exactly as proposed below: plain-Python `defaultdict` grouping +
`shapely.unary_union` per group, first-seen representative, groups emitted in
sorted `id_group` order (matches dissolve). The now-unused `geopandas` import was
dropped. Byte-identical across all four `perf_characterize.py` configs + the
7-config golden regression suite.



**Where:** `generate_cell.py:404-434`.

**Current:** builds a 504k-row `GeoDataFrame` from `cell_to_dict()`, calls
`gdf.dissolve(by="id_group", as_index=False)` (pandas groupby → per-group
`unary_union` → `.agg`), then rebuilds `Cell`s by iterating `grouped_gdf.iterrows()`.
The unions are real work, but the GeoDataFrame construction + pandas groupby +
`iterrows` are ~13s of pure overhead at this scale.

**Proposed:** group in plain Python and union per group directly:

```python
from collections import defaultdict
groups: dict[Any, list] = defaultdict(list)
rep: dict[Any, Cell] = {}
for c in updated_cells.cells:            # updated_cells already excludes "outside"/None
    groups[c.id_group].append(c.polygon)
    rep.setdefault(c.id_group, c)        # first-seen == dissolve's "first" strategy
final = CellManager()
for gid, polys in groups.items():
    poly = shapely.union_all(polys)      # same union dissolve computes
    r = rep[gid]
    final.add_cell(Cell(type=r.type, x=r.x, y=r.y, diameter=r.diameter,
                        id_cell=r.id_cell, id_layer=r.id_layer, id_group=gid,
                        angle=r.angle, radius=r.radius, area=poly.area, polygon=poly))
return final
```

**Notes / gotchas:**
- Dissolve's default aggregation is *first* per group — preserve insertion order so
  `rep[gid]` picks the same representative row. `updated_cells.cells` is already in
  seed order, so a plain first-seen dict does it.
- `area` is recomputed from the unioned polygon (dissolve did `.geometry.area`).
- No RNG, no geometry change → census hash must stay identical. Verify with
  `characterize.py check`.

## ② Vectorise the remaining per-point containment tests  *(DONE 2026-07-07 — byte-identical)*

**Applied** at all six sites in the table below (`contains(Point)` /
`intersects(Point)` → array `shapely.contains_xy` / `intersects_xy`). The two
grid loops (`_fill_medullar_rays`, `_fill_ray_parenchyma`) batch one containment
call per radius ring; the group cleanups (`fit_primary_cambium_elements`,
`_prepare_medullar_rays`, `_remove_stele_engulfed_by_xylem`) batch over each
cell/group's seeds via the new `DicotRootAnatomy._groups_intersecting` helper.
Byte-identical across `perf_characterize.py` + the golden regression suite.



`generate_cells_info` and `resolve_cell_border_overlaps` were already vectorised
(see "Already done"). These fill/mask loops were **not** — they still build a
`shapely.Point` per candidate and call `prep.contains(...)` one at a time (the
1.66M `shapely.points` in the profile). Replace each with a single
`shapely.contains_xy(polygon, xs, ys)` (or `intersects_xy`) over numpy arrays.

Sites (all hot on secondary growth):

| file:line | call | fix |
|---|---|---|
| `tissue_class.py:235,249` (`fill_by_rings`) | `filter_z.contains(Point(pt))` per seed & per border pt | collect candidate xy per ring, one `contains_xy` |
| `root_dicot_class.py:727` (`_fill_medullar_rays`) | `geom_prep.contains(Point(px,py))` per lane×radius | batch the (px,py) grid, one `contains_xy` |
| `root_dicot_class.py:831` (`_fill_ray_parenchyma`) | `ray_zone.contains(Point(px,py))` per lane | same |
| `root_dicot_class.py:940` (`_prepare_medullar_rays`) | `mr_cambium_zone.contains(Point(c.x,c.y))` per cambium cell | arrays of cell x/y, one `contains_xy`, boolean-mask the list |
| `root_class.py` `_remove_stele_engulfed_by_xylem` | `xylem_star_prep.contains(Point(c.x,c.y))` per stele cell | same array pattern |
| `root_dicot_class.py:323,334` (`fit_primary_cambium_elements`, primary only) | `thin_ring.intersects(Point(c.x,c.y))` per cell | `intersects_xy` over arrays |

**Notes / gotchas:**
- `contains_xy`/`intersects_xy` exist in shapely ≥2.0 (already used by
  `CellManager.remove_cells_in_polygon`, `cell_manager.py`). Match that pattern.
- Predicate result is identical to per-`Point` `.contains` → byte-identical.
  Keep the prepared/`prep()` geometry only where a non-`_xy` predicate is still
  needed; `contains_xy` prepares internally.

## ③ Loosen `maximum_inscribed_circle` tolerance  *(est. −2 to −4s, changes golden)*

**Where:** `geometry_collection.py:613` — `tolerance = max(extent) * 1e-4`.
Called ~593× (once per packed circle in `pack_circles`) at ~9ms each.

**Proposed:** `1e-3` (or expose a coarser tolerance for the secondary-xylem
packing path). GEOS runtime drops roughly with looser tolerance; the centre moves
by ≤ tolerance so vessel positions shift slightly → **rebaseline the golden** and
eyeball a section. Small win, only worth bundling with a golden update.

## ④ Batch `Cell.jitter` / seed creation  *(DONE 2026-07-07 — byte-identical, NOT a repro break)*

**Where:** `voronoi_diagram` (`generate_cell.py`), which jittered 504k seeds via a
per-cell `Cell.jitter()` loop. **Applied** as a vectorised jitter — and, contrary
to the original worry below, it is **byte-identical**, not a reproducibility break.

The key: draw the perturbations as a single **`rng.uniform(-shift, shift,
size=(n, 2))`**. A numpy `Generator` consumes the *same* random stream whether you
call `uniform()` 2n times in a scalar loop or once with `size=(n, 2)` — the array
is filled row-major, i.e. `[cell0.x, cell0.y, cell1.x, cell1.y, …]`, exactly the
old loop's order. So every seed gets its old perturbation, and the RNG is left in
the same state → nothing downstream shifts either. Matching the float ops exactly
matters: `x + u*d`, angle `arctan2(y, x)`, radius **`sqrt(x**2 + y**2)`** (not
`hypot`, which rounds differently).

~~Vectorising the RNG draw would change the draw order → every `seed=0` golden
changes.~~ **False for the `(n, 2)` formulation** — only a per-axis draw
(`uniform(size=n)` for all x, then all y) would reorder the stream and break it.

**Measured:** isolated microbench on 504k cells, jitter+coords step
**1274 ms → 126 ms (~10×, ~1.1s per `generate_cells`)**; output coords + angle +
radius identical bit-for-bit. `characterize.py` + the 7-config golden suite stay
green. Building a full per-cell DataFrame just to slice x/y was folded away in the
same method (the ⑤ win). `Cell.jitter` is left in place as a per-cell API but is no
longer on the hot path.

## Not pursued here

- **`_build_topology` KDTree wall-snapping (22.7s):** algorithmically necessary
  (shared-wall detection across 36k cells). Only micro-opts left (vectorise the
  6-decimal vertex rounding / dict keying). High effort, modest ROI.
- **Fewer border seeds per cell** (P1 above, the ~14→8 lever): biggest single
  lever but changes every cell shape — a quality decision, not a safe speed-up.

## Suggested order

Do **① then ②** (both byte-identical, ~25–35s / ~20–25% off `dicot_nettle`,
verify with `characterize.py`). Treat ③ as a bundled golden-rebaseline follow-up.
Leave ④ unless a reproducibility break is being taken anyway.

---

# Round 2 (2026-07-07, Linux) — profile after ①/②/F1 applied

Re-profiled `dicot_nettle` on Linux with ①/②/F1 in place. Clean wall-clock
(no cProfile), `generate_cells` per-phase, one run: total ~53s (this box is much
faster than the Windows 145s baseline above). Phase split: **Voronoi grouping
18.3s** · Cell seeds 11.5s · Vascular+organ 10.7s · Voronoi diagram 8s ·
Intercellular 1.6s. cProfile `tottime` leaders: `_build_topology` 7.2s (×2),
`voronoi_diagram` 7.2s, shapely dispatch wrapper 6.6M calls, per-group
`union_all` 5.7s, `numpy.asarray` 5.3s, `maximum_inscribed_circle` 3.1s,
`Cell.jitter` 1.5s.

## ⑤ Drop the per-cell DataFrame in `voronoi_diagram`  *(DONE 2026-07-07 — byte-identical)*

**Where:** `generate_cell.py` `voronoi_diagram`. **Was:** built a full
`pd.DataFrame([cell.cell_to_dict() ...])` (every field, ~500k rows) just to slice
`[["x","y"]]` for scipy. **Now:** `np.array([(c.x, c.y) for c in cells])` straight
into `Voronoi(...)`. Voronoi input unchanged → byte-identical (`characterize` +
golden suite green). ~5s of `tottime` removed (Voronoi phase 8.9s→7.9s; total
masked by run-to-run noise but the phase win is real).

## Findings — golden-changing levers, now measured on Linux

- **P1 (seed density `n_points` 15/10 → 8/6): the single biggest lever.**
  Measured `dicot_nettle` **54.4s → 38.0s (~28%)** — Cell seeds 12→7.6s, Voronoi
  8→4.3s, grouping 18.8→11.7s all roughly halve. Biological cell count essentially
  unchanged (dicot_primary stele 2891→2889; secondary 2508→2506) but **air-space
  count drifts ~4%** (dicot_primary 845→877) and every cell boundary coarsens.
  Rebaselines all goldens; needs a rendered-section sign-off. This is the change
  to make if one golden-rebaseline is worth taking.
- **`coverage_union_all` for the per-group union: faster but NOT byte-identical.**
  The per-group Voronoi cells form a valid coverage, so `shapely.coverage_union_all`
  replaces `unary_union` (the 5.7s hotspot) — but it shifts boundary nodes enough
  to change the geometry hash *and* drop one air-space cell (dicot_secondary
  826→825). So it is a golden-changing swap, only worth bundling with a P1-style
  rebaseline, not a free win. Tested and reverted.

## Still open (ranked)

1. **P1 seed density** — ~28%, golden change, needs visual review. Biggest lever.
2. **③ `maximum_inscribed_circle` tolerance 1e-4→1e-3** — ~3s in the packing path
   (`Vascular+organ` phase), golden change. Bundle with P1.
3. **`coverage_union_all` per group** — ~few s, golden change. Bundle with P1.
4. **`_build_topology` micro-opts** (vectorise the 6-dp vertex rounding / dict
   keying) — byte-identical in principle but high effort, modest ROI (~2–3s).

*(④ batch `Cell.jitter` — DONE 2026-07-07, byte-identical after all; see above.)*
