# GRANAP Code Review
> Branch: `ganache_dilhan` — reviewed 2026-06-22

---

## Critical — will cause bugs

### 1. ✅ `extend_cells` mutates its input *(fixed)*
`CellManager.extend_cells` overwrites `id_layer`, `id_cell`, `id_group` directly on the
passed-in `Cell` objects.  If the same cell list is used again (e.g. after
`_invalidate_geometry()` forces a second `generate_cells()` call), the ids are
already offset from the first call and will be doubled.

**Fix:** copy each cell before mutating — see commit on this branch.

---

### 2. No seed control — non-deterministic output
Every run produces a different anatomy for the same parameters because several
code paths call NumPy RNG directly with no seed:

| Location | Call |
|---|---|
| `cell_class.Cell.jitter` | `np.random.uniform` |
| `generate_cell.cells_on_layer` | `np.random.uniform` (shift) |
| `generate_cell.generate_cells_info` | `np.random.random` (transfusion type) |
| `needle_class._duct_zone_data` | `np.random.choice` (duct placement) |
| `root_class` (secondary growth) | `np.random.normal`, `np.random.uniform` |
| `organ_class.add_aerenchyma` | `np.random.uniform` (start angle) |
| `geometry_collection.pack_circles` | `np.random.uniform`, `np.random.normal` |

No `seed` parameter exists anywhere in the public API.
For a scientific modelling tool this is a reproducibility problem.

**Proposed fix:** add an optional `seed: int | None = None` parameter to
`Organ.__init__` that calls `np.random.seed(seed)` (or uses a local
`np.random.Generator`) before generation starts.

---

### 3. ✅ Abstract method `_create_central_layers` has a wrong return type *(fixed)*
Implementations in `NeedleAnatomy` and `RootAnatomy` now return
`List[LayerPolygon]`, but the abstract declaration in `Organ` and the stub in
`RoiOrgan` still declare `List[Dict[str, Any]]`.  The type contract is broken.

**Fix:** update the abstract signature and `RoiOrgan` stub to `List[LayerPolygon]`.

---

## Design problems

### 4. O(n²) polygon intersection in `merge_intercellular_aerenchyma`
```python
for i in range(n_pool):
    for j in range(i + 1, n_pool):
        if merge_pool[i].polygon.touches(merge_pool[j].polygon) or \
           merge_pool[i].polygon.intersects(merge_pool[j].polygon):
```
Full pairwise test on every air-space cell.  For large aerenchyma this is the
dominant cost.  A Shapely `STRtree` spatial index would reduce it to O(n log n).

---

### 5. Monocot and dicot phloem use different polygon stores with different semantics
Monocot phloem ellipses go into `vascular_polygons`; dicot phloem goes into
`vascular_tissue_polygons["phloem"]`.  This matters because `vascular_polygons`
is also used to carve out the remaining stele area for subsequent vascular bundle
placement (`polygon = polygon.difference(unary_union(self.vascular_polygons))`).
Bufferingmonocot phloem ellipses before storing them would shrink the available
stele area and crowd adjacent protoxylem/metaxylem bundles together.  Any future
change to phloem polygon storage must respect this asymmetry.

---

### 7. ✅ `id_layer = 0` hardcoded for all vascular cells *(fixed)*
Every xylem, phloem, cambium, stomata, and resin-duct cell is created with
`id_layer=0`.  `0` is the "outside" sentinel in `layers_polygons`, so the layer
population step silently skips all vascular cells.  This works by coincidence
but makes `id_layer` meaningless for vascular cells and would break any code
that tries to use `id_layer` to identify tissue type for those cells.

---

### 6. `_dry_run_vascular` in `visualization.py` hand-rolls save/restore
```python
saved_all_cells             = organ.all_cells
saved_vascular_cells        = ...
...
finally:
    organ.all_cells = saved_all_cells
    ...
```
Every time a new state attribute is added to an `Organ` subclass, this block
must be updated.  Root cause: mutable organ state makes "preview without
side-effects" inherently fragile.  A better design would make `generate_*`
methods pure (return values, don't store results on `self`), or introduce an
explicit snapshot/restore protocol.

---

### 7. `recalculate_cell_properties` called too early
Called after `add_canal` and after `add_stomata` on seeds that have no Voronoi
polygon yet.  `cell.area` is set to `π r²` from diameter, which is overwritten
correctly after Voronoi anyway.  The early calls are wasted work and give a
false impression that properties are up to date.

---

## Remaining `print()` calls (should be `logging`)

| File | Line | Message |
|---|---|---|
| `anatomy_writer.py` | 174, 247, 348, 362, 434, 617 | save confirmations |
| `organ_class.py` | 502 | aerenchyma clamp warning |
| `roi_organ.py` | 40 | missing `.roi` files warning |
| `input_data.py` | 534 | XML missing-field defaults |
| `generate_cell.py` | 718 | debug print inside `create_stomata` (behind `if debug:`) |

---

## Minor

### 8. `process_voronoi_groups` — cell position after dissolve
After merging border-point cells of the same `id_group`, the representative
`x, y` position comes from GeoPandas `dissolve` with `aggfunc="first"` —
i.e. the first border point of the group, not the centroid of the merged polygon.
Cell positions are slightly inconsistent with their actual polygon centroids.

### 9. `RoiOrgan` stubs are incomplete
All overridden abstract methods return `None` or `[]` with no type annotations
updated to `LayerPolygon`.  Acceptable for a bypass organ, but it should be
explicitly documented rather than left as silent stubs.
