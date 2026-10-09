
# Test P. pinaster anatomy

- [x] Changing numbers of cell layers should be tackle similarly as in dicot leaf. 

- [x] The vascular ellipse seems to permute major and minor axis while cambium and Str. interstitial are well oriented.
- [x] epidermis layer is cropped and therefore stomata are wrong.
- [x] hypodermis should have 5 layers in corner, then only 2 elsewhere 
- [x] palisade mesophyll layer is cropped
- [x] spongy mesophyll has two layer while it should only be one on the adaxial part (under the central cylinder - modeling convention), two on the abaxial side.
- [x] spongy mesophyll should have the rhombi air space.
- [x] resin duct should be inside all mesophyll layer polygon
- [x] transfusion tissue should be included

## 28.08.2026

[screenshot before this round](./outputs/pinus_pinaster_test03.png)
[screenshot after this round](./outputs/pinus_pinaster_test04.png) — and a
[4-panel close-up](./outputs/pinus_pinaster_test04_zoom.png) (bundle / duct / stoma / transfusion)

- [x] inside the vascular ellipse, xylem should not be on the abaxial side of the cambium

  Off-by-one: the xylem loop ran `range(xylem_rows + 1)`, so its last row landed at
  `xylem_cell_height/2 - cambium_d` = **+0.009**, i.e. one whole file on the phloem side.
  The phloem loop had the mirror defect (`range(1, n+1)` left a 1.5-cell dead gap at the
  cambium and pushed its last row outside `ry`, where it was silently dropped). Both are now
  built symmetrically outward from `cambium_d`. Measured: xylem local_x now spans
  [-0.0658, -0.0212], phloem [+0.0212, +0.0658] in both bundles; counts 58/58 (were 84/40).
  Also added a `sin_a < 0` frame guard so the polarity is anchored to the abaxial direction
  for *any* `vascular_angle`, instead of being an accident of the 30° value.

- [x] resin duct sheath cells are too big and lumen is too small. There might be a design flaw. take inspiration from bundle sheath in other leaf organ.

  Measurement narrowed this down: the lumen and epithelium were already built exactly to
  spec (rendered/geometric area ratios 0.99 and 1.05). **Only the sheath was wrong, at 3.5x
  its geometric area** — its Voronoi grew unbounded into the carved annulus, because there
  was nothing between the 0.018 mm sheath seeds and the 0.09-0.1 mm palisade/mesophyll. Same
  "radial sunburst" failure `leaf_class._ground_cell_size_for_veins` describes and solves
  with an outer sheath. Fixed by adding a graded transition ring of intermediate-sized cells
  (geometric-mean sizing as in `_grow_bundle_sheath`, tagged `mesophyll`, only when
  host/sheath size ratio > 4), plus the scale-to-fit that `_duct_zone_data`'s docstring had
  always promised but never implemented. Sheath ratio now **1.003**, 10 cells per duct,
  sunburst gone. Example params restored to the documented measurements (lumen 0.037, sheath
  radial 0.018) — the dict had drifted to 0.031/0.02.

[](resin_duct.png)

- [x] transfusion parenchyma don't occupy enough place in the transfusion tissue. Could add more transfusion tracheid cells (which would occupy a smaller proportion of the available space).

  Packing order reversed: parenchyma now packs into the full zone first as the large
  ellipses, tracheids second into the residue, each with its own `diameter_max`
  (`parenchyma_diameter` 0.045 / `tracheids_diameter` 0.022 — both fields already existed but
  were force-zeroed on the packing path). Parenchyma area share **36% -> 63%**; tracheids
  **42 -> 136 cells** with mean cell area ~5x smaller.
  Note the realized split is not the nominal one: `transfusion_tracheids_ratio = 1.0` asks for
  50/50, but the second pass packs into a gap-constrained residue and under-fills, landing at
  ~63/37. A sweep of that knob gives 0.6 -> 78% parenchyma (tracheids reduced to slivers) and
  1.5 -> 49% (back to the original complaint), so 1.0 is the value that matches the reference.

[](./Transfusion_tissue.png)

- [x] thin air spaces should be created in the palisade mesophyll (height = the mesophyll; 0.002 width) every two cells. New feature to implement.

  New `_apply_palisade_wall_slits` / `_wall_slit`, modelled on the existing mesophyll wall
  rhombi (same STRtree -> lacuna -> `seat_air_spaces` carve pipeline, so the slit boundary and
  the carved host boundary stay vertex-for-vertex identical). Driven by new
  `slit_width`/`slit_every` parallel-list keys on `inter_cellular_spaces`. 58 slits on
  alternating walls (= floor(116/2)), mean area 0.000149 mm2; palisade cell count unchanged at
  116 and the palisade area lost exactly equals the air-space area gained (0.00862 both ways),
  i.e. a clean carve with no leakage. Cells are ordered by angle rather than `id_group` parity,
  because the stomata carve and `_restrict_zoned_layers` leave gaps in `id_group`.
  `length_fraction` is 0.9 rather than 1.0: full-length slits put their corners exactly on
  wall-junction vertices, which made the difference degenerate and crashed `_build_topology`.

- [x] the inner most hypodermis cell touching the sub stomatal chamber should be removed so larger palisade mesophyll can extend in that region.

  Done at *seed* level — `add_stomata` runs before the Voronoi pass, so deleting the seed lets
  the neighbouring palisade expand into the space on its own.
  Two things worth recording, because the obvious implementations are both wrong:
  1. An isotropic `chamber.buffer(...)` probe grows sideways as fast as it grows inward, so
     buying enough radial reach to cross the 2-layer (5-layer at corners) hypodermis band
     stripped **34% of the whole band** (449 -> 297 cells) to open every stoma. Replaced with an
     oriented *column*: tangential half-width fixed at the chamber's own footprint, and
     `chamber_clearance` controlling only the inward depth. Same 14/18 stomata opened for 65
     cells removed instead of 99, and it saturates — raising the value further changes nothing.
  2. The column's long axis must be the stoma's *local* inward normal, recovered from the
     guard-cell tangent, not the direction to the organ centroid: on this flattened dome the
     two diverge by up to ~60 degrees away from the extremes, which aims the probe past the
     cells it is meant to hit.
  4 of 18 stomata remain blocked (corner region, where the band is thickest).

- [x] sunken stomata should have an epidermal cell above the guard cells (split actual guards cell: rectangle & the shape below).

  The decomposition already existed in `create_stomata` and was being thrown away — each guard
  cell is built as an ellipse unioned with a rectangle. Under the new `sunken` flag the union is
  skipped: the ellipse stays the guard cell and the rectangle is returned as an epidermis cap.
  `spacing_poly`/chamber are now differenced against `gc1 | gc2 | caps` so the pore does not
  balloon upward into the space the rectangles vacated (verified: sunken pore area 0.0016223 <=
  default 0.0016277). `place_stomata` unpacks tolerantly, so the leaf's 5-tuple path is
  untouched — confirmed bit-for-bit identical on the default path to full float precision.

## 31.08.2026

- [x] Sunken guard cells needed two new input params: their size was pinned to the
  epidermis cell's own width, and their pit aspect ratio was a bare hardcoded factor.

  Added `guard_cell_diameter` (default `None` → falls back to `cell.width`, i.e. today's
  behavior) and `guard_cell_aspect` (default `0.5`, reproducing the previous hardcoded
  `gc_ry/2`) to `StomataParams`. `create_stomata` now sizes `gc_rx`/`gc_ry` from
  `guard_cell_diameter/2` and the ellipse's vertical radius from `gc_ry * guard_cell_aspect`,
  instead of `cell.width/2` and a fixed `/2`. Both defaults are exact no-ops — verified the
  full pinaster render's per-tissue counts/areas are byte-identical to the pre-change run
  (guard cell 36 cells / 0.00554 mm2, epidermis 293, etc.). Confirmed the knobs actually move
  guard-cell area independently of the epidermis layer (0.03 diameter -> 0.02916 mm2 vs.
  default 0.00554 mm2). One caveat worth knowing: `place_stomata` erodes every guard cell by
  a fixed `-cell_diam/5` (the *epidermis* cell diameter, not the guard cell's own), so setting
  `guard_cell_diameter` much smaller than the epidermis `cell_diameter` can erode the guard
  cell to nothing (0 cells, no error) — keep it in the same order of magnitude as the
  epidermis layer's `cell_diameter`.

### Open / for review

- `phloem.cell_diameter` and `xylem.n_per_cluster` are read from the param dict but never used
  by `vascular_elements_in_ellipses` (phloem reuses the xylem column pitch so the interstitial
  files stay aligned). So `"cell_diameter": 0.008` on phloem currently has no effect.
  - [] phloem `cell_diameter` should be use as the height of the cells, column stay unchanged but row is affected (create a row misaligninment with interstitial) 
- The `test_needle_*_golden` fixtures and `test_preset_param_dicts_unchanged` need regenerating:
  needle geometry and the param schema both changed on purpose this round. 