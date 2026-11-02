# 3D hydraulic network from the extruded section — plan

Status: **P0, P1 done (2026-11-02)**, P2–P4 to do. Plan agreed 2026-10-30. Branch: `tissue`. Scope v1: roots
(generic / monocot / dicot / woody); stems and leaves checked afterwards; needle excluded.

## Goal
`organ.generate_cells_3d()` already stacks copies of each 2D cell along Z, but the
result is a bag of meshes: no cell ids, no neighbours, no walls. Build the **3D
equivalent of the 2D hydraulic network** (`NetworkExporter.export` →
`AbstractNetwork`): same node kinds (wall / junction / cell), same MECHA ordering, same
path names (`membrane`, `wall`, `plasmodesmata`), so `export_to_adjencymatrix()` and
`fill_matrix(K, label, "cortex-endodermis")` work unchanged on it.

Requirements fixed with the user:
1. **Apoplast at full resolution (option a)**: wall pieces connect to junction *lines*
   (vertical junction pieces and horizontal rim junctions), the exact 3D analogue of the
   2D wall ↔ junction graph.
2. **Vessels can be split into elements in series** (X tubes per segment, joined by end
   walls / perforation plates), or stay one tube spanning the segment.
3. **Output = the 2D network, in 3D**: an `AbstractNetwork` (networkx graph + sparse
   matrix).
4. **Every cell type can have its own axial height** — not only every layer.
5. **Wall thickness is configurable** as in 2D, per cell type, and also for end walls.

## Key idea: lift the 2D topology, don't recompute it
Every row of the extrusion reuses the *same* 2D section, so every 3D wall is a 2D wall
(already known: its two cells, its two junctions, its length) times a height interval.
The 3D network is built from the 2D wall registry + each cell's row intervals — pure
interval arithmetic, exact, no 3D geometry matching or snapping tolerance.

## Axial heights (per cell type)

Resolution order for a cell's row height (first hit wins):
1. `axial_heights={"cortex": 0.12, "stele": 0.08, "metaxylem": 0.40, ...}` — keyed by
   **cell type** (the `type` tag), passed to `generate_cells_3d` / `export_3d_network`.
   This is what lets cells sharing one layer differ (stele parenchyma vs protoxylem vs
   metaxylem vs phloem are all in the stele layer).
2. A vessel type (`VESSEL_TYPES`, and `air space` in aerenchyma mode `before`) → one
   full-span tube. Checked *before* the layer so a vessel does not inherit the height of
   the stele layer it sits in.
3. The layer's own `axial_height` parameter (`OrganInputData`, unchanged — kept as is).
4. `default_axial_height`, then `DEFAULT_AXIAL_HEIGHT_RATIO (5) × cell_diameter`.

`Cells3DResult.height_sources` records which rule each type used.

Root types today: monocot `epidermis, exodermis, cortex, endodermis, pericycle, stele,
protoxylem, metaxylem, phloem, air space`; dicot adds `cambium, xylem` (no proto/meta).

- **Vessels in series**: a vessel type given a height is cut into elements of that
  length, with a random start offset per vessel (as the ordinary cells already have);
  consecutive elements share an **end wall** (perforation plate). No height → one tube.
- **`segment_length=`**: set the segment length directly; default stays
  `n_axial_repeats × tallest ordinary height`.
- **Warn on fallbacks**: log which types used the layer value or the default, so a
  missing entry is never silent.
- Complete `VESSEL_TYPES` (full-span by default): add `sclerenchyma`, `companion_cell`
  (underscore spelling), `duct`.

## Wall thickness
Same per-type dict as 2D: `cell_wall_thickness` (default `DEFAULT_CELL_WALL_THICKNESS`,
fallback key `"default"`), plus a new `end_wall_thickness` dict (default = same as
`cell_wall_thickness`).
- **Side wall** (cells a, b): `t(a) + t(b)`, plus `t("outerwall")` if on the organ
  surface — exactly the 2D formula.
- **End wall** between rows k, k+1 of one file (or two vessel elements):
  `2 × end_t(type)`.
- **Segment-end face** (bottom of first row, top of last row): `end_t(type)` — the other
  half belongs to the next segment.
- Warn when a type falls back to `"default"`.

## The 3D network

### Nodes (MECHA order: walls, then junctions, then cells)
| kind | node | attributes |
|---|---|---|
| wall | **side wall piece**: (2D wall w, overlap interval [z0, z1] of its two cells' rows) | area = length(w) × (z1−z0), thickness, orientation=`lateral` |
| wall | **outer side wall piece**: (outer 2D wall, row interval) | idem, + outerwall thickness |
| wall | **end wall**: (2D cell, z of a row boundary) | area = cell area, thickness, orientation=`axial` |
| wall | **segment-end face**: (2D cell, z_min or z_max) | area, thickness, `boundary=True` (axial BC) |
| junction | **vertical junction piece**: (2D junction J, interval between consecutive row boundaries of *all* cells around J) | length = interval height |
| junction | **rim junction**: (2D wall w, z of a row boundary of a cell on w) — where an end wall meets the side walls | length = length(w) |
| cell | (2D cell, row k) or (vessel, element k) | volume, area, z0, z1, cell_type, cgroup |

Positions are `(x, y, z)` (midpoints / centroids). Each node also keeps its 2D id
(`wall_2d`, `junction_2d`, `cell_2d`) and its row index, so 3D results can be mapped
back to the 2D section.

### Edges (same `path` names as 2D, plus `orientation`)
- **`membrane`** — cell row ↔ every side wall piece overlapping its row, ↔ its end walls
  above and below, ↔ its segment-end face if any.
- **`wall`** (apoplast) — side wall piece ↔ the vertical junction pieces at both ends of
  w inside [z0, z1], ↔ the rim junctions at z0 and z1; end wall / segment-end face ↔ the
  rim junctions around its cell at that z. Junctions are not linked to each other
  (as in 2D: flow goes wall → junction → wall).
- **`plasmodesmata`** — across each side wall piece with two cells (`lateral`), and
  across each end wall (`axial`, row k ↔ k+1, or vessel element ↔ element).
- Edge attributes as in 2D (`length`, `dist`, `d_vec`, `wall_thickness`,
  `lateral_distance`, …) in 3D, plus `area` and `orientation`.

`_wall_to_cells`: side piece → its two cell rows; end wall → the two rows of the file;
outer piece / segment-end face → its single row. So `fill_matrix(K, label, "cortex")`
or `"cortex-endodermis"` keeps its 2D meaning. Add an optional
`orientation=None|"lateral"|"axial"` filter to `fill_matrix` (backward compatible) to
set different axial and lateral conductivities.

## Implementation steps

**P0 — keep identity in the 3D build (`generate_cell_3d.py`)** — DONE.
Each cell dict has `cell_2d`, `row`, `z0`, `z1`; `Cells3DResult.section` is the 2D cell
table (index = `cell_2d`, = `organ.all_cells.cells` order); `axial_heights`,
`segment_length`, `height_sources`, fallback logged at INFO. Default output
(no new options) is vertex-identical to before on monocot / dicot roots; stems change
only where `sclerenchyma` / ducts are now full-span. Tests in `test/test_cells_3d.py`.
Note for P2: a MultiPolygon cell is extruded as its largest piece, while the 2D topology
sees the whole geometry; clipped first/last rows can be very thin (no minimum yet).
- Each extruded cell records `cell_2d` (index into the section's cell table), `row`,
  `z0`, `z1`, besides `type / vertices / faces / volume`.
- Per-type `axial_heights`, vessel elements in series, `segment_length`, fallback logging,
  completed `VESSEL_TYPES`.
- The 3D build regenerates its own 2D section (no intercellular space) and invalidates it
  afterwards, so the 2D wall registry must be taken **from that same section, inside the
  build**, and returned with the result — not recomputed from the organ later.
- Check the cell index used by the extrusion (`organ.all_cells.cells` order) matches the
  index of the cell table the topology is built from (the gdf, after its valid-geometry
  filter).

**P1 — share the 2D wall step (`anatomy_writer.py`)** — DONE.
`build_wall_registry(cells_gdf, cell_wall_thickness) -> WallRegistry(cells_gdf, walls,
junctions, junction_ids)`, module-level next to `NetworkExporter`; the code was moved
verbatim. The 2D graph (nodes, edges, attributes, `_wall_to_cells`) is identical
before/after on monocot root, dicot root, dicot stem, dicot leaf and needle; full suite
179 passed. Tests in `test/test_network_3d.py`.
- Move phases 0–4 of `NetworkExporter.export` (topology, junction walk, wall registry,
  thickness) into a reusable `build_wall_registry(cells_gdf, cell_wall_thickness)`.
  The needle-specific parts (protected air spaces, `air_link`) stay in `export`.
- `export` calls it: **the 2D graph must be identical** before/after (nodes, edges,
  attributes) on root, stem, leaf, needle.

**P2 — `network_3d.py`: `Network3D(AbstractNetwork)`**
- Built from the P0 result (rows per cell) + the P1 registry.
- Per 2D wall: merge the row breakpoints of its two cells → side pieces. Per 2D
  junction: merge the breakpoints of all cells around it → vertical pieces. Per row
  boundary: end wall + its rims.
- Build edge lists with numpy / plain lists and create the networkx graph once at the end.
- Entry point: `organ.export_3d_network(axial_heights=..., segment_length=...,
  cell_wall_thickness=..., end_wall_thickness=..., seed=...)` → `Network3D`
  (also exposes the `Cells3DResult`, so `export_obj` still works).

**P3 — thickness, attributes, `fill_matrix(orientation=...)`.**

**P4 — tests, example, scale check.**
- `example/root/network_3d.py`: build, print counts per node / edge kind, export OBJ.
- Time and memory on the default monocot / dicot root; then try the default monocot stem
  (~256k prisms) to see whether v1 scales or needs array-based storage.

## Tests (`test/test_network_3d.py`)
- **Areas add up**: for every 2D wall, its side pieces sum to length × segment length;
  for every cell row, its side pieces sum to perimeter × row height; end walls = cell area.
- **Reduces to 2D**: one row, no offset → same cells, walls and edges as the 2D network
  (plus the two segment-end faces per cell).
- **Heights per type**: two types in the same layer (e.g. `stele` and `protoxylem`) given
  different `axial_heights` → rows of the expected heights; unset types are logged.
- **Vessels**: no height → one node per vessel, no internal end walls; height h →
  ceil(span / h)-ish elements joined by end walls with `end_wall_thickness`.
- **Thickness**: side = t(a)+t(b) (+ outerwall on the surface); end = 2·end_t; face = end_t.
- **Graph**: connected; every row of a non-vessel file has axial plasmodesmata to its
  neighbours; `export_to_adjencymatrix` + `fill_matrix(..., orientation=...)` work.
- **No side effects**: the organ's own 2D section is unchanged after the 3D build.
- Run on monocot, dicot and woody roots.

## Out of scope (v1)
Needle and leaf air networks (`protect_topology`, `air_link`); 3D intercellular space;
sections changing along the axis (root developmental series); MECHA's solver side;
plotting the 3D network; the `to_metafspm` port (its network lives in the MPG — port
after v1 is validated on `tissue`).

## Open points
- Store `axial_heights` / `end_wall_thickness` in `OrganInputData` (saved with the
  recipe) in addition to the call arguments?
- Units: thickness in the same unit as the 2D dict (µm on `to_metafspm`); heights in mm
  like every other length.
