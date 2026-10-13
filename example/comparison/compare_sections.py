"""
GRANAP against its legacy source, side by side (devplan_metafspm.md): for each organ, the legacy cross-section
(openalea.granap.legacy, the original, frozen), GRANAP's read from its MPG, and their differences, to compare visually
as well as by the tests (in particular where a later step is allowed to move away from exactness, Q-P5c).

    left     the legacy cells (organ.generate_cells(), openalea.granap.legacy)
    middle   the MPG's Cell vertices (their polygons and tissue properties, read from organ.mpg), the stored hydraulic
             network over them: the Connections drawn between their Compartments (protoplasts at the cell centroids,
             walls at their midpoints, junctions at their positions)
    right    the differences: both sections' outlines overlaid (legacy black, GRANAP blue), and in red, for each tissue,
             the area covered by that tissue in one section and not in the other; the title gives the cell counts that
             differ by tissue and the differing area relative to the section's

The suptitle says whether the sections are identical (every cell attribute, the polygons). Run from
GRANAP/example/comparison:

    python compare_sections.py [organ ...]      default: every organ below; figures in ./outputs/sections/
    python compare_sections.py --list           the organs

Organs: every kind GRANAP builds (roots: monocot, monocot with arch xylem, monocot with aerenchyma, dicot primary,
secondary and woody; stems: dicot, continuous dicot, monocot; leaves: monocot, dicot; the needle; a section read from
a CellSet file; one read from ImageJ ROIs).
"""
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as PolygonPatch

from shapely.ops import unary_union

from openalea.granap.cellset_organ import CellSetOrgan
from openalea.granap.input_data import OrganInputData
from openalea.granap.leaf_class import LeafAnatomy
from openalea.granap.legacy.cellset_organ import CellSetOrgan as LegacyCellSet
from openalea.granap.legacy.input_data import OrganInputData as LegacyInputData
from openalea.granap.legacy.leaf_class import LeafAnatomy as LegacyLeaf
from openalea.granap.legacy.needle_class import NeedleAnatomy as LegacyNeedle
from openalea.granap.legacy.roi_organ import RoiOrgan as LegacyRoi
from openalea.granap.legacy.root_class import RootAnatomy as LegacyRoot
from openalea.granap.legacy.stem_class import StemAnatomy as LegacyStem
from openalea.granap.metafspm.build import JUNCTION, WALL
from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.roi_organ import RoiOrgan
from openalea.granap.root_class import RootAnatomy
from openalea.granap.stem_class import StemAnatomy

SEED = 0
INPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test", "inputs")


def _data(preset, changes=None):
    """An input data preset, with {(section, field): value} changes."""
    def make():
        data = getattr(OrganInputData, preset)()
        for (section, name), value in (changes or {}).items():
            data.set_value(section, name, value)
        return data
    return make


def _input_organ(original, ported, preset, changes=None):
    """(the legacy organ, GRANAP's organ), built from an input data preset (the legacy one gets its own copy)."""
    data = _data(preset, changes)
    return (lambda: original(LegacyInputData.from_dict_list(data().to_dict_list()), seed=SEED),
            lambda: ported(data(), seed=SEED))


ROI = dict(folder_path=os.path.join(INPUTS, "Hypocotyl_RoiSet"), mm_per_pixel=0.001, smooth_factor=0.05)
CELLSET = os.path.join(INPUTS, "partial_CellSet_brasicaceae.xml")
ORGANS = {          # name: (build GRANAP's organ, build the MPG-backed organ)
    "monocot_root": _input_organ(LegacyRoot, RootAnatomy, "for_root"),
    "monocot_root_arch": _input_organ(LegacyRoot, RootAnatomy, "for_root", {("xylem", "xylem_shape"): "arch"}),
    "monocot_root_aerenchyma": _input_organ(LegacyRoot, RootAnatomy, "for_root",
                                            {("aerenchyma", "aerenchyma_proportion"): 0.2}),
    "dicot_root": _input_organ(LegacyRoot, RootAnatomy, "for_dicot_root"),
    "dicot_root_secondary": _input_organ(LegacyRoot, RootAnatomy, "for_dicot_secondary"),
    "dicot_root_woody": _input_organ(LegacyRoot, RootAnatomy, "for_woody_dicot"),
    "dicot_stem": _input_organ(LegacyStem, StemAnatomy, "for_dicot_stem"),
    "dicot_stem_continuous": _input_organ(LegacyStem, StemAnatomy, "for_dicot_stem_continuous"),
    "monocot_stem": _input_organ(LegacyStem, StemAnatomy, "for_monocot_stem"),
    "monocot_leaf": _input_organ(LegacyLeaf, LeafAnatomy, "for_monocot_leaf"),
    "dicot_leaf": _input_organ(LegacyLeaf, LeafAnatomy, "for_dicot_leaf"),
    "needle": _input_organ(LegacyNeedle, NeedleAnatomy, "for_needle"),
    "cellset": (lambda: LegacyCellSet(CELLSET, seed=SEED), lambda: CellSetOrgan(CELLSET, seed=SEED)),
    "roi": (lambda: LegacyRoi(**ROI), lambda: RoiOrgan(**ROI)),
}


def _colours(types):
    palette = matplotlib.colormaps["tab20"]
    return {t: palette(i % 20) for i, t in enumerate(sorted(types))}


def _draw_polygons(ax, polygons, types, colours):
    patches, faces = [], []
    for polygon, cell_type in zip(polygons, types):
        if polygon is None or polygon.is_empty:
            continue
        parts = polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]
        for part in parts:
            patches.append(PolygonPatch(np.asarray(part.exterior.coords), closed=True))
            faces.append(colours[cell_type])
    ax.add_collection(PatchCollection(patches, facecolor=faces, edgecolor="black", linewidth=0.3, alpha=0.6))
    ax.autoscale_view()
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (mm)")


def _same(a, b):
    if a is None or b is None:
        return a is b
    if hasattr(a, "geom_type"):
        return a.equals(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def identical(old, new) -> bool:
    if len(old.all_cells.cells) != len(new.all_cells.cells):
        return False
    return all(set(vars(a)) == set(vars(b)) and all(_same(getattr(a, k), getattr(b, k)) for k in vars(a))
               for a, b in zip(old.all_cells.cells, new.all_cells.cells))


def network_lines(organ):
    """The Connections of the stored network, as segments between their Compartments' positions."""
    g, w = organ.mpg, organ._mpg_written
    px, py = g.property("section_x"), g.property("section_y")
    a, b = g.property("n_id_a"), g.property("n_id_b")
    lines = [((px[int(a[e])], py[int(a[e])]), (px[int(b[e])], py[int(b[e])])) for e in w["connections"]
             if int(a[e]) in px and int(b[e]) in px]
    return np.asarray(lines, dtype=float).reshape(-1, 2, 2)


def _outlines(polygons):
    lines = []
    for polygon in polygons:
        if polygon is None or polygon.is_empty:
            continue
        for part in (polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]):
            lines.append(np.asarray(part.exterior.coords))
    return lines


def differences(old_polygons, old_types, new_polygons, new_types):
    """Per tissue: the area covered in one section and not in the other, and the cell counts on both sides."""
    out = {}
    for t in sorted(set(old_types) | set(new_types)):
        a = unary_union([p for p, ty in zip(old_polygons, old_types) if ty == t and p is not None and not p.is_empty])
        b = unary_union([p for p, ty in zip(new_polygons, new_types) if ty == t and p is not None and not p.is_empty])
        out[t] = (a.symmetric_difference(b), old_types.count(t), new_types.count(t))
    return out


def compare(name, folder):
    build_old, build_new = ORGANS[name]
    start = time.perf_counter()
    old = build_old()
    old.generate_cells()
    t_old = time.perf_counter() - start
    start = time.perf_counter()
    new = build_new()
    new.generate_cells()
    t_new = time.perf_counter() - start

    g, w = new.mpg, new._mpg_written
    tissue = g.property("tissue")
    mpg_types = [tissue[v] for v in w["cells"]]
    mpg_polygons = [g.property("cell__polygon")[v] for v in w["cells"]]
    old_types = [c.type for c in old.all_cells.cells]
    old_polygons = [c.polygon for c in old.all_cells.cells]
    kinds = g.property("kind")
    n_walls = sum(1 for v in w.get("apoplast", []) if kinds[v] == WALL)
    n_junctions = sum(1 for v in w.get("apoplast", []) if kinds[v] == JUNCTION)
    colours = _colours(set(old_types) | set(mpg_types))

    bounds = np.array([p.bounds for p in old_polygons if p is not None and not p.is_empty])
    aspect = (bounds[:, 2].max() - bounds[:, 0].min()) / (bounds[:, 3].max() - bounds[:, 1].min())
    if aspect > 1.6:                                          # a wide section (leaves, needle): one panel per row
        fig, (left, middle, right) = plt.subplots(3, 1, figsize=(18, min(4. + 18. / aspect * 3., 24.)),
                                                  sharex=True, sharey=True)
    else:
        fig, (left, middle, right) = plt.subplots(1, 3, figsize=(24, 8.4), sharex=True, sharey=True)
    _draw_polygons(left, old_polygons, old_types, colours)
    left.set_ylabel("y (mm)")
    left.set_title(f"legacy: {len(old_types)} cells ({t_old:.1f} s)")
    _draw_polygons(middle, mpg_polygons, mpg_types, colours)
    if w.get("connections"):
        middle.add_collection(LineCollection(network_lines(new), colors="0.15", linewidths=0.15, alpha=0.5))
    middle.set_title(f"GRANAP (MPG): {len(w['cells'])} Cells in {len(w['layers'])} Layers, {n_walls} walls, "
                     f"{n_junctions} junctions, {len(w.get('connections', []))} Connections ({t_new:.1f} s)",
                     fontsize=10)

    diff = differences(old_polygons, old_types, mpg_polygons, mpg_types)
    right.add_collection(LineCollection(_outlines(old_polygons), colors="black", linewidths=0.35))
    right.add_collection(LineCollection(_outlines(mpg_polygons), colors="tab:blue", linewidths=0.2, alpha=0.8))
    _draw_polygons(right, [d for d, _, _ in diff.values()], ["differs"] * len(diff), {"differs": "red"})
    section_area = unary_union([p for p in old_polygons if p is not None and not p.is_empty]).area
    differing = sum(d.area for d, _, _ in diff.values()) / 2.
    counts = [f"{t} {a}→{b}" for t, (_, a, b) in diff.items() if a != b]
    right.set_title(f"differences: {100. * differing / section_area:.3g} % of the area"
                    + (f"; cells {', '.join(counts)}" if counts else "; same cell counts per tissue"), fontsize=10)

    handles = [matplotlib.patches.Patch(color=colours[t], label=t, alpha=0.6) for t in sorted(colours)]
    handles += [matplotlib.patches.Patch(color="red", label="differs", alpha=0.6),
                matplotlib.lines.Line2D([], [], color="black", label="legacy outlines"),
                matplotlib.lines.Line2D([], [], color="tab:blue", label="GRANAP outlines")]
    fig.legend(handles=handles, loc="center right", fontsize=8, title="cell type")
    same = identical(old, new)
    fig.suptitle(f"{name} (seed {SEED}): sections {'identical' if same else 'DIFFERENT'}", fontsize=14,
                 color="black" if same else "red")
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"{name}: {'identical' if same else 'DIFFERENT'}, {100. * differing / section_area:.3g} % of the area "
          f"differs ({t_old:.1f} s / {t_new:.1f} s) — {path}", flush=True)
    return same


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(ORGANS))
        sys.exit()
    names = sys.argv[1:] or list(ORGANS)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "sections")
    os.makedirs(folder, exist_ok=True)
    for name in names:
        compare(name, folder)
