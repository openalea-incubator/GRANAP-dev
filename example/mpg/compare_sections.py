"""
GRANAP and its MPG port, side by side (devplan_metafspm.md): for each organ, the original cross-section and the one
read from the MPG.

    left    GRANAP's cells (organ.generate_cells(), the original classes)
    right   the MPG's Cell vertices (their polygons and tissue properties, read from organ.mpg), the stored hydraulic
            network over them: the Connections drawn between their Compartments (protoplasts at the cell centroids,
            walls at their midpoints, junctions at their positions)

The titles give the counts on both sides and whether the sections are identical (every cell attribute, the
polygons). Run from GRANAP/example/mpg:

    python compare_sections.py [organ ...]      organs: monocot_root, dicot_root, needle, dicot_stem, monocot_stem
                                                (default: monocot_root dicot_root); figures in ./outputs
"""
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as PolygonPatch

from openalea.granap import mpg
from openalea.granap.input_data import OrganInputData
from openalea.granap.mpg.build import JUNCTION, PROTOPLAST, WALL
from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.root_class import RootAnatomy
from openalea.granap.stem_class import StemAnatomy

SEED = 0
ORGANS = {
    "monocot_root": (RootAnatomy, mpg.RootAnatomy, OrganInputData.for_root),
    "dicot_root": (RootAnatomy, mpg.RootAnatomy, OrganInputData.for_dicot_root),
    "needle": (NeedleAnatomy, mpg.NeedleAnatomy, OrganInputData.for_needle),
    "dicot_stem": (StemAnatomy, mpg.StemAnatomy, OrganInputData.for_dicot_stem),
    "monocot_stem": (StemAnatomy, mpg.StemAnatomy, OrganInputData.for_monocot_stem),
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
    position = g.property("node__position")
    a, b = g.property("n_id_a"), g.property("n_id_b")
    lines = [(position[int(a[e])], position[int(b[e])]) for e in w["connections"]
             if int(a[e]) in position and int(b[e]) in position]
    return np.asarray(lines, dtype=float).reshape(-1, 2, 2)


def compare(name, folder):
    original_cls, backed_cls, data = ORGANS[name]
    start = time.perf_counter()
    old = original_cls(data(), seed=SEED)
    old.generate_cells()
    t_old = time.perf_counter() - start
    start = time.perf_counter()
    new = backed_cls(data(), seed=SEED)
    new.generate_cells()
    t_new = time.perf_counter() - start

    g, w = new.mpg, new._mpg_written
    tissue = g.property("tissue")
    mpg_types = [tissue[v] for v in w["cells"]]
    mpg_polygons = [g.property("cell__polygon")[v] for v in w["cells"]]
    kinds = g.property("kind")
    n_walls = sum(1 for v in w["apoplast"] if kinds[v] == WALL)
    n_junctions = sum(1 for v in w["apoplast"] if kinds[v] == JUNCTION)
    colours = _colours({c.type for c in old.all_cells.cells} | set(mpg_types))

    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 8.2), sharex=True, sharey=True)
    _draw_polygons(left, [c.polygon for c in old.all_cells.cells], [c.type for c in old.all_cells.cells], colours)
    left.set_ylabel("y (mm)")
    left.set_title(f"GRANAP: {len(old.all_cells.cells)} cells ({t_old:.1f} s)")
    _draw_polygons(right, mpg_polygons, mpg_types, colours)
    lines = network_lines(new)
    right.add_collection(LineCollection(lines, colors="0.15", linewidths=0.15, alpha=0.5))
    right.set_title(f"MPG: {len(w['cells'])} Cells in {len(w['layers'])} Layers, {n_walls} walls, "
                    f"{n_junctions} junctions, {len(w['connections'])} Connections ({t_new:.1f} s)")
    handles = [matplotlib.patches.Patch(color=colours[t], label=t, alpha=0.6) for t in sorted(colours)]
    fig.legend(handles=handles, loc="center right", fontsize=8, title="cell type")
    same = identical(old, new)
    fig.suptitle(f"{name} (seed {SEED}): sections {'identical' if same else 'DIFFERENT'}", fontsize=14,
                 color="black" if same else "red")
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"{name}: {'identical' if same else 'DIFFERENT'} — {path}")
    return same


if __name__ == "__main__":
    names = sys.argv[1:] or ["monocot_root", "dicot_root"]
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    for name in names:
        compare(name, folder)
