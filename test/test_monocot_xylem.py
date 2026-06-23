"""Tests for monocot star-shaped xylem + pith feature."""
 
import sys
import os
import time
import matplotlib.pyplot as plt
 
sys.path.append(os.path.abspath('..'))
 
from shapely.geometry import Point
 
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData
 
 
SEED = 0


def make_star_root(**xylem_overrides) -> RootAnatomy:
    data = OrganInputData.for_root()
    data.set_value("xylem", "xylem_shape", "star")
    for field, value in xylem_overrides.items():
        data.set_value("xylem", field, value)
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()
    return root
 
 
def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts
 
 
def test_star_mode_no_pith():
    """Star mode without pith: xylem vessels exist, no cells inside a zero-radius circle."""
    root = make_star_root()
    counts = cell_type_counts(root)
    print("Star mode (no pith) cell counts:", counts)
    assert "xylem" in counts, "Expected xylem cells in star mode"
    assert counts["xylem"] > 0, "Expected at least one xylem cell"
 
 
def test_star_mode_with_pith():
    """Star mode with pith_radius=0.05: no xylem cells inside the pith circle,
    and stele/pith cells are present inside it."""
    pith_r = 0.05
    root = make_star_root(pith_radius=pith_r)
    counts = cell_type_counts(root)
    print("Star mode (pith_radius=0.05) cell counts:", counts)
 
    cx = 0.0
    cy = 0.0
    pith_circle = Point(cx, cy).buffer(pith_r)
 
    # No xylem vessels should be placed inside the pith circle
    xylem_in_pith = [
        c for c in root.all_cells.cells
        if c.type == "xylem" and pith_circle.contains(Point(c.x, c.y))
    ]
    assert len(xylem_in_pith) == 0, (
        f"Found {len(xylem_in_pith)} xylem cells inside the pith circle — expected 0"
    )
 
    # Stele/pith cells should exist inside the pith circle
    stele_in_pith = [
        c for c in root.all_cells.cells
        if c.type == "stele" and pith_circle.contains(Point(c.x, c.y))
    ]
    assert len(stele_in_pith) > 0, "Expected stele (pith) cells inside the pith circle"
 
    assert "xylem" in counts, "Expected xylem cells outside the pith"
 
 
def test_star_vs_default_both_produce_cells():
    """Both modes produce a reasonable number of cells."""
    data_default = OrganInputData.for_root()
    root_default = RootAnatomy(data_default)
    root_default.generate_cells()
    counts_default = cell_type_counts(root_default)
 
    root_star = make_star_root()
    counts_star = cell_type_counts(root_star)
 
    print("Default mode counts:", counts_default)
    print("Star mode counts:   ", counts_star)
 
    assert sum(counts_default.values()) > 10, "Default mode produced too few cells"
    assert sum(counts_star.values()) > 10, "Star mode produced too few cells"
 
 
scenarios = [
    {"label": "Star — no pith",         "kwargs": {}},
    {"label": "Star — pith_radius=0.04","kwargs": {"pith_radius": 0.04}},
    {"label": "Star — 3 arms",          "kwargs": {"n_vascular_peak": 3, "outer_radius": 0.12}},
    {"label": "Star — 7 arms, pith_radius=0.035, inner_radius=0.035",          "kwargs": {"n_vascular_peak": 7, "outer_radius": 0.18, "inner_radius": 0.035,
                                                    "pith_radius": 0.035,"arc_bottom": 0.02, "arc_top": 0.012}},
]
 
 
def test_monocot_star_visual(show=False):
    roots = []
    for s in scenarios:
        print(f"\n=== {s['label']} ===")
        t0 = time.time()
        root = make_star_root(**s["kwargs"])
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.2f}s")
        print("  Cell types:", cell_type_counts(root))
        roots.append(root)
 
    n_cols = 2
    n_rows = (len(scenarios) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axs_flat = axs.flatten()
 
    for i, (root, s) in enumerate(zip(roots, scenarios)):
        root.plot_cells(show=False, ax=axs_flat[i], title=s["label"])
        leg = axs_flat[i].get_legend()
        if leg:
            leg.remove()
 
    for j in range(len(scenarios), len(axs_flat)):
        axs_flat[j].set_visible(False)
 
    plt.suptitle("Monocot — star xylem mode", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()
 
 
if __name__ == "__main__":
    test_star_mode_no_pith()
    test_star_mode_with_pith()
    test_star_vs_default_both_produce_cells()
    test_monocot_star_visual(show=True)
 