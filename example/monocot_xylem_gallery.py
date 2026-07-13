"""Gallery: monocot xylem modes — 'default' (metaxylem ring), 'arch', and 'star'.

Two parameterisations per mode:
  default : 1 metaxylem, and 6 metaxylem.
  arch    : 15 metaxylem / 22 protoxylem poles, and 4 metaxylem / 4 poles.
  star    : 11 arms without a pith, and 6 arms with a pith.
"""

import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0

# Shared stele (cell sizing from monocot_iris.py) so every panel has the same
# stele size and the xylem modes are compared at a common scale.
STELE = {"thickness": 0.45, "cell_diameter": 0.009, "cell_diameter_center": 0.022}


def make_root(shape: str, xylem=None, phloem=None, stele=None) -> RootAnatomy:
    """Build a monocot root in the given xylem ``shape`` with per-layer overrides."""
    data = OrganInputData.for_root()
    data.set_value("xylem", "xylem_shape", shape)
    for field, value in {**STELE, **(stele or {})}.items():
        data.set_value("stele", field, value)
    for field, value in (xylem or {}).items():
        data.set_value("xylem", field, value)
    for field, value in (phloem or {}).items():
        data.set_value("phloem", field, value)
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()
    return root


def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


scenarios = [
    # ── default: ring of discrete metaxylem bundles ────────────────────────
    {"label": "Default — 1 metaxylem", "shape": "default",
     "xylem": {"n_vascular_bundles": 1}},
    {"label": "Default — 6 metaxylem", "shape": "default",
     "xylem": {"n_vascular_bundles": 6}},
    # ── arch: metaxylem ring + graded protoxylem poles ─────────────────────
    {"label": "Arch — 6 metaxylem / 10 protoxylem", "shape": "arch",
     "xylem": {"n_metaxylem": 6, "n_vascular_peak": 10,
               "vessel_diameter": 0.08, "vessel_diameter_sd": 0.005,
               "vessel_diameter_min": 0.05, "outer_radius": 0.24,
               "allow_ellipse": True, "ellipse_max_aspect": 1.6,
               "protoxylem_band_depth": 0.06, "protoxylem_diameter": 0.025,
               "protoxylem_diameter_min": 0.010,
               "protoxylem_pole_width_inner": 0.06,
               "protoxylem_pole_width_outer": 0.02,
               "gradient_inflection": 0.5, "gradient_steepness": 2},
     "phloem": {"sieve_diameter": 0.015, "cluster_width": 0.04, "cluster_height": 0.035}},
    {"label": "Arch — 4 metaxylem / 4 protoxylem", "shape": "arch",
     "xylem": {"n_metaxylem": 4, "n_vascular_peak": 4, "outer_radius": 0.24,
               "vessel_diameter": 0.08, "vessel_diameter_min": 0.05 ,
               "protoxylem_diameter_min": 0.01, "protoxylem_diameter": 0.06,
               "protoxylem_pole_width_inner" : 0.06, "protoxylem_pole_width_outer" : 0.02},
     "phloem": {"sieve_diameter": 0.015, "cluster_width": 0.04, "cluster_height": 0.035}},
    # ── star: star-shaped xylem + phloem in the valleys between arms ────────
    {"label": "Star — 11 arms, no pith", "shape": "star",
     "xylem": {"n_vascular_peak": 11, "arc_peak_side": 0.012, "arc_valley_side": 0.02},
     "phloem": {"cluster_width": 0.015, "cluster_height": 0.02, "sieve_diameter": 0.008}},
    {"label": "Star — 6 arms, with pith", "shape": "star",
     "xylem": {"n_vascular_peak": 6, "pith_radius": 0.03}},
]


def main(show=True):
    roots = []
    for s in scenarios:
        print(f"\n=== {s['label']} ===")
        t0 = time.time()
        root = make_root(s["shape"], xylem=s.get("xylem"),
                         phloem=s.get("phloem"), stele=s.get("stele"))
        print(f"  Time: {time.time() - t0:.2f}s")
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

    plt.suptitle("Monocot — xylem modes (default / arch / star)", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
