"""Smoke tests and visualisation for secondary phloem generation."""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from shapely.geometry import Point
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def make_root(**phloem_overrides) -> RootAnatomy:
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    data.set_value("stele", "thickness", 1.0)
    for field, value in phloem_overrides.items():
        data.set_value("secondary_phloem", field, value)
    return RootAnatomy(data, seed=SEED)


def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in list(root.all_cells.cells) + list(root.vascular_cells.cells):
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


scenarios = [
    {"label": "Defaults\n",          "kwargs": {}},
]


def _zone_geometry_ok(root: RootAnatomy) -> tuple[bool, bool]:
    """Return (phloem_at_valley, phloem_at_cambium_arm)."""
    phloem_polys = root.vascular_tissue_polygons.get("secondary_phloem", [])
    if not phloem_polys:
        return False, False
    zone = phloem_polys[0]
    sc = root.secondary_cambium_params
    sp = root.secondary_phloem_params
    n_peaks = root.vascular_params["n_vascular_peak"]

    r_mid         = (sc["inner_distance"] + sp["outer_distance"]) / 2
    valley_angle  = 2 * np.pi * 0.5 / n_peaks   # vessel zone centre (phloem arm)
    cam_arm_angle = 0.0                           # cambium arm centre (parenchyma ray)

    in_valley = zone.contains(Point(r_mid * np.cos(valley_angle), r_mid * np.sin(valley_angle)))
    in_arm    = zone.contains(Point(r_mid * np.cos(cam_arm_angle), r_mid * np.sin(cam_arm_angle)))
    return in_valley, in_arm


def test_secondary_phloem(show=False):
    roots = []
    for s in scenarios:
        label = s["label"].replace("\n", " | ")
        print(f"\n=== {label} ===")
        t0 = time.time()
        root = make_root(**s["kwargs"])
        root.generate_cells()
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.2f}s")

        counts = cell_type_counts(root)
        print("  Cell types:")
        for t, n in sorted(counts.items()):
            print(f"    {t:25s}: {n}")

        # ── Assertions ────────────────────────────────────────────────────────

        phloem_polys = root.vascular_tissue_polygons.get("secondary_phloem", [])
        assert phloem_polys, "No secondary_phloem polygon registered"

        # Zone must be at the vessel-zone angles (behind secondary xylem),
        # not at the cambium-arm angles (parenchyma ray positions).
        in_valley, in_arm = _zone_geometry_ok(root)
        assert in_valley, "Phloem zone not found at vessel-zone (valley) angle"
        assert not in_arm,  "Phloem zone found at cambium-arm (parenchyma ray) angle — wrong position"

        # Sieve elements are always present
        assert counts.get("phloem", 0) > 0, "No phloem (sieve) cells generated"

        # Companion cells only when alive_distance > 0
        alive_dist = s["kwargs"].get("alive_distance", 0.1)
        if alive_dist is not None and alive_dist > 0:
            assert counts.get("companion_cell", 0) > 0, \
                "Expected companion cells with alive_distance > 0"
        else:
            assert counts.get("companion_cell", 0) == 0, \
                "Companion cells should be absent when alive_distance == 0"

        # Parenchyma fills the remainder
        assert counts.get("phloem_parenchyma", 0) > 0, "No phloem_parenchyma cells"

        print(f"  [OK] all assertions passed")
        roots.append(root)

    # ── Visualisation ─────────────────────────────────────────────────────────
    n      = len(scenarios)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axs_flat = axs.flatten()

    for i, (root, s) in enumerate(zip(roots, scenarios)):
        root.plot_cells(show=False, ax=axs_flat[i], title=s["label"])
        legend = axs_flat[i].get_legend()
        if legend:
            legend.remove()

    for j in range(n, len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.suptitle("Dicot root — secondary phloem", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    test_secondary_phloem(show=True)
