
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath('..'))

from granap.root_class import RootAnatomy
from granap.input_data import OrganInputData


def make_dicot_root(**stele_kwargs) -> RootAnatomy:
    """Construct a dicot RootAnatomy with custom stele parameters."""
    data = OrganInputData.for_dicot_root()
    for field, value in stele_kwargs.items():
        data.set_value("stele", field, value)
    return RootAnatomy(data)


def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


# ── Scenarios ─────────────────────────────────────────────────────────────

# arc_top / arc_bottom ≈ 0.667 keeps L_top/L_base ≈ 0.575, which matches the
# solver's hard-coded initial guess of alpha=3.  The default SteleDicotParams
# values (0.034 / 0.04) produce a required alpha of ~1.74; the solver overshoots
# below alpha=1 (U-shaped Beta → brentq fails) and stalls.
BASE_KWARGS = {
    "arc_top_xylem": 0.04,
    "arc_bottom_xylem": 0.06,
}

scenarios = [
    {
        "label": "Triarch (3 peaks)\ndefault",
        "kwargs": {**BASE_KWARGS},
    },
    {
        "label": "Pentarch (5 peaks)",
        "kwargs": {**BASE_KWARGS, "n_vascular_peak": 5},
    },
    {
        "label": "Heptarch (7 peaks)",
        "kwargs": {**BASE_KWARGS, "n_vascular_peak": 7},
    },
    {
        "label": "Large vessels\n(diam_max=0.08)",
        "kwargs": {**BASE_KWARGS, "xylem_diameter_max": 0.08, "xylem_diameter_min": 0.05},
    },
    {
        "label": "Narrow peaks",
        "kwargs": {**BASE_KWARGS, "arc_bottom_xylem": 0.05, "arc_top_xylem": 0.05, "xylem_diameter_max": 0.07, "xylem_diameter_min": 0.04, "inner_radius_xylem": 0.04},
    },
    {
        "label": "Narrow star\n(inner=0.10, outer=0.15)",
        "kwargs": {**BASE_KWARGS, "inner_radius_xylem": 0.10, "outer_radius_xylem": 0.15},
    },
    {
        "label": "Wide star\n(inner=0.15, outer=0.20)",
        "kwargs": {**BASE_KWARGS, "inner_radius_xylem": 0.15, "outer_radius_xylem": 0.20},
    },
]

roots = []
for s in scenarios:
    print(f"\n=== {s['label'].replace(chr(10), ' | ')} ===")
    t0 = time.time()
    root = make_dicot_root(**s["kwargs"])
    root.generate_cells()
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.2f} s")

    counts = cell_type_counts(root)
    print("  Cell type counts:")
    for t, n in sorted(counts.items()):
        print(f"    {t:22s}: {n}")

    meta = counts.get("metaxylem", 0)
    stele = counts.get("stele", 0)
    assert meta > 0, f"Expected at least one metaxylem cell in scenario '{s['label']}'"
    print(f"  metaxylem / stele-in-star: {meta} / {stele}")

    # Verify the size-based classification: check there are cells with actual
    # diameter below xylem_diameter_min that were classified as stele
    data_defaults = OrganInputData.for_dicot_root()
    for f, v in s["kwargs"].items():
        data_defaults.set_value("stele", f, v)
    dmin = data_defaults.get("stele").xylem_diameter_min
    small_meta = [
        c for c in root.all_cells.cells
        if c.type == "metaxylem" and c.diameter < dmin
    ]
    assert len(small_meta) == 0, (
        f"Found {len(small_meta)} metaxylem cell(s) with diameter < xylem_diameter_min ({dmin})"
    )

    roots.append(root)

# ── Visualisation ─────────────────────────────────────────────────────────

n = len(scenarios)
n_cols = 4
n_rows = (n + n_cols - 1) // n_cols
fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axs_flat = axs.flatten()

for i, (root, s) in enumerate(zip(roots, scenarios)):
    root.plot_cells(show=False, ax=axs_flat[i], title=s["label"])
    legend = axs_flat[i].get_legend()
    if legend:
        legend.remove()

for j in range(n, len(axs_flat)):
    axs_flat[j].set_visible(False)

plt.suptitle("Dicot root — star-shaped xylem with Apollonian packing", fontsize=14)
plt.tight_layout()
plt.show()
