"""Build a maize-like monocot stem (*Zea mays*) and plot it.

Maize stem anatomy — an *atactostele*: collateral 'face' bundles scattered through
a large parenchymatous ground tissue, under a thin epidermis + 2-layer hypodermis
(no cortex — the ground tissue / 'stele' runs right up to the hypodermis).  The
vasculature is **three bundle bands** (measured sizes, µm):

* **large** (``placement="random"``, 76) — 93 × 72 (h × w), 2 metaxylem Ø20,
  1 protoxylem Ø12, lacuna Ø27, phloem 42 × 29 (w × h); scattered inside;
* **medium** (``placement="even"``, 64) — 50 × 66, 2 metaxylem Ø20, 1 protoxylem
  Ø8, lacuna Ø10, phloem 25 × 15; on a peripheral ring, half-step offset so one
  sits *between* each small bundle;
* **small** (``placement="even"``, 64) — 30 × 39, 2 metaxylem Ø14, 1 protoxylem
  Ø6, **no lacuna**; on the outermost ring just under the hypodermis.

Ground-tissue cell diameter: 7.6 µm under the hypodermis -> 28 µm at 130 µm depth
-> 55 µm at 360 µm depth (then flat to the centre).
"""

import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import OrganInputData, VascularBundleParams

SEED = 0


# ---------------------------------------------------------------------------
# Bundle bands.  BASE is the shared 'face' (monocot, closed collateral) recipe;
# each band overrides only its size, xylem detail, placement and radial position.
# Sizes are in mm.  Stem radius 1.725 mm = ground tissue 1.7125 + 2 hypodermis
# layers (2 x 0.0036) + epidermis (0.0053).
# ---------------------------------------------------------------------------

STEM_RADIUS = 1.725                                  # 3.45 mm diameter
PITH_RADIUS = STEM_RADIUS - 0.0053 - 2 * 0.0036    # 1.7125 mm

BASE = dict(
    bundle_type="collateral", has_cambium=False,     # monocot: closed, no cambium
    xylem_layout="face", phloem_outward=True, shape="ellipse",
    n_metaxylem=2,               # the two big "eyes"
    prop_vessel=0.55, prop_sieve=0.5,
    companion_cell_diameter=0.003, companion_cell_width=0.003,
    parenchyma_diameter=0.005, parenchyma_width=0.005,
    sheath="both", sheath_thickness=0.004,           # sclerenchyma fibre caps + ring
    sclerenchyma_cell_diameter=0.004, sclerenchyma_cell_width=0.004,
)

BANDS = [
    # large — 76 bundles, TALLER than wide (93 x 72 µm), scattered at random
    # through the inner ground tissue; protoxylem + lacuna.
    {**BASE, **dict(
        radius_min=0.0, radius_max=PITH_RADIUS - 0.21, placement="random", n_bundles=56,
        n_caps_layers_outward=1, n_caps_layers_inward=1,
        width=0.072, height=0.093, metaxylem_gap=0.008,
        metaxylem_diameter=0.020, metaxylem_diameter_sd=0.002, metaxylem_diameter_min=0.015,
        n_protoxylem=1, protoxylem_diameter=0.012, protoxylem_diameter_min=0.010,
        protoxylem_width=0.012, protoxylem_height=0.012, protoxylem_relative_distance=0.3,
        lacuna=True, lacuna_width=0.027, lacuna_height=0.027,
        phloem_width=0.042, phloem_height=0.029, phloem_relative_distance=0.5,
    )},
    # medium — 64 bundles, WIDER than tall (50 x 66 µm), on a peripheral ring
    # half-step offset (angle = 180 / n_bundles) so one sits *between* each small
    # bundle; protoxylem + lacuna.
    {**BASE, **dict(
        radius=PITH_RADIUS - 0.075, placement="even", angle=180.0 / 64, n_bundles=64,
        n_caps_layers_outward=1, n_caps_layers_inward=1,
        width=0.066, height=0.050, metaxylem_gap=0.008,
        metaxylem_diameter=0.020, metaxylem_diameter_sd=0.002, metaxylem_diameter_min=0.015,
        n_protoxylem=1, protoxylem_diameter=0.008, protoxylem_diameter_min=0.006,
        protoxylem_width=0.008, protoxylem_height=0.008, protoxylem_relative_distance=0.3,
        lacuna=True, lacuna_width=0.010, lacuna_height=0.010,
        phloem_width=0.025, phloem_height=0.015, phloem_relative_distance=0.5,
    )},
    # small — 64 bundles, WIDER than tall (30 x 39 µm), on the outermost ring just
    # under the hypodermis; protoxylem but NO lacuna.
    {**BASE, **dict(
        radius=PITH_RADIUS - 0.024, placement="even", angle=0.0, n_bundles=64,
        n_caps_layers_outward=1, n_caps_layers_inward=1,
        width=0.039, height=0.030, metaxylem_gap=0.004,
        metaxylem_diameter=0.014, metaxylem_diameter_sd=0.001, metaxylem_diameter_min=0.010,
        n_protoxylem=1, protoxylem_diameter=0.006, protoxylem_diameter_min=0.005,
        protoxylem_width=0.006, protoxylem_height=0.006, protoxylem_relative_distance=0.3,
        lacuna=False,
        phloem_width=0.016, phloem_height=0.009, phloem_relative_distance=0.4,
    )},
]


def build_maize() -> OrganInputData:
    """Assemble the maize-stem ``OrganInputData`` (monocot preset + 3 bundle bands)."""
    data = OrganInputData.for_monocot_stem()

    # -- Ground tissue ('stele'): solid, radius PITH_RADIUS.  Cell diameter grows
    #    from 7.6 µm under the hypodermis to 28 µm at 130 µm depth and 55 µm at
    #    360 µm depth, flat beyond.  The 5PL shape is fitted to those three points
    #    (edge ~8.3, 130 µm -> ~27.8, 360 µm -> ~52.9); ``cell_diameter`` is the 5PL
    #    lower asymptote, set below 7.6 so the edge cell comes out near 7.6.
    data.set_value("pith", "thickness",                2 * PITH_RADIUS)
    data.set_value("pith", "cell_diameter",            0.005)
    data.set_value("pith", "cell_diameter_center",     0.055)
    data.set_value("pith", "size_gradient_function",   "five_pl")
    data.set_value("pith", "size_gradient_inflection", 1.0)
    data.set_value("pith", "size_gradient_steepness",  19.0)
    data.set_value("pith", "size_gradient_asymmetry",  3.9)
    data.set_value("pith", "cavity_radius",            0.0)

    # -- Rind: no cortex, no sclerenchyma ring — just a 2-layer hypodermis
    #    (Ø 3.6 µm) and the epidermis (D 5.3 x W 7.8 µm) ------------------------
    for name in ("cortex", "sclerenchyma", "inter_cellular_spaces", "aerenchyma"):
        data.remove_param(name)
    data.params.append({
        "name": "hypodermis",
        "cell_diameter": 0.0036, "cell_width": 0.0036, "n_layers": 2, "shift": 0.5, "order": 5,
    })
    data.set_values("epidermis", cell_diameter=0.0053, cell_width=0.0078, order=6)

    # -- Vasculature: drop the preset's single bundle spec, add the three bands ---
    data.params = [p for p in data.params if getattr(p, "name", None) != "vascular_bundle"]
    for band in BANDS:
        data.params.append(VascularBundleParams(**band))

    return data


def main(show=True):
    data = build_maize()
    print("=== maize monocot stem ===")
    t0 = time.time()
    stem = StemAnatomy(data, seed=SEED)
    stem.generate_cells()

    dt = time.time() - t0
    counts = {}
    for c in stem.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    n_req = sum(b["n_bundles"] for b in BANDS)
    n_got = len(stem.vascular_tissue_polygons.get("bundle", []))
    print(f"  Time: {dt:.2f}s   bundles requested: {n_req}   placed: {n_got}   "
          f"cells: {len(stem.all_cells.cells)}")
    for t in sorted(counts):
        print(f"    {t:16s} {counts[t]}")

    fig, ax = plt.subplots(figsize=(10, 10))
    stem.plot_cells(show=False, ax=ax, title="Maize monocot stem — atactostele (3 bundle bands)")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("tissue")
        for txt in leg.get_texts():
            txt.set_fontsize(7)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
