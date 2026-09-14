"""Snowdrop (*Galanthus*) leaf cross-section — a measured monocot leaf.

All lengths are in mm.  Built from real measurements:

* length (cross-section width) 6.05; thick-keeled thickness profile (0.87 at the
  centre, 0.48 at 0.75, 0.38 at 2.5), tapering to the margins;
* 17 bundles in the sequence 2 minor / 6 medium / 1 major / 6 medium / 2 minor,
  evenly spaced, the outermost 0.27 from the leaf edge;
* the major bundle sits 0.34 from the adaxial face / 0.53 from the abaxial, with
  an abaxial-only rib (the keel); face bundles with protoxylem + phloem, no
  metaxylem, no sclerenchyma;
* inter-bundle aerenchyma leaving ~2 mesophyll cells beside each bundle and 4 / 3
  cells from the adaxial / abaxial faces;
* amphistomatous, denser adaxially (1 per 0.3 mm) than abaxially (1 per 0.8 mm).

"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.leaf_class import LeafAnatomy

SEED = 0

LENGTH = 6.05
HALF = LENGTH / 2.0
mesophyll_CELL = 0.027                       # mesophyll cell size, the aerenchyma-margin unit


def _bundle(**kw):
    """A snowdrop face bundle: protoxylem + phloem, no metaxylem, no sclerenchyma."""
    b = dict(name="vascular_bundle", placement="explicit",
             bundle_type="collateral", has_cambium=False, phloem_outward=True,
             xylem_layout="face", lacuna=False, sheath="none", shape="ellipse",
             n_metaxylem=0, n_protoxylem=1, relative_distance=0.55)
    b.update(kw)
    return b


def build_snowdrop():
    x_out = HALF - 0.27                                   # outermost bundle
    xs = np.linspace(-x_out, x_out, 17)                   # even 2-6-1-6-2 row
    minor_x = [float(xs[i]) for i in (0, 1, 15, 16)]
    medium_x = [float(xs[i]) for i in list(range(2, 8)) + list(range(9, 15))]
    major_x = [float(xs[8])]                              # centre

    minor = _bundle(x_positions=minor_x, width=0.05, height=0.085,
                    protoxylem_diameter=0.008, protoxylem_diameter_min=0.006,
                    protoxylem_width=0.025, protoxylem_height=0.02,
                    phloem_width=0.02, phloem_height=0.016)
    medium = _bundle(x_positions=medium_x, width=0.054, height=0.25,
                     protoxylem_diameter=0.022, protoxylem_diameter_min=0.013,
                     protoxylem_width=0.03, protoxylem_height=0.075,
                     phloem_width=0.03, phloem_height=0.02)
    major = _bundle(x_positions=major_x, width=0.073, height=0.265, relative_distance=0.58,
                    protoxylem_diameter=0.025, protoxylem_diameter_min=0.010,
                    protoxylem_width=0.04, protoxylem_height=0.13,
                    phloem_width=0.03, phloem_height=0.066, rib_abaxial_height=0.105, rib_abaxial_width=0.176, 
                    rib_adaxial_height=0.0)  

    return [
        {"name": "planttype", "value": 1, "organ": "leaf",
         "width": LENGTH,
         # Folded, keeled section.  Measured from the straight tip-to-tip chord, at the
         # midrib the adaxial surface is 0.1 mm below it and the abaxial 1.0 mm below —
         # so the leaf is 0.9 mm thick there (thickness_profile centre) and the mid-line
         # sags fold_sag = (0.1 + 1.0) / 2 = 0.55 mm.  The fold spans the whole width.
         "thickness_profile": [[0.0, 0.8], [0.75, 0.48], [2.5, 0.38], [3, 0.15],[HALF, 0.0]],
         "fold_sag": 0.55, "fold_width": LENGTH, "edge_radius": 0.03},
        {"name": "epidermis", "cell_diameter": 0.03, "cell_width": 0.022,
         "n_layers": 1, "shift": 0.3, "order": 3},
        {"name": "mesophyll", "cell_diameter": mesophyll_CELL, "cell_width": 0.033},
        minor, medium, major,
        {"name": "xylem"},
        {"name": "phloem", "sieve_diameter": 0.008,
         "sieve_diameter_min": 0.006, "sieve_diameter_sd": 0.001},
        {"name": "cambium"},
        {"name": "inter_bundle_aerenchyma",
         "side_margin": 2 * mesophyll_CELL, "adaxial_margin": 5 * mesophyll_CELL, "abaxial_margin": 3 * mesophyll_CELL},
        {"name": "stomata",
         "n_adaxial": int(round(LENGTH / 0.8)), "n_abaxial": int(round(LENGTH / 0.3)),
         "width": 0.03, "depth": 0.025, "sub_chamber": 0.05},
    ]


def main(show=True):
    print("=== snowdrop (Galanthus) leaf ===")
    t0 = time.time()
    leaf = LeafAnatomy(build_snowdrop(), seed=SEED)
    leaf.generate_cells()
    counts = {}
    for c in leaf.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    n_b = len(leaf.vascular_tissue_polygons.get("bundle", []))
    print(f"  Time: {time.time() - t0:.2f}s   cells: {len(leaf.all_cells.cells)}   bundles: {n_b}")
    for t in sorted(counts):
        print(f"    {t:14s} {counts[t]}")

    fig, ax = plt.subplots(figsize=(15, 3.0))
    leaf.plot_cells(show=False, ax=ax, title="Snowdrop (Galanthus) leaf — measured monocot")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("tissue")
        for txt in leg.get_texts():
            txt.set_fontsize(7)
    ax.set_aspect("equal")
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
