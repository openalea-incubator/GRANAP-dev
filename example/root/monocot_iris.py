"""Build a iris-like monocot root from the ``for_root`` preset and plot it.

Configuration model: an organ is configured through its ``OrganInputData`` and
then **built once** — construction snapshots the params and parses the vascular
geometry. So all tuning is applied to ``data`` *before* the organ is built, using
``data.set_value(...)`` for existing entries and ``data.params.append(...)`` for
new tissue layers.

Iris (Iris germanica) root anatomy
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


def anatomy_metrics(root: RootAnatomy) -> dict:
    """Geometry summary of a generated root: root/stele diameters and total
    xylem (metaxylem + protoxylem) cross-sectional area.

    Diameters are area-equivalent (``2 * sqrt(area / pi)``), robust to
    non-circular outlines.

    Stele note: ``generate_layer_polygons`` stores each ring at its cell
    *centre-line*, and the stele is a stack of nested rings. So neither the first
    "stele" ring (inset by ``stele_cell/2``) nor the enclosing layer's centre-line
    (out by ``pericycle_cell/2``) is the true boundary. The stele<->pericycle
    interface is the enclosing layer's *inner* edge = its centre-line radius minus
    half its cell diameter, which recovers the nominal ``stele.thickness``.
    """
    root_area = root.generate_base_shape().area

    polys = root.generate_layer_polygons()
    i0 = next((i for i, p in enumerate(polys) if p["name"] == "stele"), None)
    if i0 is not None and i0 > 0:
        enclosing = polys[i0 - 1]
        stele_radius = np.sqrt(enclosing["polygon"].area / np.pi) - enclosing["cell_diameter"] / 2
    else:
        stele_radius = float("nan")

    xylem_area = sum(c.area for c in root.all_cells.cells if "xylem" in c.type)
    return {
        "root_diameter":  np.sqrt(root_area / np.pi) * 2,
        "stele_diameter": stele_radius * 2,
        "xylem_area":   xylem_area,
    }


def main(show=True):
    iris = OrganInputData.for_root()  # monocot preset (planttype=1)

    # -- Stele size + radial parenchyma (pith) gradient ---------------------
    iris.set_value("stele", "thickness",            0.7)
    iris.set_value("stele", "cell_diameter",        0.009)
    iris.set_value("stele", "cell_diameter_center", 0.022)

    # -- Vasculature: ARCH mode ---------------------------------------------
    # Layout is set by just outer_radius (pericycle) + protoxylem_band_depth.
    iris.set_value("xylem", "xylem_shape", "arch")
    iris.set_value("xylem", "n_vascular_peak", 19)          # 19 poles (+ 19 phloem valleys)
    iris.set_value("xylem", "n_metaxylem", 15)              # exactly 15 metaxylem
    iris.set_value("xylem", "vessel_diameter", 0.11)        # metaxylem size
    iris.set_value("xylem", "vessel_diameter_sd", 0.005)     # per-vessel size variation
    iris.set_value("xylem", "vessel_diameter_min", 0.08)    # smallest metaxylem allowed
    iris.set_value("xylem", "outer_radius", 0.350)          # poles reach the pericycle
    iris.set_value("xylem", "pith_radius", 0.0)             # metaxylem ring runs to the centre
    iris.set_value("xylem", "allow_ellipse", True)
    iris.set_value("xylem", "ellipse_max_aspect", 1.6)      # elongate metaxylem where they don't fit

    # Protoxylem chain per pole (outer band), sized by the gradient:
    iris.set_value("xylem", "protoxylem_band_depth", 0.08)  # outer 0.08 = protoxylem + phloem
    iris.set_value("xylem", "protoxylem_diameter", 0.030)   # largest (inner edge of band)
    iris.set_value("xylem", "protoxylem_diameter_min", 0.010)  # smallest (at pericycle)
    iris.set_value("xylem", "protoxylem_pole_width_inner", 0.07)  # pole width near metaxylem
    iris.set_value("xylem", "protoxylem_pole_width_outer", 0.02)  # pole width near pericycle
    iris.set_value("xylem", "gradient_inflection", 0.5)
    iris.set_value("xylem", "gradient_steepness", 2)
    iris.set_value("xylem", "gradient_asymmetry", 1)

    # -- Phloem in the valleys between the poles ----------------------------
    iris.set_value("phloem", "sieve_diameter", 0.015)
    iris.set_value("phloem", "cluster_width",  0.045)
    iris.set_value("phloem", "cluster_height", 0.040)

    # -- Endodermis / pericycle (stele boundary) ----------------------------
    iris.set_value("endodermis", "cell_diameter", 0.019)
    iris.set_value("endodermis", "cell_width",    0.019)
    iris.set_value("pericycle", "cell_diameter",  0.039)
    iris.set_value("pericycle", "cell_width",     0.020)

    # -- Cortex layers (inner / main / outer) -------------------------------
    iris.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.022,
        "cell_width": 0.026,
        "n_layers": 1,
        "shift": 0.5,
        "order": 3.5,
    })

    iris.set_value("cortex", "cell_diameter", 0.0392)
    iris.set_value("cortex", "cell_width",    0.0392)
    iris.set_value("cortex", "n_layers",      25)

    iris.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.03,
        "cell_width": 0.025,
        "n_layers": 3,
        "shift": 0.5,
        "order": 4.5,
    })

    # -- Exodermis / epidermis ----------------------------------------------
    iris.set_value("exodermis", "cell_diameter", 0.036)
    iris.set_value("exodermis", "cell_width",    0.030)
    iris.set_value("exodermis", "n_layers",    3)
    iris.set_value("epidermis", "cell_diameter", 0.016)
    iris.set_value("epidermis", "cell_width",    0.027)

    # -- Intercellular spaces across the cortex tissues ---------------------
    iris.set_value("inter_cellular_spaces", "smoothness", 0.05)
    iris.set_value("inter_cellular_spaces", "tissue",
                   ["inner_cortex", "cortex", "outer_cortex"])

    # Build once, after all configuration is in place.
    iris_anatomy = RootAnatomy(iris, seed=SEED)
    iris_anatomy.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    iris_anatomy.retag_cells("inner_cortex", "cortex")
    iris_anatomy.retag_cells("outer_cortex", "cortex")

    iris_roots = []
    iris_roots.append(iris_anatomy)

    n_cols = 1
    n_rows = 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs_flat = np.atleast_1d(axs).flatten()

    labels = ["Iris"]
    for i, root in enumerate(iris_roots):
        m = anatomy_metrics(root)
        title = (
            f"{labels[i]}\n"
            f"root d = {m['root_diameter']:.3f}   "
            f"stele d = {m['stele_diameter']:.3f}   "
            f"xylem area = {m['xylem_area']:.4f}"
        )
        root.plot_cells(show=False, ax=axs_flat[i], title=title)
        legend = axs_flat[i].get_legend()
        if legend:
            legend.remove()

    for j in range(2, len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.suptitle("Iris anatomy - Cross section", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
