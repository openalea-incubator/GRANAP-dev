import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def _dicot_secondary():
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    data.set_value("stele", "thickness", 1.2)
    return RootAnatomy(data, seed=SEED)


def test_plot_tissues():
    """Verify plot_tissues and build_anatomy_tissues for monocot, dicot, and needle:
    both kinds of tissue are returned, no polygon is empty, and the dry-run (both
    build_anatomy_tissues and plot_tissues) never materialises cells."""

    organs = [
        ("monocot",         RootAnatomy(OrganInputData.for_root(),        seed=SEED)),
        ("dicot",           RootAnatomy(OrganInputData.for_dicot_root(),  seed=SEED)),
        ("dicot secondary", _dicot_secondary()),
        ("needle",          NeedleAnatomy(seed=SEED)),
    ]

    for name, organ in organs:
        # build_anatomy_tissues must return a non-empty list with both kinds
        tissues = organ.build_anatomy_tissues()
        assert len(tissues) > 0, f"{name}: build_anatomy_tissues returned empty list"
        kinds = {t["kind"] for t in tissues}
        assert "layer" in kinds,    f"{name}: no layer entries"
        assert "vascular" in kinds, f"{name}: no vascular entries"
        for t in tissues:
            assert not t["polygon"].is_empty, f"{name}: empty polygon for '{t['name']}'"

        # state must be untouched after the dry-run
        assert organ.all_cells.cells == [], f"{name}: all_cells was modified"

        # plot_tissues must render onto axes (both modes) without materialising cells
        fig, (ax_rings, ax_fused) = plt.subplots(1, 2, figsize=(16, 8))
        organ.plot_tissues(ax=ax_rings, show=False, labels=True, fuse=False)
        organ.plot_tissues(ax=ax_fused, show=False, labels=True, fuse=True)
        assert organ.all_cells.cells == [], f"{name}: all_cells was modified by plot_tissues"
        plt.close(fig)
