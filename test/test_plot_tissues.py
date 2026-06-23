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


def test_plot_tissues(show=False):
    """
    Verify plot_tissues and build_anatomy_tissues for monocot, dicot, and needle.
    Shows each organ as two side-by-side plots: individual rings vs fused tissues.
    """

    organs = [
        ("monocot",         RootAnatomy(OrganInputData.for_root(),        seed=SEED)),
        ("dicot",           RootAnatomy(OrganInputData.for_dicot_root(),  seed=SEED)),
        ("dicot secondary", _dicot_secondary()),
        ("needle",          NeedleAnatomy(seed=SEED)),
    ]

    for name, organ in organs:
        print(f"\n--- {name} ---")

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

        # Side-by-side: individual rings on the left, fused tissues on the right
        fig, (ax_rings, ax_fused) = plt.subplots(1, 2, figsize=(16, 8))

        organ.plot_tissues(ax=ax_rings, show=False, labels=True, fuse=False)
        ax_rings.set_title(f"{name} - individual rings")

        organ.plot_tissues(ax=ax_fused, show=False, labels=True, fuse=True)
        ax_fused.set_title(f"{name} - fused tissues")

        assert organ.all_cells.cells == [], f"{name}: all_cells was modified by plot_tissues"

        print(f"  layers:            {[t['name'] for t in tissues if t['kind'] == 'layer']}")
        print(f"  vascular polygons: {sum(1 for t in tissues if t['kind'] == 'vascular')}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    print("\nAll plot_tissues tests passed.")


if __name__ == "__main__":
    test_plot_tissues(show=True)
