"""Gallery: root & needle cells and connectivity networks.

Top row shows cells, bottom row shows the adjacency network, first for default
programmatic organs and then for organs built from XML / param-list input.
"""

import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../test'))  # for the param_pinus fixture

from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData
from openalea.granap.organ_class import Organ


def main(show=True):
    # --- default programmatic organs ---
    needle = NeedleAnatomy()
    root = RootAnatomy()

    root.export_to_adjencymatrix()
    needle.export_to_adjencymatrix()

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    root.plot_cells(show=False, ax=axs[0, 0], title="Root Cells")
    needle.plot_cells(show=False, ax=axs[0, 1], title="Needle Cells")
    root.plot_network(ax=axs[1, 0], show=False, title="Root Network")
    needle.plot_network(ax=axs[1, 1], show=False, title="Needle Network")
    plt.tight_layout()

    # --- organs built from XML and from a param list ---
    root_sim = Organ.create_from_input(
        OrganInputData.from_xml("../test/inputs/root_monocot_simpl.xml")
    )
    from inputs.param_pinus import params_pinaster  # noqa: E402
    needle_sim = Organ.create_from_input(OrganInputData.from_dict_list(params_pinaster))

    root_sim.export_to_adjencymatrix()
    needle_sim.export_to_adjencymatrix()

    fig2, axs2 = plt.subplots(2, 2, figsize=(20, 20))
    root_sim.plot_cells(show=False, ax=axs2[0, 0], title="Root Cells (XML)")
    needle_sim.plot_cells(show=False, ax=axs2[0, 1], title="Needle Cells (Param)")
    root_sim.plot_network(ax=axs2[1, 0], show=False, title="Root Network (XML)")
    needle_sim.plot_network(ax=axs2[1, 1], show=False, title="Needle Network (Param)")
    plt.tight_layout()

    if show:
        plt.show()


if __name__ == "__main__":
    main()
