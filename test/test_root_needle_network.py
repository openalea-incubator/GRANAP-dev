import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.root_class import RootAnatomy
from openalea.granap.visualization import plot_layers_simple, plot_section
from openalea.granap.input_data import OrganInputData
from openalea.granap.organ_class import Organ

def test_root_and_needle_networks(show=False):
    # Create a needle anatomy
    needle = NeedleAnatomy()
    root = RootAnatomy()

    # Plot the needle and root anatomy in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(20, 20), sharex=False, sharey=False)

    # Generate adjacency matrices to build the graph before plotting network
    mat_root = root.export_to_adjencymatrix()  # builds graph + matrix (1s for connectivity)
    mat_needle = needle.export_to_adjencymatrix()  # builds graph + matrix (1s for connectivity)

    # Plot Root and Needle cells on top row
    root.plot_cells(show=show, ax=axs[0, 0], title="Root Cells")
    needle.plot_cells(show=show, ax=axs[0, 1], title="Needle Cells")

    # Plot Root and Needle networks on bottom row
    root.plot_network(ax=axs[1, 0], show=show, title="Root Network")
    needle.plot_network(ax=axs[1, 1], show=show, title="Needle Network")
    
    if show:
        plt.tight_layout()
        plt.show()

    # From XML:
    input_data_root = OrganInputData.from_xml("inputs/root_monocot_simpl.xml")
    root_sim = Organ.create_from_input(input_data_root)

    # From param list:
    from inputs.param_pinus import params_pinaster
    input_data_needle = OrganInputData.from_dict_list(params_pinaster)
    needle_sim = Organ.create_from_input(input_data_needle)

    # Plot the needle and root anatomy in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(20, 20), sharex=False, sharey=False)

    # Generate adjacency matrices to build the graph before plotting network
    mat_root_sim = root_sim.export_to_adjencymatrix()  # builds graph + matrix (1s for connectivity)
    mat_needle_sim = needle_sim.export_to_adjencymatrix()  # builds graph + matrix (1s for connectivity)

    # Plot Root and Needle cells on top row
    root_sim.plot_cells(show=show, ax=axs[0, 0], title="Root Cells (XML)")
    needle_sim.plot_cells(show=show, ax=axs[0, 1], title="Needle Cells (Param)")

    # Plot Root and Needle networks on bottom row
    root_sim.plot_network(ax=axs[1, 0], show=show, title="Root Network (XML)")
    needle_sim.plot_network(ax=axs[1, 1], show=show, title="Needle Network (Param)")
    n_metaxylem = len([cell for cell in root_sim.all_cells.cells if cell.type == "metaxylem"])
    n_airspace = len([cell for cell in root_sim.all_cells.cells if cell.type == "air space"])
    # assert that root from xml has 3 metaxylem
    assert n_metaxylem == 3
    assert n_airspace == 0
    
    if show:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_root_and_needle_networks(show=True)
