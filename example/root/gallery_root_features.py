
import sys
import os
import time
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


def main(show=False):
    t_start = time.time()
    # Configure the input data, then build once.
    data = OrganInputData.for_root()
    data.set_value("cortex", "n_layers", 5)
    # Set the scalar smoothness before narrowing `tissue` so the cross-field
    # length validator (smoothness vs tissue) stays satisfied at each step.
    data.set_value("inter_cellular_spaces", "smoothness", 0.05)
    data.set_value("inter_cellular_spaces", "tissue", ["cortex"])
    data.set_value("aerenchyma", "aerenchyma_proportion", 0.1)
    data.set_value("aerenchyma", "n_files", 20)

    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()
    t_end = time.time()
    print("Time to generate cells:", t_end - t_start)

    # Plot the needle and root anatomy in a 1x2 grid
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharex=False, sharey=False)

    # Plot the root cells
    t_start = time.time()
    root.plot_cells(show=show, title=f"Root Cells", ax=axs[0])
    t_end = time.time()
    print("Time to plot cells:", t_end - t_start)

    # Export to adjacency matrix
    t_start = time.time()
    _ = root.export_to_adjencymatrix()
    t_end = time.time()
    print("Time to export to adjacency matrix:", t_end - t_start)

    # Plot the root network
    t_start = time.time()
    root.plot_network(show=show, title="Root Network", ax=axs[1])
    t_end = time.time()
    print("Time to plot network:", t_end - t_start)
    
    if show:
        plt.show()


if __name__ == "__main__":
    main(show=True)
