
import sys
import os
import time
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy

def test_root_features(show=False):
    t_start = time.time()
    # Create a root anatomy
    root = RootAnatomy()
    root.update_params("cortex", "n_layers", 5)
    root.update_params("inter_cellular_spaces", "tissue", "cortex")
    root.update_params("inter_cellular_spaces", "smoothness", 0.05)
    root.update_params("aerenchyma", "aerenchyma_proportion", 0.1)
    root.update_params("aerenchyma", "n_files", 20)
    # root.plot_layers(show=True, title=f"Root Layers")

    root.generate_cells()
    t_end = time.time()
    print("Time to generate cells:", t_end - t_start)

    # Plot the needle and root anatomy in a 1x2 grid
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharex=False, sharey=False)

<<<<<<< HEAD
# Plot the root cells
t_start = time.time()
root.plot_cells(show=False, title=f"Root Cells", ax=axs[0])
axs[0].get_legend().remove()
t_end = time.time()
print("Time to plot cells:", t_end - t_start)
=======
    # Plot the root cells
    t_start = time.time()
    root.plot_cells(show=show, title=f"Root Cells", ax=axs[0])
    t_end = time.time()
    print("Time to plot cells:", t_end - t_start)
>>>>>>> origin/develop

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
    test_root_features(show=True)
