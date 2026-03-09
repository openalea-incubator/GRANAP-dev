import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from granap.needle_class import NeedleAnatomy
from granap.root_class import RootAnatomy
from granap.visualization import plot_layers_simple, plot_section

# Create a needle anatomy
needle = NeedleAnatomy()
root = RootAnatomy()

# Plot the needle and root anatomy in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(20, 20), sharex=False, sharey=False)

# Generate adjacency matrices to build the graph before plotting network
mat_root = root.export_to_adjencymatrix()  # builds graph + matrix (1s for connectivity)
mat_needle = needle.export_to_adjencymatrix()  # builds graph + matrix (1s for connectivity)

# Plot Root and Needle cells on top row
root.plot_cells(show=False, ax=axs[0, 0], title="Root Cells")
needle.plot_cells(show=False, ax=axs[0, 1], title="Needle Cells")

# Plot Root and Needle networks on bottom row
root.plot_network(ax=axs[1, 0], title="Root Network")
needle.plot_network(ax=axs[1, 1], title="Needle Network")

plt.tight_layout()
plt.show()
