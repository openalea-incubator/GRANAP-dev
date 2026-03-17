
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from granap.root_class import RootAnatomy
from granap.visualization import plot_layers_simple, plot_section

# Create a needle anatomy
root = RootAnatomy()

root.update_params("inter_cellular_space", "aerenchyma_proportion", 0.1)
root.update_params("inter_cellular_space", "cortex", 0.01)
root.update_params("inter_cellular_space", "n_files", 5)
root.update_params("inter_cellular_space", "aerenchyma_type", 2)
# root.plot_layers(show=True, title=f"Root Layers")

root.plot_cells(show=True, title=f"Root Cells")


_ = root.export_to_adjencymatrix()
root.plot_network(show=True, title="Root Network")