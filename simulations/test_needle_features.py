import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from granap.needle_class import NeedleAnatomy
from granap.visualization import plot_layers_simple, plot_section

# Create a needle anatomy
needle = NeedleAnatomy()

needle.update_params("resin_duct", "n_files", 2)
needle.update_params("stomata", "n_files", 10)
needle.plot_layers(show=True, title=f"Needle Layers")

needle.plot_cells(show=True, title=f"Needle Cells")


_ = needle.export_to_adjencymatrix()
needle.plot_network(show=True, title="Needle Network")