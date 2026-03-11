import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from granap.needle_class import NeedleAnatomy
from granap.visualization import plot_layers_simple, plot_section

# Create a needle anatomy
needle = NeedleAnatomy()
for i in range(1,10):
    needle.update_params("resin_duct", "n_files", i)
    needle.plot_cells(show=True, title="Needle Cells")


