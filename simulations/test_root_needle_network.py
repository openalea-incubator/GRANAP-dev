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

# Plot the needle anatomy
root.plot_cells()
needle.plot_cells()

# After generating the organ anatomy:
mat_root = root.export_to_adjencymatrix()  # builds graph + matrix (1s for connectivity)
mat_needle = needle.export_to_adjencymatrix()  # builds graph + matrix (1s for connectivity)
root.plot_network()
needle.plot_network()


