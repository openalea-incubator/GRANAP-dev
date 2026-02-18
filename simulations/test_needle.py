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

# Plot the needle anatomy
needle.plot_cells()

# After generating the organ anatomy:
mat = needle.export_to_adjencymatrix()  # builds graph + matrix (1s for connectivity)

# Fill apoplastic conductivities for cortex-cortex walls:
needle.fill_matrix(K=1e-5, label="apoplastic", cell_type="mesophyl")

# Fill for cortex-endodermis interface:
needle.fill_matrix(K=2e-10, label="apoplastic", cell_type="mesophyl-endodermis")

# Fill transmembrane conductivities:
needle.fill_matrix(K=1e-5, label="transmembrane", cell_type="mesophyl")

# Fill symplastic (plasmodesmata) conductivities:
needle.fill_matrix(K=1e-3, label="symplastic", cell_type="mesophyl")

print(needle.graph)
print(needle._matrix)