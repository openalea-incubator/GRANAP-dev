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
plot_layers_simple(needle)
plot_section(needle)