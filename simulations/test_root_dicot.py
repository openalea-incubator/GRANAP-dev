
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from granap.root_class import RootAnatomy
from granap.visualization import plot_layers_simple, plot_section

from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData

# Create a needle anatomy
root = RootAnatomy()
root.update_params("planttype", "value", 2)
root.generate_cells()
root.plot_cells(ax=axs[0, 0], title="Root Cells")