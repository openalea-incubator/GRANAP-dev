import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.input_data import OrganInputData
from openalea.granap.visualization import plot_layers_simple, plot_section

def main(show=False):
    # Configure the input data, then build once.
    data = OrganInputData.for_needle()
    data.set_value("resin_duct", "n_files", 2)
    data.set_value("stomata", "n_files", 10)
    needle = NeedleAnatomy(data)

    needle.plot_layers(show=show, title=f"Needle Layers")

    needle.plot_cells(show=show, title=f"Needle Cells")

    _ = needle.export_to_adjencymatrix()
    needle.plot_network(show=show, title="Needle Network")

if __name__ == "__main__":
    main(show=True)
