import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from granap.root_class import RootAnatomy
from granap.input_data import OrganInputData

data = OrganInputData.for_dicot_root()
root = RootAnatomy(data)
root.generate_cells()

types = {c.type for c in root.all_cells.cells}
print("Cell types:", types)

n_cambium = sum(1 for c in root.all_cells.cells if c.type == "cambium")
print(f"Cambium cell seeds: {n_cambium}")

root.plot_cells(show=True, title="Dicot root — primary cambium")
