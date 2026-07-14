import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def main(show=False):
    data = OrganInputData.for_dicot_root()
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()

    types = {c.type for c in root.all_cells.cells}
    print("Cell types:", types)

    n_cambium = sum(1 for c in root.all_cells.cells if c.type == "cambium")
    print(f"Cambium cell seeds: {n_cambium}")

    root.plot_cells(show=show, title="Dicot root — primary cambium")


if __name__ == "__main__":
    main(show=True)
