
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from granap.root_class import RootAnatomy
from granap.visualization import plot_layers_simple, plot_section

# Create a needle anatomy
root = RootAnatomy()

root.update_params("inter_cellular_space", "cortex", 0.1)
root.update_params("inter_cellular_space", "n_files", 2)
root.update_params("inter_cellular_space", "aerenchyma_type", 1)

root.update_params("stele", "n_vascular_bundles", 3)

asked_proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
actual_proportions_type1 = []
actual_proportions_type2 = []
for aer in asked_proportions:
    print(aer)
    root.update_params("inter_cellular_space", "aerenchyma_type", 1)
    root.update_params("inter_cellular_space", "cortex", aer)
    root.update_params("inter_cellular_space", "n_files", int(np.ceil(10*aer)*2))
    root.generate_cells()
    cortex_cells = root.all_cells.get_cells_by_type('cortex')
    air_spaces = root.all_cells.get_cells_by_type('air space')
    total_cortex_area = sum(c.polygon.area for c in cortex_cells if c.polygon is not None)
    total_air_area = sum(c.polygon.area for c in air_spaces if c.polygon is not None)

    print('Aerenchyma proportion:', total_air_area/(total_cortex_area+total_air_area))
    actual_proportions_type1.append(total_air_area/(total_cortex_area+total_air_area))

    root.update_params("inter_cellular_space", "aerenchyma_type", 2)
    root.generate_cells()
    cortex_cells = root.all_cells.get_cells_by_type('cortex')
    air_spaces = root.all_cells.get_cells_by_type('air space')

    total_cortex_area = sum(c.polygon.area for c in cortex_cells if c.polygon is not None)
    total_air_area = sum(c.polygon.area for c in air_spaces if c.polygon is not None)

    print('Aerenchyma proportion:', total_air_area/(total_cortex_area+total_air_area))
    actual_proportions_type2.append(total_air_area/(total_cortex_area+total_air_area))

plt.plot(asked_proportions, actual_proportions_type1, label='Type 1', marker='o', linestyle='--', color='red')
plt.plot(asked_proportions, actual_proportions_type2, label='Type 2', marker='x', linestyle='--', color='blue')
plt.plot(asked_proportions, asked_proportions, label='Ideal', marker='*', linestyle='--', color='black')
plt.legend()
plt.xlabel("Asked proportion")
plt.ylabel("Actual proportion")
plt.title("Aerenchyma proportion")
plt.show()
