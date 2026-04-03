
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from granap.root_class import RootAnatomy
from granap.visualization import plot_layers_simple, plot_section

from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData

# Create a needle anatomy
root = RootAnatomy()

root.update_params("inter_cellular_space", "cortex", 0.1)
root.update_params("inter_cellular_space", "n_files", 2)
root.update_params("inter_cellular_space", "aerenchyma_type", 1)

root.update_params("stele", "n_vascular_bundles", 3)

results = {}
asked_proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
results['asked_proportions'] = asked_proportions
results['actual_proportions_type1'] = []
results['actual_proportions_type2'] = []
results['k_root_type1'] = []
results['k_root_type2'] = []

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
    results['actual_proportions_type1'].append(total_air_area/(total_cortex_area+total_air_area))

    root.write_to_xml("test_ganache_type1_"+str(aer)+".xml")

    root.update_params("inter_cellular_space", "aerenchyma_type", 2)
    root.generate_cells()
    cortex_cells = root.all_cells.get_cells_by_type('cortex')
    air_spaces = root.all_cells.get_cells_by_type('air space')

    total_cortex_area = sum(c.polygon.area for c in cortex_cells if c.polygon is not None)
    total_air_area = sum(c.polygon.area for c in air_spaces if c.polygon is not None)

    print('Aerenchyma proportion:', total_air_area/(total_cortex_area+total_air_area))
    results['actual_proportions_type2'].append(total_air_area/(total_cortex_area+total_air_area))

    root.write_to_xml("test_ganache_type2_"+str(aer)+".xml")

    Ganache_1 = InData(cellset_file="test_ganache_type1_"+str(aer)+".xml")
    Ganache_1.geometry.set_maturity_stages([1])
    mecha_ganache_1 = Mecha(Ganache_1)

    print("mecha ganache 1")
    mecha_ganache_1.compute_conductivities()
    for i in range(len(mecha_ganache_1.root_hydraulic_properties)):
        print(mecha_ganache_1.root_hydraulic_properties[i])
        results['k_root_type1'].append(mecha_ganache_1.root_hydraulic_properties[i]['kr'])

    Ganache_2 = InData(cellset_file="test_ganache_type2_"+str(aer)+".xml")
    Ganache_2.geometry.set_maturity_stages([1])
    mecha_ganache_2 = Mecha(Ganache_2)

    print("mecha ganache 2")
    mecha_ganache_2.compute_conductivities()
    for i in range(len(mecha_ganache_2.root_hydraulic_properties)):
        print(mecha_ganache_2.root_hydraulic_properties[i])
        results['k_root_type2'].append(mecha_ganache_2.root_hydraulic_properties[i]['kr'])


plt.plot(asked_proportions, results['actual_proportions_type1'], label='Type 1', marker='o', linestyle='--', color='red')
plt.plot(asked_proportions, results['actual_proportions_type2'], label='Type 2', marker='x', linestyle='--', color='blue')
plt.plot(asked_proportions, asked_proportions, label='Ideal', marker='*', linestyle='--', color='black')
plt.legend()
plt.xlabel("Asked proportion")
plt.ylabel("Actual proportion")
plt.title("Aerenchyma proportion")
plt.show()

plt.plot(asked_proportions, results['k_root_type1'], label='Type 1', marker='o', linestyle='--', color='red')
plt.plot(asked_proportions, results['k_root_type2'], label='Type 2', marker='x', linestyle='--', color='blue')
plt.legend()
plt.xlabel("Asked proportion")
plt.ylabel("k_root")
plt.title("Root hydraulic conductivity")
plt.show()

os.remove("test_ganache_type1_0.1.xml")
os.remove("test_ganache_type2_0.1.xml")
os.remove("test_ganache_type1_0.2.xml")
os.remove("test_ganache_type2_0.2.xml")
os.remove("test_ganache_type1_0.3.xml")
os.remove("test_ganache_type2_0.3.xml")
os.remove("test_ganache_type1_0.4.xml")
os.remove("test_ganache_type2_0.4.xml")
os.remove("test_ganache_type1_0.5.xml")
os.remove("test_ganache_type2_0.5.xml")
os.remove("test_ganache_type1_0.6.xml")
os.remove("test_ganache_type2_0.6.xml")
os.remove("test_ganache_type1_0.7.xml")
os.remove("test_ganache_type2_0.7.xml")
os.remove("test_ganache_type1_0.8.xml")
os.remove("test_ganache_type2_0.8.xml")
os.remove("test_ganache_type1_0.9.xml")
os.remove("test_ganache_type2_0.9.xml")
os.remove("test_ganache_type1_1.xml")
os.remove("test_ganache_type2_1.xml")
