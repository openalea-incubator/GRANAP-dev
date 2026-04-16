
import sys
import os
import time
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from granap.root_class import RootAnatomy

t_start = time.time()
# Create a needle anatomy
root = RootAnatomy()
root.update_params(param_name="aerenchyma", attribute="aerenchyma_proportion", value = 0.8)
root.update_params(param_name="aerenchyma", attribute="n_files", value = 20)
root.update_params(param_name="cortex", attribute="n_layers", value = 15)
root.update_params(param_name="endodermis", attribute="n_layers", value = 10)
print(root.params)


root.generate_cells()
t_end = time.time()
print("Time to generate cells:", t_end - t_start)

# Plot the needle and root anatomy in a 1x2 grid
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharex=False, sharey=False)

# Plot the root cells
t_start = time.time()
root.plot_cells(show=False, title=f"Root Cells", ax=axs[0])
t_end = time.time()
print("Time to plot cells:", t_end - t_start)

# Export to adjacency matrix
t_start = time.time()
_ = root.export_to_adjencymatrix()
t_end = time.time()
print("Time to export to adjacency matrix:", t_end - t_start)

# Plot the root network
t_start = time.time()
root.plot_network(show=False, title="Root Network", ax=axs[1])
t_end = time.time()
print("Time to plot network:", t_end - t_start)

plt.show()