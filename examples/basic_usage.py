# examples/basic_usage.py
from xml_reader import ReadXML
from xml_writer import AnatomyWriter
from plot_2d import RootPlotter
from root_anatomy import RootAnatomy
import matplotlib.pyplot as plt

# 1. Load parameters
params = ReadXML("data/monocot.xml")

# 2. Generate anatomy
root_generator = RootAnatomy(params, verbatim=True)
anatomy = root_generator.generate()

# 3. Visualize
fig, ax = RootPlotter.plot(anatomy, color_by='type')
plt.savefig("root_cross_section.png", dpi=300)
plt.show()

# 4. Export to XML
AnatomyWriter.write_xml(anatomy, "output_anatomy.xml")

# 5. Statistics
print(f"Total cells: {len(anatomy.cells)}")
print(f"Tissue distribution:\n{anatomy.nodes['type'].value_counts()}")


