"""
Capability tests of the MPG port: GRANAP anatomies populating a plant's root segments
through the structural component, the library per class, vessel ids, junctions between sections and the hydraulic
graph. Run from test/: pytest mpg_capabilities
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
