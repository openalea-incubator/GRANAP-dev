======
GRANAP
======

GRANAP generates plant organs' anatomical cross-sections (roots, stems, leaves, needles: their tissues, cells,
walls and intercellular spaces) and their hydraulic network, the Python translation of GRANAR. This version builds
them in metafspm's multiscale plant graph (MPG): ``GranapAnatomy`` (a StructuralComponent) gives each segment of a
plant its anatomy, from a library of templates per class, and follows the plant's growth; ``RootAxis`` and
``Needle`` are the plant structures of its examples. MECHA computes the water flows on these anatomies.

The tutorial is ``example/tutorials/tutorial_metafspm_01_cross_sections.ipynb``; the README gives the installation.

.. toctree::
    :maxdepth: 1

    components
    equations
    legacy
    performance
