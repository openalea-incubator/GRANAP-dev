======
GRANAP
======

GRANAP generates the anatomical cross-sections of plant organs (roots, stems, leaves, needles): their tissues, cells,
walls and air spaces, and the network they form, on which MECHA computes the water flows. It is the Python
translation of GRANAR (Heymans et al., 2020), ready to be combined with other plant models (metafspm). The README
gives the installation.

User guide
==========

For newcomers to the model, to plant anatomy models or to Python modelling: generate a section, change its
parameters, explore them, use measured traits. Each tutorial is a notebook in ``example/tutorials/user/``, shown here
with its results.

.. toctree::
    :maxdepth: 1

    notebooks/user/01_first_run
    notebooks/user/02_change_a_parameter
    notebooks/user/03_combine_parameters
    notebooks/user/04_sensitivity_analysis
    notebooks/user/05_experimental_data
    parameters

Developer guide
===============

For those who read, check, change or couple the model: how it is written as a metafspm component
(``GranapAnatomy``: each segment of a plant gets its anatomy, from a library of templates per class, and follows the
plant's growth; ``RootAxis`` and ``Needle`` are the plant structures of its examples), how it was ported and
validated, its equations and its performance. Best read after the first user tutorial.

.. toctree::
    :maxdepth: 1

    notebooks/developer/00_overview
    notebooks/developer/01_cross_sections_in_metafspm
    notebooks/developer/02_extend_the_component
    components
    equations
    legacy
    performance
