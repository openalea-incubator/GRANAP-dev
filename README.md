# GRANAR package translated in Python

GRANAP generates the anatomical cross-sections of plant organs: roots, stems, leaves and needles. From a description
of the tissues (the number of layers of each, the size of their cells, the diameter of the stele, the number and size
of the xylem vessels), it draws every cell, wall and air space of the section, and the network they form, on which
[MECHA](https://github.com/openalea-incubator/MECHA-dev) computes the water flows. It is the Python translation of
GRANAR, ready to be combined with other plant models ([metafspm](https://github.com/openalea/metafspm)).

Heymans A., Couvreur V., LaRue T., Paez-Garcia A., Lobet G. (2020). GRANAR, a computational tool to better understand
the functional importance of monocotyledon root anatomy. *Plant Physiology* 182, 707–720.

## Installation

GRANAP and metafspm (and MECHA, which uses GRANAP's anatomies) are not installable from the `openalea3` channel yet
in the versions used here: clone the repositories on their working branches, next to each other, and install them
from source (`pip install -e`) in one environment.

```bash
mkdir fspm && cd fspm                       # any folder: the repositories side by side
git clone -b data_structure_api https://github.com/openalea/metafspm.git metafspm
git clone -b to_metafspm https://github.com/openalea-incubator/GRANAP-dev.git GRANAP

mamba env create -f GRANAP/conda/environment.yaml        # GRANAP's dependencies and metafspm's
mamba activate granap
pip install -e metafspm
pip install -e GRANAP
```

Keep the folder names above: `GRANAP/conda/environment.yaml` expects `metafspm` next to GRANAP. To work with MECHA
too, clone it alongside and use its environment instead, which covers the three
(`git clone -b to_metafspm https://github.com/openalea-incubator/MECHA-dev.git MECHA-dev`, then MECHA-dev's README).

## Getting started

Generate a root cross-section with the default parameters (a monocot root), and look at it:

```python
from openalea.granap.simulation import simulate

section = simulate()
print(section.summary)                         # per tissue: cells, cell diameter, area, share
section.draw()                                 # the section, coloured by tissue
section = simulate({"cortex": {"n_layers": 3}})   # with three cortex layers instead of five
```

### User tutorials

Step by step, for newcomers to the model, to plant anatomy models or to Python modelling, in
`example/tutorials/user/`:

1. [Your first root cross-section](example/tutorials/user/01_first_run.ipynb): generate a section, read its
   tissues, draw it;
2. [Change a parameter](example/tutorials/user/02_change_a_parameter.ipynb): the parameters, chance, a dicot root;
3. [Combine parameters](example/tutorials/user/03_combine_parameters.ipynb): two parameters together;
4. [Sensitivity analysis](example/tutorials/user/04_sensitivity_analysis.ipynb): which parameters matter most;
5. [Experimental data](example/tutorials/user/05_experimental_data.ipynb): a section from measured traits,
   observations, fitting.

Then MECHA's tutorials compute the water flows on these sections:
[MECHA-dev/example/tutorials](https://github.com/openalea-incubator/MECHA-dev/tree/to_metafspm/example/tutorials).
Run them locally with `jupyter lab example/tutorials/`.

## For developers

### metafspm (MPG data structure)

This version generates every cross-section into metafspm's multiscale plant graph (MPG): its cells, walls, junctions
and hydraulic network are the MPG's vertices (`openalea.granap.metafspm`; the documentation in `doc/`: its components, equations, the legacy compared, performance). metafspm is
therefore required, on its `data_structure_api` branch (the anatomy scales), until it is merged.

The original source is kept, frozen, as `openalea.granap.legacy` (the reference the port is compared with:
`test/comparison/`, `example/comparison/`). New work goes into the main package.

Warning : 
> - Intercellular space are not well implemented
> - There is a problem for the polygon creation

### Developer tutorials

In `example/tutorials/developer/`, after the first user tutorial:

0. [Overview](example/tutorials/developer/00_overview.ipynb): the modules, the original code, the validation;
1. [Cross sections as a metafspm component](example/tutorials/developer/01_cross_sections_in_metafspm.ipynb):
   - GRANAP's anatomy component in a plant model, its outputs in the DataStructure;
   - the parametrization by the plants' scenarios (`root_anatomy`, `stem_anatomy`, `leaf_anatomy`);
   - classes within a plant;
   - how the sections are joined axially (the multiscale traversal);
   - growth;
2. [Extend the component](example/tutorials/developer/02_extend_the_component.ipynb): a new variable and process.

Examples with the component: [`example/mpg/`](example/mpg/) (a root, a root system, a seedling); GRANAP's organs on
their own: [`example/root/`](example/root/), [`example/stem/`](example/stem/), [`example/leaf/`](example/leaf/),
[`example/needle/`](example/needle/); the port against the original: [`example/comparison/`](example/comparison/).
Docs (`doc/`, Sphinx): a user guide (the user tutorials, the parameters in plain words) and a developer guide (the
developer tutorials, the component's declarations, the equations, the legacy compared, the performance). Tests:
`pytest test`.

## Code structure

For an interactive browsing, use the following link: TODO 

