# GRANAR package translated in Python

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

### metafspm (MPG data structure)

This version generates every cross-section into metafspm's multiscale plant graph (MPG): its cells, walls, junctions
and hydraulic network are the MPG's vertices (`openalea.granap.metafspm`, see `devplan_metafspm.md`). metafspm is
therefore required, on its `data_structure_api` branch (the anatomy scales), until it is merged.

The original source is kept, frozen, as `openalea.granap.legacy` (the reference the port is compared with:
`test/comparison/`, `example/comparison/`). New work goes into the main package.

Warning : 
> - Intercellular space are not well implemented
> - There is a problem for the polygon creation

## Tutorials

- [Cross sections as a metafspm component](example/tutorials/tutorial_metafspm_01_cross_sections.ipynb):
  - GRANAP's anatomy component in a plant model, its outputs in the DataStructure;
  - the parametrization by the plants' scenarios (`root_anatomy`, `stem_anatomy`, `leaf_anatomy`);
  - classes within a plant;
  - how the sections are joined axially (the multiscale traversal);
  - growth.

Then MECHA's tutorials, on these anatomies:
[MECHA-dev/example/tutorials](https://github.com/openalea-incubator/MECHA-dev/tree/to_metafspm/example/tutorials).
Run them locally with `jupyter lab example/tutorials/`. Examples with the component: [`example/mpg/`](example/mpg/)
(a root, a root system, a seedling); GRANAP's organs on their own: [`example/root/`](example/root/),
[`example/stem/`](example/stem/), [`example/leaf/`](example/leaf/), [`example/needle/`](example/needle/); the port
against the original: [`example/comparison/`](example/comparison/).

## Code structure

For an interactive browsing, use the following link: TODO 

