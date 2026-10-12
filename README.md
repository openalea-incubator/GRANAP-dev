# GRANAR package translated in Python

## Installation

In GRANAP root directory, install the environment with
```bash
mamba create -f ./conda/environment.yaml -y
mamba activate granap 
```

For source to be recognized in the environment, run the following at the root of the GRANAP directory
```bash
pip install -e . 
```

### metafspm (MPG data structure)

This version stores the anatomies in metafspm's multiscale plant graph (`openalea.granap.mpg`, see
`devplan_metafspm.md`). metafspm's latest version is not packaged yet: clone it next to GRANAP and install it from
source in the same environment.
```bash
cd ..                                       # the folder holding GRANAP
git clone https://github.com/openalea/metafspm.git metafspm
cd metafspm && git checkout data_structure_api   # the branch holding the anatomy scales, until it is merged
pip install .                               # or pip install -e . to follow its development
```
`conda/environment.yaml` already installs it this way (`-e ../../metafspm`), when it is cloned next to GRANAP.

Warning : 
> - Intercellular space are not well implemented
> - There is a problem for the polygon creation

## Code structure

For an interactive browsing, use the following link: TODO 

