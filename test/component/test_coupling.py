"""
GRANAP's anatomy component coupled the metafspm way, as MECHA will be: a component
declares GRANAP's outputs as inputs (by="GranapAnatomy") and reads them on every node and edge of a root system, the
axial junctions between sections included (their geometry, Q-C1a). The 3D quantities follow the segments' lengths.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable
from openalea.metafspm.data_structure.configs import ScalesConfig

from openalea.granap.metafspm.columns import JUNCTION_PATH, PATH_CODES
from openalea.granap.metafspm.component import GranapAnatomy
from openalea.granap.metafspm.library import AnatomyLibrary

from roots import SEGMENT, SmallRootSystem, initiated, planting

S = ScalesConfig()
DOC = dict(unit="", unit_comment="", description="", min_value=-1e12, max_value=1e12, value_comment="",
           references="", DOI=[])


@dataclass
class HydraulicsInputs(FunctionalComponent):
    """What a MECHA-like model reads from GRANAP: the network's codes, geometry and 3D quantities."""
    network_path: float = input_variable(**DOC, by="GranapAnatomy", scale=S.Connection, location="edge")
    edge_length: float = input_variable(**DOC, by="GranapAnatomy", scale=S.Connection, location="edge")
    membrane_surface: float = input_variable(**DOC, by="GranapAnatomy", scale=S.Connection, location="edge")
    cgroup: float = input_variable(**DOC, by="GranapAnatomy", scale=S.Compartment, location="node")
    wall_surface: float = input_variable(**DOC, by="GranapAnatomy", scale=S.Compartment, location="node")
    cell_volume: float = input_variable(**DOC, by="GranapAnatomy", scale=S.Compartment, location="node")


class Anatomy(GranapAnatomy):
    library = AnatomyLibrary()


@pytest.fixture(scope="module")
def coupled():
    population = initiated(planting(), SmallRootSystem, Anatomy)
    mtg = population.data_structure.structure
    data_structure = population.data_structure
    anatomy = Anatomy(data_structure=data_structure)
    anatomy()                                                    # one step: the 3D quantities computed
    return mtg, data_structure, anatomy, HydraulicsInputs(data_structure=data_structure)


def test_every_edge_has_its_path_and_length_the_junctions_included(coupled):
    mtg, data_structure, anatomy, mecha = coupled
    path, length = data_structure.get("network_path"), data_structure.get("edge_length")
    assert not np.isnan(path).any() and not np.isnan(length).any()
    junctions = path == JUNCTION_PATH
    assert junctions.sum() == len(mtg.junction_vids()) > 0
    np.testing.assert_allclose(length[junctions], SEGMENT * 1000.)  # between successive 5 mm sections' centres


def test_the_3d_quantities_are_the_2d_ones_times_the_segment_length(coupled):
    mtg, data_structure, anatomy, mecha = coupled
    np.testing.assert_allclose(data_structure.get("wall_surface"), data_structure.get("wall_length") * SEGMENT * 1000.)
    np.testing.assert_allclose(data_structure.get("cell_volume"), data_structure.get("cell_area") * SEGMENT * 1000.)
    path = data_structure.get("network_path")
    membranes = (path == PATH_CODES["membrane"]) | (path == PATH_CODES["wall_air"])
    surface = data_structure.get("membrane_surface")
    np.testing.assert_allclose(surface[membranes], data_structure.get("edge_length")[membranes] * SEGMENT * 1000.)
    assert (surface[~membranes] == 0.).all()
    owner = data_structure.entity_ids("SubOrgan")[data_structure.owner("SubOrgan")]
    total = np.bincount(np.searchsorted(data_structure.entity_ids("SubOrgan"), owner), weights=data_structure.get("wall_surface"))
    assert np.isfinite(total).all() and (total > 0).all()        # extensive: the segments' sums are defined


def test_the_coupled_model_reads_them(coupled):
    mtg, data_structure, anatomy, mecha = coupled
    for name in ("network_path", "edge_length", "membrane_surface"):
        assert mecha.data_structure.get(name).shape == (data_structure.n_edges(),)
    for name in ("cgroup", "wall_surface", "cell_volume"):
        assert mecha.data_structure.get(name).shape == (data_structure.n_nodes(),)


def test_the_3d_quantities_follow_elongation(coupled):
    mtg, data_structure, anatomy, mecha = coupled
    length = data_structure.get("length").copy()
    data_structure.set("length", length * 2.)                               # as the architecture model would, on every node
    anatomy()
    np.testing.assert_allclose(data_structure.get("wall_surface"), data_structure.get("wall_length") * SEGMENT * 2000.)
    path = data_structure.get("network_path")
    np.testing.assert_allclose(data_structure.get("edge_length")[path == JUNCTION_PATH], SEGMENT * 2000.)
