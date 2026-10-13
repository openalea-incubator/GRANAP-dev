"""
C2 (devplan_metafspm.md, §43): the converted pipeline asks the organ for its topology (MPGBacked.section_topology)
instead of calling CellGenerator._build_topology, so nothing needs to replace the class method for the process. With
the class method made to fail, a converted organ still generates, exports and writes its XML, the same as GRANAP's.
"""
import pytest

from openalea.granap import mpg
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy

from test_cross_check import SEED, _check_cells, _check_network


@pytest.fixture
def class_method_forbidden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CellGenerator._build_topology called")
    monkeypatch.setattr(CellGenerator, "_build_topology", staticmethod(forbidden))


def test_a_converted_root_never_calls_the_class_method(class_method_forbidden, tmp_path):
    new = mpg.RootAnatomy(OrganInputData.for_root(), seed=SEED)
    new.exact_speedups = False                    # no patch at all: only the organ's own topology is asked
    new.generate_cells()
    new.write_to_xml(str(tmp_path / "new.xml"))
    new.export_to_adjencymatrix(bridges=False)
    assert getattr(new, "mpg_topology_reads", 0) >= 1


def test_and_gives_granaps_section_network_and_xml(tmp_path):
    new = mpg.RootAnatomy(OrganInputData.for_root(), seed=SEED)
    new.exact_speedups = False
    old = RootAnatomy(OrganInputData.for_root(), seed=SEED)
    for organ, name in ((old, "old.xml"), (new, "new.xml")):
        organ.generate_cells()
        organ.write_to_xml(str(tmp_path / name))
        organ.export_to_adjencymatrix()
    _check_cells(old, new)
    _check_network(old, new)
    assert (tmp_path / "old.xml").read_bytes() == (tmp_path / "new.xml").read_bytes()
