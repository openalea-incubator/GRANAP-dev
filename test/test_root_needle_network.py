"""Tests for organ construction from XML / param-list input.

Visual network gallery lives in ``example/root_needle_network_gallery.py``.
"""

import os
import sys

sys.path.append(os.path.abspath('..'))

from openalea.granap.input_data import OrganInputData
from openalea.granap.organ_class import Organ


def test_root_from_xml_has_expected_vasculature():
    """A monocot root loaded from XML has 3 metaxylem cells and no air spaces."""
    input_data_root = OrganInputData.from_xml("inputs/root_monocot_simpl.xml")
    root_sim = Organ.create_from_input(input_data_root)
    root_sim.export_to_adjencymatrix()  # builds graph + ensures cells

    n_xylem = len([c for c in root_sim.all_cells.cells if c.type == "metaxylem"])
    n_airspace = len([c for c in root_sim.all_cells.cells if c.type == "air space"])
    assert n_xylem == 3
    assert n_airspace == 0


def test_needle_from_param_list_builds():
    """A needle built from a raw param list constructs and produces cells."""
    from inputs.param_pinus import params_pinaster

    input_data_needle = OrganInputData.from_dict_list(params_pinaster)
    needle_sim = Organ.create_from_input(input_data_needle)
    needle_sim.export_to_adjencymatrix()
    assert len(needle_sim.all_cells.cells) > 0


if __name__ == "__main__":
    test_root_from_xml_has_expected_vasculature()
    test_needle_from_param_list_builds()
    print("ROOT/NEEDLE NETWORK INPUT TESTS PASSED")
