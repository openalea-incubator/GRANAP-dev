"""
P2.1: the pipeline's cells written into the MPG at each stage boundary (grouping, intercellular, aerenchyma,
fused_gaps, final). At every stage the section in the MPG is the live cell list (types, attributes, polygons, rows),
surviving cells keep their vertex, removed cells leave no vertex or property behind, the Layers are the current
tissue regions, and the final cells are the original pipeline's.
"""
import numpy as np
import pytest

from openalea.granap.root_class import RootAnatomy as Root
from openalea.granap.input_data import OrganInputData
from openalea.granap.metafspm.build import UID, tissue_regions
from openalea.granap.metafspm.cells import CellView, MPGCells, attributes
from openalea.granap.legacy.root_class import RootAnatomy

from test_cross_check import legacy, _check_cells, _same

STAGES = ["grouping", "intercellular", "aerenchyma", "fused_gaps", "final"]


def _aerenchymatous_root():
    data = OrganInputData.for_root()
    data.set_value("aerenchyma", "aerenchyma_proportion", 0.2)
    return data


def _record_stages(organ):
    """Hook: at each stage, compare the MPG with the live cells, and keep what the next stage is checked against."""
    records = []

    def hook(organ, stage):
        g, written = organ.mpg, organ._mpg_written
        cells = organ.all_cells.cells
        assert len(written["cells"]) == len(cells) == len(written["protoplasts"])
        for row, (cell, vid, protoplast) in enumerate(zip(cells, written["cells"], written["protoplasts"])):
            assert g.complex(protoplast) == vid and g.property("cell_row")[vid] == row
            assert g.property(UID)[vid] == getattr(cell, UID)
            for name, value in attributes(cell).items():
                assert _same(g.property("cell__" + name)[vid], value), f"{stage}: cell {row}.{name}"
        if stage != "grouping":                                # P5.1: the cells are views of the MPG from here on
            assert isinstance(organ.all_cells, MPGCells) and all(isinstance(c, CellView) for c in cells)
        regions = tissue_regions(cells)
        layer_of = [g.complex(v) for v in written["cells"]]
        assert len(set(layer_of)) == len(set(regions.tolist())) == len(written["layers"])
        same_region = regions[:, None] == regions[None, :] if len(cells) < 3000 else None
        if same_region is not None:
            assert (same_region == (np.asarray(layer_of)[:, None] == np.asarray(layer_of)[None, :])).all()
        alive = set(written["cells"]) | set(written["protoplasts"]) | set(written["layers"])
        for name, values in g.properties().items():
            if name.startswith("cell__"):
                assert set(values) <= alive, f"{stage}: properties of removed cells in {name}"
        records.append((stage, dict(zip([getattr(c, UID) for c in cells], written["cells"]))))

    organ.mpg_stage_hooks = (hook,)
    return records


@pytest.mark.slow
def test_each_stage_is_written_into_the_mpg_and_cells_keep_their_vertex():
    organ = Root(_aerenchymatous_root(), seed=0)
    records = _record_stages(organ)
    organ.generate_cells()
    assert [stage for stage, _ in records] == STAGES
    removed = 0
    for (_, before), (_, after) in zip(records[:-1], records[1:]):
        for uid, vid in after.items():
            if uid in before:
                assert before[uid] == vid                      # a surviving cell keeps its Cell vertex
        removed += len(set(before) - set(after))
    assert removed > 0                                         # the aerenchyma fused some cells away
    original = RootAnatomy(legacy(_aerenchymatous_root()), seed=0)
    original.generate_cells()
    _check_cells(original, organ)
    assert not any(hasattr(c, UID) for c in organ.all_cells.cells)


def test_the_stages_of_the_default_root_are_written():
    organ = Root(OrganInputData.for_root(), seed=0)
    records = _record_stages(organ)
    organ.generate_cells()
    assert [stage for stage, _ in records] == STAGES
