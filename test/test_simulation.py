"""The users' entry point (openalea.granap.simulation): its tables agree with the DataStructure the component wrote,
and its parameters act as GRANAP's rules say."""
import numpy as np
import pytest

from openalea.granap.simulation import parameters, simulate


@pytest.fixture(scope="module")
def default():
    return simulate()


def test_the_tables_are_the_components_cells(default):
    ds = default.data_structure
    cell = np.asarray(ds.get("kind")) == 0
    cells = default.cells()
    assert len(cells) == int(cell.sum())
    assert cells["area"].sum() == pytest.approx(float(np.asarray(ds.get("cell_area"))[cell].sum()), rel=1e-12)
    summary = default.summary
    assert summary["cells"].sum() == len(cells)
    assert summary["area_share"].sum() == pytest.approx(1.)
    assert summary["tissue"].iloc[0] == "epidermis"                  # from the outside in
    totals = default.totals
    assert totals["cells"] == len(cells) - int((cells["tissue"] == "air space").sum())
    assert totals["vessels"] == int(cells["tissue"].isin(("metaxylem", "protoxylem", "xylem")).sum())


def test_the_radius_is_the_layers_sum(default):
    """GRANAP's root radius: half the stele's diameter plus each layer's cells (_calculate_root_radius)."""
    p = parameters().set_index(["section", "field"])["value"]
    layers = sum(p[(name, "cell_diameter")] * p[(name, "n_layers")]
                 for name in ("pericycle", "endodermis", "cortex", "exodermis", "epidermis"))
    assert default.totals["radius"] == pytest.approx(p[("stele", "thickness")] / 2 + layers, rel=0.01)


def test_a_parameter_changes_the_section(default):
    thinner = simulate({"cortex": {"n_layers": 3}})
    assert thinner.totals["radius"] == pytest.approx(default.totals["radius"] - 2 * 0.04, rel=1e-3)
    assert thinner.parameters == {"cortex": {"n_layers": 3}}


def test_parameters_table():
    table = parameters({"cortex": {"n_layers": 3}})
    row = table[(table["section"] == "cortex") & (table["field"] == "n_layers")]
    assert row["value"].item() == 3 and row["unit"].item() == "count"
    assert table.loc[(table["section"] == "cortex") & (table["field"] == "cell_diameter"), "unit"].item() == "mm"
    assert table["main"].iloc[0] and not table["main"].iloc[-1]
    assert parameters(main_only=True)["main"].all()


def test_the_seed(default):
    assert simulate(seed=0).totals.equals(default.totals)
    assert simulate(seed=1).totals["cells"] != default.totals["cells"]


def test_draw(default):
    import matplotlib
    matplotlib.use("Agg")
    ax = default.draw()
    assert ax.collections and ax.get_legend() is not None
