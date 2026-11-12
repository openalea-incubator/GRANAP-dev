"""The users' entry point (openalea.granap.simulation): its tables agree with the DataStructure the component wrote,
and its parameters act as GRANAP's rules say."""
import numpy as np
import pytest

from openalea.granap.simulation import parameters, simulate


@pytest.fixture(scope="module")
def default():
    return simulate()


def test_the_tables_are_the_components_cells(default):
    data_structure = default.data_structure
    cell = np.asarray(data_structure.get("compartment_type")) == 0
    cells = default.cells()
    assert len(cells) == int(cell.sum())
    assert cells["area"].sum() == pytest.approx(float(np.asarray(data_structure.get("cell_area"))[cell].sum()), rel=1e-12)
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


def test_the_run_configuration_is_the_scenes():
    """compute= (metafspm's Compute) is the Scene's run configuration."""
    from openalea.metafspm.compute import Compute
    compute = Compute(threads=1, memory="keep")
    assert simulate(compute=compute).scene.compute is compute


def _every_value(scene) -> dict:
    """Every numeric variable of every population, by population and name."""
    found = {}
    for population in scene.populations:
        data_structure = population.data_structure
        for name in data_structure.available_vars():
            values = np.asarray(data_structure.get(name))
            if values.dtype != object:
                found[f"{population.name} {name}"] = values.copy()
    return found


def _two_populations_at(processes, build, steps) -> dict:
    """*build*(Compute)'s scene run *steps* steps: every value of its populations."""
    from openalea.metafspm.compute import Compute
    scene = build(Compute(threads=1, processes=processes))
    try:
        for _ in range(steps):
            scene.run()
        assert type(scene.runner).__name__ == ("ProcessRunner" if processes > 1 else "SequentialRunner")
        return _every_value(scene)
    finally:
        scene.finish()


def _assert_same_bits(one: dict, two: dict) -> None:
    assert one.keys() == two.keys() and one
    for key, values in one.items():
        np.testing.assert_array_equal(two[key], values, err_msg=key)


def test_two_populations_in_two_processes_give_the_bits_of_one():
    """Two sections (seeds 0 and 1) in two populations (planting(population=)): Compute(processes=2) runs each in its
    worker process, with the values of one process."""
    from openalea.metafspm.scene.population import Planting, planting
    from openalea.metafspm.scene.scene import Scene

    from openalea.granap.simulation import _plant_model

    def build(compute):
        table = planting([Planting(), Planting(x=0.1)], _plant_model(),
                         plant_scenarios=[{"root_anatomy": {"preset": "for_root", "seed": seed}} for seed in (0, 1)],
                         population=["first", "second"])
        return Scene(table, compute=compute)
    _assert_same_bits(_two_populations_at(1, build, 2), _two_populations_at(2, build, 2))
