"""Tests for the needle transfusion-tissue network bridge.

See ``example/needle/Transfusion_network.md`` for the physiological
requirements and ``example/needle/transfusion_network.py`` for the visual
deliverable. All tests use the measured Pinus pinaster needle
(``example/needle/pinus_pinaster.py::build_pinaster``, ``pack_circles=True``)
at a fixed ``seed=0`` so transfusion parenchyma/tracheid actually get
differentiated cell types (the default needle preset does not pack circles
and never produces "transfusion parenchyma"/"transfusion tracheid" — only a
single undifferentiated "transfusion" ring type).
"""

import os
import sys

import networkx as nx
import pytest

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "example", "needle"))

from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.anatomy_writer import NetworkExporter

from pinus_pinaster import build_pinaster

SEED = 0

SYMPLASTIC_CONTINUITY_TYPES = {
    "transfusion parenchyma", "parenchyma", "Strasburger cell", "endodermis", "phloem",
}


@pytest.fixture(scope="module")
def needle():
    n = NeedleAnatomy(build_pinaster(), seed=SEED)
    n.export_to_adjencymatrix()
    return n


def _cell_nodes_by_type(graph, cell_type):
    return [nd for nd, d in graph.nodes(data=True) if d.get("cell_type") == cell_type]


def test_no_symplastic_tracheid(needle):
    """No plasmodesmata edge is incident to any transfusion-tracheid node."""
    g = needle.graph
    tracheid_nodes = _cell_nodes_by_type(g, "transfusion tracheid")
    assert tracheid_nodes, "expected at least one transfusion tracheid cell"

    bad_edges = [
        (nd, nb) for nd in tracheid_nodes for nb in g.neighbors(nd)
        if g.edges[nd, nb].get("path") == "plasmodesmata"
    ]
    assert bad_edges == [], f"tracheid nodes must never carry a plasmodesmata edge: {bad_edges}"


def test_tracheids_keep_apoplast(needle):
    """Every tracheid node still has at least one membrane edge."""
    g = needle.graph
    tracheid_nodes = _cell_nodes_by_type(g, "transfusion tracheid")
    assert tracheid_nodes

    for nd in tracheid_nodes:
        has_membrane = any(g.edges[nd, nb].get("path") == "membrane" for nb in g.neighbors(nd))
        assert has_membrane, f"tracheid node {nd} lost its apoplastic (membrane) connection"


def test_symplastic_continuity(needle):
    """Every transfusion parenchyma cell reaches, via plasmodesmata edges only,
    at least one endodermis cell AND at least one Strasburger/parenchyma/phloem cell.

    This is the test that fails before the Phase-9 bridge (stage C) and
    passes after it: stripping tracheids of their symplastic edges alone
    (stage B) would otherwise leave many parenchyma ellipses as isolated
    symplastic islands, embedded individually in the tracheid matrix.
    """
    g = needle.graph
    sym_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("path") == "plasmodesmata"]
    sym_graph = nx.Graph()
    sym_graph.add_edges_from(sym_edges)

    parenchyma_nodes = _cell_nodes_by_type(g, "transfusion parenchyma")
    assert parenchyma_nodes, "expected at least one transfusion parenchyma cell"

    endodermis_nodes = set(_cell_nodes_by_type(g, "endodermis"))
    other_nodes = set(
        n for t in ("Strasburger cell", "parenchyma", "phloem") for n in _cell_nodes_by_type(g, t)
    )
    assert endodermis_nodes and other_nodes

    unreached_endodermis = []
    unreached_other = []
    for nd in parenchyma_nodes:
        if nd not in sym_graph:
            unreached_endodermis.append(nd)
            unreached_other.append(nd)
            continue
        reachable = nx.node_connected_component(sym_graph, nd)
        if not (reachable & endodermis_nodes):
            unreached_endodermis.append(nd)
        if not (reachable & other_nodes):
            unreached_other.append(nd)

    assert unreached_endodermis == [], (
        f"{len(unreached_endodermis)}/{len(parenchyma_nodes)} transfusion parenchyma "
        "cells cannot reach any endodermis cell via plasmodesmata"
    )
    assert unreached_other == [], (
        f"{len(unreached_other)}/{len(parenchyma_nodes)} transfusion parenchyma "
        "cells cannot reach any Strasburger cell/parenchyma/phloem cell via plasmodesmata"
    )


def test_cgroups(needle):
    """cgroups: parenchyma nodes = 17, tracheid nodes = 18, endodermis = 3, xylem = 13."""
    g = needle.graph
    expected = {
        "transfusion parenchyma": 17,
        "transfusion tracheid": 18,
        "endodermis": 3,
        "xylem": 13,
    }
    for cell_type, cgroup in expected.items():
        nodes = _cell_nodes_by_type(g, cell_type)
        assert nodes, f"expected at least one {cell_type} cell"
        bad = [nd for nd in nodes if g.nodes[nd].get("cgroup") != cgroup]
        assert bad == [], f"{cell_type} node(s) {bad} have the wrong cgroup (want {cgroup})"


def test_mecha_index_contract(needle):
    """n_total == number of graph nodes; wall ids < n_walls; cell ids >=
    n_walls + n_junctions; every edge path is one MECHA actually understands
    (or, for wall_air/air_link, one that is at least deliberately excluded)."""
    g = needle.graph
    n = needle

    assert n.n_total == g.number_of_nodes()

    wall_junction_nodes = [nd for nd, d in g.nodes(data=True) if d.get("type") == "apo"]
    cell_nodes = [nd for nd, d in g.nodes(data=True) if d.get("type") == "cell"]

    assert wall_junction_nodes and cell_nodes
    assert all(nd < n.n_walls + n.n_junctions for nd in wall_junction_nodes)
    assert all(nd >= n.n_walls + n.n_junctions for nd in cell_nodes)
    assert all(nd < n.n_total for nd in cell_nodes)

    allowed_paths = {"wall", "membrane", "plasmodesmata", "wall_air", "air_link"}
    seen_paths = {d.get("path") for _, _, d in g.edges(data=True)}
    assert seen_paths <= allowed_paths, f"unexpected edge path(s): {seen_paths - allowed_paths}"


def test_bridges_inert_without_export():
    """Cell census from generate_cells() is unchanged by a later
    export_to_adjencymatrix() call -- bridge rows are flagged ``bridge=True``
    precisely so a census can filter them back out, and a
    generate_cells()-only run (e.g. the golden regression suite, which never
    calls export_to_adjencymatrix()) never sees them at all."""
    n = NeedleAnatomy(build_pinaster(), seed=SEED)
    gdf_before = n.generate_cells()
    census_before = gdf_before["type"].value_counts().to_dict()
    n_all_cells_before = len(n.all_cells.cells)

    n.export_to_adjencymatrix()

    # A fresh NeedleAnatomy that never exports must be untouched -- the
    # bridge-building code lives entirely inside NetworkExporter.export and
    # is never reached by generate_cells() alone.
    fresh = NeedleAnatomy(build_pinaster(), seed=SEED)
    fresh_census = fresh.generate_cells()["type"].value_counts().to_dict()
    assert fresh_census == census_before
    assert len(fresh.all_cells.cells) == n_all_cells_before

    # The same organ, post-export, must carry visibly-flagged bridge rows
    # that a census can filter back out to recover the original counts.
    gdf_after = n.generate_cells()
    assert "bridge" in gdf_after.columns
    assert gdf_after["bridge"].sum() > 0
    non_bridge_census = gdf_after[~gdf_after["bridge"].astype(bool)]["type"].value_counts().to_dict()
    assert non_bridge_census == census_before

    # Calling export a second time (a fresh NetworkExporter on the same,
    # already-exported organ) must not append the bridge cells again --
    # and, critically, must not corrupt the graph either. A prior bug fed
    # the previous call's own bridge rows back into Phase 9 as fresh
    # source/target/blocker candidates (since ``generate_cells()`` returns
    # the cached, already-augmented frame), silently bridging the bridges:
    # n_walls grew past the already-finalised junction-node id range
    # (corrupting those nodes' attributes) while n_cells grew past what
    # ``all_cells`` actually has entries for -- exactly the MECHA
    # positional-walk breakage stage D exists to prevent.
    n_walls_before = n.n_walls
    n_junctions_before = n.n_junctions
    n_cells_before = n.n_cells
    n_nodes_before = n.graph.number_of_nodes()
    n_all_cells_after_first_export = len(n.all_cells.cells)
    bridge_positions_before = sorted(
        (nd, d["position"]) for nd, d in n.graph.nodes(data=True) if d.get("bridge")
    )

    NetworkExporter(n).export(n)

    assert len(n.all_cells.cells) == n_all_cells_after_first_export
    assert n.n_walls == n_walls_before
    assert n.n_junctions == n_junctions_before
    assert n.n_cells == n_cells_before
    assert n.graph.number_of_nodes() == n_nodes_before

    # Not just equal counts -- the repeat export must reproduce the exact
    # same bridge node ids AND the exact same positions (tracheid
    # centroids), i.e. Phase 9a re-derives an identical set of bridges
    # rather than a shifted one.
    bridge_positions_after = sorted(
        (nd, d["position"]) for nd, d in n.graph.nodes(data=True) if d.get("bridge")
    )
    assert bridge_positions_after == bridge_positions_before


def test_bridge_nodes_have_no_polygon():
    """The bridge is a NODE, not a cell with a polygon (per the reference
    diagram, ``example/needle/Transfusion_tissue_network.png``): every
    bridge row's geometry must be None, and each still carries a position
    (the tracheid's own centroid) and a nominal area for MECHA's
    capacitance/volume terms."""
    n = NeedleAnatomy(build_pinaster(), seed=SEED)
    n.export_to_adjencymatrix()
    gdf = n.generate_cells()
    bridges = gdf[gdf.get("bridge", False).astype(bool)]
    assert len(bridges) > 0, "expected at least one bridge node"

    assert bridges["geometry"].isna().all() or (bridges["geometry"] == None).all()  # noqa: E711

    g = n.graph
    bridge_nodes = [nd for nd, d in g.nodes(data=True) if d.get("bridge")]
    assert len(bridge_nodes) == len(bridges)
    for nd in bridge_nodes:
        d = g.nodes[nd]
        assert d["position"] is not None
        assert d["area"] is not None and d["area"] > 0


def test_bridges_never_add_walls_or_junctions():
    """Bridges are reused EXISTING wall nodes only -- no new wall, no new
    junction. ``n_walls``/``n_junctions`` with bridges enabled must equal
    the pre-bridge topology (``bridges=False``) exactly. ``NeedleAnatomy``
    (via ``Organ``) *is* an ``AbstractNetwork``, so both are built directly
    on organ instances."""
    n_no_bridges = NeedleAnatomy(build_pinaster(), seed=SEED)
    n_no_bridges.export_to_adjencymatrix(bridges=False)

    n_with_bridges = NeedleAnatomy(build_pinaster(), seed=SEED)
    n_with_bridges.export_to_adjencymatrix()

    assert n_no_bridges.n_walls == n_with_bridges.n_walls
    assert n_no_bridges.n_junctions == n_with_bridges.n_junctions
    # bridges only ever add cell nodes
    assert n_with_bridges.n_cells > n_no_bridges.n_cells


def test_bridge_edges_are_only_membrane_and_plasmodesmata():
    """Every edge touching a bridge node has path in {membrane,
    plasmodesmata} -- never ``wall`` (bridges never touch a junction) and
    never anything MECHA's solver would silently drop."""
    n = NeedleAnatomy(build_pinaster(), seed=SEED)
    n.export_to_adjencymatrix()
    g = n.graph

    bridge_nodes = [nd for nd, d in g.nodes(data=True) if d.get("bridge")]
    assert bridge_nodes

    for nd in bridge_nodes:
        paths = {g.edges[nd, nb].get("path") for nb in g.neighbors(nd)}
        assert paths, f"bridge node {nd} has no edges at all"
        assert paths <= {"membrane", "plasmodesmata"}, (
            f"bridge node {nd} has an unexpected edge path {paths - {'membrane', 'plasmodesmata'}}"
        )


def test_bridge_nodes_shared_across_paths():
    """One virtual node is shared per tracheid across every path that
    traverses it (not one-per-pair, not one-per-path). The
    ``_bridge_report`` counts must therefore satisfy: number of graph
    bridge nodes == the shared-per-tracheid count, and that count is <=
    the one-per-path (no sharing) count -- with '<' whenever any tracheid
    actually serves more than one path (true for this measured needle)."""
    n = NeedleAnatomy(build_pinaster(), seed=SEED)
    n.export_to_adjencymatrix()
    g = n.graph

    report = getattr(n, "_bridge_report", None)
    assert report is not None, "expected NetworkExporter.export to record a bridge report"

    bridge_nodes = [nd for nd, d in g.nodes(data=True) if d.get("bridge")]
    assert len(bridge_nodes) == report["n_virtual_shared"]
    assert report["n_virtual_shared"] <= report["n_virtual_per_path"]
    assert report["n_virtual_shared"] < report["n_virtual_per_path"], (
        "expected at least some tracheid sharing across multiple paths in "
        "this measured needle -- if this now legitimately fails because the "
        "geometry changed, the '<=' assertion above is the one that matters"
    )
