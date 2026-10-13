"""
GRANAP's objects rebuilt from the MPG (devplan_metafspm.md, P1 and Q6): the cells (Cell objects with every attribute
they had), the cell GeoDataFrame and the MECHA network (networkx graph, counts and wall-to-cells map), so that code
written against GRANAP's organ classes reads the MPG.
"""
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from openalea.granap.cell_class import Cell
from openalea.granap.mpg import columns


def _by_row(g, vids, prefix):
    """{row: {attribute: value}} of the vertices *vids*, from their properties named prefix + attribute."""
    props = {name[len(prefix):]: values for name, values in g.properties().items() if name.startswith(prefix)}
    rows = g.property("cell_row")
    out = {}
    for v in vids:
        out[int(rows[v])] = {name: values[v] for name, values in props.items() if v in values}
    return out


def cells_from_mpg(g, written) -> list:
    """The GRANAP Cells of the section, in their order, each with exactly the attributes it was written with."""
    attributes = _by_row(g, written["cells"], "cell__")
    cells = []
    for row in range(len(attributes)):
        cell = Cell.__new__(Cell)
        cell.__dict__.update(attributes[row])
        cells.append(cell)
    return cells


def gdf_from_cells(cells, n_real=None) -> gpd.GeoDataFrame:
    """
    The cell GeoDataFrame as Organ.generate_cells builds it (one row per Cell.cell_to_dict, plus geometry); with
    *n_real*, the cells after it are the exporter's bridge cells (a bridge column, their ids as row labels).
    """
    real = cells if n_real is None else cells[:n_real]
    dicts = [dict(c.cell_to_dict(), geometry=c.polygon) for c in real]
    gdf = gpd.GeoDataFrame(dicts)
    if n_real is None:
        return gdf
    gdf["bridge"] = False
    bridges = cells[n_real:]
    if bridges:
        extra = gpd.GeoDataFrame([dict(c.cell_to_dict(), geometry=c.polygon, bridge=True) for c in bridges],
                                 index=[c.id_cell for c in bridges])
        gdf = pd.concat([gdf, extra], ignore_index=False)
    return gdf


def network_from_mpg(g, written) -> nx.Graph:
    """The exporter's networkx graph: its nodes (MECHA indices, attributes) and edges (attributes), rebuilt from the
    MPG's network columns (columns.py)."""
    mecha, kinds, tissue = g.property("mecha_index"), g.property("kind"), g.property("tissue")
    graph = nx.Graph()
    compartments = sorted(mecha, key=lambda v: mecha[v])
    for v, attrs in zip(compartments, columns.node_attributes(g, compartments, [int(kinds[v]) for v in compartments],
                                                             mecha, tissue)):
        graph.add_node(int(mecha[v]), **attrs)
    u_of, v_of = g.property("edge_u"), g.property("edge_v")
    connections = written["connections"]
    for e, attrs in zip(connections, columns.edge_attributes(g, connections)):
        graph.add_edge(int(u_of[e]), int(v_of[e]), **attrs)
    return graph
