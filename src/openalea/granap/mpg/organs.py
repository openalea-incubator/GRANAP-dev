"""
GRANAP's organ classes backed by the MPG (devplan_metafspm.md, P1).

    from openalea.granap.mpg import RootAnatomy                 # same constructor and API as GRANAP's
    root = RootAnatomy(OrganInputData.for_root(), seed=0)
    root.generate_cells()            # the GRANAP pipeline, then the section written into root.mpg and read back
    root.export_to_adjencymatrix()   # the exporter, then its walls, junctions and edges written and read back
    root.mpg, root.segment           # the MPG and the section's SubOrgan

The pipeline itself is GRANAP's, unchanged; its cells are written into the MPG at each stage boundary (P2: the
Voronoi grouping, the intercellular spaces, the aerenchyma, the fused gaps, the final cells), each cell keeping its
vertex through the stages. What the rest of GRANAP reads afterwards (all_cells, the layers' cells, the cell
GeoDataFrame, the network graph, its counts and wall map) is rebuilt from the MPG, so every result is the MPG's. The
MPG becomes the storage itself in P5.
"""
from types import SimpleNamespace

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix

from openalea.granap.cellset_organ import CellSetOrgan as _CellSetOrgan
from openalea.granap.leaf_class import (DicotLeafAnatomy as _DicotLeaf, LeafAnatomy as _Leaf,
                                        MonocotLeafAnatomy as _MonocotLeaf)
from openalea.granap.needle_class import NeedleAnatomy as _Needle
from openalea.granap.organ_class import Organ as _Organ
from openalea.granap.roi_organ import RoiOrgan as _RoiOrgan
from openalea.granap.root_class import RootAnatomy as _Root
from openalea.granap.root_dicot_class import DicotRootAnatomy as _DicotRoot
from openalea.granap.root_monocot_class import MonocotRootAnatomy as _MonocotRoot
from openalea.granap.stem_class import StemAnatomy as _Stem
from openalea.granap.stem_dicot_class import DicotStemAnatomy as _DicotStem
from openalea.granap.stem_dicot_continuous_class import ContinuousDicotStemAnatomy as _ContinuousStem
from openalea.granap.stem_monocot_class import MonocotStemAnatomy as _MonocotStem

from openalea.granap.mpg.build import single_plant_mpg, write_cells, write_network
from openalea.granap.cell_manager import CellManager
from openalea.granap.mpg.cells import CellStore, MPGCells
from openalea.granap.mpg.network import build_network
from openalea.granap.mpg.pipeline.aerenchyma import add_aerenchyma_to_proportion
from openalea.granap.mpg.pipeline.generate import generate_cells
from openalea.granap.mpg.pipeline.merge import merge_intercellular_aerenchyma
from openalea.granap.mpg.speedups import exact_speedups
from openalea.granap.mpg.topology import store_topology, stored_topology
from openalea.granap.mpg.views import cells_from_mpg, gdf_from_cells, network_from_mpg


class MPGBacked:
    """The organ's cells and network written into an MPG and read back from it (mixin, before the organ class)."""

    mpg = None
    segment = None
    _mpg_written = None
    mpg_stage_hooks = ()             # callables (organ, stage) after each stage write (tests, diagnostics)
    _converted_generation = False    # N2: within the converted generate_cells, which writes the stages itself
    exact_speedups = True            # P4: GRANAP's pipeline with exact speed-ups (identical outputs); False: as is
    converted_pipeline = True        # P5: the converted copies of GRANAP's modules (mpg/pipeline/); False: GRANAP's
    native_network = True            # N1: the stored network built in the MPG; False: the exporter, copied into it

    def generate_cells(self):
        """GRANAP's: the cells generated (once), and their GeoDataFrame, a view of the MPG built when first read."""
        self.generate_section()
        return self._cells_gdf

    def generate_section(self) -> None:
        """The section generated once into the MPG: the pipeline stage by stage, the final cells, the stored network."""
        if self.__dict__.get("_mpg_generated"):
            return
        converted = self.converted_pipeline and _inherited(type(self), "generate_cells") is _Organ.generate_cells
        self._converted_generation = converted
        try:
            if self.exact_speedups:
                with exact_speedups(self):
                    self._run_pipeline(converted)
            else:
                self._run_pipeline(converted)
        finally:
            self._converted_generation = False
        self._mpg_stage("final")
        self._read_cells_back()
        self.__dict__["_mpg_generated"] = True                  # the cells are final; the network follows them
        self._store_network()

    def _run_pipeline(self, converted: bool) -> None:
        if converted:
            generate_cells(self)                               # N2: GRANAP's, the stages explicit, no GeoDataFrame
        else:
            super().generate_cells()

    @property
    def _cells_gdf(self):
        """The cell GeoDataFrame: once generated, built from the cells at its first read (then cached)."""
        gdf = self.__dict__.get("_gdf")
        if gdf is None and self.__dict__.get("_mpg_generated"):
            gdf = self.__dict__["_gdf"] = gdf_from_cells(self.all_cells.cells, self.__dict__.get("_gdf_n_real"))
        return gdf

    @_cells_gdf.setter
    def _cells_gdf(self, value):
        self.__dict__["_gdf"] = value                          # GRANAP's own writes (bridge rows, retagging)
        if value is None:                                      # Organ._invalidate_geometry: to be generated again
            self.__dict__["_mpg_generated"] = False

    # ── Stages (P2): the pipeline's cells written into the MPG at each boundary ──

    def _mpg_stage(self, stage: str) -> None:
        """Write the current cells into the section's MPG (created at the first stage) as the record of *stage*."""
        if self.mpg is None:
            self.mpg, self.segment = single_plant_mpg()
        self._mpg_written = write_cells(self.mpg, self.segment, self, self._mpg_written, stage)
        store = getattr(self, "_cell_store", None)
        if store is not None:                                    # views of removed cells are forgotten
            alive = self._mpg_written["cell_of_uid"]
            for uid in [u for u in store.views if u not in alive]:
                del store.views[uid]
        for hook in getattr(self, "mpg_stage_hooks", ()):
            hook(self, stage)

    def _grouped(self) -> None:
        """The Voronoi groups, simplified, nested cells removed: stage "grouping"; then the cells become views."""
        self._mpg_stage("grouping")
        if self.converted_pipeline:                    # P5.1: from here on, the cells are views of the MPG
            self._cell_store = CellStore(self.mpg, self.segment, self._mpg_written)
            self.all_cells = MPGCells(self._cell_store, self.all_cells.cells)

    def add_intercellular_spaces(self):
        if not self._converted_generation:             # GRANAP's generate_cells: the grouping ends here
            self._grouped()
        return super().add_intercellular_spaces()

    def _add_aerenchyma_to_proportion(self, *args, **kwargs):
        self._mpg_stage("intercellular")
        inherited = _inherited(type(self), "_add_aerenchyma_to_proportion")
        if self.converted_pipeline and inherited is _Organ.__dict__["_add_aerenchyma_to_proportion"]:
            result = add_aerenchyma_to_proportion(self, *args, **kwargs)        # P5.3: GRANAP's, on shallow copies
        else:
            result = super()._add_aerenchyma_to_proportion(*args, **kwargs)     # an organ's own (e.g. leaves)
        self._mpg_stage("aerenchyma")
        return result

    def merge_intercellular_aerenchyma(self):
        """B2a: the converted merge (simplification and carving in bulk) when the organ's is Organ's."""
        if self.converted_pipeline and _inherited(type(self), "merge_intercellular_aerenchyma") is \
                _Organ.__dict__["merge_intercellular_aerenchyma"]:
            return merge_intercellular_aerenchyma(self)
        return super().merge_intercellular_aerenchyma()

    def fuse_gaps(self, *args, **kwargs):
        fused = super().fuse_gaps(*args, **kwargs)
        if not self._converted_generation:             # the converted generation writes the stage itself
            self._mpg_stage("fused_gaps")
        return fused

    def _read_cells_back(self, n_real=None):
        """all_cells and the layers' cell lists from the MPG (the same cells, the same order); the GeoDataFrame
        follows them (with *n_real*, the cells after it are the exporter's bridge cells)."""
        old = list(self.all_cells.cells)
        cells = cells_from_mpg(self.mpg, self._mpg_written)
        replaced = {id(a): b for a, b in zip(old, cells)}
        if isinstance(self.all_cells, MPGCells):               # the pipeline is over: plain Cells, as since P1
            self.all_cells = CellManager()
            self._cell_store = None
        self.all_cells.cells[:] = cells
        for layer in getattr(self.layer_manager, "layers", []):
            layer.cells[:] = [replaced.get(id(c), c) for c in layer.cells]
        self.__dict__["_gdf"], self.__dict__["_gdf_n_real"] = None, n_real      # the view, rebuilt when read

    def _store_network(self) -> None:
        """
        The hydraulic network of the final cells, stored in the MPG when the cells are generated (P2.2), without
        bridges (which would add cells to the organ), so that every section carries its walls, junctions and edges,
        and the export reads them instead of recomputing them. Built in the MPG from the stored rings (N1), or by
        the exporter into a scratch network copied into the MPG (native_network False).
        """
        store_topology(self.mpg, self.segment, self, self._mpg_written)          # the one snapping of the section
        if self.native_network:
            build_network(self)
            return
        NetworkExporter = self._network_exporter()
        scratch = SimpleNamespace(graph=nx.Graph(), n_walls=0, n_junctions=0, n_cells=0, _wall_to_cells={})
        with stored_topology(self):
            NetworkExporter(self).export(scratch, bridges=False)
        write_network(self.mpg, self.segment, self, self._mpg_written, network=scratch)

    def write_to_xml(self, path: str, **kwargs):
        """GRANAP's XML writer, its walls numbered from the topology stored in the MPG (P2.3)."""
        self.generate_section()
        with stored_topology(self):
            return super().write_to_xml(path, **kwargs)

    def _network_exporter(self):
        """The network exporter: the converted copy (P5.2) or GRANAP's."""
        if self.converted_pipeline:
            from openalea.granap.mpg.pipeline.network import NetworkExporter
        else:
            from openalea.granap.anatomy_writer import NetworkExporter
        return NetworkExporter

    def _stored_network_matches(self, air_link_radius, exporter_kwargs) -> bool:
        """Whether the export asked for is the stored one: default exporter settings, and no bridge to add."""
        if air_link_radius is not None or set(exporter_kwargs) - {"bridges", "bridge_radius"}:
            return False
        wants_bridges = exporter_kwargs.get("bridges", True)
        return not (wants_bridges and self.network_bridge_specs())

    # ── The network (N1): stored in the MPG; GRANAP's graph and matrix are views of it ──

    @property
    def graph(self):
        """The exporter's networkx graph, built from the MPG at its first read after the network is (re)built."""
        graph = self.__dict__.get("_graph")
        if graph is None:
            graph = self.__dict__["_graph"] = network_from_mpg(self.mpg, self._mpg_written)
        return graph

    @graph.setter
    def graph(self, value):
        self.__dict__["_graph"] = value

    def export_to_adjencymatrix(self, air_link_radius=None, **exporter_kwargs):
        """GRANAP's adjacency matrix (a 1 per edge, both ways), from the section's Connections."""
        self.generate_section()
        if self._matrix is not None:
            return self._matrix
        if not self.__dict__.get("_network_built"):
            self._build_anatnetwork(air_link_radius=air_link_radius, **exporter_kwargs)
        g, written = self.mpg, self._mpg_written
        u_of, v_of = g.property("edge_u"), g.property("edge_v")
        u = np.array([u_of[e] for e in written["connections"]], dtype=np.int64)
        v = np.array([v_of[e] for e in written["connections"]], dtype=np.int64)
        n = self.n_total
        self._matrix = coo_matrix((np.ones(2 * u.size), (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tolil()
        return self._matrix

    def _build_anatnetwork(self, air_link_radius=None, **exporter_kwargs):
        self.generate_section()
        written = self._mpg_written
        if not self._stored_network_matches(air_link_radius, exporter_kwargs):
            self.graph = nx.Graph()                                               # the exporter's own target
            with stored_topology(self):                                           # the stored rings, not a new snap
                self._network_exporter()(self).export(self, air_link_radius=air_link_radius, **exporter_kwargs)
            written = write_network(self.mpg, self.segment, self, written)        # replaces the stored one
            n_real = len(written["cells"]) - len(written.get("bridge_cells", []))
            if len(written["cells"]) > n_real or "bridge" in self._cells_gdf.columns:
                self._read_cells_back(n_real)
        self.graph = None                                                         # a view of the MPG, when read
        self.n_walls, self.n_junctions, self.n_cells = written["n_walls"], written["n_junctions"], written["n_cells"]
        self._wall_to_cells = dict(written["wall_to_cells"])
        self._network_built = True


def _inherited(cls, name):
    """The method *name* that *cls* inherits past MPGBacked."""
    return next(c.__dict__[name] for c in cls.__mro__ if name in c.__dict__ and c is not MPGBacked)


def _backed(cls):
    return type(cls.__name__, (MPGBacked, cls), {"__module__": __name__, "__doc__": cls.__doc__})


MonocotRootAnatomy, DicotRootAnatomy = _backed(_MonocotRoot), _backed(_DicotRoot)
MonocotStemAnatomy, DicotStemAnatomy = _backed(_MonocotStem), _backed(_DicotStem)
ContinuousDicotStemAnatomy = _backed(_ContinuousStem)
MonocotLeafAnatomy, DicotLeafAnatomy = _backed(_MonocotLeaf), _backed(_DicotLeaf)
NeedleAnatomy, CellSetOrgan, RoiOrgan = _backed(_Needle), _backed(_CellSetOrgan), _backed(_RoiOrgan)
_CONCRETE = {_MonocotRoot: MonocotRootAnatomy, _DicotRoot: DicotRootAnatomy, _MonocotStem: MonocotStemAnatomy,
             _DicotStem: DicotStemAnatomy, _ContinuousStem: ContinuousDicotStemAnatomy,
             _MonocotLeaf: MonocotLeafAnatomy, _DicotLeaf: DicotLeafAnatomy}


def _factory(base):
    """GRANAP's factory class (RootAnatomy, StemAnatomy, LeafAnatomy), returning the MPG-backed concrete class."""
    def __new__(cls, *args, **kwargs):
        instance = base.__new__(base, *args, **kwargs)
        backed = _CONCRETE.get(type(instance))
        if backed is None:
            raise TypeError(f"no MPG-backed class for {type(instance).__name__}")
        organ = object.__new__(backed)
        organ.__init__(*args, **kwargs)
        return organ
    return type(base.__name__, (base,), {"__new__": __new__, "__module__": __name__, "__doc__": base.__doc__})


RootAnatomy, StemAnatomy, LeafAnatomy = _factory(_Root), _factory(_Stem), _factory(_Leaf)


_ORIGINAL_CREATE = _Organ.__dict__["create_from_input"].__func__     # GRANAP's own, even if replaced on Organ


def create_from_input(input_data, seed=None):
    """
    Organ.create_from_input, returning the MPG-backed class of the organ it chooses, with *seed* (Q8, your answer
    Q-P5h: GRANAP's passes none, so its organs were never reproducible; seed=None keeps that behaviour).
    """
    chosen = type(_ORIGINAL_CREATE(_Organ, input_data))
    backed = {**_CONCRETE, _Needle: NeedleAnatomy}.get(chosen)
    if backed is None:
        raise TypeError(f"no MPG-backed class for {chosen.__name__}")
    return backed(input_data, seed=seed)
