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
from openalea.granap.mpg.pipeline.aerenchyma import add_aerenchyma_to_proportion
from openalea.granap.mpg.speedups import exact_speedups
from openalea.granap.mpg.topology import store_topology, stored_topology
from openalea.granap.mpg.views import cells_from_mpg, gdf_from_cells, network_from_mpg


class MPGBacked:
    """The organ's cells and network written into an MPG and read back from it (mixin, before the organ class)."""

    mpg = None
    segment = None
    _mpg_written = None
    mpg_stage_hooks = ()             # callables (organ, stage) after each stage write (tests, diagnostics)
    exact_speedups = True            # P4: GRANAP's pipeline with exact speed-ups (identical outputs); False: as is
    converted_pipeline = True        # P5: the converted copies of GRANAP's modules (mpg/pipeline/); False: GRANAP's

    def generate_cells(self):
        if getattr(self, "_cells_gdf", None) is not None:
            return self._cells_gdf
        if self.exact_speedups:
            with exact_speedups(self):
                super().generate_cells()
        else:
            super().generate_cells()
        self._mpg_stage("final")
        self._read_cells_back()
        self._store_network()
        return self._cells_gdf

    # ── Stages (P2): the pipeline's cells written into the MPG at each boundary, GRANAP's own calls unchanged ──

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

    def add_intercellular_spaces(self):
        self._mpg_stage("grouping")                    # the Voronoi groups, simplified, nested cells removed
        if self.converted_pipeline:                    # P5.1: from here on, the cells are views of the MPG
            self._cell_store = CellStore(self.mpg, self.segment, self._mpg_written)
            self.all_cells = MPGCells(self._cell_store, self.all_cells.cells)
        return super().add_intercellular_spaces()

    def _add_aerenchyma_to_proportion(self, *args, **kwargs):
        self._mpg_stage("intercellular")
        inherited = next(c.__dict__["_add_aerenchyma_to_proportion"] for c in type(self).__mro__[1:]
                         if "_add_aerenchyma_to_proportion" in c.__dict__ and c is not MPGBacked)
        if self.converted_pipeline and inherited is _Organ.__dict__["_add_aerenchyma_to_proportion"]:
            result = add_aerenchyma_to_proportion(self, *args, **kwargs)        # P5.3: GRANAP's, on shallow copies
        else:
            result = super()._add_aerenchyma_to_proportion(*args, **kwargs)     # an organ's own (e.g. leaves)
        self._mpg_stage("aerenchyma")
        return result

    def fuse_gaps(self, *args, **kwargs):
        fused = super().fuse_gaps(*args, **kwargs)
        self._mpg_stage("fused_gaps")
        return fused

    def _read_cells_back(self, n_real=None):
        """all_cells, the layers' cell lists and the GeoDataFrame, from the MPG (the same cells, the same order)."""
        old = list(self.all_cells.cells)
        cells = cells_from_mpg(self.mpg, self._mpg_written)
        replaced = {id(a): b for a, b in zip(old, cells)}
        if isinstance(self.all_cells, MPGCells):               # the pipeline is over: plain Cells, as since P1
            self.all_cells = CellManager()
            self._cell_store = None
        self.all_cells.cells[:] = cells
        for layer in getattr(self.layer_manager, "layers", []):
            layer.cells[:] = [replaced.get(id(c), c) for c in layer.cells]
        self._cells_gdf = gdf_from_cells(cells, n_real)

    def _store_network(self) -> None:
        """
        The hydraulic network of the final cells, stored in the MPG when the cells are generated (P2.2): GRANAP's
        exporter run once into a scratch network, without bridges (which would add cells to the organ), so that
        every section carries its walls, junctions and edges, and the export reads them instead of recomputing them.
        """
        NetworkExporter = self._network_exporter()
        store_topology(self.mpg, self.segment, self, self._mpg_written)          # the one snapping of the section
        scratch = SimpleNamespace(graph=nx.Graph(), n_walls=0, n_junctions=0, n_cells=0, _wall_to_cells={})
        with stored_topology(self):
            NetworkExporter(self).export(scratch, bridges=False)
        write_network(self.mpg, self.segment, self, self._mpg_written, network=scratch)

    def write_to_xml(self, path: str, **kwargs):
        """GRANAP's XML writer, its walls numbered from the topology stored in the MPG (P2.3)."""
        self.generate_cells()
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

    def _build_anatnetwork(self, air_link_radius=None, **exporter_kwargs):
        if self.mpg is None:
            self.generate_cells()
        written = self._mpg_written
        if not self._stored_network_matches(air_link_radius, exporter_kwargs):
            with stored_topology(self):                                           # the stored rings, not a new snap
                self._network_exporter()(self).export(self, air_link_radius=air_link_radius, **exporter_kwargs)
            written = write_network(self.mpg, self.segment, self, written)        # replaces the stored one
            n_real = len(written["cells"]) - len(written.get("bridge_cells", []))
            if len(written["cells"]) > n_real or "bridge" in self._cells_gdf.columns:
                self._read_cells_back(n_real)
        self.graph = network_from_mpg(self.mpg, written)
        self.n_walls, self.n_junctions = written["n_walls"], written["n_junctions"]
        self.n_cells = self.graph.number_of_nodes() - self.n_walls - self.n_junctions
        self._wall_to_cells = dict(written["wall_to_cells"])


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


def create_from_input(input_data):
    """Organ.create_from_input, returning the MPG-backed class of the organ it chooses (no seed, as GRANAP's)."""
    chosen = type(_ORIGINAL_CREATE(_Organ, input_data))
    backed = {**_CONCRETE, _Needle: NeedleAnatomy}.get(chosen)
    if backed is None:
        raise TypeError(f"no MPG-backed class for {chosen.__name__}")
    return backed(input_data)
