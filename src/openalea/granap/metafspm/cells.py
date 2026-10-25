"""
The pipeline's cells as views of the MPG.

From the Voronoi grouping on, the organ's cell list (organ.all_cells) holds CellViews: Cell objects that keep nothing
themselves, their attributes read from and written to their Cell vertex's properties (cell__<name>). GRANAP's
algorithms run unchanged on them: setting cell.type or cell.polygon edits the MPG.

Where GRANAP brings plain Cells (an air space appended, fused cells extended, the aerenchyma calibration restoring its
copies), the list adopts them: a cell already in the section (the same uid, e.g. a copy) updates its vertex, only the
attributes whose values differ; a new one gets a Cell vertex and a protoplast, in a provisional Layer until the next
stage regroups the Layers. Copying a view (copy.copy, copy.deepcopy) gives a detached plain Cell with its values
and uid. Cells dropped from the list are removed from the MPG at the next stage write (build.write_cells).

At the end of the pipeline the organ gets plain Cells back, rebuilt from the MPG (views.cells_from_mpg), as since P1.
"""
import copy

from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.metafspm.build import PROTOPLAST, UID, _props

_MISSING = object()


class CellStore:
    """The section's Cell vertices as attribute stores: one per cell uid, its attributes in cell__<name>."""

    def __init__(self, g, segment, written):
        self.g, self.segment, self.written = g, segment, written
        self._props = g.properties()
        self._names = {}                                        # attribute -> its property dict
        self.views = {}                                         # uid -> CellView (one object per cell)
        self.next_uid = written["next_uid"]

    def _prop(self, name):
        prop = self._names.get(name)
        if prop is None:
            prop = self._names[name] = self._props.setdefault("cell__" + name, {})
        return prop

    def read(self, vid, name):
        if name == UID:
            return self._props[UID][vid]
        value = self._prop(name).get(vid, _MISSING)
        if value is _MISSING:
            raise AttributeError(name)
        return value

    def write(self, vid, name, value):
        if name == UID:
            raise AttributeError("a cell's uid is fixed")
        self._prop(name)[vid] = value

    def record(self, vid) -> dict:
        out = {}
        for name, values in self._props.items():
            if name.startswith("cell__") and vid in values:
                out[name[6:]] = values[vid]
        return out

    def view(self, uid) -> "CellView":
        view = self.views.get(uid)
        if view is None:
            view = self.views[uid] = CellView(self, self.written["cell_of_uid"][uid])
        return view

    def adopt(self, cell):
        """The view of *cell*: itself if a view, else its section vertex updated (same uid) or a new vertex."""
        if isinstance(cell, CellView):
            return cell
        attributes = {k: v for k, v in vars(cell).items() if k != UID}
        uid = vars(cell).get(UID)
        vid = self.written["cell_of_uid"].get(uid) if uid is not None else None
        if vid is None:                                         # a new cell: a vertex and a protoplast
            if uid is None:
                uid = self.written["next_uid"]
                self.written["next_uid"] += 1
            layer = self._provisional_layer()
            s, L = self.g.scales, self.g.labels
            vid = self.g.add_components_bulk(layer, 1, **_props(s.Cell))[0]
            protoplast = self.g.add_components_bulk([vid], 1, **_props(s.Compartment, kind=PROTOPLAST,
                                                                          label=L.Compartment.Symplastic))[0]
            self.written["cell_of_uid"][uid], self.written["protoplast_of_uid"][uid] = vid, protoplast
            self._props.setdefault(UID, {})[vid] = uid
        for name, value in attributes.items():                  # only what differs (a copy restored: nothing)
            prop = self._prop(name)
            current = prop.get(vid, _MISSING)
            if current is _MISSING or current is not value and not _equal(current, value):
                prop[vid] = value
        for name, prop in self._names.items():                  # an attribute the cell no longer has
            if name not in attributes and vid in prop:
                del prop[vid]
        return self.view(uid)

    def _provisional_layer(self):
        layer = self.written.get("provisional_layer")
        if layer is None or layer not in self.g._scale:
            s = self.g.scales
            layer = self.g.add_components_bulk(self.segment, 1, **_props(s.Layer, tissue="provisional"))[0]
            self.written["provisional_layer"] = layer
        return layer


def _equal(a, b) -> bool:
    try:
        result = a == b
        return bool(result) if not hasattr(result, "all") else bool(result.all())
    except Exception:
        return False


class CellView(Cell):
    """A Cell whose attributes live in the MPG (its Cell vertex's properties)."""
    __slots__ = ("_store", "_vid")
    _mpg_view = True

    def __init__(self, store, vid):                             # not Cell.__init__: the values are in the MPG
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_vid", vid)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._store.read(self._vid, name)

    def __setattr__(self, name, value):
        self._store.write(self._vid, name, value)

    def __delattr__(self, name):
        prop = self._store._prop(name)
        if self._vid not in prop:
            raise AttributeError(name)
        del prop[self._vid]

    def detached(self) -> Cell:
        cell = Cell.__new__(Cell)
        cell.__dict__.update(self._store.record(self._vid))
        cell.__dict__[UID] = self._store.read(self._vid, UID)
        return cell

    def __copy__(self):
        return self.detached()

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.detached(), memo)


class ViewList(list):
    """A cell list that adopts the plain Cells it receives (append, extend, insert, item and slice assignment)."""

    def __init__(self, store, cells=()):
        self._store = store
        super().__init__(store.adopt(c) for c in cells)

    def append(self, cell):
        super().append(self._store.adopt(cell))

    def extend(self, cells):
        super().extend(self._store.adopt(c) for c in cells)

    def insert(self, index, cell):
        super().insert(index, self._store.adopt(cell))

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            super().__setitem__(index, [self._store.adopt(c) for c in value])
        else:
            super().__setitem__(index, self._store.adopt(value))

    def __iadd__(self, cells):
        self.extend(cells)
        return self


class MPGCells(CellManager):
    """The organ's cell manager during the pipeline: its cells (_cells) are a ViewList of the section in the MPG."""

    def __init__(self, store, cells=()):
        self._store = store
        super().__init__()
        self.cells = list(cells)

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, value):
        CellManager.cells.fset(self, ViewList(self._store, value))      # keeps CellManager's id_group cache


def attributes(cell) -> dict:
    """A cell's attributes, a view's from the MPG, a plain Cell's its own (its pipeline uid left out)."""
    if getattr(type(cell), "_mpg_view", False):
        return cell._store.record(cell._vid)
    return {k: v for k, v in vars(cell).items() if k != UID}
