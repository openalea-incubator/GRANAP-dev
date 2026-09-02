"""
Layer module for plant anatomy representation.
Provides the Layer class representing individual tissue layers,
and LayerPolygon which carries the computed geometry for one layer ring.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from openalea.granap.cell_class import Cell
from shapely.geometry import Polygon
from openalea.granap.input_data import LayerDefaultParams


@dataclass
class LayerPolygon:
    """
    Typed representation of a layer polygon produced by _build_layer_polygons.

    Replaces the ad-hoc plain dicts that were previously passed around.
    Backward-compatible dict-style access (``lp["key"]``, ``lp.get("key")``)
    is kept so callers that have not yet been migrated continue to work.
    """
    name: str
    polygon: Polygon
    cell_diameter: float
    cell_width: float = 0.0
    id_layer: int = 0
    shift: float = 0.0
    # Out-of-plane (longitudinal) cell extent — see ROOT_3D_PLAN. None = no 3D
    # axial subdivision configured (2D behaviour unaffected).
    axial_height: Optional[float] = None
    # Transfusion-tissue fields (needle-specific)
    transfusion_type: bool = False
    tt_diameter: float = 0.0
    tp_diameter: float = 0.0
    p_tt: float = 0.0

    # ------------------------------------------------------------------
    # Backward-compatible dict-style access
    # ------------------------------------------------------------------
    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


@dataclass
class Layer:
    """
    Represents a single tissue layer in plant anatomy.
    
    Attributes:
        name: Identifier for the layer type (e.g., 'epidermis', 'mesophyll')
        cell_diameter: Diameter of cells in this layer (mm)
        n_layers: Number of sub-layers (default: 1)
        order: Rendering order (higher = outer layers)
        cell_width: Optional cell width if different from diameter (mm)
        additional_params: Dictionary for layer-specific parameters
    """
    name: str
    cell_diameter: float
    n_layers: int = 1
    order: int = 0
    cell_width: Optional[float] = None
    shift: float = 0.0
    # Out-of-plane (longitudinal) cell extent — see ROOT_3D_PLAN. None = no 3D
    # axial subdivision configured (2D behaviour unaffected).
    axial_height: Optional[float] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    cells: List[Cell] = field(default_factory=list)
    polygon: Optional[Polygon] = None
    
    def __post_init__(self):
        """Validate layer parameters."""
        if self.cell_diameter <= 0:
            raise ValueError(f"cell_diameter must be positive, got {self.cell_diameter}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {self.n_layers}")
    
    def get_total_thickness(self) -> float:
        """Calculate total thickness of this layer."""
        return self.cell_diameter * self.n_layers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert layer to dictionary representation."""
        result = {
            "name": self.name,
            "cell_diameter": self.cell_diameter,
            "n_layers": self.n_layers,
            "order": self.order,
            "shift": self.shift
        }
        if self.cell_width is not None:
            result["cell_width"] = self.cell_width
        if self.axial_height is not None:
            result["axial_height"] = self.axial_height
        result.update(self.additional_params)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Layer':
        """Create a Layer from dictionary representation."""
        # Extract known fields
        _defaults = LayerDefaultParams()
        name = data.get("name", _defaults.name)
        cell_diameter = data.get("cell_diameter", _defaults.cell_diameter_default)
        n_layers = data.get("n_layers", _defaults.n_layers_default)
        order = data.get("order", _defaults.order_default)
        cell_width = data.get("cell_width", data.get("cell_diameter", _defaults.cell_width_default))
        shift = data.get("shift", _defaults.shift_default)
        axial_height = data.get("axial_height", _defaults.axial_height_default)

        # Everything else goes into additional_params
        additional_params = {
            k: v for k, v in data.items()
            if k not in ["name", "cell_diameter", "n_layers", "order", "cell_width", "shift", "axial_height"]
        }

        return cls(
            name=name,
            cell_diameter=cell_diameter,
            n_layers=n_layers,
            order=order,
            cell_width=cell_width,
            shift=shift,
            axial_height=axial_height,
            additional_params=additional_params
        )
    
    def __repr__(self) -> str:
        return f"Layer(name='{self.name}', diameter={self.cell_diameter:.4f}, n_layers={self.n_layers}, order={self.order})"
