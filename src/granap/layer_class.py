"""
Layer module for plant anatomy representation.
Provides the Layer class representing individual tissue layers.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from granap.cell_class import Cell
from shapely.geometry import Polygon


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
        result.update(self.additional_params)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Layer':
        """Create a Layer from dictionary representation."""
        # Extract known fields
        name = data.get("name", "unknown")
        cell_diameter = data.get("cell_diameter", 0.01)
        n_layers = data.get("n_layers", 1)
        order = data.get("order", 0)
        cell_width = data.get("cell_width", cell_diameter)
        shift = data.get("shift", 0.0)
        
        # Everything else goes into additional_params
        additional_params = {
            k: v for k, v in data.items() 
            if k not in ["name", "cell_diameter", "n_layers", "order", "cell_width", "shift"]
        }
        
        return cls(
            name=name,
            cell_diameter=cell_diameter,
            n_layers=n_layers,
            order=order,
            cell_width=cell_width,
            shift=shift,
            additional_params=additional_params
        )
    
    def __repr__(self) -> str:
        return f"Layer(name='{self.name}', diameter={self.cell_diameter:.4f}, n_layers={self.n_layers}, order={self.order})"
