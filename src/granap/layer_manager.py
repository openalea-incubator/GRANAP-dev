"""
Layer manager module for handling collections of tissue layers.
"""

from typing import List, Optional, Dict, Any
from granap.layer_class import Layer


class LayerManager:
    """
    Manages a collection of tissue layers with add/remove operations.
    
    This class handles the dynamic management of layers, ensuring proper
    ordering and validation of layer operations.
    """
    
    def __init__(self):
        """Initialize an empty layer collection."""
        self._layers: List[Layer] = []
    
    def add_layer(self, layer: Layer, position: Optional[int] = None) -> None:
        """
        Add a layer to the collection.
        
        Args:
            layer: Layer object to add
            position: Optional position index (None = append to end)
        
        Raises:
            ValueError: If layer name already exists or position invalid
        """
        if self.has_layer(layer.name):
            raise ValueError(f"Layer with name '{layer.name}' already exists")
        
        if position is None:
            self._layers.append(layer)
        else:
            if position < 0 or position > len(self._layers):
                raise ValueError(f"Invalid position {position}, must be 0-{len(self._layers)}")
            self._layers.insert(position, layer)
    
    def remove_layer(self, name: str) -> Layer:
        """
        Remove a layer by name.
        
        Args:
            name: Name identifier of the layer to remove
        
        Returns:
            The removed Layer object
        
        Raises:
            ValueError: If layer name not found
        """
        for i, layer in enumerate(self._layers):
            if layer.name == name:
                return self._layers.pop(i)
        raise ValueError(f"Layer '{name}' not found")
    
    def get_layer(self, name: str) -> Optional[Layer]:
        """
        Retrieve a layer by name.
        
        Args:
            name: Name identifier of the layer
        
        Returns:
            Layer object or None if not found
        """
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    def get_layer_by_order(self, order:int) -> Optional[Layer]:
        for layer in self._layers:
            if layer.order == order:
                return layer
        return None
    
    def has_layer(self, name: str) -> bool:
        """Check if a layer exists."""
        return self.get_layer(name) is not None
    
    def get_layers(self) -> List[Layer]:
        """Get all layers in current order."""
        return self._layers.copy()
    
    def get_ordered_layers(self, reverse: bool = True) -> List[Layer]:
        """
        Get layers sorted by their order attribute.
        
        Args:
            reverse: If True, sort descending (outer to inner)
        
        Returns:
            Sorted list of layers
        """
        # Filter layers that have an order attribute > 0
        ordered = [l for l in self._layers if l.order > 0]
        ordered.sort(key=lambda x: x.order, reverse=reverse)
        return ordered
    
    def expand_layers(self) -> List[Dict[str, Any]]:
        """
        Expand layers with n_layers > 1 into individual layer entries.
        Used for iterative processing during geometry generation.
        
        Returns:
            List of dictionaries with layer information
        """
        expanded = []
        for layer in self.get_ordered_layers():
            for i in range(layer.n_layers):
                expanded.append({
                    "name": layer.name,
                    "cell_diameter": layer.cell_diameter,
                    "cell_width": layer.cell_width or 0,
                    "shift": layer.shift
                })
        return expanded
    
    def get_layers_params(self) -> List[Dict[str, Any]]:
        """
        Get parameters of all layers.
        
        Returns:
            List of dictionaries with layer parameters
        """
        return [{"name": layer.name, "cell_diameter": layer.cell_diameter, "cell_width": layer.cell_width or 0, "shift": layer.shift, "n_layers": layer.n_layers, "order": layer.order} for layer in self._layers]
    
    def clear(self) -> None:
        """Remove all layers."""
        self._layers.clear()
    
    def __len__(self) -> int:
        """Return number of layers."""
        return len(self._layers)
    
    def __iter__(self):
        """Make LayerManager iterable."""
        return iter(self._layers)
    
    def __repr__(self) -> str:
        return f"LayerManager({len(self._layers)} layers)"
