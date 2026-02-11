"""
Needle anatomy implementation.
"""

import numpy as np
from typing import List, Dict, Any
from shapely.geometry import Polygon

from granap.organ_class import Organ
from granap.layer_class import Layer
from granap.geometry_collection import GeometryProcessor


class NeedleAnatomy(Organ):
    """
    Needle cross-sectional anatomy.
    
    Implements the specific structure of gymnosperm needle leaves,
    including transfusion tissue and resin ducts.
    """
    
    def __init__(self, randomness: float = 1.0):
        """
        Initialize needle anatomy.
        
        Args:
            randomness: Degree of randomness in cell placement (0-3)
        """
        super().__init__(randomness)
        self._initialize_default_layers()
        
        # Needle specific parameters
        self.central_cylinder_params = {
            "cell_diameter": 0.0063,
            "layer_thickness": 0.2,
            "layer_length": 0.4,
            "transfusion_layers": 3,
            "transfusion_tracheids_ratio": 0.5,
            "transfusion_tracheids_diameter": 0.015,
            "transfusion_parenchyma_diameter": 0.025,
            "shape": "ellipse"
        }

        self.resin_duct_params = {
            "cell_diameter": 0.0063,
            "inner_diameter": 0.01,
            "n_files":3,
        }

        self.stomata_params = {
            "cell_diameter": 0.0063,
            "depth": 0.01,
            "n_files_adaxial":3,
            "n_files_abaxial":3,
        }
    
    def _initialize_default_layers(self) -> None:
        """Initialize default needle layers."""
        # Outer to inner (order: higher = outer)
        self.layer_manager.add_layer(Layer(
            name="epidermis",
            cell_diameter=0.025,
            n_layers=1,
            order=6
        ))
        
        self.layer_manager.add_layer(Layer(
            name="hypodermis",
            cell_diameter=0.025,
            n_layers=3,
            order=5
        ))
        
        self.layer_manager.add_layer(Layer(
            name="mesophyll",
            cell_diameter=0.03,
            cell_width=0.03,
            n_layers=3,
            order=4
        ))
        
        self.layer_manager.add_layer(Layer(
            name="endodermis",
            cell_diameter=0.010,
            cell_width=0.03,
            n_layers=1,
            order=3
        ))
    
    def _create_base_shape(self) -> Polygon:
        """
        Create the half-ellipse shape of a needle cross-section.
        
        Returns:
            Half-ellipse polygon
        """
        width = self._calculate_needle_width()
        thickness = self._calculate_needle_thickness()
        
        return GeometryProcessor.half_ellipse_polygon(width, thickness)
    
    def _calculate_needle_width(self) -> float:
        """Calculate total needle width from layers."""
        # width of vascular cylinder
        width_vascular = self.central_cylinder_params["layer_length"]
        # width of all supplementary layers
        width_layer = 0
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, 'n_layers'):
                width_layer += layer.get_total_thickness()
            elif hasattr(layer, 'cell_diameter'):
                width_layer += layer.cell_diameter
        # thickness of vascular cylinder
        thickness_vascular = self.central_cylinder_params["layer_thickness"]
        # thickness of all supplementary layers which is equal to width_layer
        thickness_layer = width_layer
        thickness_total = (2 * thickness_layer) + thickness_vascular
        
        width = 2 * np.sqrt((width_vascular/2 + width_layer)**2 / 
                            (1-(thickness_layer/thickness_total)**2))

        return width
    
    def _calculate_needle_thickness(self) -> float:
        """Calculate total needle thickness from layers."""
        thickness = self.central_cylinder_params["layer_thickness"]
        
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, 'n_layers'):
                thickness += 2 * layer.get_total_thickness()
            elif hasattr(layer, 'cell_diameter'):
                thickness += 2 * layer.cell_diameter
        
        return thickness
    
    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create transfusion tissue and parenchyma layers.
        
        Args:
            current_polygon: Current inner polygon boundary
            params: Parameter dictionaries
        
        Returns:
            List of central layer polygon dictionaries
        """
        central_layers = []
        space_increment = self.central_cylinder_params["cell_diameter"] / 2
        transfusion_layers_remaining = self.central_cylinder_params["transfusion_layers"]
        
        tt_diameter = self.transfusion_params["tracheids_diameter"]
        tp_diameter = self.transfusion_params["parenchyma_diameter"]
        parenchyma_diameter = self.central_cylinder_params["cell_diameter"]
        
        i_layer = len(params)
        
        while current_polygon.area > (parenchyma_diameter / 2)**2 * np.pi:
            if transfusion_layers_remaining > 0:
                # Transfusion tissue
                avg_diameter = (tp_diameter + tt_diameter) / 2 
                transfusion_layers_remaining -= 1
                
                current_polygon = GeometryProcessor.buffer_polygon(
                    current_polygon,
                    -space_increment - avg_diameter / 2,
                    smooth_factor=0.6
                )
                
                space_increment = avg_diameter / 2
                
                central_layers.append({
                    "name": "transfusion",
                    "polygon": current_polygon,
                    "cell_diameter": avg_diameter,
                    "id_layer": i_layer + 1,
                    "cell_width": 0
                })
            else:
                # Parenchyma
                current_polygon = GeometryProcessor.buffer_polygon(
                    current_polygon,
                    -space_increment - parenchyma_diameter / 2,
                    smooth_factor=0.7
                )
                
                space_increment = parenchyma_diameter / 2
                
                central_layers.append({
                    "name": "parenchyma",
                    "polygon": current_polygon,
                    "cell_diameter": parenchyma_diameter,
                    "id_layer": i_layer + 1,
                    "cell_width": 0
                })
            
            i_layer += 1
        
        return central_layers
    
    def set_central_cylinder_params(self, **kwargs) -> None:
        """
        Update central cylinder parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        self.central_cylinder_params.update(kwargs)
        self._invalidate_geometry()
    
    def set_transfusion_params(self, **kwargs) -> None:
        """
        Update transfusion tissue parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        self.transfusion_params.update(kwargs)
        self._invalidate_geometry()
