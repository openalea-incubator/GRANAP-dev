"""
Unit tests for the plant anatomy framework.

Run with: python -m pytest test_anatomy.py -v
Or: python test_anatomy.py
"""

import unittest
import numpy as np
from layer import Layer
from layer_manager import LayerManager
from geometry_processor import GeometryProcessor
from pine_needle_anatomy import PineNeedleAnatomy
from root_anatomy import RootAnatomy
from shapely.geometry import Polygon


class TestLayer(unittest.TestCase):
    """Test Layer class functionality."""
    
    def test_layer_creation(self):
        """Test basic layer creation."""
        layer = Layer(
            name="test_layer",
            cell_diameter=0.02,
            n_layers=2,
            order=3
        )
        
        self.assertEqual(layer.name, "test_layer")
        self.assertEqual(layer.cell_diameter, 0.02)
        self.assertEqual(layer.n_layers, 2)
        self.assertEqual(layer.order, 3)
    
    def test_layer_validation(self):
        """Test that invalid parameters raise errors."""
        with self.assertRaises(ValueError):
            Layer(name="bad", cell_diameter=-0.01, n_layers=1, order=1)
        
        with self.assertRaises(ValueError):
            Layer(name="bad", cell_diameter=0.01, n_layers=0, order=1)
    
    def test_total_thickness(self):
        """Test thickness calculation."""
        layer = Layer(name="test", cell_diameter=0.05, n_layers=3, order=1)
        self.assertAlmostEqual(layer.get_total_thickness(), 0.15, places=6)
    
    def test_layer_to_dict(self):
        """Test dictionary conversion."""
        layer = Layer(
            name="test",
            cell_diameter=0.02,
            n_layers=2,
            order=3,
            cell_width=0.01
        )
        
        d = layer.to_dict()
        self.assertEqual(d['name'], "test")
        self.assertEqual(d['cell_diameter'], 0.02)
        self.assertEqual(d['cell_width'], 0.01)
    
    def test_layer_from_dict(self):
        """Test creating layer from dictionary."""
        data = {
            "name": "epidermis",
            "cell_diameter": 0.015,
            "n_layers": 1,
            "order": 5
        }
        
        layer = Layer.from_dict(data)
        self.assertEqual(layer.name, "epidermis")
        self.assertEqual(layer.cell_diameter, 0.015)


class TestLayerManager(unittest.TestCase):
    """Test LayerManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = LayerManager()
        self.layer1 = Layer(name="layer1", cell_diameter=0.01, order=1)
        self.layer2 = Layer(name="layer2", cell_diameter=0.02, order=2)
    
    def test_add_layer(self):
        """Test adding layers."""
        self.manager.add_layer(self.layer1)
        self.assertEqual(len(self.manager), 1)
        self.assertTrue(self.manager.has_layer("layer1"))
    
    def test_add_duplicate_layer(self):
        """Test that duplicate names raise error."""
        self.manager.add_layer(self.layer1)
        with self.assertRaises(ValueError):
            self.manager.add_layer(self.layer1)
    
    def test_remove_layer(self):
        """Test removing layers."""
        self.manager.add_layer(self.layer1)
        self.manager.add_layer(self.layer2)
        
        removed = self.manager.remove_layer("layer1")
        self.assertEqual(removed.name, "layer1")
        self.assertEqual(len(self.manager), 1)
        self.assertFalse(self.manager.has_layer("layer1"))
    
    def test_remove_nonexistent_layer(self):
        """Test removing non-existent layer raises error."""
        with self.assertRaises(ValueError):
            self.manager.remove_layer("nonexistent")
    
    def test_get_layer(self):
        """Test retrieving layers."""
        self.manager.add_layer(self.layer1)
        retrieved = self.manager.get_layer("layer1")
        self.assertEqual(retrieved.name, "layer1")
    
    def test_get_ordered_layers(self):
        """Test ordering layers."""
        self.manager.add_layer(self.layer2)  # order=2
        self.manager.add_layer(self.layer1)  # order=1
        
        ordered = self.manager.get_ordered_layers(reverse=True)
        self.assertEqual(ordered[0].order, 2)
        self.assertEqual(ordered[1].order, 1)
    
    def test_expand_layers(self):
        """Test expanding multi-layer entries."""
        layer = Layer(name="cortex", cell_diameter=0.04, n_layers=3, order=1)
        self.manager.add_layer(layer)
        
        expanded = self.manager.expand_layers()
        self.assertEqual(len(expanded), 3)
        self.assertEqual(expanded[0]['name'], 'cortex')


class TestGeometryProcessor(unittest.TestCase):
    """Test GeometryProcessor utility methods."""
    
    def test_half_ellipse_creation(self):
        """Test half-ellipse polygon generation."""
        polygon = GeometryProcessor.half_ellipse_polygon(1.0, 0.5)
        
        self.assertIsInstance(polygon, Polygon)
        self.assertTrue(polygon.is_valid)
        self.assertGreater(polygon.area, 0)
    
    def test_circle_creation(self):
        """Test circle polygon generation."""
        polygon = GeometryProcessor.circle_polygon(0.5)
        
        self.assertIsInstance(polygon, Polygon)
        self.assertTrue(polygon.is_valid)
        # Area should be approximately π * r²
        expected_area = np.pi * 0.5**2
        self.assertAlmostEqual(polygon.area, expected_area, places=2)
    
    def test_resample_coords(self):
        """Test coordinate resampling."""
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        resampled = GeometryProcessor.resample_coords(coords, target_n_points=10)
        
        self.assertEqual(len(resampled), 10)
        self.assertEqual(resampled.shape[1], 2)
    
    def test_buffer_polygon(self):
        """Test polygon buffering."""
        polygon = GeometryProcessor.circle_polygon(1.0)
        buffered = GeometryProcessor.buffer_polygon(polygon, 0.1)
        
        self.assertGreater(buffered.area, polygon.area)


class TestPineNeedleAnatomy(unittest.TestCase):
    """Test PineNeedleAnatomy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.needle = PineNeedleAnatomy(randomness=1.0)
    
    def test_initialization(self):
        """Test that anatomy initializes with default layers."""
        layers = self.needle.list_layers()
        self.assertIn("epidermis", layers)
        self.assertIn("mesophyll", layers)
        self.assertIn("endodermis", layers)
    
    def test_add_remove_layer(self):
        """Test layer addition and removal."""
        initial_count = len(self.needle.layer_manager)
        
        new_layer = Layer(name="test_layer", cell_diameter=0.02, order=3)
        self.needle.add_layer(new_layer)
        
        self.assertEqual(len(self.needle.layer_manager), initial_count + 1)
        
        self.needle.remove_layer("test_layer")
        self.assertEqual(len(self.needle.layer_manager), initial_count)
    
    def test_base_shape_generation(self):
        """Test base shape creation."""
        polygon = self.needle.generate_base_shape()
        
        self.assertIsInstance(polygon, Polygon)
        self.assertTrue(polygon.is_valid)
        self.assertGreater(polygon.area, 0)
    
    def test_layer_polygon_generation(self):
        """Test layer polygon generation."""
        polygons = self.needle.generate_layer_polygons()
        
        self.assertIsInstance(polygons, list)
        self.assertGreater(len(polygons), 0)
        
        # Check that polygons get smaller (inner layers)
        areas = [p['polygon'].area for p in polygons[:5]]
        for i in range(len(areas) - 1):
            self.assertGreater(areas[i], areas[i + 1])
    
    def test_cell_generation(self):
        """Test cell generation."""
        cells_gdf = self.needle.generate_cells()
        
        self.assertIsNotNone(cells_gdf)
        self.assertGreater(len(cells_gdf), 0)
        self.assertIn('type', cells_gdf.columns)
        self.assertIn('geometry', cells_gdf.columns)
    
    def test_statistics(self):
        """Test statistics generation."""
        stats = self.needle.get_statistics()
        
        self.assertIn('total_cells', stats)
        self.assertIn('cell_types', stats)
        self.assertIn('total_area', stats)
        self.assertGreater(stats['total_cells'], 0)
        self.assertGreater(stats['total_area'], 0)
    
    def test_parameter_modification(self):
        """Test parameter modification invalidates cache."""
        # Generate once
        cells1 = self.needle.generate_cells()
        count1 = len(cells1)
        
        # Modify parameters
        self.needle.set_central_cylinder_params(transfusion_layers=5)
        
        # Regenerate
        cells2 = self.needle.generate_cells()
        count2 = len(cells2)
        
        # Should be different
        self.assertNotEqual(count1, count2)


class TestRootAnatomy(unittest.TestCase):
    """Test RootAnatomy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.root = RootAnatomy(randomness=1.0)
    
    def test_initialization(self):
        """Test that anatomy initializes with default layers."""
        layers = self.root.list_layers()
        self.assertIn("epidermis", layers)
        self.assertIn("cortex", layers)
        self.assertIn("endodermis", layers)
    
    def test_circular_shape(self):
        """Test that root has circular cross-section."""
        polygon = self.root.generate_base_shape()
        
        # A circle should have area ≈ π*r²
        # We can't test exact circularity, but can check validity
        self.assertIsInstance(polygon, Polygon)
        self.assertTrue(polygon.is_valid)
    
    def test_cell_generation(self):
        """Test cell generation for root."""
        cells_gdf = self.root.generate_cells()
        
        self.assertGreater(len(cells_gdf), 0)
        # Root should have cortex
        cortex = cells_gdf[cells_gdf['type'] == 'cortex']
        self.assertGreater(len(cortex), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_complete_workflow_needle(self):
        """Test complete workflow for pine needle."""
        needle = PineNeedleAnatomy()
        
        # Add layer
        needle.add_layer(Layer(name="test", cell_diameter=0.02, order=3))
        
        # Generate cells
        cells = needle.generate_cells()
        
        # Get statistics
        stats = needle.get_statistics()
        
        # Export
        needle.export_to_csv('/tmp/test_needle.csv')
        
        # Verify
        self.assertGreater(len(cells), 0)
        self.assertGreater(stats['total_cells'], 0)
    
    def test_complete_workflow_root(self):
        """Test complete workflow for root."""
        root = RootAnatomy()
        
        # Modify layer
        root.remove_layer("cortex")
        root.add_layer(Layer(name="cortex", cell_diameter=0.05, n_layers=2, order=4))
        
        # Generate cells
        cells = root.generate_cells()
        
        # Get statistics
        stats = root.get_statistics()
        
        # Verify
        self.assertGreater(len(cells), 0)
        self.assertEqual(stats['n_layers'], 4)  # epidermis, cortex (modified), endodermis, pericycle


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
