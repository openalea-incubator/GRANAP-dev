import os
import sys

from granap.roi_organ import RoiOrgan
from granap.anatomy_writer import AnatomyWriter

try:
    from cramic.organ_to_domain import OrganDomain
    CRAMIC_AVAILABLE = True
except ImportError:
    CRAMIC_AVAILABLE = False


wall_details = {
    "parenchyma": 0.1,  # um
    "outerwall": 4,     # um
}

def test_manual_roi_loading():
    """
    Manual testing script for RoiOrgan behavior.
    Update the folder_path to point to a local directory with .roi files.
    """
    # ====== CONFIGURE YOUR PATH HERE ======
    folder_path = "test/inputs/Pavement_RoiSet/"  
    mm_per_pixel = 0.00273
    smooth_factor = 0.05
    # ======================================
    
    if folder_path == "path/to/rois" or not os.path.exists(folder_path):
        print(f"Please update 'folder_path' in {__file__} to a real directory.")
        return

    print(f"Loading Organ from ROIs in: {folder_path}")
    organ = RoiOrgan(folder_path=folder_path, mm_per_pixel=mm_per_pixel, smooth_factor=smooth_factor)
    organ.plot_cells()
    
    # 1. Check generated GeoDataFrame
    gdf = organ.generate_cells()
    print(f"\n--- ROI LOAD SUMMARY ---")
    print(f"Total polygons mapped: {len(gdf)}")
    
    print("\nCell distribution:")
    if 'type' in gdf.columns:
        print(gdf['type'].value_counts().to_string())
        
    print(f"Convex Hull area: {organ.generate_base_shape().area:.2f} um^2\n")

    # 2. Test writing GEO layout 
    out_dir = os.path.join(folder_path, "output_tests")
    os.makedirs(out_dir, exist_ok=True)
    
    geo_path = os.path.join(out_dir, "roi_tissue.geo")
    writer = AnatomyWriter(organ)
    writer.write_to_geo(geo_path, cell_wall_thickness=wall_details)
    print(f"[*] Anatomy GEO successfully written to: {geo_path}")
    
    # 3. Test CRAMIC Integration
    if CRAMIC_AVAILABLE:
        print("\n[*] Initializing CRAMIC OrganDomain...")
        try:
            domain = OrganDomain(
                organ, 
                cell_wall_thickness=wall_details,
                celldomain = False, 
                cell_size=10, 
                dim=2, 
                symplast=False, 
                clear=True,
                simplify_tol = 0.02
            )
            # Try to build mesh if gmsh engine is accessible
            domain.discretize()
            print("\nMesh Physical Groups Mapping:")
            print(domain.sub_domain_names)
            print("CRAMIC mapping instantiated without geometric layout faults.")
            
        except Exception as e:
            print(f"[!] Error using CRAMIC OrganDomain: {e}")
    else:
        print("\n[!] CRAMIC is not found in the environment, skipping Domain tests.")


if __name__ == "__main__":
    test_manual_roi_loading()
