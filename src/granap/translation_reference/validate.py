import CreateAnatomy
import pandas as pd
import numpy as np

def validate_granap():
    # 1. Load the R reference data
    try:
        r_df = pd.read_csv("node_r.csv")
    except FileNotFoundError:
        print("Please provide the node_r.csv file from GRANAR.")
        return

    # 2. Generate the same anatomy in Python
    # Use the EXACT same parameters as you did in R
    py_nodes = CreateAnatomy("./inst/extdata/root_monocot.xml") 

    # 3. Compare Number of Cells
    print(f"R Cells: {len(r_df)} | Python Cells: {len(py_nodes)}")
    
    # 4. Compare Coordinates (Statistical Check)
    # We check if the average distance from center is the same
    r_dist = np.sqrt(r_df['x']**2 + r_df['y']**2).mean()
    py_dist = np.sqrt(py_nodes['x']**2 + py_nodes['y']**2).mean()

    print(f"Mean distance from center - R: {r_dist:.4f} | Python: {py_dist:.4f}")

    # 5. Visual Overlap Check
    import matplotlib.pyplot as plt
    plt.scatter(r_df['x'], r_df['y'], color='blue', alpha=0.5, label='R (GRANAR)')
    plt.scatter(py_nodes['x'], py_nodes['y'], color='red', marker='x', label='Python (GRANAP)')
    plt.legend()
    plt.title("Comparison: R vs Python Cell Centers")
    plt.show()

    if np.isclose(r_dist, py_dist, rtol=1e-02):
        print("SUCCESS: The geometry logic matches within 1% tolerance.")
    else:
        print("WARNING: Significant difference detected.")
