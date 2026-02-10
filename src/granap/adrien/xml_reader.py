import pandas as pd
from lxml import etree
import os

def ReadXML(path: str = None) -> pd.DataFrame:
    """
    Read the parameters for PyGRANAR from an XML file.
    
    Args:
        path (str): The path to the XML file with the parameters.
        
    Returns:
        pd.DataFrame: A long-format DataFrame containing parameter names, 
                      types (attributes), and numeric values.
    """
    if path is None:
        raise ValueError("No path specified for the XML parameter file.")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Parse the XML
    try:
        tree = etree.parse(path)
        root = tree.getroot()
    except Exception as e:
        raise Exception(f"Error parsing XML file: {e}")

    # Quality checks: Ensure required tags exist
    required_tags = [
        "planttype", 
        "randomness",
        "secondarygrowth",
        "stele",
        "pericycle",
        "endodermis",
        "exodermis",
        "epidermis",
        "aerenchyma",
        "cortex",
        "xylem", 
        "phloem", 
        "hair", 
        "inter_cellular_space", 
        "pith"
    ]
    
    for tag in required_tags:
        if tree.find(f".//{tag}") is None:
            print(f"Warning: Could not find the '{tag}' tag in the XML file")

    # Data collection
    data = []
    
    # Iterate through all elements that have children/attributes
    for element in tree.xpath("//*"):
        # Get tag name
        tag_name = element.tag
        
        # Get attributes (e.g., <tag param_name="1.5">)
        for attr_name, attr_value in element.attrib.items():
            data.append({
                "name": tag_name,
                "type": attr_name,
                "value": attr_value
            })

    # Create DataFrame
    params = pd.DataFrame(data)

    if params.empty:
        return params
    
    # errors='coerce' will turn non-numeric strings into NaN
    params['value'] = pd.to_numeric(params['value'], errors='coerce')

    return params

# Example usage:
# df = read_param_xml("parameters.xml")
# print(df.head())
