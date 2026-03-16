import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class OrganInputData:
    """
    A unified data structure for handling Organ initialization parameters
    from different sources (e.g., Python dict lists, XML files).
    """
    params: List[Dict[str, Any]]

    @classmethod
    def from_dict_list(cls, dict_list: List[Dict[str, Any]]) -> "OrganInputData":
        """
        Create OrganInputData directly from a list of parameter dictionaries.
        """
        return cls(params=dict_list)

    @classmethod
    def from_xml(cls, xml_path: str) -> "OrganInputData":
        """
        Parse an XML file (e.g., granar roots) into a list of parameter dictionaries.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        params = []
        for child in root:
            # Use the tag as the parameter name (e.g., 'stele', 'xylem')
            param_dict = {"name": child.tag}
            
            # Attributes hold the actual configuration values
            for key, value in child.attrib.items():
                try:
                    # Attempt to convert numeric strings to floats
                    param_dict[key] = float(value)
                except ValueError:
                    # Keep as string if it's not a number
                    param_dict[key] = value
            params.append(param_dict)
        return cls(params=params)
