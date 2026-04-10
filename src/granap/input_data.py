import xml.etree.ElementTree as ET
import copy
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, model_validator


# ===========================================================================
# Base config
# ===========================================================================

class BaseParams(BaseModel):
    model_config = {"validate_assignment": True}


# ===========================================================================
# Layer defaults
# ===========================================================================

class LayerDefaultParams(BaseParams):
    name                : str   = "default"
    cell_diameter_default: float = Field(default=0.01,  ge=0.00001) # No upper limit
    cell_width_default  : float = Field(default=0.01,  ge=0.00001) # No upper limit
    shift_default       : float = Field(default=0.0,   ge=0.0, le=1.0)
    n_layers_default    : int   = Field(default=1,     ge=1) # No upper limit
    order_default       : int   = Field(default=0,     ge=0) # No upper limit


# ===========================================================================
# Root anatomy defaults
# ===========================================================================

class PlantTypeParams(BaseParams):
    name  : str = "planttype"
    value : int = 1
    organ : str = "root"


class SteleParams(BaseParams):
    name               : str   = "stele"
    thickness          : float = Field(default=0.27,  ge=0.00001, title = "Thickness") # No upper limit
    cell_diameter      : float = Field(default=0.01,  ge=0.00001, title = "Cell Diameter") # No upper limit
    xylem_diameter     : float = Field(default=0.063, ge=0.00001, title = "Xylem Diameter") # No upper limit
    protoxylem_diameter: float = Field(default=0.02,  ge=0.00001, title = "Protoxylem Diameter") # No upper limit
    phloem_diameter    : float = Field(default=0.012, ge=0.00001, title = "Phloem Diameter") # No upper limit
    n_vascular_bundles : int   = Field(default=5,     ge=1, title = "Number of Vascular Bundles") # No upper limit
    ratio_proto_meta   : float = Field(default=2.2,   ge=0.0, title = "Ratio of Protoxylem to Metaxylem") # No upper limit


class InterCellularSpacesParams(BaseParams):
    name      : str   = "inter_cellular_spaces"
    tissue    : str   = "stele"
    inter_cellular_space_proportion : float = Field(default=0.1, ge=0.0, le=1.0, title = "Intercellular Space Proportion", description = "Proportion of intercellular spaces in the tissue from 0 to 1")
    smoothness: float = Field(default=0.05, ge=0.0, le=1.0, title = "Smoothness", description = "Smoothness of the inter cellular spaces from 0 to 1")


class AerenchymaParams(BaseParams):
    name                  : str   = "aerenchyma"
    tissue                : str   = "cortex"
    aerenchyma_proportion : float = Field(default=0.1, ge=0.0, le=1.0, title = "Aerenchyma Proportion", description = "Proportion of aerenchyma in the cortex from 0 to 1")
    aerenchyma_type       : int   = Field(default=1, ge=1, le=2, title = "Aerenchyma Type", description = "Type of aerenchyma to generate (1 or 2)")
    n_files               : int   = Field(default=2,   ge=1, title = "Number of Files", description = "Number of files to generate aerenchyma from")


class EpidermisParams(BaseParams):
    name         : str   = "epidermis"
    cell_diameter: float = Field(default=0.015, ge=0.00001, title = "Cell Diameter", description = "Diameter of the epidermal cells")
    n_layers     : int   = Field(default=1,     ge=1, title = "Number of Layers", description = "Number of epidermal layers")
    shift        : float = Field(default=0.5, ge=0.0, le=1.0, title = "Shift", description = "Shift of the epidermal cells from 0 to 1")
    order        : int   = Field(default=6, ge=0, title = "Order", description = "Order of the epidermal cells")


class ExodermisParams(BaseParams):
    name         : str   = "exodermis"
    cell_diameter: float = Field(default=0.03, ge=0.00001, title = "Cell Diameter", description = "Diameter of the exodermal cells")
    n_layers     : int   = Field(default=1,    ge=1, title = "Number of Layers", description = "Number of exodermal layers")
    shift        : float = Field(default=0.0, ge=0.0, le=1.0, title = "Shift", description = "Shift of the exodermal cells from 0 to 1")
    order        : int   = Field(default=5, ge=0, title = "Order", description = "Order of the exodermal cells")


class CortexParams(BaseParams):
    name         : str   = "cortex"
    cell_diameter: float = Field(default=0.04, ge=0.00001, title = "Cell Diameter", description = "Diameter of the cortical cells")
    n_layers     : int   = Field(default=5,    ge=1, title = "Number of Layers", description = "Number of cortical layers")
    shift        : float = Field(default=0.0, ge=0.0, le=1.0, title = "Shift", description = "Shift of the cortical cells from 0 to 1")
    order        : int   = Field(default=4, ge=0, title = "Order", description = "Order of the cortical cells")


class EndodermisParams(BaseParams):
    name         : str   = "endodermis"
    cell_diameter: float = Field(default=0.02,  ge=0.00001, title = "Cell Diameter", description = "Diameter of the endodermal cells")
    cell_width   : float = Field(default=0.03,  ge=0.00001, title = "Cell Width", description = "Width of the endodermal cells")
    n_layers     : int   = Field(default=1,     ge=1, title = "Number of Layers", description = "Number of endodermal layers")
    shift        : float = Field(default=0.0, ge=0.0, le=1.0, title = "Shift", description = "Shift of the endodermal cells from 0 to 1")
    order        : int   = Field(default=3, ge=0, title = "Order", description = "Order of the endodermal cells")


class PericycleParams(BaseParams):
    name         : str   = "pericycle"
    cell_diameter: float = Field(default=0.01,  ge=0.00001, title = "Cell Diameter", description = "Diameter of the pericycle cells")
    cell_width   : float = Field(default=0.009, ge=0.00001, title = "Cell Width", description = "Width of the pericycle cells")
    n_layers     : int   = Field(default=1,     ge=1, title = "Number of Layers", description = "Number of pericycle layers")
    shift        : float = Field(default=0.0, ge=0.0, le=1.0, title = "Shift", description = "Shift of the pericycle cells from 0 to 1")
    order        : int   = Field(default=2, ge=0, title = "Order", description = "Order of the pericycle cells")


# ===========================================================================
# Needle anatomy defaults
# ===========================================================================

class NeedlePlantTypeParams(BaseParams):
    name     : str   = "planttype"
    value    : int   = 3
    organ    : str   = "needle"
    width    : float = Field(default=1.8, ge=0.00001, title = "Width", description = "Width of the needle")
    thickness: float = Field(default=1.1, ge=0.00001, title = "Thickness", description = "Thickness of the needle")


class RandomnessParams(BaseParams):
    name      : str   = "randomness"
    value     : float = Field(default=1.0, ge=0.0, le=3.0, title = "Value", description = "Value of the randomness")
    smoothness: float = Field(default=0.3, ge=0.0, le=1.0, title = "Smoothness", description = "Smoothness of the randomness")


class CentralCylinderParams(BaseParams):
    name            : str   = "central_cylinder"
    shape           : str   = "half_ellipse"
    cell_diameter   : float = Field(default=0.02,  ge=0.00001, title = "Cell Diameter", description = "Diameter of the central cylinder cells")
    layer_thickness : float = Field(default=0.43,  ge=0.00001, title = "Layer Thickness", description = "Thickness of the central cylinder layers")
    layer_length    : float = Field(default=1.05,  ge=0.00001, title = "Layer Length", description = "Length of the central cylinder layers")
    vascular_width  : float = Field(default=0.15,  ge=0.00001, title = "Vascular Width", description = "Width of the vascular bundles")
    vascular_height : float = Field(default=0.2,   ge=0.00001, title = "Vascular Height", description = "Height of the vascular bundles")


class TransfusionTissueParams(BaseParams):
    name                        : str   = "transfusion_tissue"
    tracheids_diameter          : float = Field(default=0.05, ge=0.00001, title = "Tracheids Diameter", description = "Diameter of the tracheids")
    parenchyma_diameter         : float = Field(default=0.03, ge=0.00001, title = "Parenchyma Diameter", description = "Diameter of the parenchyma cells")
    transfusion_tracheids_ratio : float = Field(default=0.5,  ge=0.0,  le=1.0, title = "Transfusion Tracheids Ratio", description = "Ratio of transfusion tracheids to parenchyma cells")
    n_layers                    : int   = Field(default=2,    ge=1, title = "Number of Layers", description = "Number of transfusion tissue layers")


class XylemParams(BaseParams):
    name         : str   = "xylem"
    n_files      : int   = Field(default=10,    ge=1, title = "Number of Xylem Vessels", description = "Number of xylem vessels to generate")
    cell_diameter: float = Field(default=0.007, ge=0.00001, title = "Cell Diameter", description = "Diameter of the xylem vessels")
    n_clusters   : int   = Field(default=4,     ge=1, title = "Number of Clusters", description = "Number of clusters of xylem vessels")
    n_per_cluster: int   = Field(default=3,     ge=1, title = "Number of Vessels per Cluster", description = "Number of xylem vessels per cluster")


class PhloemParams(BaseParams):
    name         : str   = "phloem"
    n_files      : int   = Field(default=8,     ge=1, title = "Number of Phloem Bundles", description = "Number of phloem bundles to generate")
    cell_diameter: float = Field(default=0.003, ge=0.00001, title = "Cell Diameter", description = "Diameter of the phloem cells")


class CambiumParams(BaseParams):
    name         : str   = "cambium"
    cell_diameter: float = Field(default=0.002, ge=0.00001, title = "Cell Diameter", description = "Diameter of the cambium cells")


class StrasburgerCellsParams(BaseParams):
    name          : str   = "Strasburger cells"
    layer_diameter: float = Field(default=0.002, ge=0.00001, title = "Layer Diameter", description = "Diameter of the Strasburger cells")
    cell_diameter : float = Field(default=0.05,  ge=0.00001, title = "Cell Diameter", description = "Diameter of the Strasburger cells")


class ResinDuctParams(BaseParams):
    name         : str   = "resin_duct"
    diameter     : float = Field(default=0.1,  ge=0.00001, title = "Diameter", description = "Diameter of the resin duct")
    n_files      : int   = Field(default=3,    ge=1, title = "Number of Resin Ducts", description = "Number of resin ducts to generate")
    cell_diameter: float = Field(default=0.02, ge=0.00001, title = "Cell Diameter", description = "Diameter of the resin duct cells")


class NeedleInterCellularSpacesParams(BaseParams):
    name      : str   = "inter_cellular_spaces"
    tissue    : str   = "mesophyll"
    smoothness: float = Field(default=0.01, ge=0.0, le=1.0, title = "Smoothness", description = "Smoothness of the inter cellular spaces")


class NeedleAerenchymaParams(BaseParams):
    name                 : str   = "aerenchyma"
    tissue               : str   = "mesophyll"
    aerenchyma_proportion: float = Field(default=0.0, ge=0.0, le=1.0, title = "Aerenchyma Proportion", description = "Proportion of aerenchyma in the mesophyll")
    aerenchyma_type      : int   = 1
    n_files              : int   = Field(default=2,   ge=1, title = "Number of Aerenchyma", description = "Number of aerenchyma to generate")


class StomataParams(BaseParams):
    name       : str   = "stomata"
    n_files    : int   = Field(default=4,     ge=1, title = "Number of Stomata", description = "Number of stomata to generate")
    width      : float = Field(default=0.025, ge=0.00001, title = "Width", description = "Width of the stomata")
    depth      : float = Field(default=0.06,  ge=0.00001, title = "Depth", description = "Depth of the stomata")
    sub_chamber: float = Field(default=0.04,  ge=0.00001, title = "Sub Chamber", description = "Sub chamber of the stomata")


class NeedleEndodermisParams(BaseParams):
    name         : str   = "endodermis"
    cell_diameter: float = Field(default=0.02,  ge=0.00001, title = "Cell Diameter", description = "Diameter of the endodermal cells")
    cell_width   : float = Field(default=0.05,  ge=0.00001, title = "Cell Width", description = "Width of the endodermal cells")
    n_layers     : int   = Field(default=1,     ge=1, title = "Number of Layers", description = "Number of endodermal layers")
    shift        : float = Field(default=0.5, ge=0.0, le=1.0, title = "Shift", description = "Shift of the endodermal cells from 0 to 1")
    order        : int   = Field(default=3, ge=1, title = "Order", description = "Order of the endodermal cells")


class MesophyllParams(BaseParams):
    name         : str   = "mesophyll"
    cell_diameter: float = Field(default=0.08,   ge=0.00001, title = "Cell Diameter", description = "Diameter of the mesophyll cells")
    cell_width   : float = Field(default=0.045,  ge=0.00001, title = "Cell Width", description = "Width of the mesophyll cells")
    n_layers     : int   = Field(default=3,      ge=1, title = "Number of Layers", description = "Number of mesophyll layers")
    shift        : float = Field(default=0.5, ge=0.0, le=1.0, title = "Shift", description = "Shift of the mesophyll cells from 0 to 1")
    order        : int   = Field(default=4, ge=1, title = "Order", description = "Order of the mesophyll cells")


class HypodermisParams(BaseParams):
    name         : str   = "hypodermis"
    cell_diameter: float = Field(default=0.0225, ge=0.00001, title = "Cell Diameter", description = "Diameter of the hypodermal cells")
    n_layers     : int   = Field(default=2,      ge=1, title = "Number of Layers", description = "Number of hypodermal layers")
    shift        : float = Field(default=0.5, ge=0.0, le=1.0, title = "Shift", description = "Shift of the hypodermal cells from 0 to 1")
    order        : int   = Field(default=5, ge=1, title = "Order", description = "Order of the hypodermal cells")


class NeedleEpidermisParams(BaseParams):
    name         : str   = "epidermis"
    cell_diameter: float = Field(default=0.02, ge=0.00001, title = "Cell Diameter", description = "Diameter of the epidermal cells")
    n_layers     : int   = Field(default=1,    ge=1, title = "Number of Layers", description = "Number of epidermal layers")
    shift        : float = Field(default=0.5, ge=0.0, le=1.0, title = "Shift", description = "Shift of the epidermal cells from 0 to 1")
    order        : int   = Field(default=6, ge=1, title = "Order", description = "Order of the epidermal cells")


# ===========================================================================
# OrganInputData
# ===========================================================================

class OrganInputData(BaseModel):
    """
    A unified data structure for handling Organ initialization parameters
    from different sources (Python dicts, XML files, or built-in defaults).

    `params` holds either Pydantic BaseParams models (typed, validated) or
    plain dicts (from XML / from_dict_list). Use to_dict_list() to get a
    uniform list of plain dicts for downstream consumption.
    """
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    params: List[Any]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Return params as a plain list of dicts (for backward compatibility)."""
        result = []
        for p in self.params:
            if isinstance(p, BaseModel):
                result.append(p.model_dump())
            else:
                result.append(p)
        return result

    def get(self, name: str) -> Optional[Any]:
        """Retrieve a param entry by its `name` field."""
        for p in self.params:
            n = p.get("name") if isinstance(p, dict) else getattr(p, "name", None)
            if n == name:
                return p
        return None

    def set_value(self, name: str, field: str, value: Any) -> None:
        """
        Update a field on a named param entry.
        Pydantic will validate the new value automatically if validate_assignment=True.
        """
        entry = self.get(name)
        if entry is None:
            raise KeyError(f"No param with name='{name}' found.")
        if isinstance(entry, BaseModel):
            setattr(entry, field, value)  # triggers Pydantic validation
        else:
            entry[field] = value          # raw dict — no validation

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict_list(cls, dict_list: List[Dict[str, Any]]) -> "OrganInputData":
        """Create OrganInputData from a plain list of dicts (no validation)."""
        return cls(params=dict_list)

    @classmethod
    def from_xml(cls, xml_path: str) -> "OrganInputData":
        """Parse a GRANAR-style XML file into OrganInputData.

        Each parsed dict is merged with the defaults from the corresponding
        Pydantic model (if one exists for that name), so all fields are
        guaranteed to be present downstream.
        """
        # Map param name → Pydantic model that holds its defaults
        _DEFAULTS_BY_NAME: Dict[str, BaseParams] = {
            "planttype":             PlantTypeParams(),
            "stele":                 SteleParams(),
            "inter_cellular_spaces": InterCellularSpacesParams(),
            "aerenchyma":            AerenchymaParams(),
            "epidermis":             EpidermisParams(),
            "exodermis":             ExodermisParams(),
            "cortex":                CortexParams(),
            "endodermis":            EndodermisParams(),
            "pericycle":             PericycleParams(),
        }

        tree = ET.parse(xml_path)
        root = tree.getroot()
        params = []
        for child in root:
            param_dict: Dict[str, Any] = {"name": child.tag}
            for key, value in child.attrib.items():
                try:
                    param_dict[key] = float(value)
                except ValueError:
                    param_dict[key] = value
            # Fill in any missing fields from the Pydantic defaults
            if child.tag in _DEFAULTS_BY_NAME:
                defaults = _DEFAULTS_BY_NAME[child.tag].model_dump()
                missing = {k: v for k, v in defaults.items() if k not in param_dict}
                if missing:
                    lines = "\n".join(f"    {k} = {v}" for k, v in missing.items())
                    import warnings
                    warnings.warn(
                        f"[from_xml] '{child.tag}': the following fields were not found in the XML "
                        f"and have been set to their defaults:\n{lines}",
                        UserWarning, stacklevel=2,
                    )
                param_dict = {**defaults, **param_dict}
            params.append(param_dict)
        return cls(params=params)

    @classmethod
    def for_root(cls) -> "OrganInputData":
        """Return OrganInputData pre-loaded with default root anatomy parameters."""
        return cls(params=[
            PlantTypeParams(),
            SteleParams(),
            InterCellularSpacesParams(),
            AerenchymaParams(),
            EpidermisParams(),
            ExodermisParams(),
            CortexParams(),
            EndodermisParams(),
            PericycleParams(),
        ])

    @classmethod
    def for_needle(cls) -> "OrganInputData":
        """Return OrganInputData pre-loaded with default needle anatomy parameters."""
        return cls(params=[
            NeedlePlantTypeParams(),
            RandomnessParams(),
            CentralCylinderParams(),
            TransfusionTissueParams(),
            XylemParams(),
            PhloemParams(),
            CambiumParams(),
            StrasburgerCellsParams(),
            ResinDuctParams(),
            NeedleInterCellularSpacesParams(),
            NeedleAerenchymaParams(),
            StomataParams(),
            NeedleEndodermisParams(),
            MesophyllParams(),
            HypodermisParams(),
            NeedleEpidermisParams(),
        ])

    @classmethod
    def for_layer(cls) -> "OrganInputData":
        """Return OrganInputData pre-loaded with default layer parameters."""
        return cls(params=[LayerDefaultParams()])

    # ------------------------------------------------------------------
    # Validation summary  (this gives a human-readable report for dict-based params)
    # ------------------------------------------------------------------

    def validate_params(self) -> List[str]:
        """
        For raw-dict params (e.g. from XML), there is no automatic range check.
        This method returns a list of warnings for any non-Pydantic entries.
        Pydantic-backed entries are always valid by construction.
        """
        warnings = []
        for p in self.params:
            if isinstance(p, dict):
                warnings.append(
                    f"[{p.get('name', '?')}] is a raw dict — no range validation applied."
                )
        return warnings
