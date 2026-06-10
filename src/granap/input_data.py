import xml.etree.ElementTree as ET
import copy
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union
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

class InterCellularSpacesParams(BaseParams):
    name      : str             = "inter_cellular_spaces"
    tissue    : List[str]       = Field(default=["cortex", "exodermis"], title="Tissue", description="One or more tissue names to apply intercellular spaces to. Adjacent tissues in the list will have spaces generated at their shared boundary.")
    inter_cellular_space_proportion : float = Field(default=0.1, ge=0.0, le=1.0, title="Intercellular Space Proportion", description="Proportion of intercellular spaces in the tissue from 0 to 1")
    smoothness: Union[float, List[float]] = Field(default=[0.05, 0.05], title="Smoothness", description="Smoothness per tissue (0–1). Provide a single float applied to all tissues, or a list with one value per tissue.")

    @model_validator(mode="after")
    def _check_smoothness_length(self) -> "InterCellularSpacesParams":
        if isinstance(self.smoothness, list):
            if len(self.smoothness) != len(self.tissue):
                raise ValueError(
                    f"smoothness has {len(self.smoothness)} value(s) but tissue has {len(self.tissue)} entry/entries — "
                    "lengths must match, or provide a single float applied to all tissues."
                )
        return self


class AerenchymaParams(BaseParams):
    name                  : str   = "aerenchyma"
    tissue                : str   = "cortex"
    aerenchyma_proportion : float = Field(default=0.01, ge=0.0, le=1.0, title = "Aerenchyma Proportion", description = "Proportion of aerenchyma from 0 to 1")
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

# Monocotyledon-specific layers
class SteleParams(BaseParams):
    name                     : str   = "stele"
    thickness                : float = Field(default=0.27,  ge=0.00001, title="Thickness")
    cell_diameter            : float = Field(default=0.01,  ge=0.00001, title="Cell Diameter (edge)",   description="Lower asymptote of the 5PL gradient — cell diameter at the stele periphery.")
    cell_diameter_center     : float = Field(default=0.02,  ge=0.00001, title="Cell Diameter (center)", description="Upper asymptote of the 5PL gradient — cell diameter at the stele center. Set equal to cell_diameter to disable the gradient.")
    size_gradient_inflection : float = Field(default=0.3,   ge=0.001, le=1.0, title="Size Gradient Inflection", description="Normalized radial position of the 5PL inflection point (0 = center, 1 = edge).")
    size_gradient_steepness  : float = Field(default=3.0,   ge=0.1,          title="Size Gradient Steepness",   description="Hill coefficient b of the 5PL. Higher values produce a sharper size transition.")
    size_gradient_asymmetry  : float = Field(default=1.0,   ge=0.1,          title="Size Gradient Asymmetry",   description="Asymmetry exponent m of the 5PL.")


class RootXylemParams(BaseParams):
    name                    : str   = "xylem"
    cell_diameter           : float = Field(default=0.06,   ge=0.00001, title="Cell Diameter",              description="Metaxylem vessel diameter.")
    cell_diameter_sd        : float = Field(default=0.005,  ge=0.0,     title="Cell Diameter SD",           description="Standard deviation of metaxylem vessel diameter (sampled per vessel).")
    protoxylem_diameter     : float = Field(default=0.01,   ge=0.00001, title="Protoxylem Diameter",        description="Diameter of protoxylem elements.")
    protoxylem_diameter_sd  : float = Field(default=0.002,  ge=0.0,     title="Protoxylem Diameter SD",     description="Standard deviation of protoxylem element diameter.")
    n_vascular_bundles      : int   = Field(default=5,      ge=1,       title="Number of Vascular Bundles", description="Number of metaxylem vessels.")
    n_protoxylem_per_bundle : int   = Field(default=2,      ge=1,       title="Protoxylem per Bundle",      description="Number of protoxylem elements per bundle.")
    ratio_proto_meta        : float = Field(default=2.2,    ge=0.0,     title="Ratio Protoxylem/Metaxylem", description="Ratio controlling protoxylem bundle count relative to metaxylem vessels.")


class RootPhloemParams(BaseParams):
    name             : str   = "phloem"
    cell_diameter    : float = Field(default=0.005,  ge=0.00001, title="Cell Diameter",    description="Diameter of phloem sieve elements.")
    cell_diameter_sd : float = Field(default=0.001,  ge=0.0,     title="Cell Diameter SD", description="Standard deviation of phloem cell diameter.")
    n_per_bundle     : int   = Field(default=5,      ge=1,       title="Cells per Bundle", description="Number of sieve elements packed inside each phloem bundle.")

# Dicotyledon-specific layers
class SteleDicotParams(BaseParams):
    name                     : str   = "stele"
    thickness                : float = Field(default=0.5,   ge=0.00001, title="Thickness")
    cell_diameter            : float = Field(default=0.015, ge=0.00001, title="Cell Diameter (edge)",   description="Lower asymptote of the 5PL gradient — cell diameter at the stele periphery.")
    cell_diameter_center     : float = Field(default=0.03,  ge=0.00001, title="Cell Diameter (center)", description="Upper asymptote of the 5PL gradient — cell diameter at the stele center. Set equal to cell_diameter to disable the gradient.")
    size_gradient_inflection : float = Field(default=0.3,   ge=0.001, le=1.0, title="Size Gradient Inflection", description="Normalized radial position of the 5PL inflection point for stele parenchyma cell size (0 = center, 1 = edge).")
    size_gradient_steepness  : float = Field(default=3.0,   ge=0.1,          title="Size Gradient Steepness",   description="Hill coefficient b of the 5PL for stele parenchyma cell size.")
    size_gradient_asymmetry  : float = Field(default=1.0,   ge=0.1,          title="Size Gradient Asymmetry",   description="Asymmetry exponent m of the 5PL for stele parenchyma cell size.")


class DicotXylemParams(BaseParams):
    name                : str   = "xylem"
    n_vascular_peak     : int   = Field(default=3,     ge=2,       title="Number of Vascular Peaks", description="Number of xylem arms in the star pattern.")
    inner_radius        : float = Field(default=0.05,  ge=0.00001, title="Inner Radius",             description="Inner radius of the xylem star arms from the stele centre.")
    outer_radius        : float = Field(default=0.22,  ge=0.00001, title="Outer Radius",             description="Outer radius of the xylem star arms from the stele centre.")
    arc_top             : float = Field(default=0.03,  ge=0.00001, title="Arc Length at Tip",        description="Arc length of each arm at outer_radius (tip width).")
    arc_bottom          : float = Field(default=0.035,  ge=0.00001, title="Arc Length at Base",       description="Arc length of each arm at inner_radius (base width).")
    cell_diameter       : float = Field(default=0.09,  ge=0.00001, title="Cell Diameter (max)",      description="Maximum vessel diameter at the star centre (5PL upper asymptote).")
    cell_diameter_min   : float = Field(default=0.01,  ge=0.00001, title="Cell Diameter (min)",      description="Minimum vessel diameter at the star tips (5PL lower asymptote).")
    cell_diameter_sd    : float = Field(default=0.002, ge=0.0,     title="Cell Diameter SD",         description="Standard deviation added to each vessel diameter.")
    gradient_inflection : float = Field(default=0.7,   ge=0.001, le=1.0, title="Gradient Inflection", description="Normalized distance at which the 5PL inflects for vessel size (0 = centre, 1 = tip).")
    gradient_steepness  : float = Field(default=5.0,   ge=0.1,          title="Gradient Steepness",  description="Hill coefficient b of the 5PL for vessel size.")
    gradient_asymmetry  : float = Field(default=1.0,   ge=0.1,          title="Gradient Asymmetry",  description="Asymmetry exponent m of the 5PL for vessel size.")
    first_vessel_shift  : float = Field(default=0.7,   ge=0.0, le=1.0,  title="First Vessel Shift",  description="Maximum random displacement of the first vessel as a fraction of its inscribed radius.")


class DicotPhloemParams(BaseParams):
    name             : str   = "phloem"
    cell_diameter    : float = Field(default=0.005,  ge=0.00001, title="Cell Diameter",    description="Diameter of phloem sieve elements.")
    cell_diameter_sd : float = Field(default=0.001,  ge=0.0,     title="Cell Diameter SD", description="Standard deviation of phloem cell diameter.")
    width            : float = Field(default=0.15,   ge=0.00001, title="Width",            description="Width of the phloem bundle region.")
    height           : float = Field(default=0.2,    ge=0.00001, title="Height",           description="Height of the phloem bundle region.")


class DicotCambiumParams(BaseParams):
    name             : str   = "cambium"
    cell_diameter    : float = Field(default=0.015,  ge=0.00001, title="Cell Diameter",    description="Diameter of cambium cells.")
    cell_width       : float = Field(default=0.03,   ge=0.00001, title="Cell Width",       description="Width of cambium cells (tangential).")
    n_layers         : int   = Field(default=1,      ge=1,       title="Number of Layers", description="Number of cambium layers.")
    minimal_distance : float = Field(default=0.16,   ge=0.00001, title="Minimal Distance", description="Inner radius of the cambium ring from the stele centre.")
    maximal_distance : float = Field(default=0.18,   ge=0.00001, title="Maximal Distance", description="Outer clip radius of the cambium ring from the stele centre.")

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
    name      : str             = "inter_cellular_spaces"
    tissue    : List[str]       = Field(default=["mesophyll", "endodermis"], title="Tissue", description="One or more tissue names to apply intercellular spaces to. Adjacent tissues in the list will have spaces generated at their shared boundary.")
    smoothness: Union[float, List[float]] = Field(default=[0.01, 0.01], title="Smoothness", description="Smoothness per tissue (0–1). Provide a single float applied to all tissues, or a list with one value per tissue.")

    @model_validator(mode="after")
    def _check_smoothness_length(self) -> "NeedleInterCellularSpacesParams":
        if isinstance(self.smoothness, list):
            if len(self.smoothness) != len(self.tissue):
                raise ValueError(
                    f"smoothness has {len(self.smoothness)} value(s) but tissue has {len(self.tissue)} entry/entries — "
                    "lengths must match, or provide a single float applied to all tissues."
                )
        return self


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
        # Map param name to Pydantic model that holds its defaults
        _DEFAULTS_BY_NAME: Dict[str, BaseParams] = {
            "planttype":             PlantTypeParams(),
            "stele":                 SteleParams(),
            "xylem":                 RootXylemParams(),
            "phloem":                RootPhloemParams(),
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
            RootXylemParams(),
            RootPhloemParams(),
            InterCellularSpacesParams(),
            AerenchymaParams(),
            EpidermisParams(),
            ExodermisParams(),
            CortexParams(),
            EndodermisParams(),
            PericycleParams(),
        ])
    
    @classmethod
    def for_dicot_root(cls) -> "OrganInputData":
        """Return OrganInputData pre-loaded with default dicot root anatomy parameters."""
        return cls(params=[
            PlantTypeParams(value=2),
            SteleDicotParams(),
            DicotXylemParams(),
            DicotPhloemParams(),
            DicotCambiumParams(),
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
