import xml.etree.ElementTree as ET
import copy
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union, Literal
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

class BaseShapeParams(BaseParams):
    name         : str = "base_shape"
    shape        : Literal["circle", "ellipse", "square", "rectangle", "triangle", "star"] = Field(
        default="circle", title="Base Shape",
        description="Outline of the organ cross-section. 'circle' (default) is auto-sized from the layers; box shapes use width/height; 'star' uses the inner/outer radius and arc parameters below.")
    width        : float = Field(default=0.0, ge=0.0, title="Width",
        description="Total width (x extent). 0 = auto (match the default circle's diameter). Used by ellipse/square/rectangle/triangle.")
    height       : float = Field(default=0.0, ge=0.0, title="Height",
        description="Total height (y extent). 0 = auto (match the default circle's diameter). Used by ellipse/rectangle/triangle.")
    # Star outline (mirrors the xylem star parameters).
    n_peaks      : int   = Field(default=5,    ge=2,       title="Star Peaks",        description="Number of star arms. Only used when shape='star'.")
    radius_valley_side : float = Field(default=0.4,  ge=0.00001, title="Star Valley Radius", description="Valley radius between arms. Only used when shape='star'.")
    radius_peak_side   : float = Field(default=0.6,  ge=0.00001, title="Star Peak Radius",   description="Tip (peak) radius of each arm. Only used when shape='star'.")
    arc_peak_side      : float = Field(default=0.05, ge=0.00001, title="Star Arc at Peak",   description="Arc length of each arm at radius_peak_side. Only used when shape='star'.")
    arc_valley_side    : float = Field(default=0.10, ge=0.00001, title="Star Arc at Valley", description="Arc length of each arm at radius_valley_side. Only used when shape='star'.")

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
    tissue                : Union[str, List[str]] = Field(default="cortex", title="Tissue", description="One or more tissue names to convert to aerenchyma. A list is treated as a single contiguous region (only the innermost ring of that combined region is preserved).")
    aerenchyma_proportion : float = Field(default=0.01, ge=0.0, le=1.0, title = "Aerenchyma Proportion", description = "Proportion of aerenchyma from 0 to 1")
    aerenchyma_type       : int   = Field(default=1, ge=1, le=2, title = "Aerenchyma Type", description = "Type of aerenchyma to generate (1 or 2)")
    n_files               : int   = Field(default=2,   ge=1, title = "Number of Files", description = "Number of files to generate aerenchyma from")


class EpidermisParams(BaseParams):
    name         : str   = "epidermis"
    cell_diameter: float = Field(default=0.015, ge=0.00001, title = "Cell Diameter", description = "Diameter of the epidermal cells")
    cell_width: float = Field(default=0.015, ge=0.00001, title = "Cell Width", description = "Width of the pidermal cells")
    n_layers     : int   = Field(default=1,     ge=1, title = "Number of Layers", description = "Number of epidermal layers")
    shift        : float = Field(default=0.5, ge=0.0, le=1.0, title = "Shift", description = "Shift of the epidermal cells from 0 to 1")
    order        : int   = Field(default=6, ge=0, title = "Order", description = "Order of the epidermal cells")


class ExodermisParams(BaseParams):
    name         : str   = "exodermis"
    cell_diameter: float = Field(default=0.03, ge=0.00001, title = "Cell Diameter", description = "Diameter of the exodermal cells")
    cell_width: float = Field(default=0.03, ge=0.00001, title = "Cell Width", description = "Width of the exodermal cells")
    n_layers     : int   = Field(default=1,    ge=1, title = "Number of Layers", description = "Number of exodermal layers")
    shift        : float = Field(default=0.0, ge=0.0, le=1.0, title = "Shift", description = "Shift of the exodermal cells from 0 to 1")
    order        : int   = Field(default=5, ge=0, title = "Order", description = "Order of the exodermal cells")


class CortexParams(BaseParams):
    name         : str   = "cortex"
    cell_diameter: float = Field(default=0.04, ge=0.00001, title = "Cell Diameter", description = "Diameter of the cortical cells")
    cell_width: float = Field(default=0.04, ge=0.00001, title = "Cell Width", description = "Width of the cortical cells")
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

class PhellemParams(BaseParams):
    name         : str   = "phellem"
    cell_diameter: float = Field(default=0.015,  ge=0.00001, title = "Cell Diameter", description = "Diameter of the phellem cells")
    cell_width   : float = Field(default=0.025, ge=0.00001, title = "Cell Width", description = "Width of the phellem cells")
    n_layers     : int   = Field(default=3,     ge=1, title = "Number of Layers", description = "Number of phellem layers")
    shift        : float = Field(default=0.0, ge=0.0, le=1.0, title = "Shift", description = "Shift of the phellem cells from 0 to 1")
    order        : int   = Field(default=4, ge=0, title = "Order", description = "Order of the phellem cells")

class PhellogenParams(BaseParams):
    name         : str   = "phellogen"
    cell_diameter: float = Field(default=0.01,  ge=0.00001, title = "Cell Diameter", description = "Diameter of the phellogen cells")
    cell_width   : float = Field(default=0.02, ge=0.00001, title = "Cell Width", description = "Width of the phellogen cells")
    n_layers     : int   = Field(default=1,     ge=1, title = "Number of Layers", description = "Number of phellogen layers")
    shift        : float = Field(default=0.0, ge=0.0, le=1.0, title = "Shift", description = "Shift of the phellogen cells from 0 to 1")
    order        : int   = Field(default=3, ge=0, title = "Order", description = "Order of the phellogen cells")

class PhellodermParams(BaseParams):
    name         : str   = "phelloderm"
    cell_diameter: float = Field(default=0.01,  ge=0.00001, title = "Cell Diameter", description = "Diameter of the phelloderm cells")
    cell_width   : float = Field(default=0.015, ge=0.00001, title = "Cell Width", description = "Width of the phelloderm cells")
    n_layers     : int   = Field(default=4,     ge=1, title = "Number of Layers", description = "Number of phelloderm layers")
    shift        : float = Field(default=0.0, ge=0.0, le=1.0, title = "Shift", description = "Shift of the phelloderm cells from 0 to 1")
    order        : int   = Field(default=2, ge=0, title = "Order", description = "Order of the phelloderm cells")

# Stem central ground tissue (pith).  Mirrors SteleParams: the central region
# thickness + a radial cell-size gradient.  Used by StemAnatomy (monocot ground
# tissue / dicot pith) the way the "stele" param drives the root centre.
class PithParams(BaseParams):
    name                     : str                        = "pith"
    thickness                : float                      = Field(default=0.5,        ge=0.00001,        title="Thickness")
    cavity_radius            : float                      = Field(default=0.0,        ge=0.0,            title="Medullary Cavity Radius", description="Radius (mm) of the central medullary cavity (hollow/fistular pith). 0 = solid pith of parenchyma cells; >0 hollows the centre out to this radius (the wheat-culm / bamboo case). The cavity is a true void — no cells — like a large protoxylem lacuna.")
    cell_diameter            : float                      = Field(default=0.02,       ge=0.00001,        title="Cell Diameter (edge)",        description="Cell diameter at the pith periphery (lower bound of the size gradient).")
    cell_diameter_center     : float                      = Field(default=0.05,       ge=0.00001,        title="Cell Diameter (center)",      description="Cell diameter at the pith center (upper bound). Set equal to cell_diameter to disable the gradient.")
    size_gradient_function   : Literal["five_pl", "linear"] = Field(default="five_pl",                  title="Size Gradient Function",      description="Shape function used for the radial cell-size gradient.")
    size_gradient_inflection : float                      = Field(default=0.3,        ge=0.001, le=1.0,  title="Size Gradient Inflection",    description="Normalized radial position of the gradient inflection point (0 = center, 1 = edge). Used by five_pl.")
    size_gradient_steepness  : float                      = Field(default=3.0,        ge=0.1,            title="Size Gradient Steepness",     description="Hill coefficient — sharpness of the size transition. Used by five_pl.")
    size_gradient_asymmetry  : float                      = Field(default=1.0,        ge=0.1,            title="Size Gradient Asymmetry",     description="Asymmetry exponent of the size gradient. Used by five_pl.")


# Sclerenchyma (fibres / sclereids): a structural tissue of small, densely
# packed cells.  As an *ordered layer* it forms a subepidermal / hypodermal fibre
# ring; the same tissue also appears as a bundle sheath / cap (see
# VascularBundleParams.sheath), built by the bundle machinery, not as a layer.
class SclerenchymaParams(BaseParams):
    name         : str   = "sclerenchyma"
    cell_diameter: float = Field(default=0.008, ge=0.00001, title="Cell Diameter", description="Diameter of the sclerenchyma (fibre) cells — small, thick-walled.")
    cell_width   : float = Field(default=0.008, ge=0.00001, title="Cell Width",    description="Tangential width of the sclerenchyma cells.")
    n_layers     : int   = Field(default=2,     ge=1,       title="Number of Layers", description="Number of sclerenchyma layers in the subepidermal ring.")
    shift        : float = Field(default=0.0, ge=0.0, le=1.0, title="Shift",       description="Shift of the sclerenchyma cells from 0 to 1")
    order        : int   = Field(default=5, ge=0, title="Order",                   description="Order of the sclerenchyma ring (between cortex and epidermis).")


# One vascular bundle's internal arrangement (the *topology* of xylem / phloem /
# cambium within an envelope).  Cell-level sizing (vessel / sieve / cambium cell
# diameters + gradients) is read from the reused xylem / phloem / cambium params;
# this class only says how those tissues are laid out.  Fields are grouped by the
# mode that uses them (same convention as RootXylemParams' default/arch/star).
class VascularBundleParams(BaseParams):
    name             : str = "vascular_bundle"
    # -- type ---------------------------------------------------------------
    bundle_type      : Literal["collateral", "bicollateral", "concentric"] = Field(default="collateral", title="Bundle Type", description="'collateral' = xylem inner / phloem outer (± cambium between); 'bicollateral' = phloem on both sides of the xylem; 'concentric' = one tissue rings the other (see concentric_type).")
    concentric_type  : Literal["amphivasal", "amphicribral"] = Field(default="amphivasal", title="Concentric Type", description="Concentric only. 'amphivasal' = xylem surrounds a phloem core; 'amphicribral' = phloem surrounds a xylem core.")
    has_cambium      : bool  = Field(default=True,  title="Has Cambium", description="Banded types only. True = open bundle (a fascicular cambium strip between xylem and outer phloem); False = closed (no cambium, e.g. monocots).")
    # -- envelope -----------------------------------------------------------
    width            : float = Field(default=0.12, ge=0.00001, title="Envelope Width",  description="Tangential extent of the bundle envelope (mm).")
    height           : float = Field(default=0.18, ge=0.00001, title="Envelope Height", description="Radial extent of the bundle envelope (mm).")
    shape            : Literal["ellipse", "circle"] = Field(default="ellipse", title="Envelope Shape", description="Envelope outline. 'circle' is natural for concentric bundles.")
    phloem_outward   : bool  = Field(default=True, title="Phloem Outward", description="True (default) = phloem faces the organ surface, xylem faces the centre (normal orientation); False flips it.")
    xylem_maturation : Literal["endarch", "exarch"] = Field(default="endarch", title="Xylem Maturation", description="Which pole the protoxylem sits at. 'endarch' (stems) = protoxylem toward the centre; 'exarch' (roots) = protoxylem toward the surface.")
    # -- banded layout (collateral / bicollateral): radial shares, auto-normalised
    xylem_fraction   : float = Field(default=0.5,  ge=0.0, le=1.0, title="Xylem Fraction",   description="Banded types: radial share of the envelope given to xylem.")
    phloem_fraction  : float = Field(default=0.35, ge=0.0, le=1.0, title="Phloem Fraction",  description="Banded types: radial share given to the (outer) phloem.")
    cambium_fraction : float = Field(default=0.08, ge=0.0, le=1.0, title="Cambium Fraction", description="Banded open bundles: radial share given to the fascicular cambium strip.")
    inner_phloem_fraction : float = Field(default=0.0, ge=0.0, le=1.0, title="Inner Phloem Fraction", description="Bicollateral only: radial share given to the inner phloem band.")
    inner_cambium    : bool  = Field(default=True, title="Inner Cambium", description="Bicollateral only: add a cambium strip on the inner phloem side too, so the cambium flanks both faces of the xylem (default). Set False for the textbook arrangement where only the outer face has a fascicular cambium.")
    # -- concentric layout --------------------------------------------------
    core_width       : float = Field(default=0.08, ge=0.00001, title="Core Width",  description="Concentric only: tangential extent of the inner core (mm). Size it (vs the vessel/sieve diameter) so the core holds a cluster of conducting cells, not a single one.")
    core_height      : float = Field(default=0.08, ge=0.00001, title="Core Height", description="Concentric only: radial extent of the inner core (mm). Size it (vs the vessel/sieve diameter) so the core holds a cluster of conducting cells, not a single one.")
    # -- monocot 'face' xylem (xylem_layout = 'face') -----------------------
    xylem_layout     : Literal["packed", "face"] = Field(default="packed", title="Xylem Layout", description="'packed' = many size-graded vessels (dicot / concentric); 'face' = the monocot mask — a few discrete metaxylem 'eyes' + protoxylem + optional lacuna.")
    n_metaxylem      : int   = Field(default=2, ge=1, title="Number of Metaxylem", description="Face layout: number of large metaxylem vessels (the 'eyes').")
    metaxylem_diameter    : float = Field(default=0.024, ge=0.00001, title="Metaxylem Diameter", description="Face layout: metaxylem vessel diameter (mm) — the prominent 'eyes'. Keep it under the bundle size so there is room for the protoxylem, lacuna and phloem.")
    metaxylem_diameter_sd : float = Field(default=0.003, ge=0.0,     title="Metaxylem Diameter SD", description="Face layout: SD of metaxylem diameter.")
    metaxylem_gap    : float = Field(default=0.012, ge=0.0, title="Metaxylem Gap", description="Face layout: tangential spacing between the two metaxylem eyes (mm).")
    n_protoxylem     : int   = Field(default=1, ge=0, title="Number of Protoxylem", description="Face layout: number of small protoxylem vessels toward the protoxylem pole. Default 1 (with the 2 metaxylem 'eyes' this gives the canonical monocot bundle). Each protoxylem gets its own lacuna just below it when lacuna=True.")
    protoxylem_diameter    : float = Field(default=0.008, ge=0.00001, title="Protoxylem Diameter", description="Face layout: protoxylem vessel diameter (mm) — smaller than the metaxylem eyes.")
    protoxylem_diameter_sd : float = Field(default=0.0015, ge=0.0,     title="Protoxylem Diameter SD", description="Face layout: SD of protoxylem diameter.")
    lacuna           : bool  = Field(default=False, title="Protoxylem Lacuna", description="Face layout: carve an air cavity just below each protoxylem vessel (the 'mouth' of the mask), as forms when the protoxylem tears during elongation.")
    lacuna_width     : float = Field(default=0.014,  ge=0.00001, title="Lacuna Width",  description="Face layout: tangential extent of each protoxylem lacuna (mm).")
    lacuna_height    : float = Field(default=0.011, ge=0.00001, title="Lacuna Height", description="Face layout: radial extent of each protoxylem lacuna (mm).")
    # -- sclerenchyma sheath ------------------------------------------------
    sheath           : Literal["none", "ring", "caps", "both"] = Field(default="none", title="Sclerenchyma Sheath", description="Fibre sheath around the bundle. 'ring' = full envelope ring; 'caps' = fibre caps at the two radial poles; 'both' = caps + thin ring; 'none' = no fibres, but a thin parenchyma bundle-sheath ring is still placed (every bundle gets a sheath).")
    sheath_thickness : float = Field(default=0.012, ge=0.00001, title="Sheath Thickness", description="Radial/tangential depth of the bundle sheath ring / caps (mm).")
    sclerenchyma_cell_diameter : float = Field(default=0.008, ge=0.00001, title="Sheath Cell Diameter", description="Diameter (radial) of the sclerenchyma (fibre) cells in the sheath.")
    sclerenchyma_cell_width : float = Field(default=0.008, ge=0.00001, title="Sheath Cell Width", description="Tangential width of the sclerenchyma (fibre) cells in the sheath. Raise it (with the diameter) to use fewer, larger fibres.")
    # -- ground parenchyma + phloem composition (cells that fill the bundle) -
    prop_vessel      : float = Field(default=0.55, ge=0.0, le=1.0, title="Proportion Vessels", description="Fraction of the (packed) xylem zone occupied by vessels; the rest is xylem parenchyma packed around them.")
    prop_sieve       : float = Field(default=0.45, ge=0.0, le=1.0, title="Proportion Sieve", description="Fraction of the phloem ellipse occupied by sieve elements + companion cells together; the rest is parenchyma.")
    phloem_width     : float = Field(default=0.045, ge=0.00001, title="Phloem Ellipse Width",  description="Tangential extent of the phloem ellipse (the sieve-element + companion-cell cluster) in a banded bundle (mm).")
    phloem_height    : float = Field(default=0.02, ge=0.00001, title="Phloem Ellipse Height", description="Radial extent of the phloem ellipse (mm). Keep it small relative to the bundle height to leave room for phloem_relative_distance to move the cluster.")
    phloem_relative_distance : float = Field(default=0.5, ge=0.0, le=1.0, title="Phloem Relative Distance", description="Where the phloem ellipse sits along the bundle's radial axis within the phloem region: 0 = inner edge (near the metaxylem), 1 = outer edge (near the surface). Lets the sieve elements be placed near or far from the metaxylem.")
    parenchyma_diameter : float = Field(default=0.012, ge=0.00001, title="Parenchyma Diameter", description="Diameter of the ground parenchyma cells that fill the bundle around the conducting cells (and the non-fibre bundle sheath).")
    parenchyma_width : float = Field(default=0.012, ge=0.00001, title="Parenchyma Width", description="Tangential width of the ground parenchyma cells.")
    sieve_diameter_min : float = Field(default=0.006, ge=0.00001, title="Sieve Diameter (min)", description="Lower bound of the sieve-element diameter when packing the phloem.")
    companion_diameter : float = Field(default=0.007, ge=0.00001, title="Companion Cell Diameter", description="Diameter of the companion cell placed beside each sieve element (sieve elements are far smaller than xylem vessels).")
    # -- placement ----------------------------------------------------------
    n_bundles        : int   = Field(default=8, ge=0, title="Number of Bundles", description="How many bundles: evenly spaced ring slots (dicot eustele) or scattered count (monocot atactostele).")


# Monocotyledon-specific layers
class SteleParams(BaseParams):
    name                     : str                        = "stele"
    thickness                : float                      = Field(default=0.27,       ge=0.00001,        title="Thickness")
    cell_diameter            : float                      = Field(default=0.01,       ge=0.00001,        title="Cell Diameter (edge)",        description="Cell diameter at the stele periphery (lower bound of the size gradient).")
    cell_diameter_center     : float                      = Field(default=0.02,       ge=0.00001,        title="Cell Diameter (center)",      description="Cell diameter at the stele center (upper bound). Set equal to cell_diameter to disable the gradient.")
    size_gradient_function   : Literal["five_pl", "linear"] = Field(default="five_pl",                  title="Size Gradient Function",      description="Shape function used for the radial cell-size gradient.")
    size_gradient_inflection : float                      = Field(default=0.3,        ge=0.001, le=1.0,  title="Size Gradient Inflection",    description="Normalized radial position of the gradient inflection point (0 = center, 1 = edge). Used by five_pl.")
    size_gradient_steepness  : float                      = Field(default=3.0,        ge=0.1,            title="Size Gradient Steepness",     description="Hill coefficient — sharpness of the size transition. Used by five_pl.")
    size_gradient_asymmetry  : float                      = Field(default=1.0,        ge=0.1,            title="Size Gradient Asymmetry",     description="Asymmetry exponent of the size gradient. Used by five_pl.")


class RootXylemParams(BaseParams):
    name                    : str   = "xylem"
    vessel_diameter         : float = Field(default=0.06,   ge=0.00001, title="Vessel Diameter",              description="Metaxylem vessel diameter.")
    vessel_diameter_sd      : float = Field(default=0.005,  ge=0.0,     title="Vessel Diameter SD",           description="Standard deviation of metaxylem vessel diameter (sampled per vessel).")
    protoxylem_diameter    : float = Field(default=0.01,   ge=0.00001, title="Protoxylem Diameter",        description="Diameter of protoxylem elements.")
    protoxylem_diameter_sd : float = Field(default=0.001,  ge=0.0,     title="Protoxylem Diameter SD",     description="Standard deviation of protoxylem element diameter.")
    protoxylem_cluster_width       : float = Field(default=0.015,   ge=0.00001, title="Protoxylem Bundle Width",    description="Tangential width of the protoxylem ellipse.")
    protoxylem_cluster_height      : float = Field(default=0.01,   ge=0.00001, title="Protoxylem Bundle Height",   description="Radial height of the protoxylem ellipse.")
    n_vascular_bundles     : int   = Field(default=5,      ge=1,       title="Number of Vascular Bundles", description="Number of metaxylem vessels.")
    ratio_proto_meta       : float = Field(default=2.2,    ge=0.0,     title="Ratio Protoxylem/Metaxylem", description="Ratio controlling protoxylem bundle count relative to metaxylem vessels.")
    # Arch xylem mode (xylem_shape = "arch"): an evenly-spaced ring of metaxylem
    # (circle, or radial ellipse where a vessel doesn't fit) with a stele sheath,
    # a graded protoxylem chain per pole directed to its nearest metaxylem, and
    # phloem in the valleys between poles.  The whole layout is set by just
    # outer_radius (the pericycle) and protoxylem_band_depth (the outer band).
    xylem_shape            : Literal["default", "arch", "star"] = Field(default="default", title="Xylem Shape", description="'default' = ring of discrete vessels; 'arch' = evenly-spaced metaxylem ring (circle, or radial ellipse where a vessel doesn't fit) with a stele sheath, a graded protoxylem chain per pole directed to its nearest metaxylem, and phloem in the valleys between poles; 'star' = star-shaped (actinostele) xylem — vessels packed into the arms with a radial size gradient and phloem strands in the valleys between arms (no cambium).")
    n_metaxylem            : int = Field(default=0, ge=0, title="Number of Metaxylem", description="Arch mode only. Number of metaxylem, evenly spaced in the central ring; each is a circle or, where it doesn't fit, a radial ellipse. 0 defaults to n_vascular_peak.")
    n_vascular_peak        : int   = Field(default=5,     ge=1,       title="Number of Poles",         description="Number of protoxylem poles, alternating with the phloem valleys (arch mode only).")
    outer_radius           : float = Field(default=0.15,  ge=0.00001, title="Outer Radius",            description="Radius of the pericycle side, where the poles reach; capped at the stele radius (arch mode only).")
    protoxylem_band_depth  : float = Field(default=0.0, ge=0.0, title="Protoxylem Band Depth", description="Arch mode only. Radial depth of the outer band (inward from outer_radius) holding the protoxylem chains + phloem; everything inside it is the metaxylem ring. 0 defaults to 35%% of the [pith_radius, outer_radius] span.")
    pith_radius            : float = Field(default=0.0,   ge=0.0,   title="Pith Radius",             description="Inner radius of the metaxylem ring — the central pith left free of vessels (arch mode only; 0 = ring runs to the centre).")
    vessel_diameter_min    : float = Field(default=0.01,  ge=0.00001, title="Vessel Diameter (min)",   description="Minimum metaxylem diameter floor (arch mode only).")
    allow_ellipse          : bool  = Field(default=True,  title="Allow Ellipse Vessels", description="If True, when a metaxylem sector is too narrow for a target-diameter circle, fit an area-matched radial ellipse instead of shrinking the vessel (arch mode only).")
    ellipse_max_aspect     : float = Field(default=2.0,   ge=1.0, title="Ellipse Max Aspect", description="Maximum major/minor axis ratio for ellipse metaxylem, so they don't become slivers (arch mode only).")
    protoxylem_diameter_min: float = Field(default=0.0, ge=0.0, title="Protoxylem Diameter (min)", description="Arch mode only. Smallest protoxylem diameter, at the outer (pericycle) edge of the band; protoxylem_diameter is the largest, at the inner edge. 0 defaults to 0.4 * protoxylem_diameter.")
    protoxylem_pole_width_inner: float = Field(default=0.0, ge=0.0, title="Protoxylem Pole Width (inner)", description="Arch mode only. Tangential width of each protoxylem pole at its INNER end (near the metaxylem). The pole is a tapered trapezoid between this and protoxylem_pole_width_outer. 0 defaults to 3 * protoxylem_diameter.")
    protoxylem_pole_width_outer: float = Field(default=0.0, ge=0.0, title="Protoxylem Pole Width (outer)", description="Arch mode only. Tangential width of each protoxylem pole at its OUTER end (near the pericycle / stele edge). Narrower poles leave more room for the phloem in the valleys between them. 0 defaults to 3 * protoxylem_diameter.")
    gradient_function      : Literal["five_pl", "linear"] = Field(default="five_pl", title="Gradient Function",    description="Shape function for the protoxylem size gradient across the band (arch mode only).")
    gradient_inflection    : float = Field(default=0.7,   ge=0.001, le=1.0, title="Gradient Inflection",  description="Inflection point of the protoxylem gradient (arch mode only).")
    gradient_steepness     : float = Field(default=5.0,   ge=0.1,   title="Gradient Steepness",          description="Hill coefficient of the protoxylem gradient (arch mode only).")
    gradient_asymmetry     : float = Field(default=1.0,   ge=0.1,   title="Gradient Asymmetry",          description="Asymmetry exponent of the protoxylem gradient (arch mode only).")
    # Star xylem mode (xylem_shape = "star"): a star-shaped (actinostele) xylem
    # region — vessels packed into the arms with a radial size gradient (small
    # proto-like vessels at the arm tips), phloem strands in the valleys between
    # arms.  No cambium: the phloem band is positioned relative to the xylem
    # star's valley radius.  Field names mirror the dicot xylem star; reuses
    # pith_radius / n_vascular_peak / vessel_diameter* / gradient_* / allow_ellipse
    # / ellipse_max_aspect above (the gradient params serve both arch and star).
    radius_valley_side     : float = Field(default=0.05,  ge=0.00001, title="Valley Radius",            description="Star mode only. Valley-side radius of the xylem star arms from the stele centre.")
    radius_peak_side       : float = Field(default=0.22,  ge=0.00001, title="Peak Radius",              description="Star mode only. Peak-side radius of the xylem star arms (the arm tips) from the stele centre.")
    arc_peak_side          : float = Field(default=0.03,  ge=0.00001, title="Arc Length at Peak",       description="Star mode only. Arc length of each arm at radius_peak_side (peak width).")
    arc_valley_side        : float = Field(default=0.03,  ge=0.00001, title="Arc Length at Valley",     description="Star mode only. Arc length of each arm at radius_valley_side (valley/base width).")
    enforce_gradient_min   : float = Field(default=0.0,   ge=0.0, le=1.0, title="Enforce Gradient Minimum", description="Star mode only. Radial extent in [0, 1] over which the gradient minimum is enforced: where the local gradient position t <= this value, no vessel smaller than the gradient-prescribed diameter is placed. 0 disables it, 1 enforces everywhere.")
    packing_strategy       : Literal["space", "target"] = Field(default="space", title="Packing Strategy", description="Star mode only. 'space' (default): space-first Apollonian fill. 'target': size-first gradient-driven radial fill.")
    first_vessel_shift     : float = Field(default=0.7,   ge=0.0, le=1.0,  title="First Vessel Shift",  description="Star mode only. Maximum random displacement of the first vessel as a fraction of its inscribed radius.")
    direction              : Optional[str] = Field(default="center", title="Packing Direction",         description="Star mode only. Size gradient direction: 'center' (large near centre), 'edge' (large near tips), 'middle', None (random).")


class RootPhloemParams(BaseParams):
    name             : str   = "phloem"
    sieve_diameter    : float = Field(default=0.025,  ge=0.00001, title="Sieve Diameter",         description="Diameter of phloem sieve elements.")
    sieve_diameter_sd : float = Field(default=0.001,  ge=0.0,     title="Sieve Diameter SD",      description="Standard deviation of phloem sieve diameter.")
    cluster_width     : float = Field(default=0.025,   ge=0.00001, title="Phloem Bundle Width",   description="Tangential width of the phloem ellipse.")
    cluster_height    : float = Field(default=0.025,  ge=0.00001, title="Phloem Bundle Height",  description="Radial height of the phloem ellipse.")
    relative_distance : float = Field(default=0.5,    ge=0.0, le=1.0, title="Relative Distance",     description="Relative distance of the phloem from the xylem inner radius (star mode only).")

# Dicotyledon-specific layers
class SteleDicotParams(BaseParams):
    name                     : str                        = "stele"
    thickness                : float                      = Field(default=0.65,        ge=0.00001,        title="Thickness")
    cell_diameter            : float                      = Field(default=0.015,      ge=0.00001,        title="Cell Diameter (edge)",        description="Cell diameter at the stele periphery (lower bound of the size gradient).")
    cell_diameter_center     : float                      = Field(default=0.03,       ge=0.00001,        title="Cell Diameter (center)",      description="Cell diameter at the stele center (upper bound). Set equal to cell_diameter to disable the gradient.")
    size_gradient_function   : Literal["five_pl", "linear"] = Field(default="five_pl",                  title="Size Gradient Function",      description="Shape function used for the radial cell-size gradient.")
    size_gradient_inflection : float                      = Field(default=0.2,        ge=0.001, le=1.0,  title="Size Gradient Inflection",    description="Normalized radial position of the gradient inflection point (0 = center, 1 = edge). Used by five_pl.")
    size_gradient_steepness  : float                      = Field(default=3.0,        ge=0.1,            title="Size Gradient Steepness",     description="Hill coefficient — sharpness of the size transition. Used by five_pl.")
    size_gradient_asymmetry  : float                      = Field(default=1.0,        ge=0.1,            title="Size Gradient Asymmetry",     description="Asymmetry exponent of the size gradient. Used by five_pl.")


class DicotXylemParams(BaseParams):
    name                : str   = "xylem"
    n_vascular_peak     : int   = Field(default=3,     ge=2,       title="Number of Vascular Peaks", description="Number of xylem arms in the star pattern.")
    radius_valley_side  : float = Field(default=0.1,  ge=0.00001, title="Valley Radius",            description="Valley-side radius of the xylem star arms from the stele centre.")
    radius_peak_side    : float = Field(default=0.22,  ge=0.00001, title="Peak Radius",             description="Peak-side radius of the xylem star arms from the stele centre.")
    arc_peak_side       : float = Field(default=0.03,  ge=0.00001, title="Arc Length at Peak",       description="Arc length of each arm at radius_peak_side (peak width).")
    arc_valley_side     : float = Field(default=0.05,  ge=0.00001, title="Arc Length at Valley",     description="Arc length of each arm at radius_valley_side (valley/base width).")
    vessel_diameter     : float                        = Field(default=0.08,  ge=0.00001,        title="Vessel Diameter (max)",    description="Maximum vessel diameter at the star centre (upper bound of the size gradient).")
    vessel_diameter_min : float                        = Field(default=0.02,  ge=0.00001,        title="Vessel Diameter (min)",    description="Minimum vessel diameter at the star tips (lower bound of the size gradient).")
    vessel_diameter_sd  : float                        = Field(default=0.002, ge=0.0,            title="Vessel Diameter SD",       description="Standard deviation added to each vessel diameter.")
    gradient_function   : Literal["five_pl", "linear"] = Field(default="five_pl",               title="Gradient Function",        description="Shape function used for the centre-to-tip vessel size gradient.")
    gradient_inflection : float                        = Field(default=0.7,   ge=0.001, le=1.0,  title="Gradient Inflection",     description="Normalized distance of the gradient inflection point (0 = centre, 1 = tip). Used by five_pl.")
    gradient_steepness  : float                        = Field(default=1.0,   ge=0.1,            title="Gradient Steepness",      description="Hill coefficient — sharpness of the vessel size transition. Used by five_pl.")
    gradient_asymmetry  : float                        = Field(default=1.0,   ge=0.1,            title="Gradient Asymmetry",      description="Asymmetry exponent of the vessel size gradient. Used by five_pl.")
    enforce_gradient_min: float                        = Field(default=0.0,   ge=0.0, le=1.0, title="Enforce Gradient Minimum", description="Radial extent in [0, 1] (same axis as gradient_inflection) over which the gradient minimum is enforced: where the local gradient position t <= this value, no vessel smaller than the gradient-prescribed diameter is placed (a spot too tight for the local target is left empty). 0 disables it, 1 enforces it everywhere.")
    allow_ellipse       : bool                         = Field(default=False, title="Allow Ellipse Vessels", description="If True, when a tight/elongated spot is too narrow for a target-diameter circle, fit an area-matched ellipse elongated along the spot instead of shrinking the vessel.")
    ellipse_max_aspect  : float                        = Field(default=2.0,   ge=1.0, title="Ellipse Max Aspect", description="Maximum major/minor axis ratio for ellipse vessels, so they don't become slivers.")
    packing_strategy    : Literal["space", "target"] = Field(default="space", title="Packing Strategy", description="'space' (default): space-first Apollonian fill. 'target': size-first gradient-driven radial fill (big vessels first at the gradient radius, ellipse if too narrow, then small cells fill the rest).")
    first_vessel_shift  : float = Field(default=0.7,   ge=0.0, le=1.0,  title="First Vessel Shift",  description="Maximum random displacement of the first vessel as a fraction of its inscribed radius.")
    direction           : Optional[str] = Field(default="center", title="Packing Direction", description="Size gradient direction: 'center' (large near centre), 'edge' (large near boundary), 'middle' (large at mid-radius), None (random).")
    pith_radius         : float = Field(default=0.0,   ge=0.0,   title="Pith Radius",                 description="Radius of the central pith circle subtracted from the star. 0 = no pith (star mode only).")



class DicotPhloemParams(BaseParams):
    name             : str   = "phloem"
    sieve_diameter    : float = Field(default=0.012,  ge=0.00001, title="Sieve Diameter",    description="Diameter of phloem sieve elements.")
    sieve_diameter_sd : float = Field(default=0.001,  ge=0.0,     title="Sieve Diameter SD", description="Standard deviation of phloem sieve element diameter.")
    cluster_width            : float = Field(default=0.1,   ge=0.00001, title="Cluster Width",            description="Width of the phloem bundle region.")
    cluster_height           : float = Field(default=0.05,    ge=0.00001, title="Cluster Height",           description="Height of the phloem bundle region.")
    relative_distance : float = Field(default=0.8,    ge=0.0, le = 1.0, title="Relative Distance to cambium",           description="Relative distance to cambium (0 adjacent to cambium, 1 adjacent to the last stele layer)")


class DicotCambiumParams(BaseParams):
    name             : str   = "cambium"
    cell_diameter    : float = Field(default=0.01,  ge=0.00001, title="Cell Diameter",    description="Diameter of cambium cells.")
    cell_width       : float = Field(default=0.02,   ge=0.00001, title="Cell Width",       description="Width of cambium cells (tangential).")
    # for primary growth
    visible_distance : float = Field(default=0.8,  ge=0.0, title="Primary Visible Distance", description="Maximum radius at which primary cambium is differentiated. Cambium matures first in the valleys between xylem arms. Increase toward the stele edge for a more mature (complete ring) cambium.")
    radius_valley_side : float = Field(default=0.11,  ge=0.00001, title="Primary Valley Radius",   description="Valley-side radius of the cambium ring from the stele centre at primary growth.")
    radius_peak_side   : float = Field(default=0.28,  ge=0.00001, title="Primary Peak Radius",     description="Peak-side radius of the cambium star arms from the stele centre at primary growth. Should be close to the stele radius.")
    arc_peak_side       : float = Field(default=0.05,  ge=0.00001, title="Arc Length at Peak",       description="Arc length of each arm at radius_peak_side (peak width).")
    arc_valley_side     : float = Field(default=0.07,  ge=0.00001, title="Arc Length at Valley",     description="Arc length of each arm at radius_valley_side (valley/base width).")

# Dicotyledon-specific layers
class DicotSecondarySteleParams(BaseParams):
    name                     : str                        = "stele"
    thickness                : float                      = Field(default=1,        ge=0.00001,        title="Thickness")
    cell_diameter            : float                      = Field(default=0.015,      ge=0.00001,        title="Cell Diameter (edge)",        description="Cell diameter at the stele periphery (lower bound of the size gradient).")
    cell_diameter_center     : float                      = Field(default=0.03,       ge=0.00001,        title="Cell Diameter (center)",      description="Cell diameter at the stele center (upper bound). Set equal to cell_diameter to disable the gradient.")
    size_gradient_function   : Literal["five_pl", "linear"] = Field(default="five_pl",                  title="Size Gradient Function",      description="Shape function used for the radial cell-size gradient.")
    size_gradient_inflection : float                      = Field(default=0.2,        ge=0.001, le=1.0,  title="Size Gradient Inflection",    description="Normalized radial position of the gradient inflection point (0 = center, 1 = edge). Used by five_pl.")
    size_gradient_steepness  : float                      = Field(default=3.0,        ge=0.1,            title="Size Gradient Steepness",     description="Hill coefficient — sharpness of the size transition. Used by five_pl.")
    size_gradient_asymmetry  : float                      = Field(default=1.0,        ge=0.1,            title="Size Gradient Asymmetry",     description="Asymmetry exponent of the size gradient. Used by five_pl.")


class DicotSecondaryGrowthParams(BaseParams):
    name         : str   = "secondary_growth"
    value        : bool  = True

class DicotSecondaryXylemParams(BaseParams):
    name                : str   = "secondary_xylem"
    prop_stele          : float = Field(default=0.8,  ge=0.0, le=1.0, title="Proportion of Stele",       description="Angular fraction of each valley between xylem peaks that is occupied by a vessel pizza-slice zone (0–1). 1.0 means slices tile the full circle; 0.5 means each slice is half as wide.")
    cell_diameter       : float = Field(default=0.015,  ge=0.00001, title="Cell Diameter",                description="Diameter of axial parenchyma cells that fill the non-vessel area inside each pizza-slice zone.")
    cell_width          : float = Field(default=0.015,  ge=0.00001, title="Cell Width",                   description="Tangential width of axial parenchyma cells.")
    vessel_diameter     : float = Field(default=0.1,  ge=0.00001, title="Vessel Diameter (max)",         description="Maximum secondary xylem vessel diameter (upper bound of the size gradient).")
    vessel_diameter_sd  : float = Field(default=0.005, ge=0.0,     title="Vessel Diameter SD",           description="Standard deviation added to each vessel diameter after gradient sampling.")
    vessel_diameter_min : float = Field(default=0.03,  ge=0.00001, title="Vessel Diameter (min)",        description="Minimum secondary xylem vessel diameter (lower bound of the size gradient).")
    gradient_function   : Literal["five_pl", "linear", "uniform", "gaussian"] = Field(default="five_pl", title="Gradient Function",        description="Vessel diameter distribution: five_pl/linear use a centre-to-edge gradient; uniform samples from [min, max]; gaussian samples from N((max+min)/2, sd).")
    gradient_inflection : float = Field(default=0.5,   ge=0.001, le=1.0,  title="Gradient Inflection",  description="Normalized distance of the gradient inflection point (0 = centre, 1 = tip). Used by five_pl.")
    gradient_steepness  : float = Field(default=8.0,   ge=0.1,            title="Gradient Steepness",   description="Hill coefficient — sharpness of the vessel size transition. Used by five_pl.")
    gradient_asymmetry  : float = Field(default=1.0,   ge=0.1,            title="Gradient Asymmetry",   description="Asymmetry exponent of the vessel size gradient. Used by five_pl.")
    enforce_gradient_min: float = Field(default=0.0,   ge=0.0, le=1.0, title="Enforce Gradient Minimum", description="Radial extent in [0, 1] (same axis as gradient_inflection) over which the gradient minimum is enforced: where the local gradient position t <= this value, no vessel smaller than the gradient-prescribed diameter is placed (a spot too tight for the local target is left empty). 0 disables it, 1 enforces it everywhere.")
    allow_ellipse       : bool  = Field(default=False, title="Allow Ellipse Vessels", description="If True, when a tight/elongated spot is too narrow for a target-diameter circle, fit an area-matched ellipse elongated along the spot instead of shrinking the vessel.")
    ellipse_max_aspect  : float = Field(default=2.0,   ge=1.0, title="Ellipse Max Aspect", description="Maximum major/minor axis ratio for ellipse vessels, so they don't become slivers.")
    packing_strategy    : Literal["space", "target"] = Field(default="space", title="Packing Strategy", description="'space' (default): space-first Apollonian fill. 'target': size-first gradient-driven radial fill (big vessels first at the gradient radius, ellipse if too narrow, then small cells fill the rest).")
    n_ring              : int   = Field(default=1
                                        ,     ge=1,       title="Number of Rings",              description="Number of secondary xylem growth rings to generate.")
    prop_vessel_ring    : float = Field(default=0.3,   ge=0.0, le=1.0, title="Proportion of Vessel Ring", description="Stop packing vessels when (total vessel area) / (pizza-slice zone area) reaches this fraction.")
    must_be_adjacent    : bool  = Field(default=False, title="Must Be Adjacent",                         description="If True, each new vessel circle must be tangent to at least one already-placed circle (first circle is always placed freely).")
    parenchyma_diameter    : float = Field(default=0.025,  ge=0.00001, title="Parenchyma Diameter",      description="Mean cell diameter for ray parenchyma zones (angular gaps between pizza slices).")
    parenchyma_diameter_sd : float = Field(default=0.001, ge=0.0,     title="Parenchyma Diameter SD",   description="Standard deviation of ray parenchyma cell diameter.")
    parenchyma_width_sd    : float = Field(default=0.001, ge=0.0,     title="Parenchyma Width SD",      description="Standard deviation of ray parenchyma cell width.")
    parenchyma_width       : float = Field(default=0.015,  ge=0.00001, title="Parenchyma Width",         description="Buffer distance used to shape the transition zone near pizza-slice boundaries and as the tangential cell width near the outer cambium.")

class DicotSecondaryPhloemParams(BaseParams):
    name: str = "secondary_phloem"

    # ── Zone geometry ─────────────────────────────────────────────────────────
    height: float = Field(default=0.15, ge=0.00001, title="Phloem Height",
        description="Radial thickness of the secondary phloem band, measured outward "
                    "from the secondary cambium contour. The band is the secondary "
                    "cambium polygon buffered outward by this height; the medullar and "
                    "parenchyma rays carve it into trapezes.")
    top_width: float = Field(default=0.04, ge=0.00001, title="Phloem Tip Width",
        description="Tangential arc length of each phloem trapeze at its outer (top) "
                    "edge. Smaller than the vessel-zone base width makes the arm taper "
                    "inward toward the outside; set close to the base width for a "
                    "near-rectangular arm. Only used when shape='trapeze'.")

    shape: Literal["trapeze", "band"] = Field(default="trapeze", title="Phloem Shape",
        description="'trapeze' = one tapering arm per compartment between the medullar "
                    "rays (the botanical default; uses top_width). 'band' = a single "
                    "continuous ring, the whole secondary cambium buffered outward by "
                    "height, with no per-compartment trapeze carving. Medullar-ray strips "
                    "still subdivide the band when secondary medullar rays are configured.")

    # ── Alive sieve zone ──────────────────────────────────────────────────────
    alive_distance: float = Field(default=0.1, ge=0.0, title="Alive Distance (mm)",
        description="Radial distance from the cambium boundary within which sieve "
                    "elements are alive and have companion cells. Beyond this they are dead.")

    # ── Sieve elements (same size for living and dead) ────────────────────────
    sieve_diameter:     float = Field(default=0.022, ge=0.00001, title="Sieve Diameter")
    sieve_diameter_sd:  float = Field(default=0.001, ge=0.0,     title="Sieve Diameter SD")
    sieve_diameter_min: float = Field(default=0.020, ge=0.00001, title="Sieve Diameter Min")
    prop_sieve:         float = Field(default=0.3,  ge=0.0, le=1.0, title="Sieve Proportion",
        description="Stop packing sieve circles when their area reaches this fraction of the zone.")

    # ── Companion cells (one per living sieve element) ────────────────────────
    companion_diameter: float = Field(default=0.01, ge=0.00001, title="Companion Cell Diameter")
    companion_width:    float = Field(default=0.002, ge=0.00001, title="Companion Cell Width")

    # ── Phloem parenchyma ─────────────────────────────────────────────────────
    parenchyma_diameter: float = Field(default=0.012, ge=0.00001, title="Parenchyma Diameter")
    parenchyma_width:    float = Field(default=0.012, ge=0.00001, title="Parenchyma Width")


class DicotSecondaryCambiumParams(BaseParams):
    name             : str   = "secondary_cambium"
    cell_diameter    : float = Field(default=0.01,  ge=0.00001, title="Cell Diameter",    description="Diameter of secondary cambium cells.")
    cell_width       : float = Field(default=0.02,  ge=0.00001, title="Cell Width",       description="Tangential width of secondary cambium cells.")
    n_layers         : int   = Field(default=1,     ge=1,       title="Number of Layers", description="Number of concentric cambium cell files (the cambial zone). 1 = a single ring; higher values add rings buffered inward by one cell diameter each.")

    # Cambium contour family (orthogonal to n_vascular_peak).
    shape : Literal["star", "focus_ellipse"] = Field(default="star", title="Cambium Contour",
        description="'star' = lobed cambium (radius_valley_side/radius_peak_side + arcs). 'focus_ellipse' = a "
                    "single smooth best-fit superellipse for a mature/ring-shaped secondary "
                    "cambium: the axes come from the measured profile and one exponent is "
                    "least-squares fitted to the rest (NOT an interpolation through every point).")
    profile : List[Tuple[float, float]] = Field(default_factory=list, title="Measured Contour Profile",
        description="Only used when shape='focus_ellipse'. A list of (major_pos, minor_width) "
                    "measurements in mm: distance from the centre along the major axis and "
                    "the full minor-direction width there. The widest point sets the minor "
                    "axis, the farthest point (tip, width→0) the major axis; the superellipse "
                    "exponent is best-fitted to the rest. The major axis runs along +y.")

    # For secondary growth. Unlike the xylem / primary cambium, the secondary
    # cambium star is rotated half a period, so its peaks fall in the primary
    # xylem valleys (where it produces secondary xylem and bulges outward). The
    # radii are therefore named relative to the PRIMARY XYLEM orientation:
    # radius_peak_side / arc_peak_side are on the primary-xylem-peak side (the
    # narrower, inner side), radius_valley_side / arc_valley_side on the
    # primary-xylem-valley side (the wider, bulging side).
    radius_valley_side : float = Field(default=0.45,  ge=0.00001, title="Secondary Valley Radius", description="Radius on the primary-xylem-valley side, where the secondary cambium bulges outward producing secondary xylem. Must be ≤ the stele radius.")
    radius_peak_side   : float = Field(default=0.40,  ge=0.00001, title="Secondary Peak Radius", description="Radius on the primary-xylem-peak side (the inner side of the rotated cambium star). Must exceed the primary cambium radius_peak_side so the secondary cambium encloses it.")
    arc_peak_side      : float = Field(default=0.20,  ge=0.00001, title="Arc Length at Peak",   description="Arc length at radius_peak_side (primary-xylem-peak side width of each arm).")
    arc_valley_side    : float = Field(default=0.10,  ge=0.00001, title="Arc Length at Valley",  description="Arc length at radius_valley_side (primary-xylem-valley side width of each arm).")
    

class DicotMedularRaysParams(BaseParams):
    name               : str   = "medullar_rays"
    n_medullar         : int   = Field(default=6,     ge=0,       title="Number of Medullar Rays",  description="Initial number of medullar rays present from the primary cambium. When allow_non_vascular is False they are distributed evenly within the vessel zones; when True they are distributed uniformly around the full circle. Additional rays that appear further out are set by n_medullar_rate.")
    n_medullar_rate    : float = Field(default=0.0,   ge=0.0,     title="Medullar Ray Rate",        description="Rate of new medullar rays initiated per mm of radius across the secondary-xylem annulus, so ray density increases toward the periphery (as in real wood). E.g. 50 adds ~10 new rays every 0.2 mm. 0 disables it (fixed n_medullar).")
    start_radius       : float = Field(default=0.0,   ge=0.0, le=1.0, title="New-Ray Start Radius", description="Fraction of the secondary-xylem annulus (0 = primary cambium, 1 = secondary cambium) at which rate-driven rays begin to appear.")
    start_radius_sd    : float = Field(default=0.0,   ge=0.0,     title="New-Ray Start Radius SD",  description="Per-ray random jitter on the start radius of rate-driven rays, as a fraction of the annulus span, so new rays appear gradually rather than all at once.")
    base_width         : float = Field(default=0.005, ge=0.00001, title="Base Width",               description="Constant tangential width of each medullar ray.")
    cell_diameter      : float = Field(default=0.025, ge=0.00001, title="Cell Diameter",            description="Radial diameter of medullar ray cells.")
    cell_width         : float = Field(default=0.005, ge=0.00001, title="Cell Width",               description="Tangential width of each lane within the ray (determines number of lanes = base_width / cell_width).")
    allow_non_vascular : bool  = Field(default=False,              title="Allow Non-Vascular Area", description="If True, medullar rays span the full annular zone. If False, rays are placed only within secondary xylem vessel zones.")

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
    transfusion_tracheids_ratio : float = Field(default=0.5,  ge=0.0, title = "Transfusion Tracheids Ratio", description = "Ratio of transfusion tracheids to parenchyma cells")
    n_layers                    : int   = Field(default=2,    ge=1, title = "Number of Layers", description = "Number of transfusion tissue layers")
    transfusion_type            : bool  = Field(default=False, title = "Transfusion Type", description = "If True, differentiate into tracheids and parenchyma during tessellation using type-specific cell radii")


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
    tissue               : Union[str, List[str]] = Field(default="mesophyll", title="Tissue", description="One or more tissue names to convert to aerenchyma. A list is treated as a single contiguous region (only the innermost ring of that combined region is preserved).")
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
        """Retrieve a param entry by its `name` field, or None if absent.

        Prefer ``data[name]`` / ``data.name`` for typed access with autocomplete;
        this stays for callers that want the None-on-miss behaviour.
        """
        for p in self.params:
            n = p.get("name") if isinstance(p, dict) else getattr(p, "name", None)
            if n == name:
                return p
        return None

    def names(self) -> List[str]:
        """The `name` of every param entry (the keys accepted by ``get`` / ``[]``)."""
        return [
            (p.get("name") if isinstance(p, dict) else getattr(p, "name", None))
            for p in self.params
        ]

    def _require(self, name: str) -> Any:
        """Return the param entry named ``name`` or raise a KeyError listing the
        available names."""
        entry = self.get(name)
        if entry is None:
            raise KeyError(
                f"No param named {name!r}. Available: {', '.join(map(str, self.names()))}"
            )
        return entry

    def __getitem__(self, name: str) -> Any:
        """Typed access by name: ``data['xylem'].vessel_diameter = 0.03``.

        Returns the live param entry (Pydantic model or raw dict), so assigning
        to a field of a model triggers validation.
        """
        return self._require(name)

    def __getattr__(self, name: str) -> Any:
        """Attribute access by param name: ``data.xylem.vessel_diameter = 0.03``.

        Only reached when normal attribute lookup fails, so real fields/methods
        (``params``, ``set_value``, …) always take precedence; a param whose name
        shadows one of those stays reachable via ``data[name]``.
        """
        # Avoid recursion before ``params`` exists (during pydantic init).
        try:
            params = object.__getattribute__(self, "__dict__").get("params")
        except AttributeError:
            params = None
        if params:
            for p in params:
                n = p.get("name") if isinstance(p, dict) else getattr(p, "name", None)
                if n == name:
                    return p
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute or param named {name!r}"
        )

    def set_value(self, name: str, field: str, value: Any) -> "OrganInputData":
        """Update one field on a named param entry; returns ``self`` for chaining.

        Model-backed entries validate the value (and the *field name* — an unknown
        field raises with the list of valid fields).  Raw-dict entries (e.g. from
        ``from_xml``) can't be schema-checked, so setting a field that isn't
        already present warns (it's usually a typo) but still applies.
        """
        entry = self._require(name)
        if isinstance(entry, BaseModel):
            fields = type(entry).model_fields
            if field not in fields:
                raise AttributeError(
                    f"{name!r} has no field {field!r}. "
                    f"Valid fields: {', '.join(fields)}"
                )
            setattr(entry, field, value)  # triggers Pydantic validation
        else:
            if field != "name" and field not in entry:
                warnings.warn(
                    f"set_value({name!r}, {field!r}, …): {field!r} is not an existing "
                    f"field on this raw-dict param — adding it as a new key (typo?). "
                    f"Existing fields: {', '.join(k for k in entry if k != 'name')}",
                    stacklevel=2,
                )
            entry[field] = value          # raw dict — no schema validation
        return self

    def set_values(self, name: str, **fields: Any) -> "OrganInputData":
        """Update several fields on one param entry in a single call; returns
        ``self``.  ``data.set_values('xylem', n_vascular_peak=6, outer_radius=0.16)``."""
        for field, value in fields.items():
            self.set_value(name, field, value)
        return self

    def remove_param(self, name: str) -> "OrganInputData":
        """Drop the param entry named ``name`` if present; returns ``self`` for chaining.

        A tissue whose param entry is absent is skipped at build time (e.g. a root
        with no ``secondary_phloem`` entry builds no secondary phloem), so removing
        the entry is how you opt a tissue out entirely. Removing an absent name is a
        no-op — the goal is simply "this param is not present".
        """
        self.params = [
            p for p in self.params
            if (p.get("name") if isinstance(p, dict) else getattr(p, "name", None)) != name
        ]
        return self

    def validate(self, raise_on_error: bool = False) -> List[str]:
        """Check cross-field geometry constraints that otherwise fail silently
        (an empty or distorted render rather than an error).

        Returns a list of human-readable problem descriptions (empty when the
        config is sound).  With ``raise_on_error=True`` raises ``ValueError``
        instead.  Only constraints whose params are present are checked, so it is
        safe for any organ type.
        """
        def fld(param_name: str, field: str) -> Any:
            entry = self.get(param_name)
            if entry is None:
                return None
            return entry.get(field) if isinstance(entry, dict) else getattr(entry, field, None)

        issues: List[str] = []

        # A star's inner (valley) radius must be below its outer (tip) radius.
        for pname, inner, outer, label in (
            ("xylem", "radius_valley_side", "radius_peak_side", "xylem star"),
            ("cambium", "radius_valley_side", "radius_peak_side", "primary cambium")
        ):
            lo, hi = fld(pname, inner), fld(pname, outer)
            # inner == outer is a valid degenerate star (a circle); only an
            # inverted radius (inner > outer) is broken.
            if lo is not None and hi is not None and lo > hi:
                issues.append(f"{label}: {inner} ({lo}) must be <= {outer} ({hi}).")

        # Secondary cambium must enclose the primary cambium.
        if fld("secondary_growth", "value"):
            sc_in = fld("secondary_cambium", "radius_peak_side")
            pc_out = fld("cambium", "radius_peak_side")
            if sc_in is not None and pc_out is not None and sc_in < pc_out:
                issues.append(
                    f"secondary growth: secondary_cambium.radius_peak_side ({sc_in}) must "
                    f"exceed or equal the primary cambium.radius_peak_side ({pc_out}) so it encloses "
                    "the primary cambium."
                )

        if raise_on_error and issues:
            raise ValueError(
                "Invalid anatomy configuration:\n  - " + "\n  - ".join(issues)
            )
        return issues

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
            "base_shape":            BaseShapeParams(),
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

        # Translate legacy GRANAR XML attribute names into current field names.
        # Format: { tag_name: { old_attr: new_attr, ... } }
        _ATTR_RENAMES: Dict[str, Dict[str, str]] = {
            "planttype": {
                "param": "value",
            },
            "stele": {
                "layer_diameter": "thickness",
            },
            "xylem": {
                "cell_diameter": "protoxylem_diameter",
                "max_size":      "cell_diameter",
                "n_files":       "n_vascular_bundles",
                "ratio":         "ratio_proto_meta",
            },
            "aerenchyma": {
                "proportion": "aerenchyma_proportion",
                "type":       "aerenchyma_type",
            },
        }

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # --- Pass 1: collect raw dicts with per-tag renames applied ----------
        raw: Dict[str, Dict[str, Any]] = {}   # keyed by tag name (last wins)
        ordered_tags: List[str] = []
        for child in root:
            param_dict: Dict[str, Any] = {"name": child.tag}
            for key, value in child.attrib.items():
                try:
                    param_dict[key] = float(value)
                except ValueError:
                    param_dict[key] = value
            # Rename legacy XML attribute names to current field names
            renames = _ATTR_RENAMES.get(child.tag, {})
            for old_key, new_key in renames.items():
                if old_key in param_dict:
                    param_dict[new_key] = param_dict.pop(old_key)
            raw[child.tag] = param_dict
            ordered_tags.append(child.tag)

        # --- Cross-tag merges ------------------------------------------------
        # Old GRANAR XML splits stele geometry (<stele>) and vascular element
        # parameters (<xylem>) into separate tags.  The new SteleParams
        # consolidates both.  Map <xylem> attributes into the stele dict.
        _XYLEM_TO_STELE: Dict[str, str] = {
            "cell_diameter":      "xylem_diameter",
            "n_vascular_bundles": "n_vascular_bundles",
            "ratio_proto_meta":   "ratio_proto_meta",
        }
        if "stele" in raw and "xylem" in raw:
            xylem_raw = raw["xylem"]
            for xylem_key, stele_key in _XYLEM_TO_STELE.items():
                if xylem_key in xylem_raw and stele_key not in raw["stele"]:
                    raw["stele"][stele_key] = xylem_raw[xylem_key]

        # --- Pass 2: apply Pydantic defaults and emit warnings ---------------
        import warnings
        params = []
        for tag in ordered_tags:
            param_dict = raw[tag]
            if tag in _DEFAULTS_BY_NAME:
                defaults = _DEFAULTS_BY_NAME[tag].model_dump()
                missing = {k: v for k, v in defaults.items() if k not in param_dict}
                if missing:
                    lines = "\n".join(f"    {k} = {v}" for k, v in missing.items())
                    print(
                        f"[from_xml] '{tag}': the following fields were not found in the XML "
                        f"and have been set to their defaults:\n{lines}"
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
        """Return OrganInputData pre-loaded with default dicot root anatomy parameters.

        Secondary growth is disabled by default (DicotSecondaryGrowthParams value=False).
        To enable it, call ``data.set_value("secondary_growth", "value", True)`` and
        increase the stele thickness (SteleDicotParams.thickness ≥ 1.0 is recommended
        so that the secondary cambium fits within the stele boundary).
        """
        data = cls(params=[
            PlantTypeParams(value=2),
            SteleDicotParams(),
            DicotXylemParams(),
            DicotPhloemParams(),
            DicotCambiumParams(),
            DicotSecondaryGrowthParams(value=False),
            DicotSecondaryXylemParams(),
            DicotSecondaryCambiumParams(),
            DicotSecondaryPhloemParams(),
            InterCellularSpacesParams(),
            AerenchymaParams(),
            EpidermisParams(),
            ExodermisParams(),
            CortexParams(),
            EndodermisParams(),
            PericycleParams(),
        ])

        return data
    
    @classmethod
    def for_woody_dicot(cls) -> "OrganInputData":
        """Return OrganInputData pre-loaded with default dicot root anatomy parameters.

        Secondary growth is disabled by default (DicotSecondaryGrowthParams value=False).
        To enable it, call ``data.set_value("secondary_growth", "value", True)`` and
        increase the stele thickness (SteleDicotParams.thickness ≥ 1.0 is recommended
        so that the secondary cambium fits within the stele boundary).
        """
        data = cls(params=[
            PlantTypeParams(value=2),
            SteleDicotParams(),
            DicotXylemParams(),
            DicotPhloemParams(),
            DicotCambiumParams(),
            DicotSecondaryGrowthParams(value=False),
            DicotSecondaryXylemParams(),
            DicotSecondaryCambiumParams(),
            DicotSecondaryPhloemParams(),
            PhellemParams(),
            PhellogenParams(),
            PhellodermParams(),
            DicotMedularRaysParams(),
        ])

        data.set_value("secondary_growth", "value", True)
        data.set_value("stele", "thickness", 1.2)

        return data

    @classmethod
    def for_dicot_secondary(cls) -> "OrganInputData":
        """Dicot root preset with secondary growth enabled (single growth ring).

        Convenience over :meth:`for_dicot_root` — the schema is unchanged; this
        bundles the ``set_value`` calls that make secondary growth fit and look
        sensible (the growth *case* is a choice of values, not a new param set).
        """
        data = cls.for_dicot_root()
        data.set_value("secondary_growth", "value", True)

        return data

    @classmethod
    def for_monocot_stem(cls) -> "OrganInputData":
        """Monocot stem preset (atactostele).

        Ground tissue (``pith``) with an epidermis + narrow cortex; the vascular
        bundles are scattered *through* the ground tissue with no cambium.  The
        scattered-bundle placement itself is built by ``MonocotStemAnatomy`` (a
        scaffold at present — see that class).
        """
        return cls(params=[
            PlantTypeParams(value=1, organ="stem"),
            PithParams(),
            RootXylemParams(),
            # Stem-bundle sieve elements are small (the root default 0.025 is a
            # metaxylem-scale vessel); keep them well under the metaxylem eyes.
            RootPhloemParams(sieve_diameter=0.008, sieve_diameter_sd=0.001),
            VascularBundleParams(
                bundle_type="collateral", has_cambium=False,
                xylem_layout="face", lacuna=True,
                sheath="both", n_bundles=15,
                width=0.11, height=0.17,
            ),
            SclerenchymaParams(),
            InterCellularSpacesParams(tissue=["cortex"], smoothness=0.05),
            AerenchymaParams(tissue="cortex", aerenchyma_proportion=0.0),
            EpidermisParams(),
            CortexParams(n_layers=2),
        ])

    @classmethod
    def for_dicot_stem(cls) -> "OrganInputData":
        """Dicot stem preset (eustele).

        A central pith ringed by discrete collateral vascular bundles (xylem
        inner / phloem outer / cambium between), then cortex + epidermis.  The
        bundle-ring placement is built by ``DicotStemAnatomy`` (a scaffold at
        present — see that class).
        """
        return cls(params=[
            PlantTypeParams(value=2, organ="stem"),
            PithParams(),
            # Stem-bundle vessels are smaller than the root star-xylem defaults so
            # several graded vessels fit in each bundle (big near the cambium).
            DicotXylemParams(vessel_diameter=0.045, vessel_diameter_min=0.012,
                             vessel_diameter_sd=0.004, gradient_inflection=0.5),
            DicotPhloemParams(),
            DicotCambiumParams(),
            VascularBundleParams(
                bundle_type="collateral", has_cambium=True,
                xylem_layout="packed", sheath="none", n_bundles=8,
                width=0.13, height=0.2, prop_vessel=0.7,
            ),
            InterCellularSpacesParams(tissue=["cortex"], smoothness=0.05),
            AerenchymaParams(tissue="cortex", aerenchyma_proportion=0.0),
            EpidermisParams(),
            CortexParams(n_layers=3),
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
