import xml.etree.ElementTree as ET
import copy
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union, Literal
from pydantic import BaseModel, Field, create_model, model_validator


# ===========================================================================
# Base config
# ===========================================================================

class BaseParams(BaseModel):
    model_config = {"validate_assignment": True}


# ===========================================================================
# Shared param factories
# ===========================================================================
# Many params share the same field layout and differ only in their defaults, so
# the field definitions (type + constraints + GUI title/description) live once in
# these factories.  Each factory emits a normal, named Pydantic model via
# ``create_model`` — so ``EpidermisParams(...)`` / ``CortexParams(...)`` etc. are
# constructed and imported exactly as before, and ``to_dict_list()`` is unchanged.

def _layer_params(clsname: str, name: str, label: str, *, cell_diameter: float,
                  n_layers: int, shift: float, order: int,
                  cell_width: Optional[float] = None):
    """An ordered 'layer' tissue: a ring of cells peeled inward by ``n_layers``.

    Shared by every peeled layer (epidermis / exodermis / cortex / endodermis /
    pericycle / phellem / phellogen / phelloderm / sclerenchyma and the needle
    mesophyll / hypodermis / ...).  ``label`` is woven into the field descriptions
    (e.g. "Diameter of the cortical cells").  ``cell_width=None`` omits the width
    field (the needle epidermis / hypodermis have no separate tangential width).
    """
    fields: Dict[str, Any] = {
        "name": (str, name),
        "cell_diameter": (float, Field(default=cell_diameter, ge=0.00001,
            title="Cell Diameter", description=f"Diameter of the {label} cells")),
    }
    if cell_width is not None:
        fields["cell_width"] = (float, Field(default=cell_width, ge=0.00001,
            title="Cell Width", description=f"Width of the {label} cells"))
    fields["n_layers"] = (int, Field(default=n_layers, ge=1,
        title="Number of Layers", description=f"Number of {label} layers"))
    fields["shift"] = (float, Field(default=shift, ge=0.0, le=1.0,
        title="Shift", description=f"Shift of the {label} cells from 0 to 1"))
    fields["order"] = (int, Field(default=order, ge=0,
        title="Order", description=f"Order of the {label} cells"))
    return create_model(clsname, __base__=BaseParams, **fields)


def _ground_region_params(clsname: str, name: str, *, thickness: float,
                          cell_diameter: float, cell_diameter_center: float,
                          inflection: float, cavity_radius: Optional[float] = None):
    """A central ground region with a radial cell-size gradient.

    Shared by the root ``stele`` variants and the stem ``pith``: a central-region
    thickness plus a five_pl/linear size gradient from the edge (``cell_diameter``)
    to the centre (``cell_diameter_center``).  ``cavity_radius`` (pith only) adds the
    hollow/fistular medullary cavity field just after ``thickness``.
    """
    fields: Dict[str, Any] = {
        "name": (str, name),
        "thickness": (float, Field(default=thickness, ge=0.00001, title="Thickness")),
    }
    if cavity_radius is not None:
        fields["cavity_radius"] = (float, Field(default=cavity_radius, ge=0.0,
            title="Medullary Cavity Radius",
            description="Radius (mm) of the central medullary cavity (hollow/fistular pith). 0 = solid pith of parenchyma cells; >0 hollows the centre out to this radius (the wheat-culm / bamboo case). The cavity is a true void — no cells — like a large protoxylem lacuna."))
    fields["cell_diameter"] = (float, Field(default=cell_diameter, ge=0.00001,
        title="Cell Diameter (edge)", description="Cell diameter at the region periphery (lower bound of the size gradient)."))
    fields["cell_diameter_center"] = (float, Field(default=cell_diameter_center, ge=0.00001,
        title="Cell Diameter (center)", description="Cell diameter at the region center (upper bound). Set equal to cell_diameter to disable the gradient."))
    fields["size_gradient_function"] = (Literal["five_pl", "linear"], Field(default="five_pl",
        title="Size Gradient Function", description="Shape function used for the radial cell-size gradient."))
    fields["size_gradient_inflection"] = (float, Field(default=inflection, ge=0.001, le=1.0,
        title="Size Gradient Inflection", description="Normalized radial position of the gradient inflection point (0 = center, 1 = edge). Used by five_pl."))
    fields["size_gradient_steepness"] = (float, Field(default=3.0, ge=0.1,
        title="Size Gradient Steepness", description="Hill coefficient — sharpness of the size transition. Used by five_pl."))
    fields["size_gradient_asymmetry"] = (float, Field(default=1.0, ge=0.1,
        title="Size Gradient Asymmetry", description="Asymmetry exponent of the size gradient. Used by five_pl."))
    return create_model(clsname, __base__=BaseParams, **fields)


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
    shape        : Literal["circle", "ellipse", "square", "rectangle", "triangle", "star", "focus_ellipse"] = Field(
        default="circle", title="Base Shape",
        description="Outline of the organ cross-section. 'circle' (default) is auto-sized from the layers; box shapes use width/height; 'star' uses the inner/outer radius and arc parameters below; 'focus_ellipse' is a superellipse (measured profile or width/height + exponent).")
    width        : float = Field(default=0.0, ge=0.0, title="Width",
        description="Total width (x extent). 0 = auto (match the default circle's diameter). Used by ellipse/square/rectangle/triangle/focus_ellipse.")
    height       : float = Field(default=0.0, ge=0.0, title="Height",
        description="Total height (y extent). 0 = auto (match the default circle's diameter). Used by ellipse/rectangle/triangle/focus_ellipse.")
    # Star outline (mirrors the xylem star parameters).
    n_peaks      : int   = Field(default=5,    ge=2,       title="Star Peaks",        description="Number of star arms. Only used when shape='star'.")
    radius_valley_side : float = Field(default=0.4,  ge=0.00001, title="Star Valley Radius", description="Valley radius between arms. Only used when shape='star'.")
    radius_peak_side   : float = Field(default=0.6,  ge=0.00001, title="Star Peak Radius",   description="Tip (peak) radius of each arm. Only used when shape='star'.")
    arc_peak_side      : float = Field(default=0.05, ge=0.00001, title="Star Arc at Peak",   description="Arc length of each arm at radius_peak_side. Only used when shape='star'.")
    arc_valley_side    : float = Field(default=0.10, ge=0.00001, title="Star Arc at Valley", description="Arc length of each arm at radius_valley_side. Only used when shape='star'.")
    # Focus-ellipse (superellipse) outline.
    profile      : List[Tuple[float, float]] = Field(default_factory=list, title="Measured Contour Profile",
        description="shape='focus_ellipse' only (preferred): a list of (major_pos, minor_width) mm measurements best-fitted to one superellipse (major axis along +y). Empty = use width/height + exponent instead.")
    exponent     : float = Field(default=4.0, gt=0.0, title="Focus-Ellipse Exponent",
        description="shape='focus_ellipse' with no profile: superellipse fullness (2 = plain ellipse, >2 = fuller/blunter flanks).")

class InterCellularSpacesParams(BaseParams):
    name      : str             = "inter_cellular_spaces"
    tissue    : List[str]       = Field(default=["cortex", "exodermis"], title="Tissue", description="One or more tissue names to apply intercellular spaces to. Adjacent tissues in the list will have spaces generated at their shared boundary.")
    inter_cellular_space_proportion : float = Field(default=0.1, ge=0.0, le=1.0, title="Intercellular Space Proportion", description="Proportion of intercellular spaces in the tissue from 0 to 1")
    smoothness: Union[float, List[float]] = Field(default=[0.05, 0.05], title="Smoothness", description="Smoothness per tissue (0-1). Provide a single float applied to all tissues, or a list with one value per tissue.")

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


EpidermisParams  = _layer_params("EpidermisParams",  "epidermis",  "epidermal",  cell_diameter=0.015, cell_width=0.015, n_layers=1, shift=0.5, order=6)
ExodermisParams  = _layer_params("ExodermisParams",  "exodermis",  "exodermal",  cell_diameter=0.03,  cell_width=0.03,  n_layers=1, shift=0.0, order=5)
CortexParams     = _layer_params("CortexParams",     "cortex",     "cortical",   cell_diameter=0.04,  cell_width=0.04,  n_layers=5, shift=0.0, order=4)
EndodermisParams = _layer_params("EndodermisParams", "endodermis", "endodermal", cell_diameter=0.02,  cell_width=0.03,  n_layers=1, shift=0.0, order=3)
PericycleParams  = _layer_params("PericycleParams",  "pericycle",  "pericycle",  cell_diameter=0.01,  cell_width=0.009, n_layers=1, shift=0.0, order=2)
PhellemParams    = _layer_params("PhellemParams",    "phellem",    "phellem",    cell_diameter=0.015, cell_width=0.025, n_layers=3, shift=0.0, order=4)
PhellogenParams  = _layer_params("PhellogenParams",  "phellogen",  "phellogen",  cell_diameter=0.01,  cell_width=0.02,  n_layers=1, shift=0.0, order=3)
PhellodermParams = _layer_params("PhellodermParams", "phelloderm", "phelloderm", cell_diameter=0.01,  cell_width=0.015, n_layers=4, shift=0.0, order=2)

# Stem central ground tissue (pith).  Mirrors SteleParams: the central region
# thickness + a radial cell-size gradient (plus the optional hollow medullary
# cavity).  Used by StemAnatomy (monocot ground tissue / dicot pith) the way the
# "stele" param drives the root centre.
PithParams = _ground_region_params("PithParams", "pith",
    thickness=0.8, cell_diameter=0.01, cell_diameter_center=0.03, inflection=0.3,
    cavity_radius=0.0)


# Sclerenchyma (fibres / sclereids): a structural tissue of small, densely
# packed cells.  As an *ordered layer* it forms a subepidermal / hypodermal fibre
# ring; the same tissue also appears as a bundle sheath / cap (see
# VascularBundleParams.sheath), built by the bundle machinery, not as a layer.
SclerenchymaParams = _layer_params("SclerenchymaParams", "sclerenchyma", "sclerenchyma (fibre)",
                                   cell_diameter=0.008, cell_width=0.008, n_layers=2, shift=0.0, order=5)


# One vascular bundle's internal arrangement (the *topology* of xylem / phloem /
# cambium within an envelope).  Cell-level sizing (vessel / sieve / cambium cell
# diameters + gradients) is read from the reused xylem / phloem / cambium params;
# this class only says how those tissues are laid out.  Fields are grouped by the
# mode that uses them (same convention as RootXylemParams' default/arch/star).
class VascularBundleParams(BaseParams):
    name             : str = "vascular_bundle"
    # -- kind (dicot eustele pattern) ---------------------------------------
    kind             : str = Field(default="", title="Bundle Kind", description="Dicot eustele only. A label identifying this bundle spec (e.g. 'big', 'small') so a bundle_pattern can place several bundle geometries around one ring. '' (default) is the single-kind legacy behaviour — the ring is filled with n_bundles copies of this spec.")
    # -- stele arrangement (dicot stem only) --------------------------------
    arrangement      : Literal["fascicular", "continuous"] = Field(default="fascicular", title="Stele Arrangement", description="Dicot stem only. 'fascicular' (default) = the discrete-bundle eustele built from this spec (Helianthus, most herbaceous dicots). 'continuous' = a non-fascicular vascular cylinder — an uninterrupted ring of xylem / cambium / phloem built from the separate vascular_cylinder spec instead (Linum, Ricinus, rapidly-woody dicots); the bundle fields below (n_bundles, envelope, layout) are then ignored. Ignored under monocot (scattered) and when secondary_growth is on.")
    # -- type ---------------------------------------------------------------
    bundle_type      : Literal["collateral", "bicollateral", "concentric"] = Field(default="collateral", title="Bundle Type", description="'collateral' = xylem inner / phloem outer (+/- cambium between); 'bicollateral' = phloem on both sides of the xylem; 'concentric' = one tissue rings the other (see concentric_type).")
    concentric_type  : Literal["amphivasal", "amphicribral"] = Field(default="amphivasal", title="Concentric Type", description="Concentric only. 'amphivasal' = xylem surrounds a phloem core; 'amphicribral' = phloem surrounds a xylem core.")
    has_cambium      : bool  = Field(default=True,  title="Has Cambium", description="Banded types only. True = open bundle (a fascicular cambium strip between xylem and outer phloem); False = closed (no cambium, e.g. monocots).")
    # -- envelope -----------------------------------------------------------
    width            : float = Field(default=0.12, ge=0.00001, title="Envelope Width",  description="Tangential extent of the bundle envelope (mm).")
    height           : float = Field(default=0.18, ge=0.00001, title="Envelope Height", description="Radial extent of the bundle envelope (mm).")
    shape            : Literal["ellipse", "circle", "focus_ellipse", "egg"] = Field(default="ellipse", title="Envelope Shape", description="Envelope outline. 'circle' is natural for concentric bundles; 'focus_ellipse' is a superellipse (fuller/pointier flanks via focus_exponent); 'egg' is a teardrop whose wider lobe is offset toward one pole via egg_waist.")
    focus_exponent   : float = Field(default=4.0, ge=0.5, title="Focus Ellipse Exponent", description="shape='focus_ellipse' only: superellipse exponent. 2 = classic ellipse; >2 = fuller/blunter flanks; <2 = pointier toward a diamond. Keeps the width/height bounding box fixed.")
    egg_waist        : float = Field(default=0.6, gt=0.0, lt=1.0, title="Egg Waist", description="shape='egg' only: radial fraction of the envelope on the outer (phloem-pole) side of the widest point. 0.5 = symmetric; >0.5 puts the fatter lobe toward the organ surface (phloem), <0.5 toward the centre (protoxylem).")
    phloem_outward   : bool  = Field(default=True, title="Phloem Outward", description="True (default) = phloem faces the organ surface, xylem faces the centre (normal orientation); False flips it.")
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
    # Layout (inner=centre-facing pole -> outer=surface-facing pole):
    #   [ lacuna | protoxylem region(s) ] .... metaxylem vessel(s) at the middle
    #   .... phloem cluster (outer half).  Metaxylem is placed first (single
    #   vessels at the radial centre); the protoxylem bundle(s) + optional lacuna
    #   sit in the inner half; the phloem cluster sits in the outer half.
    xylem_layout     : Literal["packed", "files", "face"] = Field(default="packed", title="Xylem Layout", description="'packed' = many size-graded vessels in one open zone (concentric / generic); 'files' = the realistic dicot arrangement — thin radial parenchyma strips split the xylem into endarch radial files (small protoxylem inner, large metaxylem toward the cambium), vessels parenchyma-separated; 'face' = the monocot mask — metaxylem at the radial middle, a protoxylem bundle (+ optional lacuna) toward the centre, and the phloem cluster toward the surface.")
    n_xylem_files    : int   = Field(default=3, ge=1, title="Number of Xylem Files", description="xylem_layout='files' only: number of radial vessel files the xylem is split into by thin parenchyma strips. 1 = a single open compartment (endarch gradient, no strips).")
    xylem_file_jitter : float = Field(default=0.3, ge=0.0, title="Xylem File Jitter", description="xylem_layout='files' only: how much the radial file strips are perturbed in position and angle (0 = a rigid, evenly-spaced grid; ~0.3 = a light, natural irregularity).")
    n_metaxylem      : int   = Field(default=2, ge=1, title="Number of Metaxylem", description="Face layout: number of metaxylem vessels placed at the radial middle. Each is a single vessel (not a packed region); the canonical monocot bundle has 2.")
    metaxylem_diameter    : float = Field(default=0.04, ge=0.00001, title="Metaxylem Diameter", description="Metaxylem vessel diameter (mm). Keep it under the bundle size so there is room for the protoxylem, lacuna and phloem.")
    metaxylem_diameter_sd : float = Field(default=0.003, ge=0.0,     title="Metaxylem Diameter SD", description="SD of metaxylem diameter (per-vessel size jitter).")
    metaxylem_diameter_min : float = Field(default=0.02, ge=0.00001, title="Metaxylem Diameter (min)", description="Lower clip on the jittered metaxylem diameter.")
    metaxylem_gap    : float = Field(default=0.04, ge=0.0, title="Metaxylem Gap", description="Tangential spacing between the metaxylem vessels (mm).")
    n_protoxylem     : int   = Field(default=1, ge=0, title="Number of Protoxylem", description="Number of protoxylem bundles in the inner half. Each is a small region packed with protoxylem vessels; the bundles are spread tangentially. Default 1.")
    protoxylem_diameter    : float = Field(default=0.03, ge=0.00001, title="Protoxylem Diameter", description="Protoxylem vessel diameter (mm) when packing a protoxylem bundle — smaller than the metaxylem.")
    protoxylem_diameter_sd : float = Field(default=0.00001, ge=0.0,     title="Protoxylem Diameter SD", description="SD of protoxylem vessel diameter.")
    protoxylem_diameter_min : float = Field(default=0.025, ge=0.00001, title="Protoxylem Diameter (min)", description="Lower clip on the packed protoxylem vessel diameter.")
    protoxylem_width  : float = Field(default=0.032, ge=0.00001, title="Protoxylem Bundle Width",  description="Tangential extent of each protoxylem bundle region (mm) — the region packed with protoxylem vessels.")
    protoxylem_height : float = Field(default=0.032, ge=0.00001, title="Protoxylem Bundle Height", description="Radial extent of each protoxylem bundle region (mm).")
    protoxylem_relative_distance : float = Field(default=0.6, ge=0.0, le=1.0, title="Protoxylem Relative Distance", description="Where the protoxylem bundle sits along the inner (centre-facing) half of the bundle: 0 = bundle centre (near the metaxylem), 1 = inner edge (near the organ centre).")
    lacuna           : bool  = Field(default=True, title="Protoxylem Lacuna", description="Carve an air cavity adjacent to the protoxylem, on its inner (centre-facing) side, as forms when the protoxylem tears during elongation. The lacuna and protoxylem together are kept within the inner half of the bundle.")
    lacuna_width     : float = Field(default=0.03,  ge=0.00001, title="Lacuna Width",  description="Tangential extent of each protoxylem lacuna (mm).")
    lacuna_height    : float = Field(default=0.02, ge=0.00001, title="Lacuna Height", description="Radial extent of each protoxylem lacuna (mm).")
    # -- sclerenchyma sheath ------------------------------------------------
    sheath           : Literal["none", "ring", "caps", "both"] = Field(default="none", title="Sclerenchyma Sheath", description="Fibre sheath around the bundle. 'ring' = full envelope ring; 'caps' = fibre caps at the two radial poles; 'both' = caps + thin ring; 'none' = no fibres, but a thin parenchyma bundle-sheath ring is still placed (every bundle gets a sheath).")
    sheath_thickness : float = Field(default=0.0055, ge=0.00001, title="Sheath Thickness", description="Radial/tangential depth of the bundle sheath ring / caps (mm).")
    n_caps_layers_outward : int = Field(default=0, ge=0, title="Fibre Cap Layers (outward pole)", description="Asymmetric sclerenchyma cap on the OUTWARD (surface-facing) radial pole — the phloem pole when phloem_outward=True. The cap extends the bundle outside its envelope by this many fibre-cell layers (depth = n × sclerenchyma_cell_diameter), hugging the pole contour and tapering toward the flanks. 0 = no outward cap. Independent of the symmetric 'caps'/'ring' sheath; use a different count here than n_caps_layers_inward for an asymmetric cap (e.g. a fibre cap only over the phloem).")
    n_caps_layers_inward  : int = Field(default=0, ge=0, title="Fibre Cap Layers (inward pole)", description="Asymmetric sclerenchyma cap on the INWARD (centre-facing) radial pole — the xylem pole when phloem_outward=True. Extends the bundle outside its envelope by this many fibre-cell layers (depth = n × sclerenchyma_cell_diameter). 0 = no inward cap. See n_caps_layers_outward.")
    sclerenchyma_cell_diameter : float = Field(default=0.005, ge=0.00001, title="Sheath Cell Diameter", description="Diameter (radial) of the sclerenchyma (fibre) cells in the sheath.")
    sclerenchyma_cell_width : float = Field(default=0.005, ge=0.00001, title="Sheath Cell Width", description="Tangential width of the sclerenchyma (fibre) cells in the sheath. Raise it (with the diameter) to use fewer, larger fibres.")
    outer_sheath     : bool  = Field(default=True, title="Outer Bundle Sheath", description="Wrap the bundle in one extra file of 'bundle sheath' cells just outside the envelope, sized at the mean of the (inner) sheath cell and the surrounding ground-tissue cell — a transition layer between the bundle and the ground tissue. Set False to omit it.")
    outer_sheath_clearance : float = Field(default=0.5, ge=0.0, title="Outer Sheath Clearance", description="How much further out (beyond the sheath ring) the removal mask clears the surrounding ground cells, in ground-cell diameters. Raise it if the neighbouring ground cells are cut too harshly against the sheath; 0 = mask stops at the sheath ring.")
    # -- ground parenchyma + phloem composition (cells that fill the bundle) -
    prop_vessel      : float = Field(default=0.5, ge=0.0, le=1.0, title="Proportion Vessels", description="Fraction of the (packed) xylem zone occupied by vessels; the rest is xylem parenchyma packed around them.")
    prop_sieve       : float = Field(default=0.5, ge=0.0, le=1.0, title="Proportion Sieve", description="Fraction of the phloem ellipse occupied by sieve elements + companion cells together; the rest is parenchyma.")
    phloem_width     : float = Field(default=0.045, ge=0.00001, title="Phloem Ellipse Width",  description="Tangential extent of the phloem ellipse (the sieve-element + companion-cell cluster) in a banded bundle (mm).")
    phloem_height    : float = Field(default=0.035, ge=0.00001, title="Phloem Ellipse Height", description="Radial extent of the phloem ellipse (mm). Keep it small relative to the bundle height to leave room for phloem_relative_distance to move the cluster.")
    phloem_relative_distance : float = Field(default=0.5, ge=0.0, le=1.0, title="Phloem Relative Distance", description="Where the phloem ellipse sits along the bundle's radial axis within the phloem region: 0 = inner edge (near the metaxylem), 1 = outer edge (near the surface). Lets the sieve elements be placed near or far from the metaxylem.")
    parenchyma_diameter : float = Field(default=0.008, ge=0.00001, title="Parenchyma Diameter", description="Diameter of the ground parenchyma cells that fill the bundle around the conducting cells (and the non-fibre bundle sheath).")
    parenchyma_width : float = Field(default=0.008, ge=0.00001, title="Parenchyma Width", description="Tangential width of the ground parenchyma cells.")
    sieve_diameter_min : float = Field(default=0.006, ge=0.00001, title="Sieve Diameter (min)", description="Lower bound of the sieve-element diameter when packing the phloem.")
    companion_cell_diameter : float = Field(default=0.004, ge=0.00001, title="Companion Cell Diameter", description="Radial extent of the companion cell placed beside each sieve element (sieve elements are far smaller than xylem vessels).")
    companion_cell_width : float = Field(default=0.004, ge=0.00001, title="Companion Cell Width", description="Tangential extent of the companion cell placed beside each sieve element.")
    # -- eustele ring shape (dicot / ring-monocot) --------------------------
    # The ring bundles are placed along a drawn contour — the cambium ring. Each
    # bundle is oriented along the contour's outward normal with its *cambium*
    # sitting on the contour, so in secondary growth the fascicular cambia join
    # into one continuous ring (see DicotStemAnatomy._build_cambium_ring).
    ring_shape       : Literal["circle", "ellipse", "star"] = Field(default="circle", title="Cambium Ring Shape", description="Dicot eustele: outline the bundle ring follows. 'circle' = the pith/cortex boundary (radius auto-derived); 'ellipse' = that circle flattened by ring_ellipse_ratio; 'star' = a lobed ring set by the same absolute peak/valley radii + arcs as the root xylem/cambium star (below).")
    ring_ellipse_ratio : float = Field(default=0.75, gt=0.0, le=1.0, title="Ring Ellipse Ratio", description="ring_shape='ellipse' only: height/width of the ring ellipse (1 = circle, <1 = flattened vertically).")
    # Star ring (ring_shape='star'): the SAME parameterisation as the root xylem /
    # cambium and the stem secondary cambium — absolute radii (mm from the organ
    # centre) and arc lengths, not a derived-radius + amplitude shorthand.
    n_peaks            : int   = Field(default=5,    ge=2,       title="Star Peaks",          description="ring_shape='star' only: number of lobes (arms) of the bundle ring.")
    radius_peak_side   : float = Field(default=0.4,  ge=0.00001, title="Star Peak Radius",    description="ring_shape='star' only: arm-tip (peak) radius of the bundle ring, mm from the organ centre (absolute, like the root xylem star).")
    radius_valley_side : float = Field(default=0.34, ge=0.00001, title="Star Valley Radius",  description="ring_shape='star' only: valley radius between the arms, mm from the organ centre.")
    arc_peak_side      : float = Field(default=0.12, ge=0.00001, title="Star Arc at Peak",    description="ring_shape='star' only: arc length of each arm at radius_peak_side (arm-tip width).")
    arc_valley_side    : float = Field(default=0.10, ge=0.00001, title="Star Arc at Valley",  description="ring_shape='star' only: arc length of each arm at radius_valley_side (valley width).")
    # -- placement ----------------------------------------------------------
    n_bundles        : int   = Field(default=8, ge=0, title="Number of Bundles", description="How many bundles: evenly spaced ring slots (dicot eustele) or scattered count (monocot atactostele).")
    # Radial band + spacing (monocot atactostele only). A stem may carry several
    # vascular_bundle specs, each owning an annulus [radius_min, radius_max) so the
    # bundle kind can vary with radial distance; a single default spec (0, 0) fills
    # the whole ground tissue with one kind.
    radius_min       : float = Field(default=0.0, ge=0.0, title="Band Inner Radius", description="Monocot atactostele, placement 'random'/'spaced' only. Inner radius (mm from the stem centre) of the annulus this bundle spec fills; a scattered bundle uses the spec whose band contains its centre. Default 0. (placement='even' ignores this — it uses the single 'radius' instead.)")
    radius_max       : float = Field(default=0.0, ge=0.0, title="Band Outer Radius", description="Monocot atactostele, placement 'random'/'spaced' only. Outer radius (mm) of the annulus; 0 = unbounded (out to the pith edge). May exceed the pith radius to place bundles in the cortex/rind (clamped to the epidermis). A single spec at (0, 0) makes the whole ground tissue one kind — the default. (placement='even' ignores this — it uses the single 'radius' instead.)")
    radius           : float = Field(default=0.0, ge=0.0, title="Ring Radius", description="Monocot atactostele, placement 'even' only. The single radius (mm from the stem centre) of the evenly-spaced ring the bundles sit on. May exceed the pith radius to place the ring in the cortex/rind (clamped to the epidermis). 0 = fall back to half the pith radius.")
    placement        : Literal["random", "even", "spaced"] = Field(default="spaced", title="Placement Method", description="Monocot atactostele only. 'random' = bundles scattered by rejection sampling (non-overlapping, but can clump and leave uneven gaps); 'spaced' = best-candidate sampling — each bundle is placed at the roomiest spot (farthest from those already placed), so the band fills evenly while staying irregular/natural; 'even' = bundles equally spaced on a ring at the single 'radius' (circumference / n_bundles apart).")
    n_candidates     : int   = Field(default=8, ge=1, title="Placement Candidates", description="placement='spaced' only: how many candidate spots each bundle weighs before taking the roomiest. Higher = more even spacing (bundles pushed harder into the biggest gaps) but may fit fewer than n_bundles in a tight band; lower = closer to 'random' and fits more. 1 is equivalent to 'random'.")
    angle            : float = Field(default=0.0, title="Even Placement Angle", description="Monocot 'even' placement only (degrees). Angular phase of the evenly-spaced ring: rotates all slots by this offset. Give two close-radius bands a half-step offset (180/n_bundles) to interleave a bundle of one band between each bundle of the other. Ignored for 'random'.")


# A repeating angular pattern of bundle *kinds* around the dicot eustele ring.
# The kinds are separate ``vascular_bundle`` specs (each with its own geometry),
# identified by their ``kind`` label; ``sequence`` gives their order within one
# repeat and ``repeats`` tiles that group around the ring.  Present this param
# only for a mixed-kind eustele — with no ``bundle_pattern`` the ring is the plain
# n_bundles copies of the single ``vascular_bundle`` spec.
class BundlePatternParams(BaseParams):
    name          : str       = "bundle_pattern"
    sequence      : List[str] = Field(default_factory=list, title="Bundle Kind Sequence", description="Bundle 'kind' labels for one repeat, in angular order (e.g. ['big', 'small', 'small']). Each label must match a vascular_bundle spec's 'kind'. The whole tiled sequence is placed at equal angular spacing, so each kind is equidistant from its next occurrence and the others fall evenly between.")
    repeats       : int       = Field(default=1, ge=1, title="Pattern Repeats", description="How many times the sequence is tiled around the ring. sequence=['big','small','small'] with repeats=4 places 12 bundles, the 'big' ones equally spaced with two 'small' evenly between each pair.")
    spacing       : Literal["distance", "angle", "grouped"] = Field(default="distance", title="Bundle Spacing", description="How the sequence is laid out around the ring. 'distance' (default) = the whole tiled sequence at equal arc-length spacing (bundles evenly spread). 'angle' = equal angular step from the centre (differs from 'distance' on an ellipse/star). 'grouped' = each repeat's sequence is packed as one tight cluster centred in its share of the ring, leaving empty valleys between clusters — use it (with a star ring + align_to_arms) for lobed stems where each lobe carries one group of bundles (e.g. hemp).")
    angle         : float     = Field(default=0.0, title="Pattern Angle", description="Angular offset (degrees) rotating the whole pattern around the ring — orients which direction the first kind (the sequence's first bundle) points, e.g. to line the 'big' bundles up with a chosen axis. Composes with align_to_arms.")
    align_to_arms : bool      = Field(default=True, title="Align First Kind to Star Arms", description="When the ring is a 'star' and repeats == the ring's n_peaks, phase the first bundle of each repeat onto a star arm (so the leading kind sits on the arms). Ignored for circle/ellipse rings or a mismatched arm count.")


class VascularCylinderParams(BaseParams):
    """Non-fascicular (continuous) dicot-stem vascular cylinder.

    Used instead of ``vascular_bundle`` when that spec's ``arrangement`` is
    ``continuous``: an uninterrupted ring of xylem / cambium / phloem laid down as
    a cylinder from the start (rather than discrete strands that later fuse).  The
    cylinder sits on the pith/cortex boundary — an endarch xylem annulus toward the
    pith (small protoxylem inner -> large metaxylem outer), a continuous cambium ring
    on the boundary, and a phloem annulus toward the cortex.  Vessel / sieve / cambium
    *cell* sizes reuse the ``xylem`` / ``phloem`` / ``cambium`` param blocks; this
    block owns only the cylinder geometry and the ground-cell composition.
    """
    name             : str = "vascular_cylinder"
    # -- radial geometry ----------------------------------------------------
    xylem_thickness  : float = Field(default=0.13, ge=0.00001, title="Xylem Thickness", description="Radial extent (mm) of the xylem annulus, measured inward from the cambium ring toward the pith.")
    phloem_thickness : float = Field(default=0.055, ge=0.00001, title="Phloem Thickness", description="Radial extent (mm) of the phloem annulus, measured outward from the cambium ring toward the cortex.")
    # -- ring shape (shared with the eustele ring_shape family) -------------
    ring_shape       : Literal["circle", "ellipse", "star"] = Field(default="circle", title="Cylinder Ring Shape", description="Outline the cylinder follows. 'circle' = the pith/cortex boundary (radius auto-derived); 'ellipse' = flattened by ring_ellipse_ratio; 'star' = a lobed ring set by the same absolute peak/valley radii + arcs as the root/eustele star (below).")
    ring_ellipse_ratio : float = Field(default=0.75, gt=0.0, le=1.0, title="Ring Ellipse Ratio", description="ring_shape='ellipse' only: height/width of the ring ellipse (1 = circle, <1 = flattened vertically).")
    # Star ring (ring_shape='star'): same absolute peak/valley parameterisation as
    # the root and the eustele bundle ring.
    n_peaks            : int   = Field(default=5,    ge=2,       title="Star Peaks",          description="ring_shape='star' only: number of lobes (arms) of the cylinder.")
    radius_peak_side   : float = Field(default=0.4,  ge=0.00001, title="Star Peak Radius",    description="ring_shape='star' only: arm-tip (peak) radius of the cylinder, mm from the organ centre (absolute, like the root xylem star).")
    radius_valley_side : float = Field(default=0.34, ge=0.00001, title="Star Valley Radius",  description="ring_shape='star' only: valley radius between the arms, mm from the organ centre.")
    arc_peak_side      : float = Field(default=0.12, ge=0.00001, title="Star Arc at Peak",    description="ring_shape='star' only: arc length of each arm at radius_peak_side.")
    arc_valley_side    : float = Field(default=0.10, ge=0.00001, title="Star Arc at Valley",  description="ring_shape='star' only: arc length of each arm at radius_valley_side.")
    # -- xylem files (radial vessel lines) ----------------------------------
    # Optional thin radial parenchyma strips *inside the xylem annulus only* (the
    # cambium ring and the phloem annulus stay continuous).  They cut the xylem into
    # tangential compartments so the vessel packer fills each in a radial row — the
    # same 'files' texture the dicot vascular bundle's xylem uses.  Mirrors
    # VascularBundleParams.xylem_layout / n_xylem_files.
    xylem_layout     : Literal["packed", "files"] = Field(default="packed", title="Xylem Layout", description="How the xylem annulus is organised. 'packed' (default) = a fully continuous, un-lined ring of vessels + parenchyma. 'files' = the vessels are forced into radial files (lines) separated by thin parenchyma strips, like the dicot bundle's xylem.")
    n_xylem_files    : int   = Field(default=0, ge=0, title="Number of Xylem Files", description="xylem_layout='files' only: number of radial vessel files the xylem annulus is split into by thin parenchyma strips. 0 (default) = auto: as many files as fit one vessel-file per (vessel + strip) of circumference, so every xylem pole reads as its own radial line. 1 = a single open compartment (no strips).")
    xylem_file_jitter : float = Field(default=0.3, ge=0.0, title="Xylem File Jitter", description="xylem_layout='files' only: how much each file's angular position is perturbed (0 = a rigid, evenly-spaced grid) so the files don't read as mechanical rows.")
    # -- ground-cell composition (fills the annuli around the conducting cells) -
    prop_vessel      : float = Field(default=0.6, ge=0.0, le=1.0, title="Proportion Vessels", description="Fraction of the xylem annulus occupied by vessels; the rest is xylem parenchyma packed around them.")
    prop_sieve       : float = Field(default=0.5, ge=0.0, le=1.0, title="Proportion Sieve", description="Fraction of the phloem annulus occupied by sieve elements + companion cells together; the rest is phloem parenchyma.")
    parenchyma_diameter : float = Field(default=0.008, ge=0.00001, title="Parenchyma Diameter", description="Diameter of the ground parenchyma cells filling the annuli and the xylem files.")
    parenchyma_width : float = Field(default=0.008, ge=0.00001, title="Parenchyma Width", description="Tangential width of the ground parenchyma cells.")
    sieve_diameter_min : float = Field(default=0.006, ge=0.00001, title="Sieve Diameter (min)", description="Lower bound of the sieve-element diameter when packing the phloem.")
    companion_cell_diameter : float = Field(default=0.004, ge=0.00001, title="Companion Cell Diameter", description="Radial extent of the companion cell placed beside each sieve element.")
    companion_cell_width : float = Field(default=0.004, ge=0.00001, title="Companion Cell Width", description="Tangential extent of the companion cell placed beside each sieve element.")


# Monocotyledon-specific layers
SteleParams = _ground_region_params("SteleParams", "stele",
    thickness=0.27, cell_diameter=0.01, cell_diameter_center=0.02, inflection=0.3)


# --- Shared xylem field groups --------------------------------------------
# The star-xylem organs (root stele + dicot-stem eustele) and the leaf vein all
# size their vessels the same way; the field *definitions* live once in these
# group factories and each organ passes its own *defaults*.  (Descriptions/titles
# are metadata, not emitted by to_dict_list, so one canonical wording covers every
# organ.)  Each returns a ``{field: (type, FieldInfo)}`` dict for ``create_model``.

def _vessel_sizing_fields(*, vessel_diameter: float, vessel_diameter_min: float,
                          vessel_diameter_sd: float) -> Dict[str, Any]:
    """Vessel diameter + its min floor + per-vessel jitter (every xylem uses these)."""
    return {
        "vessel_diameter":     (float, Field(default=vessel_diameter, ge=0.00001, title="Vessel Diameter", description="Metaxylem vessel diameter (upper bound of the size gradient).")),
        "vessel_diameter_min": (float, Field(default=vessel_diameter_min, ge=0.00001, title="Vessel Diameter (min)", description="Lower bound / floor of the vessel size gradient.")),
        "vessel_diameter_sd":  (float, Field(default=vessel_diameter_sd, ge=0.0, title="Vessel Diameter SD", description="Standard deviation added per vessel.")),
    }


def _size_gradient_fields(*, gradient_inflection: float, gradient_steepness: float,
                          gradient_function: str = "five_pl", gradient_asymmetry: float = 1.0,
                          enforce_gradient_min: float = 0.0) -> Dict[str, Any]:
    """The centre-to-tip vessel size gradient (shared by the root & dicot-stem stars)."""
    return {
        "gradient_function":    (Literal["five_pl", "linear"], Field(default=gradient_function, title="Gradient Function", description="Shape function for the vessel size gradient.")),
        "gradient_inflection":  (float, Field(default=gradient_inflection, ge=0.001, le=1.0, title="Gradient Inflection", description="Normalized position of the gradient inflection point (0 = centre, 1 = tip).")),
        "gradient_steepness":   (float, Field(default=gradient_steepness, ge=0.1, title="Gradient Steepness", description="Hill coefficient — sharpness of the size transition.")),
        "gradient_asymmetry":   (float, Field(default=gradient_asymmetry, ge=0.1, title="Gradient Asymmetry", description="Asymmetry exponent of the size gradient.")),
        "enforce_gradient_min": (float, Field(default=enforce_gradient_min, ge=0.0, le=1.0, title="Enforce Gradient Minimum", description="Radial extent in [0, 1] over which the gradient minimum is enforced (0 disables, 1 everywhere).")),
    }


def _xylem_star_fields(*, n_vascular_peak: int, radius_valley_side: float, radius_peak_side: float,
                       arc_peak_side: float, arc_valley_side: float) -> Dict[str, Any]:
    """The star (actinostele / eustele) arm geometry — arm count + valley/peak radii + arcs."""
    return {
        "n_vascular_peak":    (int,   Field(default=n_vascular_peak, ge=1, title="Number of Poles/Arms", description="Number of xylem star arms (poles), alternating with the phloem valleys.")),
        "radius_valley_side": (float, Field(default=radius_valley_side, ge=0.00001, title="Valley Radius", description="Valley-side radius of the star arms from the centre.")),
        "radius_peak_side":   (float, Field(default=radius_peak_side, ge=0.00001, title="Peak Radius", description="Peak-side (arm-tip) radius of the star arms from the centre.")),
        "arc_peak_side":      (float, Field(default=arc_peak_side, ge=0.00001, title="Arc Length at Peak", description="Arc length of each arm at radius_peak_side (peak width).")),
        "arc_valley_side":    (float, Field(default=arc_valley_side, ge=0.00001, title="Arc Length at Valley", description="Arc length of each arm at radius_valley_side (valley/base width).")),
    }


def _vessel_packing_fields(*, allow_ellipse: bool, ellipse_max_aspect: float = 2.0,
                           packing_strategy: str = "space", first_vessel_shift: float = 0.7,
                           direction: Optional[str] = "center", pith_radius: float = 0.0) -> Dict[str, Any]:
    """How vessels are packed into the region (ellipse fallback, strategy, seeding, pith hole)."""
    return {
        "allow_ellipse":      (bool,  Field(default=allow_ellipse, title="Allow Ellipse Vessels", description="Fit an area-matched radial ellipse where a target-diameter circle is too tight, instead of shrinking the vessel.")),
        "ellipse_max_aspect": (float, Field(default=ellipse_max_aspect, ge=1.0, title="Ellipse Max Aspect", description="Maximum major/minor axis ratio for ellipse vessels.")),
        "packing_strategy":   (Literal["space", "target"], Field(default=packing_strategy, title="Packing Strategy", description="'space' = space-first Apollonian fill; 'target' = size-first gradient-driven radial fill.")),
        "first_vessel_shift": (float, Field(default=first_vessel_shift, ge=0.0, le=1.0, title="First Vessel Shift", description="Maximum random displacement of the first vessel as a fraction of its inscribed radius.")),
        "direction":          (Optional[str], Field(default=direction, title="Packing Direction", description="Size gradient direction: 'center', 'edge', 'middle', or None (random).")),
        "pith_radius":        (float, Field(default=pith_radius, ge=0.0, title="Pith Radius", description="Inner radius left free of vessels (0 = runs to the centre).")),
    }


# Root-stele xylem: the four shared star groups (root defaults) + root-only fields
# for the arch mode (protoxylem chains, metaxylem ring) and the discrete-vessel ring.
_root_xylem_extra_fields: Dict[str, Any] = {
    "protoxylem_diameter":     (float, Field(default=0.01, ge=0.00001, title="Protoxylem Diameter", description="Diameter of protoxylem elements.")),
    "protoxylem_diameter_sd":  (float, Field(default=0.001, ge=0.0, title="Protoxylem Diameter SD", description="Standard deviation of protoxylem element diameter.")),
    "protoxylem_diameter_min": (float, Field(default=0.0, ge=0.0, title="Protoxylem Diameter (min)", description="Arch mode: smallest protoxylem diameter (outer edge of the band); 0 defaults to 0.4 * protoxylem_diameter.")),
    "protoxylem_cluster_width":  (float, Field(default=0.015, ge=0.00001, title="Protoxylem Bundle Width", description="Tangential width of the protoxylem ellipse.")),
    "protoxylem_cluster_height": (float, Field(default=0.01, ge=0.00001, title="Protoxylem Bundle Height", description="Radial height of the protoxylem ellipse.")),
    "protoxylem_band_depth":   (float, Field(default=0.0, ge=0.0, title="Protoxylem Band Depth", description="Arch mode: radial depth of the outer band holding the protoxylem chains + phloem; 0 defaults to 35%% of the span.")),
    "protoxylem_pole_width_inner": (float, Field(default=0.0, ge=0.0, title="Protoxylem Pole Width (inner)", description="Arch mode: tangential width of each protoxylem pole at its inner end; 0 defaults to 3 * protoxylem_diameter.")),
    "protoxylem_pole_width_outer": (float, Field(default=0.0, ge=0.0, title="Protoxylem Pole Width (outer)", description="Arch mode: tangential width of each protoxylem pole at its outer end; 0 defaults to 3 * protoxylem_diameter.")),
    "n_vascular_bundles": (int,   Field(default=5, ge=1, title="Number of Vascular Bundles", description="Number of metaxylem vessels.")),
    "ratio_proto_meta":   (float, Field(default=2.2, ge=0.0, title="Ratio Protoxylem/Metaxylem", description="Ratio controlling protoxylem bundle count relative to metaxylem vessels.")),
    "xylem_shape":        (Literal["default", "arch", "star"], Field(default="default", title="Xylem Shape", description="'default' = ring of discrete vessels; 'arch' = metaxylem ring + graded protoxylem chains + valley phloem; 'star' = actinostele arms with a radial size gradient.")),
    "n_metaxylem":        (int,   Field(default=0, ge=0, title="Number of Metaxylem", description="Arch mode: metaxylem count in the central ring; 0 defaults to n_vascular_peak.")),
    "outer_radius":       (float, Field(default=0.15, ge=0.00001, title="Outer Radius", description="Arch mode: radius of the pericycle side where the poles reach; capped at the stele radius.")),
}

RootXylemParams = create_model("RootXylemParams", __base__=BaseParams,
    name=(str, "xylem"),
    **_vessel_sizing_fields(vessel_diameter=0.06, vessel_diameter_min=0.01, vessel_diameter_sd=0.005),
    **_size_gradient_fields(gradient_inflection=0.7, gradient_steepness=5.0),
    **_xylem_star_fields(n_vascular_peak=5, radius_valley_side=0.05, radius_peak_side=0.22,
                         arc_peak_side=0.03, arc_valley_side=0.03),
    **_vessel_packing_fields(allow_ellipse=True),
    **_root_xylem_extra_fields,
)


def _phloem_params(clsname: str, *, sieve_diameter: float, sieve_diameter_sd: float = 0.001,
                   cluster_width: Optional[float] = None, cluster_height: Optional[float] = None,
                   relative_distance: Optional[float] = None):
    """A phloem param set: sieve-element sizing (always) + an optional phloem-cluster
    ellipse (cluster_width/height + relative_distance).

    Shared by every ``phloem`` bundle spec (root star, dicot-stem bundle, leaf vein);
    the field *definitions* live here once and each organ passes its own *defaults*.
    Omit the cluster args for a sizing-only phloem (the leaf vein reads its cluster
    extent from the vascular_bundle spec instead).
    """
    fields: Dict[str, Any] = {
        "name": (str, "phloem"),
        "sieve_diameter": (float, Field(default=sieve_diameter, ge=0.00001,
            title="Sieve Diameter", description="Diameter of phloem sieve elements.")),
        "sieve_diameter_sd": (float, Field(default=sieve_diameter_sd, ge=0.0,
            title="Sieve Diameter SD", description="Standard deviation of phloem sieve diameter.")),
    }
    if cluster_width is not None:
        fields["cluster_width"] = (float, Field(default=cluster_width, ge=0.00001,
            title="Phloem Bundle Width", description="Tangential width of the phloem ellipse."))
        fields["cluster_height"] = (float, Field(default=cluster_height, ge=0.00001,
            title="Phloem Bundle Height", description="Radial height of the phloem ellipse."))
        fields["relative_distance"] = (float, Field(default=relative_distance, ge=0.0, le=1.0,
            title="Relative Distance", description="Relative distance of the phloem from the xylem inner radius / cambium."))
    return create_model(clsname, __base__=BaseParams, **fields)


RootPhloemParams = _phloem_params("RootPhloemParams", sieve_diameter=0.025,
    cluster_width=0.025, cluster_height=0.025, relative_distance=0.5)

# Dicotyledon-specific layers
SteleDicotParams = _ground_region_params("SteleDicotParams", "stele",
    thickness=0.65, cell_diameter=0.015, cell_diameter_center=0.03, inflection=0.2)


# Dicot-stem eustele xylem: the same four shared star groups, dicot defaults (no
# root-only arch/protoxylem fields).
DicotXylemParams = create_model("DicotXylemParams", __base__=BaseParams,
    name=(str, "xylem"),
    **_xylem_star_fields(n_vascular_peak=3, radius_valley_side=0.1, radius_peak_side=0.22,
                         arc_peak_side=0.03, arc_valley_side=0.05),
    **_vessel_sizing_fields(vessel_diameter=0.08, vessel_diameter_min=0.02, vessel_diameter_sd=0.002),
    **_size_gradient_fields(gradient_inflection=0.7, gradient_steepness=1.0),
    **_vessel_packing_fields(allow_ellipse=False),
)



DicotPhloemParams = _phloem_params("DicotPhloemParams", sieve_diameter=0.012,
    cluster_width=0.1, cluster_height=0.05, relative_distance=0.8)


class BundleCambiumParams(BaseParams):
    """Cambium cell sizing for a vascular bundle — the fascicular cambium of a dicot
    stem bundle or a dicot leaf vein.  Base = just the cell diameter; the dicot-stem
    ring adds its star-placement fields (see ``DicotCambiumParams``)."""
    name          : str   = "cambium"
    cell_diameter : float = Field(default=0.01, ge=0.00001, title="Cell Diameter", description="Diameter of cambium cells.")


# Leaf vein cambium: just the cell diameter (placement is handled by the bundle).
class LeafBundleCambiumParams(BundleCambiumParams):
    """Cambium cell sizing for dicot leaf veins (ignored by closed monocot veins)."""


class DicotCambiumParams(BundleCambiumParams):
    cell_width       : float = Field(default=0.02,   ge=0.00001, title="Cell Width",       description="Width of cambium cells (tangential).")
    n_layers         : int   = Field(default=2, ge=1, title="Cambium Layers", description="Stem: number of concentric cambium cell files. In the fascicular cambium strip and, under secondary growth, in the continuous cambium ring — a thicker meristematic zone reads as several cell layers.")
    # for primary growth
    visible_distance : float = Field(default=0.8,  ge=0.0, title="Primary Visible Distance", description="Maximum radius at which primary cambium is differentiated. Cambium matures first in the valleys between xylem arms. Increase toward the stele edge for a more mature (complete ring) cambium.")
    radius_valley_side : float = Field(default=0.11,  ge=0.00001, title="Primary Valley Radius",   description="Valley-side radius of the cambium ring from the stele centre at primary growth.")
    radius_peak_side   : float = Field(default=0.28,  ge=0.00001, title="Primary Peak Radius",     description="Peak-side radius of the cambium star arms from the stele centre at primary growth. Should be close to the stele radius.")
    arc_peak_side       : float = Field(default=0.05,  ge=0.00001, title="Arc Length at Peak",       description="Arc length of each arm at radius_peak_side (peak width).")
    arc_valley_side     : float = Field(default=0.07,  ge=0.00001, title="Arc Length at Valley",     description="Arc length of each arm at radius_valley_side (valley/base width).")

# Dicot secondary stele — same as SteleDicotParams but a wider default thickness
# (secondary growth needs room for the secondary xylem annulus).
DicotSecondarySteleParams = _ground_region_params("DicotSecondarySteleParams", "stele",
    thickness=1.0, cell_diameter=0.015, cell_diameter_center=0.03, inflection=0.2)


class DicotSecondaryGrowthParams(BaseParams):
    name         : str   = "secondary_growth"
    value        : bool  = True

class DicotSecondaryXylemParams(BaseParams):
    name                : str   = "secondary_xylem"
    prop_stele          : float = Field(default=0.8,  ge=0.0, le=1.0, title="Proportion of Stele",       description="Angular fraction of each valley between xylem peaks that is occupied by a vessel pizza-slice zone (0-1). 1.0 means slices tile the full circle; 0.5 means each slice is half as wide. In the dicot stem this is the angular *cap* each secondary-xylem sector flares up to (see flare_angle).")
    flare_angle         : float = Field(default=30.0, ge=0.0, le=90.0, title="Secondary Xylem Flare Angle", description="Dicot stem only. Tilt (degrees, from the radial direction) of each secondary-xylem sector's side edges: the sector starts at the vascular-bundle width against the primary xylem and flares outward at this angle until it reaches the prop_stele angular cap. So adjacent sectors stay separate near the pith and only merge into a continuous cylinder further out (0 = straight radial sides that never widen).")
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

    # -- Primary phloem remnant (dicot stem) -----------------------------------
    keep_primary: bool = Field(default=False, title="Keep Primary Phloem",
        description="Dicot stem only. If True, a thin primary-phloem remnant is placed "
                    "just outside the secondary phloem band (one arm per bundle) — the "
                    "displaced, usually crushed primary phloem. Off by default because "
                    "secondary growth crushes it; turn on to render it explicitly.")

    # -- Zone geometry ---------------------------------------------------------
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

    # -- Alive sieve zone ------------------------------------------------------
    alive_distance: float = Field(default=0.1, ge=0.0, title="Alive Distance (mm)",
        description="Radial distance from the cambium boundary within which sieve "
                    "elements are alive and have companion cells. Beyond this they are dead.")

    # -- Sieve elements (same size for living and dead) ------------------------
    sieve_diameter:     float = Field(default=0.022, ge=0.00001, title="Sieve Diameter")
    sieve_diameter_sd:  float = Field(default=0.001, ge=0.0,     title="Sieve Diameter SD")
    sieve_diameter_min: float = Field(default=0.020, ge=0.00001, title="Sieve Diameter Min")
    prop_sieve:         float = Field(default=0.3,  ge=0.0, le=1.0, title="Sieve Proportion",
        description="Stop packing sieve circles when their area reaches this fraction of the zone.")

    # -- Companion cells (one per living sieve element) ------------------------
    companion_diameter: float = Field(default=0.01, ge=0.00001, title="Companion Cell Diameter")
    companion_width:    float = Field(default=0.002, ge=0.00001, title="Companion Cell Width")

    # -- Phloem parenchyma -----------------------------------------------------
    parenchyma_diameter: float = Field(default=0.012, ge=0.00001, title="Parenchyma Diameter")
    parenchyma_width:    float = Field(default=0.012, ge=0.00001, title="Parenchyma Width")


class SecondaryCambiumParams(BaseParams):
    """Dicot secondary cambium — one contour saying where the cambium sits.

    Shared by the root (:class:`~openalea.granap.root_dicot_class.DicotRootAnatomy`)
    and the stem (:class:`~openalea.granap.stem_dicot_class.DicotStemAnatomy`).
    Both describe the cambium by its **radius** from the organ centre and pick a
    contour from the shared ``circle`` / ``ellipse`` / ``star`` / ``focus_ellipse``
    family; the secondary xylem fills the annulus between the primary cambium /
    bundle ring and this contour, the secondary phloem sits just outside it.

    Organ differences: on the **root** a ``star`` cambium is rotated half a period
    so its arms point into the primary-xylem valleys (``radius_valley_side`` is the
    outer, bulging side there) and the contour is clipped to the stele; on the
    **stem** the contour is a plain concentric outline and the stem's outer radius
    grows outward to ``radius_valley_side`` to make room.  The radii must therefore
    stay within the stele on a root and beyond the primary bundle ring on a stem.
    """
    name             : str   = "secondary_cambium"
    cell_diameter    : float = Field(default=0.01,  ge=0.00001, title="Cell Diameter",    description="Diameter of secondary cambium cells.")
    cell_width       : float = Field(default=0.02,  ge=0.00001, title="Cell Width",       description="Tangential width of secondary cambium cells.")
    n_layers         : int   = Field(default=1,     ge=1,       title="Number of Layers", description="Number of concentric cambium cell files (the cambial zone). 1 = a single ring; higher values add rings buffered inward by one cell diameter each.")

    # Contour family (shared with base_shape / the eustele ring).
    shape : Literal["circle", "ellipse", "star", "focus_ellipse"] = Field(default="star", title="Cambium Contour",
        description="Outline of the secondary cambium. 'circle' / 'ellipse' = a smooth ring sized by "
                    "radius_valley_side (ellipse flattened by ellipse_ratio); 'star' = a lobed contour "
                    "(radius_peak_side / radius_valley_side + arcs; on a root the arms point into the "
                    "primary-xylem valleys); 'focus_ellipse' = a best-fit superellipse from a measured profile.")

    # Radii (mm from the organ centre) — the cambium position for every shape.
    radius_valley_side : float = Field(default=0.45,  ge=0.00001, title="Cambium Radius / Valley Radius", description="Outer cambium radius. For 'circle'/'ellipse' this is the ring radius; for 'star' the valley-side radius (on a root the bulging, primary-xylem-valley side). Root: <= the stele radius. Stem: the radius the organ grows out to (must exceed the primary bundle ring).")
    radius_peak_side   : float = Field(default=0.40,  ge=0.00001, title="Star Peak Radius", description="shape='star' only: the peak-side radius (on a root, the inner primary-xylem-peak side; must exceed the primary cambium radius so the secondary cambium encloses it).")
    arc_peak_side      : float = Field(default=0.20,  ge=0.00001, title="Arc Length at Peak",   description="shape='star' only: arc length at radius_peak_side (peak-side width of each arm).")
    arc_valley_side    : float = Field(default=0.10,  ge=0.00001, title="Arc Length at Valley",  description="shape='star' only: arc length at radius_valley_side (valley-side width of each arm).")
    n_peaks            : int   = Field(default=0,      ge=0,       title="Star Peaks", description="shape='star' only: number of arms. 0 = follow the primary vascular pattern (root: n_vascular_peak; stem: a smooth default).")
    ellipse_ratio      : float = Field(default=0.75,  gt=0.0, le=1.0, title="Ellipse Ratio", description="shape='ellipse' only: height/width of the cambium ellipse (1 = circle, <1 = flattened).")

    # Focus-ellipse (superellipse) contour.
    profile : List[Tuple[float, float]] = Field(default_factory=list, title="Measured Contour Profile",
        description="shape='focus_ellipse' only. A list of (major_pos, minor_width) mm measurements best-fitted "
                    "to one superellipse: the widest point sets the minor axis, the farthest point (tip, width->0) "
                    "the major axis, and the exponent is least-squares fitted to the rest. Major axis along +y.")
    exponent : float = Field(default=4.0, gt=0.0, title="Focus-Ellipse Exponent",
        description="shape='focus_ellipse' with no profile: superellipse fullness (2 = plain ellipse, >2 = fuller flanks).")


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
    smoothness: Union[float, List[float]] = Field(default=[0.01, 0.01], title="Smoothness", description="Smoothness per tissue (0-1). Provide a single float applied to all tissues, or a list with one value per tissue.")

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


NeedleEndodermisParams = _layer_params("NeedleEndodermisParams", "endodermis", "endodermal", cell_diameter=0.02,   cell_width=0.05,  n_layers=1, shift=0.5, order=3)
MesophyllParams        = _layer_params("MesophyllParams",        "mesophyll",  "mesophyll",  cell_diameter=0.08,   cell_width=0.045, n_layers=3, shift=0.5, order=4)
HypodermisParams       = _layer_params("HypodermisParams",       "hypodermis", "hypodermal", cell_diameter=0.0225,                   n_layers=2, shift=0.5, order=5)
NeedleEpidermisParams  = _layer_params("NeedleEpidermisParams",  "epidermis",  "epidermal",  cell_diameter=0.02,                     n_layers=1, shift=0.5, order=6)


# ===========================================================================
# Leaf params (typed, like every other organ).  A leaf is a flat slab (see
# leaf_class.py): a folded/ribbed lamina outline + a transverse vein row + stomata.
# ===========================================================================

class LeafPlantTypeParams(PlantTypeParams):
    """Leaf global geometry (the 'planttype' entry): the slab outline and how it is
    shaped (custom thickness profile, fold, twist, rounded margins).  Extends the base
    planttype (name/value/organ) with the leaf's slab-shape fields."""
    value : int = Field(default=1, ge=1, le=2, title="Plant Type", description="1 = monocot leaf (uniform mesophyll, 'face' veins), 2 = dicot leaf (dorsiventral palisade/spongy, collateral veins).")
    organ : str = "leaf"
    width     : float = Field(default=4.0, ge=0.00001, title="Width", description="Leaf width (x extent) — the straight tip-to-tip chord, mm.")
    thickness : float = Field(default=0.45, ge=0.00001, title="Thickness", description="Lamina thickness (y extent) at the centre, mm. Used for the default elliptical profile when thickness_profile is empty.")
    thickness_profile : List[Tuple[float, float]] = Field(default_factory=list, title="Thickness Profile", description="Optional measured half-width profile: a list of (x, thickness) mm control points from the centre outward, linearly interpolated (symmetric in x, 0 beyond the last point). Overrides the elliptical thickness. Empty = plain ellipse from width/thickness.")
    fold_sag    : float = Field(default=0.0, ge=0.0, title="Fold Sag", description="Folded (keeled) leaf: how far the mid-line sags below the tip-to-tip chord at the centre, mm (a smooth raised-cosine, 0 at the tips). The adaxial surface then hugs the chord while the abaxial swings out into a keel. 0 = flat, straight leaf.")
    fold_width  : float = Field(default=0.0, ge=0.0, title="Fold Width", description="Span of the fold sag, mm (the cosine reaches 0 at ±fold_width/2). 0 = the whole leaf width.")
    edge_radius : float = Field(default=0.0, ge=0.0, title="Edge Radius", description="Morphological-opening radius (mm) that rounds the sharp convex features (the thin pointed margins / tips, a narrow keel point) so they don't render as spikes. 0 = no rounding; ~half an epidermis cell blunts the tips.")
    twist_amplitude : float = Field(default=0.0, ge=0.0, title="Twist Amplitude", description="Bend depth (mm) of the mid-line sine — a leaf that is not perfectly straight. 0 = straight.")
    twist_waves     : float = Field(default=1.0, ge=0.0, title="Twist Waves", description="Number of half-bends of the twist sine across the width.")


LeafEpidermisParams = _layer_params("LeafEpidermisParams", "epidermis", "epidermal",
                                    cell_diameter=0.030, cell_width=0.035, n_layers=1, shift=0.3, order=3)


class _LeafGroundParams(BaseParams):
    """A leaf mesophyll ground tissue filling the lamina core (no 'order' — a central
    fill, not a peeled ring)."""
    name          : str   = "mesophyll"
    cell_diameter : float = Field(default=0.045, ge=0.00001, title="Cell Diameter", description="Cell diameter (radial/vertical extent) of the ground cells.")
    cell_width    : float = Field(default=0.045, ge=0.00001, title="Cell Width", description="Cell width (tangential/horizontal extent). Set larger than the diameter for flat cells, smaller for columnar (palisade) cells.")


class LeafMesophyllParams(_LeafGroundParams):
    """Monocot leaf: a single uniform mesophyll tissue (no palisade/spongy split)."""
    name : str = "mesophyll"


class LeafPalisadeParams(_LeafGroundParams):
    """Dicot leaf: adaxial palisade — elongated columnar cells (diameter tall > width)."""
    name          : str   = "palisade"
    cell_diameter : float = Field(default=0.075, ge=0.00001, title="Cell Diameter", description="Vertical (columnar) extent of the palisade cells — larger than cell_width so the cells stand up.")
    cell_width    : float = Field(default=0.020, ge=0.00001, title="Cell Width", description="Narrow tangential width of the palisade cells.")


class LeafSpongyParams(_LeafGroundParams):
    """Dicot leaf: abaxial spongy — big roundish cells (intercellular air is added by
    an inter_cellular_spaces entry)."""
    name          : str   = "spongy"
    cell_diameter : float = Field(default=0.048, ge=0.00001, title="Cell Diameter", description="Diameter of the (big, roundish) spongy cells.")
    cell_width    : float = Field(default=0.048, ge=0.00001, title="Cell Width", description="Width of the spongy cells.")


class LeafVascularBundleParams(BaseParams):
    """One leaf vein size-class: a transverse bundle (xylem adaxial / phloem abaxial),
    the raised-cosine rib it raises on the lamina, and optional sclerenchyma girders.
    Several specs give vein size-classes (midrib / minor). Bundle-internal fields not
    listed here fall back to the ``build_bundle`` defaults."""
    name        : str = "vascular_bundle"
    # -- type ---------------------------------------------------------------
    bundle_type : Literal["collateral", "bicollateral", "concentric"] = Field(default="collateral", title="Bundle Type", description="Vein bundle type; leaves are collateral (xylem inner/adaxial, phloem outer/abaxial).")
    has_cambium : bool  = Field(default=False, title="Has Cambium", description="False (monocot 'face' vein, closed) or True (dicot collateral, a fascicular cambium between xylem and phloem).")
    xylem_layout: Literal["packed", "files", "face"] = Field(default="face", title="Xylem Layout", description="'face' = the monocot mask (metaxylem middle, protoxylem + optional lacuna toward the centre, phloem toward the abaxial surface); 'files' = the dicot radial files; 'packed' = one open zone.")
    lacuna      : bool  = Field(default=True, title="Protoxylem Lacuna", description="'face' layout: carve a protoxylem air lacuna (the substomatal/vein air space).")
    sheath      : Literal["none", "ring", "caps", "both"] = Field(default="both", title="Sclerenchyma Sheath", description="Fibre sheath around the vein ('both' = caps + ring; 'none' = only a thin parenchyma bundle sheath).")
    # -- envelope + orientation --------------------------------------------
    shape          : Literal["ellipse", "circle", "focus_ellipse", "egg"] = Field(default="ellipse", title="Envelope Shape", description="Vein envelope outline.")
    width          : float = Field(default=0.12, ge=0.00001, title="Envelope Width", description="Tangential extent of the vein envelope (mm). Widest class is placed first, so a midrib yields the minor veins around it.")
    height         : float = Field(default=0.16, ge=0.00001, title="Envelope Height", description="Radial (adaxial-abaxial) extent of the vein envelope (mm).")
    phloem_outward : bool  = Field(default=True, title="Phloem Outward", description="True = phloem faces the abaxial surface, xylem the adaxial (the normal leaf orientation).")
    relative_distance : float = Field(default=0.5, ge=0.0, le=1.0, title="Relative Distance", description="Where the vein sits through the local (ribbed) lamina thickness: 0 = abaxial (lower) face, 0.5 = mid-plane, 1 = adaxial (upper) face.")
    # -- rib (each vein raises the lamina) ----------------------------------
    rib_adaxial_height : float = Field(default=0.02, ge=0.0, title="Rib Height (adaxial)", description="How much this vein bulges the upper (adaxial) surface — a raised cosine (mm). 0 = no upper rib.")
    rib_adaxial_width  : float = Field(default=0.25, ge=0.00001, title="Rib Width (adaxial)", description="Full tangential width of the adaxial rib (mm); the bump is exactly 0 beyond ±width/2.")
    rib_abaxial_height : float = Field(default=0.06, ge=0.0, title="Rib Height (abaxial)", description="How much this vein bulges the lower (abaxial) surface — the keel side usually bulges more (mm).")
    rib_abaxial_width  : float = Field(default=0.30, ge=0.00001, title="Rib Width (abaxial)", description="Full tangential width of the abaxial rib (mm).")
    # -- sclerenchyma girders (grass-style struts to both epidermes) --------
    girder_adaxial     : bool  = Field(default=False, title="Girder (adaxial)", description="Fill a sclerenchyma triangle from this vein to the adaxial epidermis (base at the epidermis, apex at the vein).")
    girder_abaxial     : bool  = Field(default=False, title="Girder (abaxial)", description="Fill a sclerenchyma triangle from this vein to the abaxial epidermis.")
    girder_base_width  : float = Field(default=0.10, ge=0.00001, title="Girder Base Width", description="Tangential width of the girder triangle's base against the epidermis (mm).")
    girder_cell_diameter : float = Field(default=0.012, ge=0.00001, title="Girder Cell Diameter", description="Diameter of the sclerenchyma (fibre) cells filling the girder.")
    girder_cell_width    : float = Field(default=0.012, ge=0.00001, title="Girder Cell Width", description="Width of the girder fibre cells.")
    # -- placement ----------------------------------------------------------
    n_bundles     : int   = Field(default=7, ge=0, title="Number of Veins", description="How many veins of this size-class to place along the mid-plane.")
    span_fraction : float = Field(default=0.75, ge=0.0, le=1.0, title="Span Fraction", description="Fraction of the leaf width the vein row spans (centred), for 'even'/'center'/'scatter' placement.")
    placement     : Literal["even", "scatter", "center", "explicit"] = Field(default="even", title="Placement", description="'even' = endpoints-included row; 'center' = row centred on x=0 (a midrib at n=1); 'scatter' = an interleaved row offset half a step; 'explicit' = at the exact x_positions.")
    x_positions   : List[float] = Field(default_factory=list, title="Explicit Positions", description="placement='explicit' only: the exact x (mm) of each vein of this class.")
    # -- bundle internals commonly tuned per leaf ---------------------------
    prop_vessel      : float = Field(default=0.5, ge=0.0, le=1.0, title="Proportion Vessels", description="Fraction of the packed xylem zone occupied by vessels (the rest is xylem parenchyma).")
    n_metaxylem      : int   = Field(default=2, ge=0, title="Number of Metaxylem", description="'face' layout: metaxylem vessels at the radial middle (0 for a small vein with only protoxylem + phloem).")
    n_protoxylem     : int   = Field(default=1, ge=0, title="Number of Protoxylem", description="'face' layout: protoxylem bundles toward the centre.")
    protoxylem_diameter     : float = Field(default=0.03,  ge=0.00001, title="Protoxylem Diameter", description="Protoxylem vessel diameter (mm).")
    protoxylem_diameter_min : float = Field(default=0.025, ge=0.00001, title="Protoxylem Diameter (min)", description="Lower clip on the packed protoxylem diameter.")
    protoxylem_width  : float = Field(default=0.032, ge=0.00001, title="Protoxylem Bundle Width", description="Tangential extent of each protoxylem bundle region (mm).")
    protoxylem_height : float = Field(default=0.032, ge=0.00001, title="Protoxylem Bundle Height", description="Radial extent of each protoxylem bundle region (mm).")
    phloem_width      : float = Field(default=0.05, ge=0.00001, title="Phloem Ellipse Width", description="Tangential extent of the phloem (sieve + companion) cluster (mm).")
    phloem_height     : float = Field(default=0.04, ge=0.00001, title="Phloem Ellipse Height", description="Radial extent of the phloem cluster (mm).")
    sieve_diameter_min: float = Field(default=0.006, ge=0.00001, title="Sieve Diameter (min)", description="Lower bound of the sieve-element diameter when packing the phloem.")


# Leaf-vein xylem: just the shared vessel-sizing trio (leaf-scaled), read by build_bundle.
LeafBundleXylemParams = create_model("LeafBundleXylemParams", __base__=BaseParams,
    name=(str, "xylem"),
    **_vessel_sizing_fields(vessel_diameter=0.045, vessel_diameter_min=0.012, vessel_diameter_sd=0.003),
)


# Leaf-vein phloem: sieve sizing only (the cluster extent comes from the
# vascular_bundle spec's phloem_width/height).
LeafBundlePhloemParams = _phloem_params("LeafBundlePhloemParams", sieve_diameter=0.008)


class InterBundleAerenchymaParams(BaseParams):
    """Monocot leaf: an air lacuna in the mesophyll between each pair of veins, built
    root-style (the mesophyll cells inside the region are converted to air and fused).
    Two sizing modes: an ellipse (width/height) on the mid-plane, or margins holding
    the lacuna off the bundles and faces (used when any *_margin > 0)."""
    name           : str = "inter_bundle_aerenchyma"
    tissue         : str = Field(default="mesophyll", title="Tissue", description="Ground tissue the lacuna is carved from.")
    width          : float = Field(default=0.10, ge=0.0, title="Lacuna Width", description="Ellipse mode: tangential extent of the lacuna (mm).")
    height         : float = Field(default=0.24, ge=0.0, title="Lacuna Height", description="Ellipse mode: radial extent of the lacuna (mm).")
    side_margin    : float = Field(default=0.0, ge=0.0, title="Side Margin", description="Margin mode (any margin > 0): mesophyll kept beside each bundle (mm). The lacuna is an ellipse inscribed in the remaining gap.")
    adaxial_margin : float = Field(default=0.0, ge=0.0, title="Adaxial Margin", description="Margin mode: mesophyll kept below the adaxial (upper) face (mm).")
    abaxial_margin : float = Field(default=0.0, ge=0.0, title="Abaxial Margin", description="Margin mode: mesophyll kept above the abaxial (lower) face (mm).")


class LeafStomataParams(BaseParams):
    """Amphistomatous leaf stomata: counts per face (split along the mid-line), reusing
    the needle stomata geometry."""
    name        : str = "stomata"
    n_adaxial   : int = Field(default=10, ge=0, title="Stomata (adaxial)", description="Number of stomata on the upper (adaxial) face.")
    n_abaxial   : int = Field(default=10, ge=0, title="Stomata (abaxial)", description="Number of stomata on the lower (abaxial) face (usually denser in a dicot).")
    edge_margin : float = Field(default=0.12, ge=0.0, le=0.5, title="Edge Margin", description="Fraction of each epidermis run skipped at both ends, so no stoma sits at the tips where the two faces meet.")
    width       : float = Field(default=0.035, ge=0.00001, title="Width", description="Stoma width.")
    depth       : float = Field(default=0.03, ge=0.00001, title="Depth", description="Stoma depth.")
    sub_chamber : float = Field(default=0.06, ge=0.00001, title="Sub Chamber", description="Substomatal chamber (air space) size.")


def _monocot_leaf_vein(**kw: Any) -> LeafVascularBundleParams:
    """The monocot 'face' vein (metaxylem eyes + protoxylem + a lacuna), leaf-scaled —
    the class defaults already describe it; ``kw`` overrides per size-class."""
    return LeafVascularBundleParams(**kw)


def _dicot_leaf_vein(**kw: Any) -> LeafVascularBundleParams:
    """A dicot collateral vein (xylem / cambium / phloem in radial files, no metaxylem
    eyes, no lacuna); a touch more keeled underneath.  ``kw`` overrides per size-class."""
    base = dict(has_cambium=True, xylem_layout="files", lacuna=False, sheath="none",
                prop_vessel=0.7, rib_abaxial_height=0.10, rib_abaxial_width=0.32)
    base.update(kw)
    return LeafVascularBundleParams(**base)


def default_monocot_leaf_params() -> List[BaseParams]:
    """Monocot leaf: the palisade/spongy distinction is not clear, so the mesophyll
    is a **single uniform tissue** (big parenchyma) with intercellular air (and
    substomatal chambers near the stomata); amphistomatous."""
    return [
        LeafPlantTypeParams(value=1, width=4.0, thickness=0.45),
        LeafEpidermisParams(),
        LeafMesophyllParams(),
        _monocot_leaf_vein(),
        LeafBundleXylemParams(),
        LeafBundlePhloemParams(),
        LeafBundleCambiumParams(),
        # Air near the stomata comes from the substomatal chambers; aerenchyma is a
        # lacuna in the mesophyll between each pair of veins.
        InterBundleAerenchymaParams(width=0.09, height=0.16),
        LeafStomataParams(n_adaxial=10, n_abaxial=10),   # ~equal both faces
    ]


def default_dicot_leaf_params() -> List[BaseParams]:
    """Dicot leaf: dorsiventral palisade (adaxial) / spongy (abaxial); two vein
    size-classes (a central **midrib** + scattered **minor** veins), each with its
    own rib; stomata denser on the abaxial face (a few adaxial)."""
    midrib = _dicot_leaf_vein(n_bundles=1, placement="center", span_fraction=0.0,
                              width=0.18, height=0.24,
                              rib_adaxial_height=0.035, rib_adaxial_width=0.40,
                              rib_abaxial_height=0.12, rib_abaxial_width=0.50)   # big keeled midrib
    minor = _dicot_leaf_vein(n_bundles=8, placement="scatter", span_fraction=0.85,
                             width=0.08, height=0.11,
                             rib_adaxial_height=0.012, rib_adaxial_width=0.14,
                             rib_abaxial_height=0.035, rib_abaxial_width=0.20)    # small ribs
    return [
        LeafPlantTypeParams(value=2, width=4.0, thickness=0.45),
        LeafEpidermisParams(),
        # Both are central fills (no 'order'); the core is split at the mid-plane.
        LeafPalisadeParams(),   # adaxial, elongated columnar
        LeafSpongyParams(),     # abaxial, big roundish
        midrib,
        minor,
        LeafBundleXylemParams(),
        LeafBundlePhloemParams(),
        LeafBundleCambiumParams(),
        # Spongy = big parenchyma with lots of intercellular air (ICS module).
        InterCellularSpacesParams(tissue=["spongy"], smoothness=0.22),
        LeafStomataParams(n_adaxial=3, n_abaxial=12),    # abaxial-denser
    ]


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
        (``params``, ``set_value``, ...) always take precedence; a param whose name
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
                    f"set_value({name!r}, {field!r}, ...): {field!r} is not an existing "
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
        increase the stele thickness (SteleDicotParams.thickness >= 1.0 is recommended
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
            SecondaryCambiumParams(),
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
        increase the stele thickness (SteleDicotParams.thickness >= 1.0 is recommended
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
            SecondaryCambiumParams(),
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
        scattered-bundle placement is built by ``MonocotStemAnatomy``.
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
            CortexParams(n_layers=5),
        ])

    @classmethod
    def for_dicot_stem(cls) -> "OrganInputData":
        """Dicot stem preset (eustele).

        A central pith ringed by discrete collateral vascular bundles (xylem
        inner / phloem outer / cambium between), then cortex + epidermis.  The
        bundle-ring placement is built by ``DicotStemAnatomy``.
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
            # Primary growth by default: fascicular cambium only. Flip
            # secondary_growth.value True for the full secondary growth (a closed
            # secondary cambium ring producing secondary xylem inward + secondary
            # phloem outward, with parenchyma rays between the bundle positions).
            DicotSecondaryGrowthParams(value=False),
            # Radius-described (like the root): the secondary cambium sits at
            # radius_valley_side mm from the centre; the secondary-xylem annulus is
            # [primary-ring radius (pith thickness/2 = 0.4) .. 0.75], so the stem
            # grows outward by 0.35 mm + the secondary phloem.
            SecondaryCambiumParams(shape="circle", n_layers=2, radius_valley_side=0.75),
            # Stem secondary xylem/phloem reuse the root's param sets, tuned smaller
            # for a stem (used only when secondary_growth.value is True).
            DicotSecondaryXylemParams(
                prop_stele=0.85, flare_angle=30.0, n_ring=3,
                vessel_diameter=0.04, vessel_diameter_min=0.012,
                vessel_diameter_sd=0.004, prop_vessel_ring=0.1,
                cell_diameter=0.01, cell_width=0.01,
                parenchyma_diameter=0.01, parenchyma_width=0.01,
            ),
            # Secondary-xylem medullar rays (radial parenchyma files cutting the
            # xylem sectors); density can grow outward via n_medullar_rate.
            DicotMedularRaysParams(
                n_medullar=16, base_width=0.01, cell_diameter=0.01,
                cell_width=0.01, allow_non_vascular=False,
            ),
            DicotSecondaryPhloemParams(
                height=0.08, top_width=0.04, alive_distance=0.05,
                sieve_diameter=0.014, sieve_diameter_min=0.008,
                companion_diameter=0.006, companion_width=0.006,
                parenchyma_diameter=0.01, parenchyma_width=0.01,
            ),
            VascularBundleParams(
                bundle_type="collateral", has_cambium=True,
                xylem_layout="files", sheath="none", n_bundles=8,
                width=0.13, height=0.2, prop_vessel=0.7,
                ring_shape="circle",
            ),
            InterCellularSpacesParams(tissue=["cortex"], smoothness=0.05),
            AerenchymaParams(tissue="cortex", aerenchyma_proportion=0.0),
            EpidermisParams(),
            CortexParams(n_layers=3),
        ])

    @classmethod
    def for_dicot_stem_continuous(cls) -> "OrganInputData":
        """Dicot stem preset with a continuous (non-fascicular) vascular cylinder.

        The same pith / cortex / epidermis as :meth:`for_dicot_stem`, but the
        discrete bundle ring is replaced by an uninterrupted cylinder of xylem
        (endarch) / cambium / phloem, built by ``ContinuousDicotStemAnatomy`` from
        the ``vascular_cylinder`` spec.  The ``vascular_bundle`` spec is still
        present — its ``arrangement="continuous"`` is what selects the cylinder —
        but its bundle-ring fields are ignored.  Set ``vascular_cylinder.xylem_layout``
        to ``"files"`` to force the xylem vessels into radial files (lines).
        """
        return cls(params=[
            PlantTypeParams(value=2, organ="stem"),
            PithParams(),
            DicotXylemParams(vessel_diameter=0.045, vessel_diameter_min=0.012,
                             vessel_diameter_sd=0.004, gradient_inflection=0.5),
            DicotPhloemParams(),
            DicotCambiumParams(),
            DicotSecondaryGrowthParams(value=False),
            VascularBundleParams(arrangement="continuous"),
            VascularCylinderParams(
                xylem_thickness=0.13, phloem_thickness=0.055,
                ring_shape="circle", xylem_layout="packed", prop_vessel=0.6,
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
    def for_monocot_leaf(cls) -> "OrganInputData":
        """Monocot leaf preset: spongy / palisade / spongy mesophyll, an even row of
        transverse 'face' veins (xylem adaxial), amphistomatous.  Built by
        ``MonocotLeafAnatomy`` (via ``LeafAnatomy``)."""
        return cls(params=default_monocot_leaf_params())

    @classmethod
    def for_dicot_leaf(cls) -> "OrganInputData":
        """Dicot leaf preset: dorsiventral palisade (adaxial) / spongy (abaxial), a
        central midrib + scattered minor collateral veins, stomata denser abaxial.
        Built by ``DicotLeafAnatomy`` (via ``LeafAnatomy``)."""
        return cls(params=default_dicot_leaf_params())

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
