"""
GRANAP for users: a cross-section generated in a few lines, its tissues as tables.

    from openalea.granap.simulation import simulate, parameters

    section = simulate()                                  one root section, GRANAP's monocot root defaults
    section.summary                                       per tissue: cells, mean cell diameter, area, share
    section.totals                                        the section's radius, area, cells, vessels, air spaces
    section.cells()                                       one row per cell: tissue, area, position
    section.draw()                                        the section, coloured by tissue
    parameters()                                          the parameters: section, field, value, unit, description
    simulate({"cortex": {"n_layers": 3}})                 three cortex layers instead of five
    simulate(preset="for_dicot_root", seed=2)             another preset, other random draws

GRANAP's parameters are grouped by section of the anatomy (cortex, stele, xylem, ...): a change is written
{section: {field: value}}. Lengths are in mm, as GRANAP's; the summary gives cell sizes in µm.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.granap.input_data import OrganInputData, _field_tier

LENGTH_WORDS = ("diameter", "width", "thickness", "height", "radius", "length", "gap", "distance")
# the vessels' tissues (MECHA's xylem), counted in the totals
VESSELS = ("metaxylem", "protoxylem", "xylem")


def _unit(field: str, value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    if field.startswith("n_") or field == "value":
        return "count" if field.startswith("n_") else ""
    if "proportion" in field or "fraction" in field:
        return "0 to 1"
    return "mm" if any(word in field for word in LENGTH_WORDS) else ""


def parameters(values: dict = None, main_only: bool = False, preset: str = "for_root") -> pd.DataFrame:
    """The parameters of *preset* (an OrganInputData preset: for_root, for_dicot_root, for_monocot_stem,
    for_needle, ...), one row per field: its section, field, value (the preset's, or the one in *values*,
    {section: {field: value}}), unit, title, description, and whether it is a main parameter (GRANAP's primary
    fields: those the presets move; those first). *main_only*: the main ones alone."""
    rows = []
    for params in getattr(OrganInputData, preset)().params:
        section, fields = params.name, type(params).model_fields
        given = (values or {}).get(section, {})
        for field, info in fields.items():
            if field == "name":
                continue
            value = given.get(field, getattr(params, field))
            rows.append(dict(section=section, field=field, value=value, unit=_unit(field, value),
                             title=info.title or "", description=info.description or "",
                             main=_field_tier(type(params).__name__, field) == "primary"))
    table = pd.DataFrame(rows)
    table = pd.concat([table[table["main"]], table[~table["main"]]]).reset_index(drop=True)
    return table[table["main"]].reset_index(drop=True) if main_only else table


def _plant_model():
    """The plant model of the component alone: a stand-in root axis (one segment) and GranapAnatomy."""
    from openalea.metafspm.coupling.composite_wrapper import CompositeModel
    from openalea.metafspm.coupling.translator import Translator

    from openalea.granap.metafspm.architectures import RootAxis
    from openalea.granap.metafspm.component import GranapAnatomy, junction_rules

    class SectionAlone(CompositeModel):
        initiators = (RootAxis, GranapAnatomy)
        from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

        def __init__(self, data_structure, time_step):
            self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
            self.declare_data_and_couple_components(translator_path=Translator(), components=(self.anatomy,))

        def run(self):
            self.anatomy()

    return SectionAlone


@dataclass
class Section:
    """A generated cross-section: its *summary* (one row per tissue), its *totals*, and its *scene* (metafspm's,
    for more: its DataStructure holds every cell, wall and connection of the section)."""
    scene: object
    parameters: dict
    preset: str
    seed: int

    @property
    def data_structure(self):
        return self.scene.populations[0].data_structure

    def cells(self) -> pd.DataFrame:
        """The section's cells, one row each: its tissue, area (mm2), equivalent diameter (µm: the diameter of a
        disc of its area), centre (x, y, mm) and distance from the section's centre (mm). The air spaces (between
        cells, or aerenchyma) are rows of tissue "air space"."""
        ds = self.data_structure
        cell = np.asarray(ds.get("kind")) == 0
        area = np.asarray(ds.get("cell_area"))[cell]
        x, y = np.asarray(ds.get("section_x"))[cell], np.asarray(ds.get("section_y"))[cell]
        return pd.DataFrame({"tissue": np.asarray(ds.get("tissue"), dtype=object)[cell], "area": area,
                             "diameter": 2e3 * np.sqrt(area / np.pi), "x": x, "y": y, "distance": np.hypot(x, y)})

    @property
    def summary(self) -> pd.DataFrame:
        """One row per tissue, from the outside in: its number of cells, their mean equivalent diameter (µm), the
        tissue's area (mm2: its cells' areas) and its share of the section's cell area."""
        cells = self.cells()
        table = cells.groupby("tissue").agg(cells=("area", "size"), mean_cell_diameter=("diameter", "mean"),
                                            area=("area", "sum"), distance=("distance", "mean"))
        table["area_share"] = table["area"] / table["area"].sum()
        return table.sort_values("distance", ascending=False).drop(columns="distance").reset_index()

    @property
    def totals(self) -> pd.Series:
        """The section as a whole: its radius (mm: its outermost wall junction's distance from the centre), its
        area (mm2: of a disc of that radius), its cells (air spaces excluded), its xylem vessels and their mean
        diameter (µm), and the share of its cell area that is air space."""
        ds, cells = self.data_structure, self.cells()
        junction = np.asarray(ds.get("kind")) == 2
        radius = float(np.hypot(np.asarray(ds.get("section_x"))[junction],
                                np.asarray(ds.get("section_y"))[junction]).max())
        vessels = cells[cells["tissue"].isin(VESSELS)]
        air = cells["tissue"] == "air space"
        return pd.Series({"radius": radius, "area": np.pi * radius ** 2, "cells": int((~air).sum()),
                          "vessels": len(vessels), "mean_vessel_diameter": float(vessels["diameter"].mean()),
                          "air_share": float(cells.loc[air, "area"].sum() / cells["area"].sum())})

    def draw(self, ax=None, legend: bool = True):
        """The section's cells coloured by tissue; returns the matplotlib axes."""
        import matplotlib.pyplot as plt

        from openalea.granap.metafspm.plots import draw_section, tissue_colour

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        segment = int(self.data_structure.entity_ids("SubOrgan")[0])
        draw_section(ax, self.data_structure, segment)
        if legend:
            import matplotlib.patches as patches
            tissues = list(self.summary["tissue"])
            ax.legend(handles=[patches.Patch(color=tissue_colour(t), label=t) for t in tissues],
                      loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
        return ax


def simulate(parameters: dict = None, preset: str = "for_root", seed: int = 0) -> Section:
    """Generate one root section from *preset*'s parameters (an OrganInputData preset of a root: for_root, the
    monocot root, or for_dicot_root) changed by *parameters* ({section: {field: value}}, lengths in mm), with the
    random draws of *seed* (the same seed gives the same section)."""
    from openalea.metafspm.scene.scene import Scene

    anatomy = {"preset": preset, "seed": int(seed), **{section: dict(fields) for section, fields in
                                                       (parameters or {}).items()}}
    table = pd.DataFrame([dict(plant="section", model=_plant_model(), x=0., y=0., z=0., rotation=0.,
                               scenario={"parameters": {"root_anatomy": anatomy}})])
    return Section(scene=Scene(table), parameters=dict(parameters or {}), preset=preset, seed=int(seed))
