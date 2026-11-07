"""
The instructions an architecture model and a scenario give GRANAP's anatomy component:
a plant's scenario parametrizes its anatomies (root_anatomy, leaf_anatomy: {"preset": ..., section: {field: value}}),
and a class input bins a SubOrgan variable into the class and sets a GRANAP field from it. One component instance
serves both plants.
"""
from dataclasses import dataclass

import pytest

from openalea.metafspm.coupling.component import StructuralComponent, input_variable, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig
from openalea.metafspm.scene.population import Planting, planting

from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy, default_class_inputs
from openalea.granap.metafspm.library import AnatomyLibrary, ClassInput, template_registry

from roots import _axis, initiated


class TwoRoots(StructuralComponent):
    """Each plant: a root of two segments, the second with a cortex_layers given by the 'architecture'."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        segments = _axis(g, plant, 2, 0.55e-3)
        g.set_values("cortex_layers", segments[:2], [4, 2])


def _planting(*overrides, key="root_anatomy"):
    return planting([Planting(x=float(i)) for i in range(len(overrides))], None,
                    plant_scenarios=[{key: o} if o is not None else {} for o in overrides])


class OneLeafElement(StructuralComponent):
    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Shoot), (s.GrowthUnit, L.GrowthUnit.Shoot),
                             (s.Phytomer, L.Phytomer.Shoot), (s.Organ, L.Organ.Leaf)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        v = g.add_component(complex_, **PropsConfig(scale=s.SubOrgan, edge_type='/', label=L.SubOrgan.LeafElement))
        for name, value in (("radius", 1e-3), ("length", 2e-4), ("distance_from_tip", 0.)):
            g.set_values(name, [v], [value])


def test_plants_parametrized_differently_get_their_own_templates_and_alike_share():
    class Anatomy(GranapAnatomy):
        library = AnatomyLibrary()
    population = initiated(_planting({"cortex": {"n_layers": 3}}, {"cortex": {"n_layers": 3}}, None), TwoRoots,
                           Anatomy)
    g, plants = population.data_structure.structure, population.plants
    template = g.property("anatomy_template")
    per_plant = [{template[s] for s in g.components_at_scale(p, scale=g.scales.SubOrgan)} for p in plants]
    assert per_plant[0] == per_plant[1] != per_plant[2]          # alike share, the default plant has its own
    assert len(template_registry(g)) == 2                           # the population's templates


def test_a_class_input_sets_a_granap_field_from_an_architecture_variable():
    layers = ClassInput("cortex_layers", width=1., target=("cortex", "n_layers"))

    @dataclass
    class Anatomy(GranapAnatomy):
        library = AnatomyLibrary()
        root = OrganAnatomy(library, "radius", class_inputs=default_class_inputs("radius") + (layers,))
        cortex_layers: float = input_variable(unit="", description="the cortex' layers, given by the architecture",
                                              by="Architecture", scale=ScalesConfig.SubOrgan, kind="descriptor")

    population = initiated(_planting(None), TwoRoots, Anatomy)
    g, plants = population.data_structure.structure, population.plants
    segments = sorted(int(s) for s in g.property("anatomy_class"))
    assert [g.view("anatomy_class")[s][-1] for s in segments] == [4., 2.]
    assert len(template_registry(g)) == 2
    cortex = {t.key[-1]: sum(c.type == "cortex" for c in t.organ.all_cells.cells)
              for t in template_registry(g).values()}
    assert cortex[4.] > cortex[2.] > 0                            # the field set from the class input


def test_a_scenario_gives_a_kind_of_segment_its_preset():
    """leaf_anatomy's preset gives the leaf elements their organ class (for_needle: the needle), without a library;
    without the key, leaf elements get no anatomy."""
    from openalea.granap.metafspm.library import template_by_index
    from openalea.granap.needle_class import NeedleAnatomy
    population = initiated(_planting({"preset": "for_needle", "seed": 42}, None, key="leaf_anatomy"), OneLeafElement,
                           GranapAnatomy)
    g, plants = population.data_structure.structure, population.plants
    template = g.property("anatomy_template")
    with_needle, without = [[template.get(s, -1.) for s in g.components_at_scale(p, scale=g.scales.SubOrgan)]
                            for p in plants]
    assert isinstance(template_by_index(g, with_needle[0]).organ, NeedleAnatomy) and without == [-1.]


def test_a_preset_on_an_explicit_organ_raises():
    class Anatomy(GranapAnatomy):
        root = OrganAnatomy(AnatomyLibrary(), "radius")
    with pytest.raises(ValueError, match="explicit"):
        initiated(_planting({"preset": "for_dicot_root"}), TwoRoots, Anatomy)


@dataclass
class TwoStemOrgans(StructuralComponent):
    """Each plant: two stem organs of two elements each, their diameters given at the Organ scale (as Adel's)."""
    diameter: float = state_variable(unit="m", description="the stem organ's diameter", default=0.,
                                     scale=ScalesConfig.Organ, kind="descriptor")

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Shoot), (s.GrowthUnit, L.GrowthUnit.Shoot)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        for diameter in (2e-3, 3e-3):
            phytomer = g.add_component(complex_, **PropsConfig(scale=s.Phytomer, edge_type='/', label=L.Phytomer.Shoot))
            organ = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=L.Organ.StemInternode))
            g.set_values("diameter", [organ], [diameter])
            for _ in range(2):
                v = g.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                         label=L.SubOrgan.StemElement))
                for name, value in (("length", 1e-2), ("distance_from_tip", 0.)):
                    g.set_values(name, [v], [value])


def test_the_stems_are_classed_by_a_linked_stem_radius():
    """The stems' class input is GRANAP's declared stem_radius, linked in the plant model's translator from the
    architecture's organ diameter (Organ -> SubOrgan, broadcast, x 0.5): applied at initiation, before GRANAP builds
    the anatomies. An organ's elements share its class; without the link GRANAP finds no stem_radius."""
    from openalea.granap.metafspm.component import preset_library
    from openalea.metafspm.coupling.translator import Translator

    class Anatomy(GranapAnatomy):
        shoot = {"StemElement": OrganAnatomy(preset_library("for_monocot_stem", 0), "stem_radius")}

    translator = Translator().link("Anatomy", "stem_radius", "TwoStemOrgans", {"diameter": 0.5},
                                   source_scale="Organ", scale="SubOrgan", aggregation="broadcast")
    population = initiated(_planting(None), TwoStemOrgans, Anatomy, translator=translator)
    g = population.data_structure.structure
    elements = sorted(int(s) for s in g.property("anatomy_class"))
    assert [g.view("anatomy_class")[s][0] for s in elements] == [2., 2., 3., 3.]   # diameters in mm
    with pytest.raises(KeyError, match="class input 'stem_radius'"):
        initiated(_planting(None), TwoStemOrgans, Anatomy)


def test_an_undeclared_class_input_raises():
    class Anatomy(GranapAnatomy):
        root = OrganAnatomy(AnatomyLibrary(), "cortex_layers")
    with pytest.raises(ValueError, match="'cortex_layers' is not a declared input"):
        initiated(_planting(None), TwoRoots, Anatomy)


class InternodeAndSheath(StructuralComponent):
    """Each plant: an internode and a sheath, one stem element each (as Adel's: a sheath's elements are stem
    elements)."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Shoot), (s.GrowthUnit, L.GrowthUnit.Shoot), (s.Phytomer, L.Phytomer.Shoot)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        for organ_label in (L.Organ.StemInternode, L.Organ.Sheath):
            organ = g.add_component(complex_, **PropsConfig(scale=s.Organ, edge_type='/', label=organ_label))
            v = g.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/', label=L.SubOrgan.StemElement))
            for name, value in (("stem_radius", 1e-3), ("length", 1e-2), ("distance_from_tip", 0.)):
                g.set_values(name, [v], [value])


def test_a_sheath_is_a_bent_leaf_of_its_own_kind():
    """sheath_anatomy: a sheath's elements get a monocot leaf bent round the stem (fold_sag 2 mm) and its organ a
    surface; with stem_anatomy only, the internode's element gets a stem and the sheath's none."""
    from openalea.granap.leaf_class import LeafAnatomy
    from openalea.granap.metafspm.library import template_by_index
    planting_ = planting([Planting(x=0.), Planting(x=1.)], None,
                         plant_scenarios=[{"sheath_anatomy": {}}, {"stem_anatomy": {}}])
    population = initiated(planting_, InternodeAndSheath, GranapAnatomy)
    g, plants = population.data_structure.structure, population.plants
    template = g.property("anatomy_template")
    (internode, sheath), (stem, bare) = [sorted(int(v) for v in g.components_at_scale(p, scale=g.scales.SubOrgan))
                                         for p in plants]
    assert template.get(internode, -1.) == -1. and template.get(bare, -1.) == -1.
    leaf = template_by_index(g, template[sheath]).organ
    assert isinstance(leaf, LeafAnatomy) and leaf.global_params["fold_sag"] == 2.0
    assert not isinstance(template_by_index(g, template[stem]).organ, LeafAnatomy)
    organ = int(g.complex_at_scale(sheath, g.scales.Organ))
    assert organ in g.property("leaf_surface")


def test_a_plants_anatomy_field_given_dotted():
    """root_anatomy's fields are parameters of their own (parameter_table: defaults and bounds from the preset), and
    a scenario may give one dotted: "root_anatomy.cortex.n_layers"."""
    from openalea.metafspm.scene.scenarios import parameter_table
    table = parameter_table(GranapAnatomy).set_index("name")
    assert table.at["root_anatomy.cortex.n_layers", "min_value"] == 1
    population = initiated(planting([Planting(x=0.), Planting(x=1.)], None,
                                    plant_scenarios=[{"root_anatomy.cortex.n_layers": 2}, {}]), TwoRoots, GranapAnatomy)
    g, plants = population.data_structure.structure, population.plants
    template = g.property("anatomy_template")
    from openalea.granap.metafspm.library import template_by_index
    cortex = [max(sum(c.type == "cortex" for c in template_by_index(g, template[s]).organ.all_cells.cells)
                  for s in g.components_at_scale(p, scale=g.scales.SubOrgan)) for p in plants]
    assert 0 < cortex[0] < cortex[1]                              # 2 layers, the preset's 5
