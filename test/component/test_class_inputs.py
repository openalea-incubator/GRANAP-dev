"""
The instructions an architecture model and a scenario give GRANAP's anatomy component (devplan_metafspm.md, §57):
a plant's granap_overrides (its scenario) parametrize its anatomies, and a class input bins a SubOrgan variable into
the class and sets a GRANAP field from it. One component instance serves both plants.
"""
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.scene.population import build_population

from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy, default_class_inputs
from openalea.granap.metafspm.library import AnatomyLibrary, ClassInput

from roots import _axis


class TwoRoots(StructuralComponent):
    """Each plant: a root of two segments, the second with a cortex_layers given by the 'architecture'."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        segments = _axis(g, plant, 2, 0.55e-3)
        g.property("cortex_layers")[segments[0]] = 4
        g.property("cortex_layers")[segments[1]] = 2


def _planting(*overrides):
    return pd.DataFrame([dict(plant=f"p{i}", model=None, x=float(i), y=0., z=0., rotation=0.,
                              scenario={"parameters": {"granap_overrides": o} if o else {}})
                         for i, o in enumerate(overrides)])


def test_plants_parametrized_differently_get_their_own_templates_and_alike_share():
    class Anatomy(GranapAnatomy):
        library = AnatomyLibrary()
    g, plants = build_population(_planting({"cortex": {"n_layers": 3}}, {"cortex": {"n_layers": 3}}, None),
                                 initiators=(TwoRoots, Anatomy))
    template = g.property("anatomy_template")
    per_plant = [{template[s] for s in g.components_at_scale(p, scale=g.scales.SubOrgan)} for p in plants]
    assert per_plant[0] == per_plant[1] != per_plant[2]          # alike share, the default plant has its own
    assert len(Anatomy.library.templates) == 2


def test_a_class_input_sets_a_granap_field_from_an_architecture_variable():
    layers = ClassInput("cortex_layers", width=1., target=("cortex", "n_layers"))

    class Anatomy(GranapAnatomy):
        library = AnatomyLibrary()

        @classmethod
        def organs(cls):
            return {"RootSegment": OrganAnatomy(cls.library, "radius",
                                                class_inputs=default_class_inputs("radius") + (layers,))}

    g, plants = build_population(_planting(None), initiators=(TwoRoots, Anatomy))
    segments = sorted(int(s) for s in g.property("anatomy_class"))
    assert [g.property("anatomy_class")[s][-1] for s in segments] == [4., 2.]
    assert len(Anatomy.library.templates) == 2
    cortex = {key[-1]: sum(c.type == "cortex" for c in t.organ.all_cells.cells)
              for key, t in Anatomy.library.templates.items()}
    assert cortex[4.] > cortex[2.] > 0                            # the field set from the class input
