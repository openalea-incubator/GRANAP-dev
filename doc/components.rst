==========
Components
==========

GRANAP's components, rendered from their declarations (their variables by kind, their schedule). The rules are in
:doc:`equations`.

**Drawing the sections.** ``GranapAnatomy.display_triangles()`` draws each segment's section flat, across the middle
of its segment and perpendicular to it. Each protoplast is its cell's polygon, triangulated once per template. The
segment is placed by its ends (``x1`` ... ``z2``, optional inputs: on the DataStructure only when the architecture
gives them); otherwise, as with ``RootAxis`` and ``Needle``, along a straight axis from the plant's position. These shapes are for display only: metafspm's
encoding3d export writes them as a mesh of their own, which ParaView colours by any Compartment variable (tissue, water
potential, ...). Light models keep the architecture's triangles. They are computed when asked (a frame written), not
at each step.

**Reduced sections in frames.** A reduced section (the plant model's reduction, e.g. MECHA's ``scenario_reduction``:
only its conduits, its soil node and a blade's vapour path in the DataStructure) is drawn whole, in one of two ways
(metafspm's ``display=`` of ``SceneRecorder``, ``to_frame``, ``scene_to_frame``):

- ``"instances"`` (the default): ``GranapAnatomy.display_instances(owners)`` gives each full template's protoplasts
  once, flat in the section's own plane (x, y in m), each triangle owned by its protoplast's row in the full
  template, and each reduced segment's 4×4 transform (its columns the section's two axes, the segment's axis and its
  middle: where ``display_triangles`` draws it). The frame writes an instance set ``GranapAnatomy``: per segment its
  transform and its kept nodes' ``psi``; once per template and per archetype (shared parts, ``frames/parts``), the shape
  with the template's protoplasts and walls (tissue, ...) and the archetype's recovery operator; ``psi`` on every
  protoplast is a derived field, rebuilt by encoding3d (``encoding3d.evaluate``) or ParaView (Select at scale) for
  the sections chosen. A display frame of use case 1 (one plant drawn, day 3) is 0.8 MB instead of 33 MB.
- ``"full"``: the interior's protoplasts owned by their display ids (metafspm's ``solve.reduction.display_ids``),
  which the frame writes as display rows of the Compartments (``metafspm_display_only``), their values the reduced
  system's unknown recovered (MECHA's ``psi``) and their template's properties (tissue, cell area, ...), NaN
  otherwise; such a frame is as large as the full anatomy's.

The DataStructure stays reduced either way.

**Templates and checkpoints.** The anatomy templates are the population's (metafspm's registry of its MPG,
``mtg.objects("anatomy_templates")``, ``data_structure.objects`` on its DataStructure: no module-level registry, two Scenes in one
process have their own): each keyed by its index, a number of its content (``library.template_index``: the organ
class, its parameters, the seed and the class key, the same in every population and process), resolved by
``template_by_index(mtg, index)``. A Scene's checkpoint saves them with the MPG, each once (a shared part, without its
generated organ: ``saved_template``); a run resumed in a new process (``Scene.load_checkpoint``) has them back: the
segments' sections are drawn from them and a segment of a class the checkpoint has gets its template. The presets'
libraries and the derived tables (the vessels' pairs, a reduced section's interior protoplasts) are the MPG's caches
(``saved=False``), a template's triangles kept on it.

**The kinds of sections.** Each kind is a scenario key, a Plant parameter with GRANAP's sections and fields on top
of its preset; the kinds other than the roots (segments and apices) are generated when their key is given:

================= ===================================================== ===================== ================
key               sections                                              default preset        class input
================= ===================================================== ===================== ================
root_anatomy      root segments                                         ``for_root``          ``radius``
root_apex_anatomy root apices (label ``RootApex``: an axis' tip,        root_anatomy's        ``radius``
                  ArchiSimple's), on top of ``root_anatomy``
stem_anatomy      stem elements                                         ``for_monocot_stem``  ``stem_radius``
leaf_anatomy      leaf elements                                         ``for_monocot_leaf``  (one class)
sheath_anatomy    the stem elements of a Sheath organ (Adel's)          ``for_monocot_leaf``, (one class)
                                                                        bent round the stem
seed_anatomy      the seed (label ``Seed``: the seminal roots' bearer)  ``for_monocot_stem``  ``stem_radius``
================= ===================================================== ===================== ================

The seed is, for now, a conducting node of the stem's anatomy: its preset is the stems', and its class input theirs,
``stem_radius``, which the model building the seed writes on it (one input, one translator link, for the stems and
the seed). A wheat root has its own preset, ``for_wheat_root`` (``{"root_anatomy": {"preset": "for_wheat_root"}}``):
the monocot root with the cell and vessel sizes of GRANAP's wheat example (the cultivar Salmone's section, 0.41 mm in
radius); its docstring lists the values and those to calibrate (the nodal roots' among them).

**Root apices.** An architecture's axis tip (``RootApex``, ArchiSimple's: the part of the root beyond its last
segment, which ArchiSimple cuts into segments as it grows) is a root section: by default it gets the root segments'
anatomy of its class (``root_anatomy``, the explicit ``root``), its own scenario key, ``root_apex_anatomy``, on top of
it; its soil node with the segments' (``soil_node``); it is joined to its parent segment as a segment is. An apex
without length (a primordium, a new tip) has none until it grows. When the architecture relabels an apex a segment
(ArchiSimple cutting it off the tip), its anatomy is kept if the segments' kind gives it the same (the same template,
reduced or not as before), else replaced. The meristem is not represented: an apex is a root section of its
class, its maturity its barriers' (MECHA's ``Development``, by its distance from the tip; at barrier 0 MECHA's xylem is
immature, without xylem junction: an apex there takes up almost nothing, its only axial path the protophloem's).

**A reduction selects its sections by their labels.** A plant model's reduction (metafspm's ``solve.reduction``)
asks of each section its filter, ``Reduction(where=)`` (``Reduction.reduces(mtg, segment)``): its SubOrgan label
(``RootSegment``, ``RootApex``, ``StemElement``, ``LeafElement``, ``Seed``: the labels GRANAP gives an anatomy,
``REDUCIBLE_LABELS``) and its organ's (``"Organ.label"``): a sheath's elements are ``StemElement``\ s, reduced with the
stems, or apart from them, ``{"label": "StemElement", "Organ.label": "StemInternode"}``. GRANAP warns (once) of a
label it gives no anatomy. A section whose kind changes keeps its anatomy when it would be reduced as its instance
is (``Reduction.is_reduced``).

**Its named entity sets.** ``GranapAnatomy.entity_sets``: ``soil_nodes`` (a root section's soil node, its one soil
boundary), ``conduits`` (the protoplasts of families 0 and 1, which the junctions join) and ``leaf_surfaces``.
Assemblies select by these names, not by the codes: a plant model's mask of the roots' soil nodes,
``data_structure.define_mask("root_surface", entities="GranapAnatomy.soil_nodes")``, a reduction's ports and kept nodes,
``Reduction(ports="GranapAnatomy.conduits", keep="GranapAnatomy.soil_nodes")`` (evaluated on the section templates'
properties).
A reduction lumping kept nodes (``Reduction(merge=)``, MECHA's ``leaf_ports="bundles"``: a shoot section's conduits
per vascular bundle, by their ``pole_id`` and ``family``) keeps one conduit per group: the junctions (``vessel_pairs``,
by track id, then by position) then join the merged conduits, the bundles, of two sections.

**Junctions across sections without anatomy (a pass-through).** A SubOrgan without anatomy (Adel's
``HiddenElement``, the part of an organ hidden in the sheaths' tube; a kind the scenario does not give, e.g. a sheath
without ``sheath_anatomy``) does not cut the vessels. A section whose linked parent has no anatomy is joined to its
nearest ancestor with one (``anatomy_ancestor``), by the same pairing (``vessel_pairs``): the water goes from a stem
element through the hidden elements to the next element. The junction's length spans the elements between: half of
each section plus the lengths of the SubOrgans without anatomy (read where the architecture holds ``length``, at the
SubOrgans). The roots on the seed and on the crown's stem elements are joined root to stem, as any root on a stem.
A segment without length (Adel's internodes and sheaths before they grow) gets no anatomy until it has one: a
pass-through meanwhile, as a hidden element (a section of zero height would conduct nothing radially, its nodes cut
off from the network; MECHA refuses one); a leaf organ's surface is on its basal element with an anatomy. The
DataStructure rewires a new or replaced section's children only: the component joins it to its descendants across
the SubOrgans without anatomy itself (``_wire_pass_through``), their junctions to the ancestor they had removed.

.. metafspm-component:: openalea.granap.metafspm.component.GranapAnatomy

**The stand-in architectures** (``architectures.py``) declare the segments' variables GRANAP reads (``radius``,
``length``, ``distance_from_tip``: ``SegmentArchitecture``), shared by name once a plant model couples them with
``GranapAnatomy``; ``RootAxis`` also grows its axes at their tips (``root_growth``), by an array-style step returning
the new segments (metafspm's ``NewNodes``), whose anatomies ``GranapAnatomy`` adds in the same step.

.. metafspm-component:: openalea.granap.metafspm.architectures.RootAxis

.. metafspm-component:: openalea.granap.metafspm.architectures.Needle
