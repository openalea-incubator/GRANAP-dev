==========
Components
==========

GRANAP's components, rendered from their declarations (their variables by kind, their schedule). The rules are in
:doc:`equations`.

**Drawing the sections.** ``GranapAnatomy.display_triangles()`` draws each segment's section flat, across the middle
of its segment and perpendicular to it. Each protoplast is its cell's polygon, triangulated once per template. The
segment is placed by its ends (``x1`` ... ``z2``, optional inputs: on the DataStructure only when the architecture
gives them); otherwise, as with ``RootAxis`` and ``Needle``, along a straight axis from the plant's position. These shapes are for display only: metafspm's
fspm-TF export writes them as a mesh of their own, which ParaView colours by any Compartment variable (tissue, water
potential, ...). Light models keep the architecture's triangles. They are computed when asked (a frame written), not
at each step.

.. metafspm-component:: openalea.granap.metafspm.component.GranapAnatomy

**The stand-in architectures** (``architectures.py``) declare the segments' variables GRANAP reads (``radius``,
``length``, ``distance_from_tip``: ``SegmentArchitecture``), shared by name once a plant model couples them with
``GranapAnatomy``; ``RootAxis`` also grows its axes at their tips (``root_growth``), by an array-style step returning
the new segments (metafspm's ``NewNodes``), whose anatomies ``GranapAnatomy`` adds in the same step.

.. metafspm-component:: openalea.granap.metafspm.architectures.RootAxis

.. metafspm-component:: openalea.granap.metafspm.architectures.Needle
