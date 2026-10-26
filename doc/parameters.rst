==============================
The parameters in plain words
==============================

GRANAP's parameters are grouped by section of the anatomy, each with its fields: a change is written
``{section: {field: value}}``, e.g. ``simulate({"cortex": {"n_layers": 3}})``. Lengths are in millimetres (40 µm is
written ``0.04``). ``openalea.granap.simulation.parameters()`` lists every parameter of a preset with its value, unit
and description; the main ones of a root, with the defaults of the monocot root preset (``for_root``):

.. list-table::
   :header-rows: 1
   :widths: 22 22 40 16

   * - section
     - field
     - what it controls
     - default
   * - ``stele``
     - ``thickness``
     - diameter of the stele, the central cylinder
     - 270 µm
   * - ``stele``
     - ``cell_diameter``, ``cell_diameter_center``
     - size of the stele's cells at its edge and at its centre
     - 10 and 20 µm
   * - ``xylem``
     - ``vessel_diameter``
     - diameter of the largest vessels (metaxylem)
     - 60 µm
   * - ``xylem``
     - ``n_vascular_bundles``
     - number of metaxylem vessels
     - 5
   * - ``xylem``
     - ``n_protoxylem``
     - number of protoxylem vessels
     - 10
   * - ``phloem``
     - ``sieve_diameter``
     - diameter of the phloem's sieve tubes
     - 25 µm
   * - ``aerenchyma``
     - ``aerenchyma_proportion``
     - share of the cortex turned into air channels
     - 0.01
   * - ``epidermis``, ``exodermis``, ``cortex``, ``endodermis``, ``pericycle``
     - ``n_layers``
     - number of layers of cells
     - 1, 1, 5, 1, 1
   * - same
     - ``cell_diameter``
     - size of the cells across the ring
     - 15, 30, 40, 20, 10 µm
   * - same
     - ``cell_width``
     - size of the cells along the ring
     - 15, 30, 40, 30, 9 µm

The section's radius follows from them: half the stele's diameter plus, for each ring, its number of layers times its
cells' diameter (0.41 mm by default). A preset is a complete set of parameters: ``for_root`` (a monocot root),
``for_dicot_root``, and the stems', leaves' and needle's presets (used through the component, see the developer
tutorials). The cells' positions are drawn at random from a seed (``simulate(seed=...)``): the same seed gives the
same section.
